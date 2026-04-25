"""Incremental spatial memory mapper — concept-graphs edition.

Subscribes to the four run_env.py outputs:
    /isaacsim/Odometry        nav_msgs/Odometry    world pose + velocity
    /isaacsim/camera_torso    sensor_msgs/Image    RGB 1600×1200
    /isaacsim/depth_torso     sensor_msgs/Image    depth (m or mm)
    /isaacsim/lidar           sensor_msgs/PointCloud2  world-frame point cloud

Every UPDATE_EVERY RGB frames a map-update cycle fires in a background thread:
    1. YOLO-World → bounding boxes on the current RGB frame
    2. MobileSAM  → per-box pixel mask
    3. CLIP       → 512-d feature vector per masked crop (ViT-H-14)
    4. depth + cam_K + pose → unproject each mask → 3D point cloud in world frame
    5. spatial IoU + CLIP cosine → associate with existing objects in MapObjectList
    6. merge or create new objects
    7. periodic DBSCAN de-noise + filter + merge
    8. lidar snapshot fused into background occupancy (separate from objects)
    9. save to disk under SCENE_GRAPH_SAVE_ROOT (default: ~/.ros/scene_graph)

Requires ROS2 Jazzy sourced before running:
    source /opt/ros/jazzy/setup.bash

Public API (thread-safe, used by scene_graph_mcp.py in-process)
----------------------------------------------------------------
    mapper = IncrementalMapper()
    mapper.start()
    mapper.objects    : MapObjectList
    mapper.lock       : threading.RLock
    mapper.status()   : dict
    mapper.stop()
"""
from __future__ import annotations

import concurrent.futures
import gzip
import json
import math
import os
import pickle
import sys
import threading
import time
from typing import Any

import numpy as np

import logging
log = logging.getLogger(__name__)

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from .cg.slam_classes import MapObjectList, DetectionList
from .cg.slam_utils import denoise_objects, filter_objects, merge_objects
from .cg.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    merge_detections_to_objects,
)
from .paths import (
    DEBUG_IMAGE_PATH,
    GRAPH_PATH,
    LIDAR_PATH,
    MAP_PATH,
    ensure_save_dirs,
    resolve_mobile_sam_path,
    resolve_openclip_pretrained,
    resolve_yolo_world_path,
)
from .scene_relations import build_scene_graph, save_scene_graph

# ── Constants ─────────────────────────────────────────────────────────────────
ODOM_TOPIC = os.environ.get("SCENE_GRAPH_ODOM_TOPIC", "/isaacsim/Odometry")
RGB_TOPIC = os.environ.get("SCENE_GRAPH_RGB_TOPIC", "/isaacsim/camera_torso")
DEPTH_TOPIC = os.environ.get("SCENE_GRAPH_DEPTH_TOPIC", "/isaacsim/depth_torso")
LIDAR_TOPIC = os.environ.get("SCENE_GRAPH_LIDAR_TOPIC", "/isaacsim/lidar")

UPDATE_EVERY = int(os.environ.get("SCENE_GRAPH_UPDATE_EVERY", "20"))
RELATIONS_EVERY = int(os.environ.get("SCENE_GRAPH_RELATIONS_EVERY", "5"))
DEVICE = os.environ.get("SCENE_GRAPH_DEVICE", "cuda:0")

# Isaac Sim camera intrinsics (run_env.py defaults: 1600×1200, hfov=100°)
_W, _H    = 1600, 1200
_FX       = _W / (2.0 * math.tan(math.radians(100.0) * 0.5))
_FY       = _FX
_CX, _CY  = (_W - 1) * 0.5, (_H - 1) * 0.5
CAM_K     = np.array([[_FX, 0, _CX], [0, _FY, _CY], [0, 0, 1]], dtype=np.float64)

# Camera-to-robot-body extrinsic (matches run_env.py / mcp_server)
# point_body = R_CAM2BODY @ point_cam + T_CAM2BODY
_R_CAM2BODY = np.array([[0.0, 0.0, 1.0],
                         [-1.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0]], dtype=np.float64)
_T_CAM2BODY = np.array([0.12, 0.0, 0.18], dtype=np.float64)
_T_CAM2BODY_4x4 = np.eye(4, dtype=np.float64)
_T_CAM2BODY_4x4[:3, :3] = _R_CAM2BODY
_T_CAM2BODY_4x4[:3,  3] = _T_CAM2BODY

BG_CLASSES = ["wall", "floor", "ceiling"]

WAREHOUSE_CLASSES = [
    "shelf", "rack", "pallet", "box", "crate", "forklift",
    "person", "robot", "door", "table", "chair", "cart",
    "bin", "container", "post", "pillar", "conveyor", "wall", "floor",
]

_CFG_DEFAULTS: dict[str, Any] = {
    "device":             DEVICE,
    "match_method":       "sim_sum",
    "phys_bias":          0.0,
    "sim_threshold":      1.5,
    "obj_min_detections": 1,
    "obj_min_points":     10,
    "downsample_voxel_size": 0.025,
    "dbscan_remove_noise":   True,
    "dbscan_eps":            0.1,
    "dbscan_min_points":     10,
    "spatial_sim_type":      "overlap",
    "skip_bg":               True,
    "use_contain_number":    False,
    "contain_area_thresh":   0.95,
    "contain_mismatch_penalty": 0.5,
    "merge_overlap_thresh":    0.7,
    "merge_visual_sim_thresh": 0.7,
    "merge_text_sim_thresh":   0.7,
    "denoise_every":  5,
    "filter_every":   5,
    "merge_every":    10,
    "class_agnostic": True,
}


# ── Lazy model cache ──────────────────────────────────────────────────────────

class _Models:
    det        = None
    sam        = None
    clip       = None
    prep       = None
    tok        = None
    ready      = False
    cfg_device = DEVICE

    @classmethod
    def load(cls, device: str = DEVICE):
        if cls.ready:
            return
        log.info("Loading detection models (first update, ~30s)…")
        from ultralytics import YOLO, SAM
        import open_clip

        yolo_world_path = resolve_yolo_world_path()
        mobile_sam_path = resolve_mobile_sam_path()
        openclip_pretrained = resolve_openclip_pretrained()

        log.info("YOLO-World weights: %s", yolo_world_path)
        log.info("MobileSAM weights: %s", mobile_sam_path)
        log.info("OpenCLIP pretrained: %s", openclip_pretrained)

        cls.det = YOLO(str(yolo_world_path))
        cls.det.set_classes(WAREHOUSE_CLASSES)
        cls.sam = SAM(str(mobile_sam_path))
        cls.clip, _, cls.prep = open_clip.create_model_and_transforms(
            "ViT-H-14",
            pretrained=openclip_pretrained,
            # This local checkpoint is a trusted full pickle-style archive.
            # PyTorch 2.6 defaults to weights_only=True, which fails on it.
            weights_only=False,
        )
        cls.clip = cls.clip.to(device).eval()
        cls.tok = open_clip.get_tokenizer("ViT-H-14")
        cls.cfg_device = device
        cls.ready = True
        log.info("Models ready.")


# ── Image / depth decoding ────────────────────────────────────────────────────

def _decode_rgb(msg: Image) -> np.ndarray | None:
    try:
        data = bytes(msg.data)
        enc  = (msg.encoding or "").lower()
        if enc == "rgb8":
            a = np.frombuffer(data, np.uint8).reshape(msg.height, msg.width, 3)
        elif enc == "bgr8":
            a = np.frombuffer(data, np.uint8).reshape(msg.height, msg.width, 3)[:, :, ::-1]
        elif enc in ("rgba8", "bgra8"):
            a = np.frombuffer(data, np.uint8).reshape(msg.height, msg.width, 4)
            a = a[:, :, :3] if enc == "rgba8" else a[:, :, 2::-1]
        else:
            a = np.frombuffer(data, np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3]
        return np.ascontiguousarray(a, dtype=np.uint8)
    except Exception:
        return None


def _decode_depth_m(msg: Image) -> np.ndarray | None:
    try:
        data = bytes(msg.data)
        enc  = (msg.encoding or "").upper()
        if enc == "16UC1":
            return np.frombuffer(data, np.uint16).reshape(
                msg.height, msg.width).astype(np.float32) * 0.001
        elif enc == "32FC1":
            return np.frombuffer(data, np.float32).reshape(msg.height, msg.width).copy()
        elif enc == "64FC1":
            return np.frombuffer(data, np.float64).reshape(
                msg.height, msg.width).astype(np.float32)
        else:
            return np.frombuffer(data, np.float32).reshape(msg.height, msg.width).copy()
    except Exception:
        return None


# ── 3D helpers ────────────────────────────────────────────────────────────────

def _quat_to_R(qx, qy, qz, qw) -> np.ndarray:
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ], dtype=np.float64)


def _pose_to_T_c2w(pos, quat_xyzw) -> np.ndarray:
    R = _quat_to_R(*quat_xyzw)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = np.array(pos, dtype=np.float64)
    return T


def _unproject_mask(depth_m, mask, cam_K, T_c2w, max_depth=20.0) -> np.ndarray:
    fx, fy = cam_K[0, 0], cam_K[1, 1]
    cx, cy = cam_K[0, 2], cam_K[1, 2]
    valid  = mask & (depth_m > 0.05) & (depth_m < max_depth)
    if valid.sum() == 0:
        return np.zeros((0, 3), np.float32)
    rows, cols = np.where(valid)
    d = depth_m[rows, cols].astype(np.float64)
    pts_cam = np.stack(
        [(cols - cx) * d / fx, (rows - cy) * d / fy, d, np.ones_like(d)], axis=1
    )
    return (T_c2w @ pts_cam.T).T[:, :3].astype(np.float32)


def _clip_feature(image_rgb, mask, xyxy) -> np.ndarray:
    import torch
    from PIL import Image as PILImage
    x1, y1, x2, y2 = (
        max(0, int(xyxy[0])), max(0, int(xyxy[1])),
        min(image_rgb.shape[1], int(xyxy[2])), min(image_rgb.shape[0], int(xyxy[3])),
    )
    crop = image_rgb[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return np.zeros(512, np.float32)
    crop[~mask[y1:y2, x1:x2]] = 0
    inp = _Models.prep(PILImage.fromarray(crop)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        ft = _Models.clip.encode_image(inp)[0].cpu().numpy().astype(np.float32)
    n = float(np.linalg.norm(ft))
    return ft / n if n > 1e-9 else ft


def _build_detection_list(image_rgb, depth_m, T_c2w,
                          bboxes, masks, confs, class_ids, class_names) -> DetectionList:
    import open3d as o3d
    import torch
    det_list = DetectionList()
    for box, mask, conf, cid, cname in zip(bboxes, masks, confs, class_ids, class_names):
        pts = _unproject_mask(depth_m, mask.astype(bool), CAM_K, T_c2w)
        log.info("    %s mask_px=%d pts_3d=%d", cname, int(mask.sum()), len(pts))
        if len(pts) < 3:
            continue
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.voxel_down_sample(voxel_size=0.05)  # downsample to ~5cm grid
        if len(pcd.points) < 3:
            continue
        try:
            bbox3d = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(pts))
        except Exception:
            continue
        ft = _clip_feature(image_rgb, mask.astype(bool), box)
        det_list.append({
            "class_id":       [int(cid)],
            "class_name":     [cname],
            "conf":           [float(conf)],
            "xyxy":           [box.tolist()],
            "clip_ft":        torch.tensor(ft, dtype=torch.float32).to(DEVICE),
            "text_ft":        torch.tensor(ft, dtype=torch.float32).to(DEVICE),
            "pcd":            pcd,
            "bbox":           bbox3d,
            "n_points":       [len(pts)],
            "num_detections": 1,
            "inst_color":     [0.5, 0.5, 0.9],
        })
    return det_list


# ── Mapper ────────────────────────────────────────────────────────────────────

class IncrementalMapper:

    def __init__(self, cfg_overrides: dict | None = None):
        from types import SimpleNamespace
        self.cfg     = SimpleNamespace(**{**_CFG_DEFAULTS, **(cfg_overrides or {})})
        self.lock    = threading.RLock()
        self.objects = MapObjectList(device=DEVICE)

        self._msg_lock  = threading.Lock()
        self._rgb_msg   = None
        self._depth_msg = None
        self._odom_pose = None
        self._lidar_pts = None

        self._rgb_count    = 0
        self._update_count = 0
        self._total_objs   = 0
        self._last_update  = 0.0

        self._executor    = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._future: concurrent.futures.Future | None = None
        self._node        = None
        self._ros_exec    = None
        self._spin_thread = None
        self._running     = False

        ensure_save_dirs()

        if MAP_PATH.exists():
            try:
                with gzip.open(MAP_PATH, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and "objects" in data:
                    self.objects.load_serializable(data["objects"])
                    log.info("Resumed map: %d objects.", len(self.objects))
            except Exception as e:
                log.warning("Could not load existing map (%s), starting fresh.", e)

    def start(self):
        if self._running:
            return
        self._running    = True
        self._node       = _MapperNode(self)
        self._ros_exec   = MultiThreadedExecutor()
        self._ros_exec.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._ros_exec.spin, daemon=True, name="mapper_spin")
        self._spin_thread.start()
        log.info("IncrementalMapper started (update every %d RGB frames).", UPDATE_EVERY)

    def stop(self):
        self._running = False
        self._executor.shutdown(wait=False)
        if self._ros_exec:
            self._ros_exec.shutdown(wait=False)
        if self._node:
            self._node.destroy_node()
        if self._spin_thread:
            self._spin_thread.join(timeout=2.0)

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _on_rgb(self, msg: Image):
        with self._msg_lock:
            self._rgb_msg    = msg
            self._rgb_count += 1
            trigger = (self._rgb_count % UPDATE_EVERY == 0)
        if trigger and self._running:
            self._maybe_trigger_update()

    def _on_depth(self, msg: Image):
        with self._msg_lock:
            self._depth_msg = msg

    def _on_odom(self, msg: Odometry):
        q = msg.pose.pose.orientation
        p = msg.pose.pose.position
        with self._msg_lock:
            self._odom_pose = (
                [p.x, p.y, p.z],
                [q.x, q.y, q.z, q.w],
            )

    def _on_lidar(self, msg: PointCloud2):
        pts = [
            (float(p[0]), float(p[1]), float(p[2]))
            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        ]
        arr = np.asarray(pts, np.float32) if pts else np.zeros((0, 3), np.float32)
        with self._msg_lock:
            self._lidar_pts = arr

    # ── update trigger ────────────────────────────────────────────────────────

    def _maybe_trigger_update(self):
        if self._future is not None and not self._future.done():
            return
        with self._msg_lock:
            rgb_msg   = self._rgb_msg
            depth_msg = self._depth_msg
            pose      = self._odom_pose
            lidar_pts = self._lidar_pts
        if rgb_msg is None or depth_msg is None or pose is None:
            log.warning("Update skipped: rgb=%s depth=%s pose=%s",
                        rgb_msg is not None, depth_msg is not None, pose is not None)
            return
        self._future = self._executor.submit(
            self._update, rgb_msg, depth_msg, pose, lidar_pts)
        self._future.add_done_callback(
            lambda f: log.error("Update crashed: %s", f.exception(), exc_info=f.exception())
            if f.exception() else None
        )

    # ── core update ───────────────────────────────────────────────────────────

    def _update(self, rgb_msg, depth_msg, pose, lidar_pts):
        t0 = time.time()
        _Models.load(DEVICE)

        image_rgb = _decode_rgb(rgb_msg)
        depth_m   = _decode_depth_m(depth_msg)
        if image_rgb is None or depth_m is None:
            return
        valid_depth = (depth_m > 0.05) & (depth_m < 20.0)
        log.info("depth: min=%.2f max=%.2f valid_pixels=%d/%d",
                 float(depth_m[valid_depth].min()) if valid_depth.any() else 0,
                 float(depth_m[valid_depth].max()) if valid_depth.any() else 0,
                 int(valid_depth.sum()), depth_m.size)

        pos, quat = pose
        T_body2world = _pose_to_T_c2w(pos, quat)
        T_c2w = T_body2world @ _T_CAM2BODY_4x4

        # 1. YOLO-World
        results = _Models.det(image_rgb, conf=0.15, verbose=False)[0]
        boxes   = results.boxes
        if len(boxes) == 0:
            log.warning("Update skipped: YOLO detected 0 boxes (img shape=%s)", image_rgb.shape)
            return
        import cv2
        _vis = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        for _i, (_xyxy, _conf, _cls) in enumerate(zip(
                boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())):
            _cname = WAREHOUSE_CLASSES[int(_cls)] if int(_cls) < len(WAREHOUSE_CLASSES) else "?"
            log.info("  box[%d] %s conf=%.3f xyxy=[%.0f,%.0f,%.0f,%.0f]",
                     _i, _cname, _conf, *_xyxy)
            x1, y1, x2, y2 = (int(v) for v in _xyxy)
            cv2.rectangle(_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(_vis, f"{_cname} {_conf:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        _out = DEBUG_IMAGE_PATH
        cv2.imwrite(str(_out), _vis)
        log.info("YOLO vis saved → %s", _out)
        bboxes      = boxes.xyxy.cpu().numpy()
        confs       = boxes.conf.cpu().numpy()
        class_ids   = boxes.cls.cpu().numpy().astype(int).tolist()
        class_names = [
            WAREHOUSE_CLASSES[i] if i < len(WAREHOUSE_CLASSES) else "unknown"
            for i in class_ids
        ]

        # 2. MobileSAM
        from .cg.model_utils import get_sam_segmentation_from_xyxy_batched
        masks = get_sam_segmentation_from_xyxy_batched(_Models.sam, image_rgb, bboxes)

        # 3. DetectionList: CLIP + 3D unproject per mask
        fg_list = _build_detection_list(
            image_rgb, depth_m, T_c2w,
            bboxes, masks, confs, class_ids, class_names,
        )
        if len(fg_list) == 0:
            log.warning("Update skipped: all %d detections had < 5 depth points", len(bboxes))
            return

        # 4. Associate + merge
        with self.lock:
            if len(self.objects) == 0:
                for det in fg_list:
                    self.objects.append(det)
            else:
                spatial_sim = compute_spatial_similarities(self.cfg, fg_list, self.objects)
                visual_sim  = compute_visual_similarities(self.cfg, fg_list, self.objects)
                agg_sim     = aggregate_similarities(self.cfg, spatial_sim, visual_sim)
                agg_sim[agg_sim < self.cfg.sim_threshold] = float("-inf")
                self.objects = merge_detections_to_objects(
                    self.cfg, fg_list, self.objects, agg_sim)

            self._update_count += 1
            uc = self._update_count

            # 5. Periodic post-processing
            if uc % self.cfg.denoise_every == 0:
                self.objects = denoise_objects(self.cfg, self.objects)
            if uc % self.cfg.filter_every == 0:
                self.objects = filter_objects(self.cfg, self.objects)
            if uc % self.cfg.merge_every == 0:
                self.objects = merge_objects(self.cfg, self.objects)

            self._total_objs = len(self.objects)

        # 6. Lidar background accumulation
        if lidar_pts is not None and len(lidar_pts) > 0:
            self._fuse_lidar(lidar_pts)

        # 7. Rebuild scene graph relations
        if uc % RELATIONS_EVERY == 0:
            with self.lock:
                graph = build_scene_graph(self.objects)
            save_scene_graph(graph, GRAPH_PATH)
            log.info("Scene graph: %d nodes, %d edges",
                     len(graph["nodes"]), len(graph["edges"]))

        # 8. Persist
        self._save_map()
        self._last_update = time.time()

        log.info("Update #%d | frames=%d | dets=%d | objs=%d | %.2fs",
                 uc, self._rgb_count, len(fg_list), self._total_objs, time.time() - t0)

    # ── persistence ──────────────────────────────────────────────────────────

    def _save_map(self):
        with self.lock:
            data = {
                "objects":      self.objects.to_serializable(),
                "update_count": self._update_count,
                "rgb_frames":   self._rgb_count,
                "class_names":  WAREHOUSE_CLASSES,
                "saved_at":     time.time(),
            }
        with gzip.open(MAP_PATH, "wb") as f:
            pickle.dump(data, f)

    def _fuse_lidar(self, pts: np.ndarray):
        existing = np.zeros((0, 3), np.float32)
        if LIDAR_PATH.exists():
            try:
                with gzip.open(LIDAR_PATH, "rb") as f:
                    existing = pickle.load(f)
            except Exception:
                pass
        merged = np.concatenate([existing, pts], axis=0)
        if len(merged) > 500_000:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(merged.astype(np.float64))
            pcd = pcd.voxel_down_sample(0.05)
            merged = np.asarray(pcd.points, np.float32)
        with gzip.open(LIDAR_PATH, "wb") as f:
            pickle.dump(merged, f)

    # ── public ───────────────────────────────────────────────────────────────

    def status(self) -> dict:
        with self.lock:
            n = len(self.objects)
        busy = self._future is not None and not self._future.done()
        return {
            "rgb_frames_received": self._rgb_count,
            "update_cycles":       self._update_count,
            "objects_in_map":      n,
            "update_running":      busy,
            "map_path":            str(MAP_PATH),
            "last_update_ago_s":   round(time.time() - self._last_update, 1)
                                   if self._last_update else None,
        }

    def get_lidar_cloud(self) -> np.ndarray | None:
        if not LIDAR_PATH.exists():
            return None
        try:
            with gzip.open(LIDAR_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None


# ── ROS2 node ─────────────────────────────────────────────────────────────────

class _MapperNode(Node):
    def __init__(self, mapper: IncrementalMapper):
        super().__init__("incremental_mapper")
        m = mapper
        self.create_subscription(Odometry,    ODOM_TOPIC,  m._on_odom,  10)
        self.create_subscription(Image,       RGB_TOPIC,   m._on_rgb,   qos_profile_sensor_data)
        self.create_subscription(Image,       DEPTH_TOPIC, m._on_depth, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, LIDAR_TOPIC, m._on_lidar, qos_profile_sensor_data)
        log.info("Subscribed to %s %s %s %s",
                 ODOM_TOPIC, RGB_TOPIC, DEPTH_TOPIC, LIDAR_TOPIC)


# ── Standalone ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [mapper] %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    rclpy.init(args=None)
    mapper = IncrementalMapper()
    mapper.start()
    log.info("Running standalone. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(10)
            log.info("Status: %s", json.dumps(mapper.status()))
    except KeyboardInterrupt:
        pass
    finally:
        mapper.stop()
        rclpy.shutdown()
