"""ROS navigation MCP server backed by the official FastMCP library."""

from __future__ import annotations

import base64
import json
import math
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from mcp import types
from mcp.server.fastmcp import FastMCP
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, Joy, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Empty
from tf2_ros import Buffer, TransformListener


NAVIGATION_FRAME = "map"
CAMERA_OPTICAL_FRAME = "camera"
MCP_SERVER_HOST = "127.0.0.1"
MCP_SERVER_PORT = 11451
MCP_SERVER_PATH = "/mcp"
MCP_SERVER_TRANSPORT = "streamable-http"
CAMERA_TO_ROBOT_ROTATION = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
CAMERA_TO_ROBOT_TRANSLATION = np.array([0.12, 0.0, 0.18], dtype=np.float64)
CAMERA_IMAGE_WIDTH = 1600
CAMERA_IMAGE_HEIGHT = 1200
CAMERA_HORIZONTAL_FOV_DEG = 100.0
_ESTIMATED_CAMERA_FOCAL_PX = CAMERA_IMAGE_WIDTH / (
    2.0 * math.tan(math.radians(CAMERA_HORIZONTAL_FOV_DEG) * 0.5)
)
# Estimated from scripts/run_env.py defaults: 1600x1200, horizontal FOV 100 deg,
# centered principal point, square pixels, no distortion.
CAMERA_INTRINSICS = {
    "fx": _ESTIMATED_CAMERA_FOCAL_PX,
    "fy": _ESTIMATED_CAMERA_FOCAL_PX,
    "cx": (CAMERA_IMAGE_WIDTH - 1) * 0.5,
    "cy": (CAMERA_IMAGE_HEIGHT - 1) * 0.5,
    "width": CAMERA_IMAGE_WIDTH,
    "height": CAMERA_IMAGE_HEIGHT,
}
DEPTH_SCALE_BY_ENCODING = {
    "16UC1": 0.001,
    "32FC1": 1.0,
    "64FC1": 1.0,
}
DEFAULT_DEPTH_SCALE = 1.0


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _default_data_dir() -> Path:
    return Path.home() / ".ros" / "mcp_server"


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _text_content(text: str) -> types.TextContent:
    return types.TextContent(type="text", text=text)


def _image_content(raw: bytes, mime_type: str = "image/png") -> types.ImageContent:
    return types.ImageContent(
        type="image",
        data=base64.b64encode(raw).decode("ascii"),
        mimeType=mime_type,
    )


def _intrinsics_payload() -> dict[str, Any]:
    return {
        "fx": float(CAMERA_INTRINSICS["fx"]),
        "fy": float(CAMERA_INTRINSICS["fy"]),
        "cx": float(CAMERA_INTRINSICS["cx"]),
        "cy": float(CAMERA_INTRINSICS["cy"]),
        "width": int(CAMERA_INTRINSICS["width"]),
        "height": int(CAMERA_INTRINSICS["height"]),
        "frame_id": CAMERA_OPTICAL_FRAME,
    }


def _rotation_matrix_from_quaternion(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _depth_to_meters(raw_depth: Any, encoding: str | None) -> float:
    scale = DEPTH_SCALE_BY_ENCODING.get((encoding or "").upper(), DEFAULT_DEPTH_SCALE)
    return float(raw_depth) * scale


def _pose_record_from_state(pose: "PoseState") -> dict[str, Any]:
    return {
        "x": float(pose.x),
        "y": float(pose.y),
        "z": float(pose.z),
        "yaw_deg": float(math.degrees(pose.yaw)),
        "stamp_sec": float(pose.stamp_sec),
        "frame_id": pose.frame_id,
    }


def _camera_point_to_robot_relative(point_cam: np.ndarray) -> dict[str, float]:
    point_rel = CAMERA_TO_ROBOT_ROTATION @ point_cam + CAMERA_TO_ROBOT_TRANSLATION
    return {
        "x": float(point_rel[0]),
        "y": float(point_rel[1]),
        "z": float(point_rel[2]),
    }


def _robot_relative_to_world(pose: dict[str, Any], point_rel: dict[str, float]) -> dict[str, float]:
    yaw_deg = float(pose.get("yaw_deg", 0.0))
    yaw = math.radians(yaw_deg)
    x_rel = float(point_rel["x"])
    y_rel = float(point_rel["y"])
    z_rel = float(point_rel["z"])
    return {
        "x": float(pose["x"] + math.cos(yaw) * x_rel - math.sin(yaw) * y_rel),
        "y": float(pose["y"] + math.sin(yaw) * x_rel + math.cos(yaw) * y_rel),
        "z": float(pose["z"] + z_rel),
    }


@dataclass
class PoseState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    stamp_sec: float = 0.0
    frame_id: str = NAVIGATION_FRAME


@dataclass
class NavigationTask:
    id: str
    kind: str
    status: str
    created_at: float
    target: dict[str, Any]
    message: str = ""
    result: str | None = None


class RosMcpNode(Node):
    """ROS integration layer for MCP tools."""

    def __init__(self) -> None:
        super().__init__("ros_mcp_server")
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.pose = PoseState()
        self.pose_history: list[tuple[float, float]] = []
        self.latest_rgb_msg: Image | None = None
        self.latest_depth_msg: Image | None = None
        self.latest_cloud: np.ndarray | None = None
        self.latest_reach_goal = False
        self.current_task: NavigationTask | None = None
        self.turn_cancel = threading.Event()
        self.turn_thread: threading.Thread | None = None
        self.capture_root_dir = _default_data_dir() / "captures"
        self.capture_root_dir.mkdir(parents=True, exist_ok=True)
        self.capture_dir = self.capture_root_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.capture_counter = 0
        self.waypoint_path = _default_data_dir() / "waypoints.json"
        self.waypoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        self.declare_parameter("pose_topic", "/state_estimation")
        self.declare_parameter("goal_topic", "/goal_point")
        self.declare_parameter("waypoint_topic", "/way_point")
        self.declare_parameter("joy_topic", "/joy")
        self.declare_parameter("reach_goal_topic", "/far_reach_goal_status")
        self.declare_parameter("reset_visibility_graph_topic", "/reset_visibility_graph")
        self.declare_parameter("scan_topic", "/registered_scan")
        self.declare_parameter("rgb_topic", "/isaacsim/camera_torso")
        self.declare_parameter("depth_topic", "/isaacsim/depth_torso")

        pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value
        goal_topic = self.get_parameter("goal_topic").get_parameter_value().string_value
        waypoint_topic = self.get_parameter("waypoint_topic").get_parameter_value().string_value
        joy_topic = self.get_parameter("joy_topic").get_parameter_value().string_value
        reach_goal_topic = self.get_parameter("reach_goal_topic").get_parameter_value().string_value
        reset_topic = self.get_parameter("reset_visibility_graph_topic").get_parameter_value().string_value
        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value

        self.create_subscription(Odometry, pose_topic, self._pose_cb, 10)
        self.create_subscription(Bool, reach_goal_topic, self._reach_goal_cb, 10)
        self.create_subscription(PointCloud2, scan_topic, self._cloud_cb, qos_profile_sensor_data)
        self.create_subscription(Image, rgb_topic, self._rgb_cb, qos_profile_sensor_data)
        self.create_subscription(Image, depth_topic, self._depth_cb, qos_profile_sensor_data)

        self.goal_pub = self.create_publisher(PointStamped, goal_topic, 10)
        self.waypoint_pub = self.create_publisher(PointStamped, waypoint_topic, 10)
        self.joy_pub = self.create_publisher(Joy, joy_topic, 10)
        self.reset_pub = self.create_publisher(Empty, reset_topic, 10)

    def _pose_cb(self, msg: Odometry) -> None:
        quat = msg.pose.pose.orientation
        pose = PoseState(
            x=float(msg.pose.pose.position.x),
            y=float(msg.pose.pose.position.y),
            z=float(msg.pose.pose.position.z),
            yaw=_yaw_from_quaternion(quat.x, quat.y, quat.z, quat.w),
            stamp_sec=float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9,
            frame_id=msg.header.frame_id or NAVIGATION_FRAME,
        )
        with self.lock:
            self.pose = pose
            self.pose_history.append((pose.x, pose.y))
            if len(self.pose_history) > 2000:
                self.pose_history = self.pose_history[-2000:]

    def _reach_goal_cb(self, msg: Bool) -> None:
        with self.lock:
            self.latest_reach_goal = bool(msg.data)
            if msg.data and self.current_task and self.current_task.kind == "navigate":
                self.current_task.status = "completed"
                self.current_task.result = "reached_goal"
                self.current_task.message = "Navigation goal reached."

    def _cloud_cb(self, msg: PointCloud2) -> None:
        points = []
        for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append((float(p[0]), float(p[1]), float(p[2])))
        arr = np.asarray(points, dtype=np.float32) if points else np.zeros((0, 3), dtype=np.float32)
        with self.lock:
            self.latest_cloud = arr

    def _rgb_cb(self, msg: Image) -> None:
        with self.lock:
            self.latest_rgb_msg = msg

    def _depth_cb(self, msg: Image) -> None:
        with self.lock:
            self.latest_depth_msg = msg

    def _resume_autonomy_pulse(self) -> None:
        joy = Joy()
        joy.axes = [0.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        joy.buttons = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        joy.header.stamp = self.get_clock().now().to_msg()
        joy.header.frame_id = "ros_mcp_server"
        self.joy_pub.publish(joy)

    def _manual_joy(self, angular: float = 0.0, linear: float = 0.0) -> None:
        joy = Joy()
        joy.axes = [0.0, 0.0, 1.0, float(angular), float(linear), 1.0, 0.0, 0.0]
        joy.buttons = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        joy.header.stamp = self.get_clock().now().to_msg()
        joy.header.frame_id = "ros_mcp_server"
        self.joy_pub.publish(joy)

    def _publish_goal(self, x: float, y: float, z: float, *, use_waypoint_topic: bool = False) -> None:
        self._resume_autonomy_pulse()
        msg = PointStamped()
        msg.header.frame_id = NAVIGATION_FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = float(x)
        msg.point.y = float(y)
        msg.point.z = float(z)
        pub = self.waypoint_pub if use_waypoint_topic else self.goal_pub
        pub.publish(msg)
        time.sleep(0.01)
        pub.publish(msg)

    def _lookup_map_to_camera(self, camera_frame: str, stamp: Time) -> dict[str, Any]:
        try:
            transform = self.tf_buffer.lookup_transform(
                NAVIGATION_FRAME,
                camera_frame,
                stamp,
                timeout=Duration(seconds=0.2),
            )
        except Exception:
            transform = self.tf_buffer.lookup_transform(
                NAVIGATION_FRAME,
                camera_frame,
                Time(),
                timeout=Duration(seconds=0.2),
            )

        return {
            "translation": {
                "x": float(transform.transform.translation.x),
                "y": float(transform.transform.translation.y),
                "z": float(transform.transform.translation.z),
            },
            "rotation": {
                "x": float(transform.transform.rotation.x),
                "y": float(transform.transform.rotation.y),
                "z": float(transform.transform.rotation.z),
                "w": float(transform.transform.rotation.w),
            },
        }

    def get_pose_dict(self) -> dict[str, Any]:
        with self.lock:
            return _pose_record_from_state(self.pose)

    def _load_waypoints(self) -> dict[str, dict[str, Any]]:
        if not self.waypoint_path.exists():
            return {}
        try:
            return json.loads(self.waypoint_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_waypoints(self, data: dict[str, dict[str, Any]]) -> None:
        self.waypoint_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def waypoint_list(self) -> dict[str, dict[str, Any]]:
        with self.lock:
            return self._load_waypoints()

    def waypoint_create(
        self,
        name: str,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            data = self._load_waypoints()
            pose = self.pose
            entry = {
                "name": name,
                "x": float(pose.x if x is None else x),
                "y": float(pose.y if y is None else y),
                "z": float(pose.z if z is None else z),
                "created_at": time.time(),
            }
            data[name] = entry
            self._save_waypoints(data)
            return entry

    def waypoint_get(self, name: str) -> dict[str, Any] | None:
        return self.waypoint_list().get(name)

    def waypoint_update(self, name: str, fields: dict[str, Any]) -> dict[str, Any] | None:
        with self.lock:
            data = self._load_waypoints()
            if name not in data:
                return None
            data[name].update({k: v for k, v in fields.items() if v is not None})
            self._save_waypoints(data)
            return data[name]

    def waypoint_delete(self, name: str) -> bool:
        with self.lock:
            data = self._load_waypoints()
            if name not in data:
                return False
            del data[name]
            self._save_waypoints(data)
            return True

    def _start_nav_task(self, target: dict[str, Any], message: str) -> NavigationTask:
        task = NavigationTask(
            id=uuid.uuid4().hex[:8],
            kind="navigate",
            status="running",
            created_at=time.time(),
            target=target,
            message=message,
        )
        with self.lock:
            self.latest_reach_goal = False
            self.current_task = task
        return task

    def navigate_absolute(
        self,
        x: float,
        y: float,
        z: float | None = None,
        *,
        waypoint_topic: bool = False,
    ) -> NavigationTask:
        with self.lock:
            z_value = self.pose.z if z is None else z
        task = self._start_nav_task({"x": x, "y": y, "z": z_value}, "Navigation started.")
        self._publish_goal(x, y, z_value, use_waypoint_topic=waypoint_topic)
        return task

    def navigate_relative(self, dx: float, dy: float, dz: float = 0.0) -> NavigationTask:
        with self.lock:
            pose = self.pose
        wx = pose.x + math.cos(pose.yaw) * dx - math.sin(pose.yaw) * dy
        wy = pose.y + math.sin(pose.yaw) * dx + math.cos(pose.yaw) * dy
        wz = pose.z + dz
        return self.navigate_absolute(wx, wy, wz)

    def navigate_waypoint(self, name: str) -> NavigationTask | None:
        wp = self.waypoint_get(name)
        if wp is None:
            return None
        return self.navigate_absolute(
            float(wp["x"]),
            float(wp["y"]),
            float(wp.get("z", 0.0)),
            waypoint_topic=True,
        )

    def cancel_navigation(self) -> bool:
        with self.lock:
            pose = self.pose
            task = self.current_task
            if task is not None:
                task.status = "cancelled"
                task.result = "cancelled"
                task.message = "Navigation cancelled."
        self.turn_cancel.set()
        self._manual_joy(0.0, 0.0)
        self._publish_goal(pose.x, pose.y, pose.z)
        return task is not None

    def wait_navigation(self, timeout_sec: float) -> dict[str, Any]:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            with self.lock:
                task = self.current_task
                if task is None:
                    return {"status": "ended", "result": "no_active_task"}
                if task.status in {"completed", "cancelled", "failed"}:
                    return {"status": "ended", "result": task.result or task.status, "task_id": task.id}
            time.sleep(0.1)
        return {"status": "timeout"}

    def _turn_loop(self, target_yaw: float, relative: bool) -> None:
        with self.lock:
            current = self.pose.yaw
            if relative:
                target = _normalize_angle(current + target_yaw)
            else:
                target = _normalize_angle(target_yaw)
            task = NavigationTask(
                id=uuid.uuid4().hex[:8],
                kind="turn",
                status="running",
                created_at=time.time(),
                target={"yaw_deg": float(math.degrees(target)), "relative": relative},
                message="Turning started.",
            )
            self.current_task = task

        stable_hits = 0
        while rclpy.ok() and not self.turn_cancel.is_set():
            with self.lock:
                yaw = self.pose.yaw
                active = self.current_task
            if active is None or active.id != task.id:
                return
            error = _normalize_angle(target - yaw)
            if abs(error) < math.radians(3.0):
                stable_hits += 1
                self._manual_joy(0.0, 0.0)
                if stable_hits >= 3:
                    with self.lock:
                        if self.current_task and self.current_task.id == task.id:
                            self.current_task.status = "completed"
                            self.current_task.result = "reached_yaw"
                            self.current_task.message = "Turn completed."
                    return
            else:
                stable_hits = 0
                angular = max(-1.0, min(1.0, error / math.radians(30.0)))
                self._manual_joy(angular=angular, linear=0.0)
            time.sleep(0.1)

        self._manual_joy(0.0, 0.0)
        with self.lock:
            if self.current_task and self.current_task.id == task.id:
                self.current_task.status = "cancelled"
                self.current_task.result = "cancelled"
                self.current_task.message = "Turn cancelled."

    def start_turn(self, angle_deg: float, *, relative: bool) -> NavigationTask:
        angle_rad = math.radians(float(angle_deg))
        self.turn_cancel.set()
        if self.turn_thread and self.turn_thread.is_alive():
            self.turn_thread.join(timeout=1.0)
        self.turn_cancel = threading.Event()
        self.turn_thread = threading.Thread(
            target=self._turn_loop,
            args=(angle_rad, relative),
            daemon=True,
        )
        self.turn_thread.start()
        time.sleep(0.05)
        with self.lock:
            assert self.current_task is not None
            return self.current_task

    def reset_visibility_graph(self) -> None:
        self.reset_pub.publish(Empty())

    def capture_image(self) -> tuple[str, bytes, str]:
        with self.lock:
            rgb_msg = self.latest_rgb_msg
            depth_msg = self.latest_depth_msg
            pose = _pose_record_from_state(self.pose)
            self.capture_counter += 1
            capture_id = str(self.capture_counter)

        if rgb_msg is None:
            raise RuntimeError("No RGB image received yet.")

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        rgb_path = self.capture_dir / f"{capture_id}_rgb.png"
        cv2.imwrite(str(rgb_path), rgb)

        depth_encoding = None
        if depth_msg is not None:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            np.save(self.capture_dir / f"{capture_id}_depth.npy", depth)
            depth_encoding = depth_msg.encoding or None

        meta = {
            "capture_id": capture_id,
            "capture_dir": str(self.capture_dir),
            "rgb_path": str(rgb_path),
            "pose": pose,
            "rgb_frame": rgb_msg.header.frame_id,
            "camera_frame": rgb_msg.header.frame_id or CAMERA_OPTICAL_FRAME,
            "stamp_sec": float(rgb_msg.header.stamp.sec) + float(rgb_msg.header.stamp.nanosec) * 1e-9,
            "camera_intrinsics": _intrinsics_payload(),
        }
        if depth_encoding:
            meta["depth_encoding"] = depth_encoding
        try:
            meta["map_to_camera"] = self._lookup_map_to_camera(
                rgb_msg.header.frame_id or CAMERA_OPTICAL_FRAME,
                Time.from_msg(rgb_msg.header.stamp),
            )
        except Exception:
            pass

        (self.capture_dir / f"{capture_id}_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        ok, encoded = cv2.imencode(".png", rgb)
        if not ok:
            raise RuntimeError("Failed to encode RGB image.")
        return capture_id, encoded.tobytes(), str(rgb_path)

    def estimate_pixel(self, capture_id: str, u: int, v: int, navigate: bool = False) -> dict[str, Any]:
        meta_path = self.capture_dir / f"{capture_id}_meta.json"
        depth_path = self.capture_dir / f"{capture_id}_depth.npy"
        if not meta_path.exists():
            raise RuntimeError(f"Unknown capture id: {capture_id}")
        if not depth_path.exists():
            raise RuntimeError(f"No depth saved for capture id: {capture_id}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        intrinsics = meta.get("camera_intrinsics")
        if not intrinsics:
            raise RuntimeError("No camera intrinsics saved with capture.")

        depth = np.load(depth_path)
        if v < 0 or u < 0 or v >= depth.shape[0] or u >= depth.shape[1]:
            raise RuntimeError("Pixel is out of image bounds.")

        ray_depth = _depth_to_meters(depth[v, u], meta.get("depth_encoding"))
        if not np.isfinite(ray_depth) or ray_depth <= 0:
            raise RuntimeError("Depth at the given pixel is invalid.")

        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        x_cam = (float(u) - cx) * ray_depth / fx
        y_cam = (float(v) - cy) * ray_depth / fy
        point_cam = np.array([x_cam, y_cam, ray_depth], dtype=np.float64)

        relative = _camera_point_to_robot_relative(point_cam)
        world = _robot_relative_to_world(meta["pose"], relative)
        if navigate:
            self.navigate_absolute(world["x"], world["y"])

        return {
            "capture_id": capture_id,
            "camera_frame": meta.get("camera_frame", CAMERA_OPTICAL_FRAME),
            "u": int(u),
            "v": int(v),
            "ray_depth": float(ray_depth),
            "depth_encoding": meta.get("depth_encoding"),
            "camera_point": {
                "x": float(point_cam[0]),
                "y": float(point_cam[1]),
                "z": float(point_cam[2]),
            },
            "relative_coordinate": relative,
            "world_coordinate": world,
            "navigating": bool(navigate and world is not None),
        }

    def render_local_map(self, size_m: float = 8.0, resolution: float = 0.05) -> tuple[bytes, dict[str, Any]]:
        with self.lock:
            cloud = None if self.latest_cloud is None else self.latest_cloud.copy()
            pose = asdict(self.pose)
            pose_public = _pose_record_from_state(self.pose)
            trajectory = list(self.pose_history)

        if cloud is None:
            raise RuntimeError("No point cloud received yet.")

        side_px = int(round(size_m / resolution))
        side_px = max(side_px, 200)
        img = np.full((side_px, side_px, 3), 255, dtype=np.uint8)
        center = side_px // 2
        half = size_m / 2.0
        yaw = pose["yaw"]
        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)

        for meter in range(-int(half), int(half) + 1):
            offset = int(round((meter / size_m) * side_px))
            x = center + offset
            y = center - offset
            cv2.line(img, (x, 0), (x, side_px - 1), (220, 220, 220), 1)
            cv2.line(img, (0, y), (side_px - 1, y), (220, 220, 220), 1)
            if meter != 0:
                cv2.putText(
                    img,
                    f"{meter}m",
                    (x + 2, center - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (100, 100, 100),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    f"{meter}m",
                    (center + 4, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (100, 100, 100),
                    1,
                    cv2.LINE_AA,
                )

        if cloud.size > 0:
            rel_x = cloud[:, 0] - pose["x"]
            rel_y = cloud[:, 1] - pose["y"]
            local_x = cos_yaw * rel_x - sin_yaw * rel_y
            local_y = sin_yaw * rel_x + cos_yaw * rel_y
            mask = (
                (local_x >= -half)
                & (local_x <= half)
                & (local_y >= -half)
                & (local_y <= half)
            )
            local_x = local_x[mask]
            local_y = local_y[mask]
            px = ((local_x + half) / size_m * (side_px - 1)).astype(np.int32)
            py = ((half - local_y) / size_m * (side_px - 1)).astype(np.int32)
            img[py, px] = (30, 30, 30)

        pts = []
        for tx, ty in trajectory[-500:]:
            rel_x = tx - pose["x"]
            rel_y = ty - pose["y"]
            local_x = cos_yaw * rel_x - sin_yaw * rel_y
            local_y = sin_yaw * rel_x + cos_yaw * rel_y
            if -half <= local_x <= half and -half <= local_y <= half:
                px = int(round((local_x + half) / size_m * (side_px - 1)))
                py = int(round((half - local_y) / size_m * (side_px - 1)))
                pts.append((px, py))
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, (255, 0, 0), 2)

        cv2.circle(img, (center, center), 6, (0, 0, 255), -1)
        heading = (int(center + 20), center)
        cv2.arrowedLine(img, (center, center), heading, (0, 0, 255), 2, tipLength=0.3)
        cv2.putText(
            img,
            "0,0",
            (center + 8, center - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 180),
            1,
            cv2.LINE_AA,
        )

        ok, encoded = cv2.imencode(".png", img)
        if not ok:
            raise RuntimeError("Failed to encode map image.")
        return encoded.tobytes(), {"size_m": size_m, "resolution": resolution, "pose": pose_public}


def build_mcp_server(ros: RosMcpNode) -> FastMCP:
    server = FastMCP(
        "ros_navigation_mcp",
        host=MCP_SERVER_HOST,
        port=MCP_SERVER_PORT,
        streamable_http_path=MCP_SERVER_PATH,
    )

    @server.tool(
        description="Render an 8x8 local top-down occupancy view from point cloud with trajectory and labeled relative grid.",
        structured_output=False,
    )
    def render_local_map():
        raw, meta = ros.render_local_map()
        return [_image_content(raw), _text_content(_json_text(meta))]

    @server.tool(description="Get current pose as x, y, z, yaw_deg.")
    def get_pose() -> str:
        return _json_text(ros.get_pose_dict())

    @server.tool(description="List all saved waypoints.")
    def waypoint_list() -> str:
        return _json_text(ros.waypoint_list())

    @server.tool(description="Create a waypoint. Absolute coordinates default to current pose when omitted.")
    def waypoint_create(
        name: str,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
    ) -> str:
        return _json_text(ros.waypoint_create(name, x, y, z))

    @server.tool(description="Get a waypoint by name.")
    def waypoint_get(name: str) -> str:
        entry = ros.waypoint_get(name)
        if entry is None:
            raise RuntimeError("Waypoint not found.")
        return _json_text(entry)

    @server.tool(description="Update a waypoint by name.")
    def waypoint_update(
        name: str,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
    ) -> str:
        entry = ros.waypoint_update(name, {"x": x, "y": y, "z": z})
        if entry is None:
            raise RuntimeError("Waypoint not found.")
        return _json_text(entry)

    @server.tool(description="Delete a waypoint by name.")
    def waypoint_delete(name: str) -> str:
        return _json_text({"deleted": ros.waypoint_delete(name)})

    @server.tool(description="Navigate to an absolute map coordinate.")
    def navigate_to_absolute(x: float, y: float, z: float | None = None) -> str:
        return _json_text(asdict(ros.navigate_absolute(x, y, z)))

    @server.tool(description="Navigate to a coordinate relative to the robot frame.")
    def navigate_to_relative(dx: float, dy: float, dz: float = 0.0) -> str:
        return _json_text(asdict(ros.navigate_relative(dx, dy, dz)))

    @server.tool(description="Navigate to a saved waypoint.")
    def navigate_to_waypoint(name: str) -> str:
        task = ros.navigate_waypoint(name)
        if task is None:
            raise RuntimeError("Waypoint not found.")
        return _json_text(asdict(task))

    @server.tool(description="Turn by a relative angle in degrees.")
    def turn_relative(angle_deg: float) -> str:
        return _json_text(asdict(ros.start_turn(angle_deg, relative=True)))

    @server.tool(description="Turn to an absolute yaw angle in degrees.")
    def turn_absolute(yaw_deg: float) -> str:
        return _json_text(asdict(ros.start_turn(yaw_deg, relative=False)))

    @server.tool(description="Wait for the current navigation or turn task to finish.")
    def wait_navigation(timeout_sec: float = 30.0) -> str:
        return _json_text(ros.wait_navigation(float(timeout_sec)))

    @server.tool(description="Cancel the current navigation or turn task.")
    def cancel_navigation() -> str:
        return _json_text({"cancelled": ros.cancel_navigation()})

    @server.tool(description="Publish /reset_visibility_graph.")
    def reset_visibility_graph() -> str:
        ros.reset_visibility_graph()
        return "Visibility graph reset published."

    @server.tool(
        description="Capture the latest RGB image, store matching depth and pose under the same id, and return the image.",
        structured_output=False,
    )
    def capture_image():
        capture_id, raw, rgb_path = ros.capture_image()
        return [_image_content(raw), _text_content(_json_text({"capture_id": capture_id, "rgb_path": rgb_path}))]

    @server.tool(
        description="Estimate 3D coordinates from a captured image id and pixel coordinate, and optionally navigate there."
    )
    def estimate_pixel(capture_id: str, u: int, v: int, navigate: bool = False) -> str:
        return _json_text(ros.estimate_pixel(capture_id, int(u), int(v), bool(navigate)))

    return server


def main() -> None:
    os.environ.setdefault("RCUTILS_LOGGING_USE_STDOUT", "0")
    rclpy.init(args=None)
    ros = RosMcpNode()
    executor = MultiThreadedExecutor()
    executor.add_node(ros)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    server = build_mcp_server(ros)
    try:
        server.run(transport=MCP_SERVER_TRANSPORT)
    finally:
        executor.shutdown()
        ros.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
