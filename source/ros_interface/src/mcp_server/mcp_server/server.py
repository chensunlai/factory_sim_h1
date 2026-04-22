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
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, Joy, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Empty
from tf2_ros import Buffer, TransformListener

cv2.ocl.setUseOpenCL(False)


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
CAPTURE_AROUND_SAMPLE_COUNT = 16
CAPTURE_AROUND_DEFAULT_ANGULAR_CMD = 1.2
CAPTURE_AROUND_DEFAULT_ROTATION_TIMEOUT_SEC = 15.0
CAPTURE_AROUND_FRAME_TIMEOUT_SEC = 2.0
CAPTURE_AROUND_CONTROL_PERIOD_SEC = 0.05
CAPTURE_AROUND_START_MOTION_THRESHOLD_DEG = 1.0
CAPTURE_AROUND_RECORDED_FRAME_LIMIT = 2000
CAPTURE_AROUND_STITCH_SCALE = 0.5
LOCAL_MAP_DEFAULT_SIZE_M = 16.0
LOCAL_MAP_GROUND_INTENSITY_THRESHOLD = 0.15
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
LOCAL_MAP_PLANNER_PATH_FRAME = "vehicle"


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


def _stamp_to_sec(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _image_stamp_sec(msg: Image | None) -> float:
    if msg is None:
        return 0.0
    return _stamp_to_sec(msg.header.stamp)


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
        self.latest_global_goal: dict[str, Any] | None = None
        self.latest_planner_path: list[tuple[float, float, float]] = []
        self.latest_planner_path_frame_id = ""
        self.latest_short_term_target: dict[str, Any] | None = None
        self.latest_rgb_msg: Image | None = None
        self.latest_depth_msg: Image | None = None
        self.latest_cloud: np.ndarray | None = None
        self.rgb_recording = False
        self.rgb_recorded_frames: list[dict[str, Any]] = []
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
        self.declare_parameter("scan_topic", "/terrain_map_ext")
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
        self.create_subscription(PointStamped, goal_topic, self._global_goal_cb, 10)
        self.create_subscription(NavPath, "/path", self._path_cb, 10)
        self.create_subscription(PointStamped, waypoint_topic, self._short_term_target_cb, 10)
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
        field_names = {field.name for field in msg.fields}
        points = []
        if "intensity" in field_names:
            for p in point_cloud2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
                intensity = float(p[3])
                if intensity <= LOCAL_MAP_GROUND_INTENSITY_THRESHOLD:
                    continue
                points.append((float(p[0]), float(p[1]), float(p[2])))
        else:
            for p in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                points.append((float(p[0]), float(p[1]), float(p[2])))
        arr = np.asarray(points, dtype=np.float32) if points else np.zeros((0, 3), dtype=np.float32)
        with self.lock:
            self.latest_cloud = arr

    def _global_goal_cb(self, msg: PointStamped) -> None:
        with self.lock:
            self.latest_global_goal = {
                "x": float(msg.point.x),
                "y": float(msg.point.y),
                "z": float(msg.point.z),
                "frame_id": msg.header.frame_id or "",
            }

    def _path_cb(self, msg: NavPath) -> None:
        points = [
            (
                float(entry.pose.position.x),
                float(entry.pose.position.y),
                float(entry.pose.position.z),
            )
            for entry in msg.poses
        ]
        with self.lock:
            self.latest_planner_path = points
            self.latest_planner_path_frame_id = msg.header.frame_id or ""

    def _short_term_target_cb(self, msg: PointStamped) -> None:
        with self.lock:
            self.latest_short_term_target = {
                "x": float(msg.point.x),
                "y": float(msg.point.y),
                "z": float(msg.point.z),
                "frame_id": msg.header.frame_id or "",
            }

    def _rgb_cb(self, msg: Image) -> None:
        with self.lock:
            self.latest_rgb_msg = msg
            if self.rgb_recording:
                stamp_sec = _image_stamp_sec(msg)
                if not self.rgb_recorded_frames or stamp_sec > self.rgb_recorded_frames[-1]["stamp_sec"] + 1e-6:
                    self.rgb_recorded_frames.append({"stamp_sec": stamp_sec, "msg": msg})
                    if len(self.rgb_recorded_frames) > CAPTURE_AROUND_RECORDED_FRAME_LIMIT:
                        self.rgb_recorded_frames = self.rgb_recorded_frames[-CAPTURE_AROUND_RECORDED_FRAME_LIMIT:]

    def _depth_cb(self, msg: Image) -> None:
        with self.lock:
            self.latest_depth_msg = msg

    def _begin_rgb_recording(self) -> None:
        with self.lock:
            self.rgb_recorded_frames = []
            self.rgb_recording = True

    def _end_rgb_recording(self) -> list[dict[str, Any]]:
        with self.lock:
            frames = list(self.rgb_recorded_frames)
            self.rgb_recorded_frames = []
            self.rgb_recording = False
        return frames

    def _wait_for_recorded_rgb_frame(
        self,
        frame_count: int = 1,
        timeout_sec: float = CAPTURE_AROUND_FRAME_TIMEOUT_SEC,
    ) -> dict[str, Any]:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            with self.lock:
                if len(self.rgb_recorded_frames) >= frame_count:
                    frame = self.rgb_recorded_frames[frame_count - 1]
                    return {"stamp_sec": float(frame["stamp_sec"]), "msg": frame["msg"]}
            time.sleep(0.05)
        raise RuntimeError("Timed out waiting for a recorded RGB frame.")

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

    def _cancel_current_task(
        self,
        *,
        expected_task_id: str | None = None,
        message: str = "Current task cancelled.",
    ) -> tuple[bool, str | None]:
        with self.lock:
            pose = self.pose
            task = self.current_task
            if task is None:
                return False, None
            if expected_task_id is not None and task.id != expected_task_id:
                return False, task.id
            task.status = "cancelled"
            task.result = "cancelled"
            task.message = message
            task_id = task.id
        self.turn_cancel.set()
        self._manual_joy(0.0, 0.0)
        self._publish_goal(pose.x, pose.y, pose.z)
        return True, task_id

    def cancel_navigation(self) -> bool:
        cancelled, _ = self._cancel_current_task()
        return cancelled

    def wait_navigation(self, timeout_sec: float, stop_on_timeout: bool = False) -> dict[str, Any]:
        deadline = time.time() + timeout_sec
        task_id: str | None = None
        while time.time() < deadline:
            with self.lock:
                task = self.current_task
                if task is None:
                    return {"status": "ended", "result": "no_active_task"}
                task_id = task.id
                if task.status in {"completed", "cancelled", "failed"}:
                    return {"status": "ended", "result": task.result or task.status, "task_id": task.id}
            time.sleep(0.1)
        response: dict[str, Any] = {"status": "timeout"}
        if task_id is not None:
            response["task_id"] = task_id
        if stop_on_timeout:
            cancelled, _ = self._cancel_current_task(
                expected_task_id=task_id,
                message="Current task stopped after wait_navigation timeout.",
            )
            response["stopped"] = cancelled
            if cancelled:
                response["result"] = "cancelled"
        return response

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
            if abs(error) < math.radians(5.0):
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

    def _trim_capture_around_panorama(self, panorama: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        points = cv2.findNonZero(mask)
        if points is None:
            raise RuntimeError("capture_around panorama is empty after stitching.")
        x, y, w, h = cv2.boundingRect(points)
        return panorama[y : y + h, x : x + w]

    def _align_capture_around_panorama_to_front(
        self,
        panorama: np.ndarray,
        front_image: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        pano_height, pano_width = panorama.shape[:2]
        front_height, front_width = front_image.shape[:2]
        template_width = max(32, min(front_width, int(round(front_width * 0.45))))
        template_height = max(32, min(front_height, int(round(front_height * 0.5))))
        x0 = (front_width - template_width) // 2
        y0 = (front_height - template_height) // 2
        template = front_image[y0 : y0 + template_height, x0 : x0 + template_width]

        if pano_height != front_height:
            scaled_width = max(1, int(round(template.shape[1] * float(pano_height) / float(front_height))))
            template = cv2.resize(template, (scaled_width, pano_height), interpolation=cv2.INTER_AREA)

        if pano_width <= template.shape[1]:
            return panorama, {"front_match_score": None, "front_match_center_x": None, "front_shift_px": 0}

        pano_gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        tiled = np.concatenate([pano_gray, pano_gray], axis=1)
        result = cv2.matchTemplate(tiled, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_value, _, max_location = cv2.minMaxLoc(result)
        match_center_x = (float(max_location[0] % pano_width) + template.shape[1] * 0.5)
        shift_px = int(round(pano_width * 0.5 - match_center_x))
        aligned = np.roll(panorama, shift_px, axis=1)
        return aligned, {
            "front_match_score": float(max_value),
            "front_match_center_x": float((match_center_x + shift_px) % pano_width),
            "front_shift_px": int(shift_px),
        }

    def _capture_around_stitch_order(self, capture_count: int) -> list[int]:
        midpoint = (capture_count + 1) // 2
        return list(range(midpoint)) + list(range(capture_count - 1, midpoint - 1, -1))

    def _draw_capture_around_heading_arrow(self, panorama: np.ndarray) -> np.ndarray:
        annotated = panorama.copy()
        height, width = annotated.shape[:2]
        center_x = width // 2
        bottom_margin = max(10, int(round(height * 0.04)))
        arrow_height = max(28, int(round(height * 0.18)))
        head_height = max(16, int(round(arrow_height * 0.42)))
        shaft_half_width = max(8, int(round(width * 0.015)))
        head_half_width = max(18, int(round(width * 0.035)))
        base_y = min(height - 1, height - bottom_margin)
        tip_y = max(0, base_y - arrow_height)
        head_base_y = min(base_y - 1, tip_y + head_height)
        arrow_points = np.array(
            [
                [center_x, tip_y],
                [center_x + head_half_width, head_base_y],
                [center_x + shaft_half_width, head_base_y],
                [center_x + shaft_half_width, base_y],
                [center_x - shaft_half_width, base_y],
                [center_x - shaft_half_width, head_base_y],
                [center_x - head_half_width, head_base_y],
            ],
            dtype=np.int32,
        )
        outline_thickness = max(2, int(round(min(width, height) * 0.006)))
        cv2.fillPoly(annotated, [arrow_points], (0, 0, 255), lineType=cv2.LINE_AA)
        cv2.polylines(
            annotated,
            [arrow_points],
            isClosed=True,
            color=(0, 0, 0),
            thickness=outline_thickness,
            lineType=cv2.LINE_AA,
        )
        return annotated

    def _build_capture_around_panorama(
        self,
        captures: list[dict[str, Any]],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not captures:
            raise RuntimeError("No captures available for panorama.")

        images: list[np.ndarray] = []
        source_width: int | None = None
        source_height: int | None = None
        for entry in captures:
            image = self.bridge.imgmsg_to_cv2(entry["msg"], desired_encoding="bgr8")
            if image is None:
                raise RuntimeError("Failed to decode a capture_around frame.")
            if source_width is None or source_height is None:
                source_height, source_width = image.shape[:2]
            elif image.shape[:2] != (source_height, source_width):
                raise RuntimeError("All capture_around images must share the same resolution.")
            images.append(image)

        assert source_width is not None and source_height is not None
        stitch_order = self._capture_around_stitch_order(len(images))
        ordered_images = [images[idx] for idx in stitch_order]
        ordered_captures = [captures[idx] for idx in stitch_order]
        stitch_inputs = [
            cv2.resize(
                src=image,
                dsize=None,
                fx=CAPTURE_AROUND_STITCH_SCALE,
                fy=CAPTURE_AROUND_STITCH_SCALE,
                interpolation=cv2.INTER_AREA,
            )
            for image in ordered_images
        ]
        anchor_stitch_image = stitch_inputs[0]

        stitcher = cv2.Stitcher().create()
        stitch_status, stitched_panorama = stitcher.stitch(stitch_inputs)
        if int(stitch_status) != 0 or stitched_panorama is None or stitched_panorama.size == 0:
            raise RuntimeError(f"capture_around stitching failed with status={int(stitch_status)}")

        trimmed = self._trim_capture_around_panorama(stitched_panorama)
        resized_height = max(1, int(round(trimmed.shape[0] * float(source_width) / float(trimmed.shape[1]))))
        resized_panorama = cv2.resize(trimmed, (int(source_width), int(resized_height)), interpolation=cv2.INTER_AREA)
        panorama, align_meta = self._align_capture_around_panorama_to_front(resized_panorama, anchor_stitch_image)
        panorama = self._draw_capture_around_heading_arrow(panorama)

        meta = {
            "selected_source_stamps_sec": [float(entry["source_stamp_sec"]) for entry in captures],
            "stitch_source_stamps_sec": [float(entry["source_stamp_sec"]) for entry in ordered_captures],
            "stitch_input_order": [int(idx) for idx in stitch_order],
            "source_width": int(source_width),
            "source_height": int(source_height),
            "raw_panorama_width": int(trimmed.shape[1]),
            "raw_panorama_height": int(trimmed.shape[0]),
            "panorama_width": int(panorama.shape[1]),
            "panorama_height": int(panorama.shape[0]),
            "front_center_x": int(round(panorama.shape[1] * 0.5)),
            "stitch_scale": float(CAPTURE_AROUND_STITCH_SCALE),
            "stitch_status": int(stitch_status),
            "frame_order_prior": "first_half_forward_second_half_reverse",
            "center_anchor_source": "sample_0_center_patch",
        }
        meta.update(align_meta)
        return panorama, meta

    def capture_around(
        self,
        rotation_timeout_sec: float = CAPTURE_AROUND_DEFAULT_ROTATION_TIMEOUT_SEC,
    ) -> tuple[str, bytes, str, dict[str, Any]]:
        rotation_timeout_sec = float(rotation_timeout_sec)
        if rotation_timeout_sec <= 0.0:
            raise RuntimeError("rotation_timeout_sec must be positive.")

        with self.lock:
            rgb_msg = self.latest_rgb_msg
            task = self.current_task
        if rgb_msg is None:
            raise RuntimeError("No RGB image received yet.")
        if task is not None and task.status == "running":
            raise RuntimeError("Cannot run capture_around while another task is running.")

        self._begin_rgb_recording()
        recorded_frames: list[dict[str, Any]] = []
        motion_start_stamp: float | None = None
        motion_end_stamp: float | None = None
        accumulated_rotation = 0.0
        rotation_direction = 0.0

        try:
            self._wait_for_recorded_rgb_frame()
            with self.lock:
                pose = self.pose
            last_pose_stamp = float(pose.stamp_sec)
            last_yaw = float(pose.yaw)

            deadline = time.time() + rotation_timeout_sec
            while time.time() < deadline:
                self._manual_joy(angular=CAPTURE_AROUND_DEFAULT_ANGULAR_CMD, linear=0.0)
                with self.lock:
                    pose = self.pose
                if pose.stamp_sec > last_pose_stamp + 1e-6:
                    delta_yaw = _normalize_angle(float(pose.yaw) - last_yaw)
                    if rotation_direction == 0.0 and abs(delta_yaw) > math.radians(0.2):
                        rotation_direction = 1.0 if delta_yaw > 0.0 else -1.0
                    signed_delta = delta_yaw * (rotation_direction if rotation_direction != 0.0 else 1.0)
                    if signed_delta > 0.0:
                        accumulated_rotation += signed_delta
                        if (
                            motion_start_stamp is None
                            and accumulated_rotation >= math.radians(CAPTURE_AROUND_START_MOTION_THRESHOLD_DEG)
                        ):
                            motion_start_stamp = float(pose.stamp_sec)
                    last_pose_stamp = float(pose.stamp_sec)
                    last_yaw = float(pose.yaw)
                    if accumulated_rotation >= 2.0 * math.pi:
                        motion_end_stamp = float(pose.stamp_sec)
                        break
                time.sleep(CAPTURE_AROUND_CONTROL_PERIOD_SEC)
            else:
                raise RuntimeError("capture_around rotation timed out before completing a full turn.")
        finally:
            self._manual_joy(0.0, 0.0)
            recorded_frames = self._end_rgb_recording()

        if motion_start_stamp is None or motion_end_stamp is None:
            raise RuntimeError("Failed to measure the capture_around rotation interval.")

        recorded_frames = sorted(recorded_frames, key=lambda item: item["stamp_sec"])
        if not recorded_frames:
            raise RuntimeError("No RGB frames were recorded during capture_around.")

        motion_frames = [frame for frame in recorded_frames if frame["stamp_sec"] >= motion_start_stamp - 1e-6]
        if not motion_frames:
            raise RuntimeError("No RGB frames were recorded after capture_around motion started.")

        rotation_duration_sec = motion_end_stamp - motion_start_stamp
        if rotation_duration_sec <= 0.0:
            raise RuntimeError("capture_around rotation duration must be positive.")

        selected_frames = []
        for idx in range(CAPTURE_AROUND_SAMPLE_COUNT):
            target_stamp_sec = motion_start_stamp + rotation_duration_sec * idx / CAPTURE_AROUND_SAMPLE_COUNT
            nearest_frame = min(motion_frames, key=lambda frame: abs(frame["stamp_sec"] - target_stamp_sec))
            selected_frames.append(
                {
                    "sample_index": int(idx),
                    "target_stamp_sec": float(target_stamp_sec),
                    "source_stamp_sec": float(nearest_frame["stamp_sec"]),
                    "time_error_sec": float(nearest_frame["stamp_sec"] - target_stamp_sec),
                    "msg": nearest_frame["msg"],
                }
            )

        panorama, panorama_meta = self._build_capture_around_panorama(selected_frames)
        panorama_id = f"around_{uuid.uuid4().hex[:8]}"
        panorama_path = self.capture_dir / f"{panorama_id}_panorama.png"
        meta_path = self.capture_dir / f"{panorama_id}_panorama.json"
        if not cv2.imwrite(str(panorama_path), panorama):
            raise RuntimeError("Failed to save panorama image.")
        ok, encoded = cv2.imencode(".png", panorama)
        if not ok:
            raise RuntimeError("Failed to encode panorama image.")

        meta = {
            "capture_id": panorama_id,
            "type": "capture_around",
            "panorama_path": str(panorama_path),
            "capture_dir": str(self.capture_dir),
            "sample_count": int(CAPTURE_AROUND_SAMPLE_COUNT),
            "rotation_timeout_sec": float(rotation_timeout_sec),
            "angular_cmd": float(CAPTURE_AROUND_DEFAULT_ANGULAR_CMD),
            "motion_start_stamp_sec": float(motion_start_stamp),
            "motion_end_stamp_sec": float(motion_end_stamp),
            "rotation_duration_sec": float(rotation_duration_sec),
            "rotation_direction": "positive_yaw" if rotation_direction >= 0.0 else "negative_yaw",
            "center_is_front": False,
            "center_is_first_sample": True,
            "selected_frames": [
                {
                    "sample_index": int(entry["sample_index"]),
                    "target_stamp_sec": float(entry["target_stamp_sec"]),
                    "source_stamp_sec": float(entry["source_stamp_sec"]),
                    "time_error_sec": float(entry["time_error_sec"]),
                }
                for entry in selected_frames
            ],
        }
        meta.update(panorama_meta)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return panorama_id, encoded.tobytes(), str(panorama_path), meta

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

    def render_local_map(
        self,
        size_m: float = LOCAL_MAP_DEFAULT_SIZE_M,
        resolution: float = 0.05,
    ) -> tuple[bytes, str, dict[str, Any]]:
        with self.lock:
            cloud = None if self.latest_cloud is None else self.latest_cloud.copy()
            pose = asdict(self.pose)
            pose_public = _pose_record_from_state(self.pose)
            trajectory = list(self.pose_history)
            global_goal = None if self.latest_global_goal is None else dict(self.latest_global_goal)
            planner_path = list(self.latest_planner_path)
            planner_path_frame_id = self.latest_planner_path_frame_id
            short_term_target = (
                None if self.latest_short_term_target is None else dict(self.latest_short_term_target)
            )
            capture_dir = self.capture_dir

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
        pose_frame_id = pose.get("frame_id") or NAVIGATION_FRAME

        def to_local_xy(x: float, y: float, frame_id: str) -> tuple[float, float] | None:
            if frame_id in {"", pose_frame_id, NAVIGATION_FRAME}:
                rel_x = x - pose["x"]
                rel_y = y - pose["y"]
                return (
                    cos_yaw * rel_x - sin_yaw * rel_y,
                    sin_yaw * rel_x + cos_yaw * rel_y,
                )
            if frame_id == LOCAL_MAP_PLANNER_PATH_FRAME:
                return float(x), float(y)
            return None

        def local_xy_to_pixel(local_x: float, local_y: float) -> tuple[int, int] | None:
            if not (-half <= local_x <= half and -half <= local_y <= half):
                return None
            px = int(round((local_x + half) / size_m * (side_px - 1)))
            py = int(round((half - local_y) / size_m * (side_px - 1)))
            px = max(0, min(side_px - 1, px))
            py = max(0, min(side_px - 1, py))
            return px, py

        def draw_labeled_marker(
            pixel: tuple[int, int],
            fill_color: tuple[int, int, int],
            text: str,
            text_color: tuple[int, int, int],
        ) -> None:
            cv2.circle(img, pixel, 7, fill_color, -1)
            cv2.circle(img, pixel, 9, (0, 0, 0), 1)
            cv2.putText(
                img,
                text,
                (pixel[0] + 8, max(14, pixel[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                text_color,
                1,
                cv2.LINE_AA,
            )

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
            px = np.clip(px, 0, side_px - 1)
            py = np.clip(py, 0, side_px - 1)
            img[py, px] = (30, 30, 30)

        pts = []
        for tx, ty in trajectory[-500:]:
            local_xy = to_local_xy(tx, ty, pose_frame_id)
            if local_xy is None:
                continue
            pixel = local_xy_to_pixel(local_xy[0], local_xy[1])
            if pixel is not None:
                pts.append(pixel)
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, (255, 0, 0), 2)

        planner_pts = []
        for px_world, py_world, _ in planner_path:
            local_xy = to_local_xy(px_world, py_world, planner_path_frame_id)
            if local_xy is None:
                continue
            pixel = local_xy_to_pixel(local_xy[0], local_xy[1])
            if pixel is not None:
                planner_pts.append(pixel)
        if len(planner_pts) >= 2:
            cv2.polylines(img, [np.array(planner_pts, dtype=np.int32)], False, (0, 165, 255), 2)
        elif len(planner_pts) == 1:
            cv2.circle(img, planner_pts[0], 3, (0, 165, 255), -1)

        global_goal_visible = False
        if global_goal is not None:
            local_xy = to_local_xy(
                float(global_goal["x"]),
                float(global_goal["y"]),
                str(global_goal.get("frame_id") or ""),
            )
            if local_xy is not None:
                pixel = local_xy_to_pixel(local_xy[0], local_xy[1])
                if pixel is not None:
                    global_goal_visible = True
                    draw_labeled_marker(pixel, (180, 0, 180), "goal", (120, 0, 120))

        short_term_target_visible = False
        if short_term_target is not None:
            local_xy = to_local_xy(
                float(short_term_target["x"]),
                float(short_term_target["y"]),
                str(short_term_target.get("frame_id") or ""),
            )
            if local_xy is not None:
                pixel = local_xy_to_pixel(local_xy[0], local_xy[1])
                if pixel is not None:
                    short_term_target_visible = True
                    draw_labeled_marker(pixel, (0, 180, 0), "short", (0, 120, 0))

        legend_x = 10
        legend_y = 18
        cv2.line(img, (legend_x, legend_y), (legend_x + 18, legend_y), (255, 0, 0), 2)
        cv2.putText(img, "traj", (legend_x + 24, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
        legend_y += 18
        cv2.line(img, (legend_x, legend_y), (legend_x + 18, legend_y), (0, 165, 255), 2)
        cv2.putText(img, "path", (legend_x + 24, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
        legend_y += 18
        cv2.circle(img, (legend_x + 9, legend_y), 5, (180, 0, 180), -1)
        cv2.putText(
            img,
            "goal",
            (legend_x + 24, legend_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
        legend_y += 18
        cv2.circle(img, (legend_x + 9, legend_y), 5, (0, 180, 0), -1)
        cv2.putText(
            img,
            "short",
            (legend_x + 24, legend_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )

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
        local_map_path = capture_dir / f"local_map_{uuid.uuid4().hex[:8]}.png"
        local_map_path.write_bytes(encoded.tobytes())
        return encoded.tobytes(), str(local_map_path), {
            "size_m": size_m,
            "resolution": resolution,
            "pose": pose_public,
            "planner_path_frame_id": planner_path_frame_id,
            "planner_path_point_count": len(planner_path),
            "planner_path_visible_point_count": len(planner_pts),
            "global_goal": global_goal,
            "global_goal_visible": global_goal_visible,
            "short_term_target": short_term_target,
            "short_term_target_visible": short_term_target_visible,
            "selected_target": short_term_target,
            "selected_target_visible": short_term_target_visible,
            "ground_filter_intensity_threshold": LOCAL_MAP_GROUND_INTENSITY_THRESHOLD,
            "local_map_path": str(local_map_path),
        }


def build_mcp_server(ros: RosMcpNode) -> FastMCP:
    server = FastMCP(
        "ros_navigation_mcp",
        host=MCP_SERVER_HOST,
        port=MCP_SERVER_PORT,
        streamable_http_path=MCP_SERVER_PATH,
    )

    @server.tool(
        description=(
            "Render a 16x16 local top-down occupancy view. "
            "Black points are terrain obstacles, blue is the robot trajectory, "
            "orange is the planned path, purple is the global goal, "
            "green is the short-term goal, and red marks the robot pose."
        ),
        structured_output=False,
    )
    def render_local_map():
        raw, local_map_path, meta = ros.render_local_map()
        payload = dict(meta)
        payload["local_map_path"] = local_map_path
        return [_image_content(raw), _text_content(_json_text(payload))]

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

    @server.tool(
        description=(
            "Wait for the current navigation or turn task to finish. "
            "A typical timeout is 10 seconds. "
            "Set stop_on_timeout=true to automatically stop the task when the timeout expires."
        )
    )
    def wait_navigation(timeout_sec: float = 30.0, stop_on_timeout: bool = False) -> str:
        return _json_text(ros.wait_navigation(float(timeout_sec), bool(stop_on_timeout)))

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
        description=(
            "Rotate in place for one full turn, capture a panorama image, "
            "and mark the heading with an upward arrow at the lower center."
        ),
        structured_output=False,
    )
    def capture_around(rotation_timeout_sec: float = CAPTURE_AROUND_DEFAULT_ROTATION_TIMEOUT_SEC):
        capture_id, raw, panorama_path, meta = ros.capture_around(float(rotation_timeout_sec))
        payload = {"capture_id": capture_id, "panorama_path": panorama_path}
        payload.update(meta)
        return [_image_content(raw), _text_content(_json_text(payload))]

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
