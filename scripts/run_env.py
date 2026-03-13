# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates policy inference in a prebuilt USD environment.

In this example, we use a locomotion policy to control the H1 robot. The robot was trained
using Isaac-Velocity-Rough-H1-v0. The robot velocity target is read from ROS2 topic /cmd_vel.

.. code-block:: bash

    # Run the script
    python scripts/run_env.py --checkpoint /path/to/policy.pt

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on inferencing a policy on an H1 robot in a warehouse.")
parser.add_argument("--checkpoint", type=str, default="weight/policy.pt", help="Path to model checkpoint exported as jit.")
parser.add_argument("--cmd_vel_topic", type=str, default="/cmd_vel", help="ROS2 topic for geometry_msgs/Twist.")
parser.add_argument("--cmd_vel_timeout", type=float, default=0.5, help="Zero command timeout in seconds.")
parser.add_argument("--camera_topic", type=str, default="/camera/rgb", help="ROS2 camera topic.")
parser.add_argument("--camera_frame", type=str, default="camera", help="ROS2 camera frame id.")
parser.add_argument("--camera_frame_skip", type=int, default=0, help="Frame skip count for camera publisher.")
parser.add_argument("--camera_width", type=int, default=1600, help="Camera image width in pixels.")
parser.add_argument("--camera_height", type=int, default=1200, help="Camera image height in pixels (4:3 enforced).")
parser.add_argument("--camera_hfov_deg", type=float, default=120.0, help="Camera horizontal field-of-view in degrees.")
parser.add_argument("--camera_near_clip", type=float, default=0.01, help="Camera near clipping plane in meters.")
parser.add_argument("--camera_far_clip", type=float, default=300.0, help="Camera far clipping plane in meters.")
parser.add_argument(
    "--camera_roll_deg",
    type=float,
    default=-90.0,
    help="Counter-clockwise image roll in degrees around optical axis.",
)
parser.add_argument("--lidar_topic", type=str, default="/lidar/points", help="ROS2 lidar point cloud topic.")
parser.add_argument("--lidar_frame", type=str, default="lidar", help="ROS2 lidar frame id.")
parser.add_argument("--lidar_hz", type=float, default=10.0, help="Target lidar publish frequency in Hz.")
parser.add_argument(
    "--lidar_config",
    type=str,
    default="OS1_REV6_32ch10hz1024res",
    help="RTX lidar config token. Example: OS1_REV6_32ch10hz1024res or Example_Rotary",
)
parser.add_argument("--odom_topic", type=str, default="/odom", help="ROS2 odometry topic.")
parser.add_argument("--odom_frame", type=str, default="odom", help="ROS2 odometry frame id.")
parser.add_argument("--base_frame", type=str, default="base_link", help="ROS2 base frame id for odom.")
parser.add_argument("--odom_hz", type=float, default=200.0, help="Target odometry and TF publish frequency in Hz.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import io
import math
import os
import threading
import time

import torch

import omni
import omni.graph.core as og
import omni.kit.app
import omni.kit.commands
import omni.usd

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import H1_CFG
from factory_sim_h1.tasks.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY

CAMERA_PARENT_KEYWORDS = ("head", "torso", "pelvis", "trunk", "base")
LIDAR_PARENT_KEYWORDS = ("head", "torso", "pelvis", "trunk", "base")
CAMERA_LOCAL_T = (0.12, 0.0, 0.18)
# Base aim (look-forward) for a USD Camera (which looks along local -Z).
CAMERA_AIM_RPY_DEG = (0.0, -90.0, 0.0)
LIDAR_LOCAL_T = (0.05, 0.0, 0.22)
LIDAR_LOCAL_RPY_DEG = (0.0, 0.0, 0.0)
TERRAIN_COLLISION_ROOTS = ("/World/ground/terrain", "/World/ground")


class CmdVelSubscriber:
    """ROS2 /cmd_vel subscriber with background spinning."""

    def __init__(self, topic: str, timeout_s: float, odom_topic: str):
        import rclpy
        from geometry_msgs.msg import Twist
        from rclpy.executors import SingleThreadedExecutor

        self._timeout_s = timeout_s
        self._lock = threading.Lock()
        self._cmd = (0.0, 0.0, 0.0)
        self._stamp = time.monotonic()

        self._rclpy = rclpy
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=None)

        self._node = rclpy.create_node("h1_cmd_vel_listener")
        self._sub = self._node.create_subscription(Twist, topic, self._callback, 10)
        self._odom_msg_type = None
        self._odom_pub = None
        try:
            from nav_msgs.msg import Odometry

            self._odom_msg_type = Odometry
            self._odom_pub = self._node.create_publisher(Odometry, odom_topic, 10)
        except Exception as exc:
            print(f"[WARN] nav_msgs/Odometry unavailable, odom topic disabled: {exc}")
        self._tf_msg_type = None
        self._tf_array_msg_type = None
        self._tf_broadcaster = None
        self._tf_pub = None
        try:
            from geometry_msgs.msg import TransformStamped
            from tf2_ros import TransformBroadcaster

            self._tf_msg_type = TransformStamped
            self._tf_broadcaster = TransformBroadcaster(self._node)
        except Exception as exc:
            print(f"[WARN] tf2_ros TransformBroadcaster unavailable, try TFMessage fallback: {exc}")
            try:
                from geometry_msgs.msg import TransformStamped
                from tf2_msgs.msg import TFMessage

                self._tf_msg_type = TransformStamped
                self._tf_array_msg_type = TFMessage
                self._tf_pub = self._node.create_publisher(TFMessage, "/tf", 100)
                print("[INFO] TFMessage fallback enabled: publishing base_link->lidar on /tf.")
            except Exception as exc2:
                print(f"[WARN] TFMessage fallback unavailable, TF publishing disabled: {exc2}")
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        self._stop_evt = threading.Event()
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

    def _spin(self):
        while not self._stop_evt.is_set() and self._rclpy.ok():
            self._executor.spin_once(timeout_sec=0.05)

    def _callback(self, msg):
        with self._lock:
            self._cmd = (float(msg.linear.x), float(msg.linear.y), float(msg.angular.z))
            self._stamp = time.monotonic()

    def get(self) -> tuple[float, float, float]:
        with self._lock:
            cmd = self._cmd
            stamp = self._stamp
        if self._timeout_s > 0.0 and (time.monotonic() - stamp) > self._timeout_s:
            return (0.0, 0.0, 0.0)
        return cmd

    def close(self):
        self._stop_evt.set()
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)
        self._node.destroy_subscription(self._sub)
        self._executor.remove_node(self._node)
        self._node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()

    def publish_transform(self, parent_frame: str, child_frame: str, translation_xyz, quat_wxyz, stamp_sec: float | None = None):
        if (self._tf_broadcaster is None and self._tf_pub is None) or self._tf_msg_type is None:
            return
        msg = self._tf_msg_type()
        if stamp_sec is None:
            msg.header.stamp = self._node.get_clock().now().to_msg()
        else:
            sec = int(math.floor(float(stamp_sec)))
            nsec = int(round((float(stamp_sec) - float(sec)) * 1.0e9))
            if nsec >= 1_000_000_000:
                sec += 1
                nsec -= 1_000_000_000
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = max(0, nsec)
        msg.header.frame_id = parent_frame
        msg.child_frame_id = child_frame
        msg.transform.translation.x = float(translation_xyz[0])
        msg.transform.translation.y = float(translation_xyz[1])
        msg.transform.translation.z = float(translation_xyz[2])

        q_w = float(quat_wxyz[0])
        q_x = float(quat_wxyz[1])
        q_y = float(quat_wxyz[2])
        q_z = float(quat_wxyz[3])
        q_norm = math.sqrt(q_w * q_w + q_x * q_x + q_y * q_y + q_z * q_z)
        if q_norm > 1.0e-9:
            q_w /= q_norm
            q_x /= q_norm
            q_y /= q_norm
            q_z /= q_norm
        else:
            q_w, q_x, q_y, q_z = 1.0, 0.0, 0.0, 0.0

        msg.transform.rotation.x = q_x
        msg.transform.rotation.y = q_y
        msg.transform.rotation.z = q_z
        msg.transform.rotation.w = q_w
        if self._tf_broadcaster is not None:
            self._tf_broadcaster.sendTransform(msg)
            return
        if self._tf_pub is not None and self._tf_array_msg_type is not None:
            tf_msg = self._tf_array_msg_type()
            tf_msg.transforms = [msg]
            self._tf_pub.publish(tf_msg)

    def publish_odom(self, odom_frame: str, base_frame: str, pos_xyz, quat_wxyz, lin_vel_xyz, ang_vel_xyz):
        if self._odom_pub is None or self._odom_msg_type is None:
            return
        msg = self._odom_msg_type()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = odom_frame
        msg.child_frame_id = base_frame
        msg.pose.pose.position.x = float(pos_xyz[0])
        msg.pose.pose.position.y = float(pos_xyz[1])
        msg.pose.pose.position.z = float(pos_xyz[2])
        msg.pose.pose.orientation.w = float(quat_wxyz[0])
        msg.pose.pose.orientation.x = float(quat_wxyz[1])
        msg.pose.pose.orientation.y = float(quat_wxyz[2])
        msg.pose.pose.orientation.z = float(quat_wxyz[3])
        msg.twist.twist.linear.x = float(lin_vel_xyz[0])
        msg.twist.twist.linear.y = float(lin_vel_xyz[1])
        msg.twist.twist.linear.z = float(lin_vel_xyz[2])
        msg.twist.twist.angular.x = float(ang_vel_xyz[0])
        msg.twist.twist.angular.y = float(ang_vel_xyz[1])
        msg.twist.twist.angular.z = float(ang_vel_xyz[2])
        self._odom_pub.publish(msg)


def _apply_cmd_vel(command_term, cmd_vel: tuple[float, float, float]):
    command_term.vel_command_b[:, 0] = cmd_vel[0]
    command_term.vel_command_b[:, 1] = cmd_vel[1]
    command_term.vel_command_b[:, 2] = cmd_vel[2]
    if hasattr(command_term, "is_standing_env"):
        command_term.is_standing_env[:] = False
    if hasattr(command_term, "is_heading_env"):
        command_term.is_heading_env[:] = False


def _enable_ros2_sensor_extensions():
    mgr = omni.kit.app.get_app().get_extension_manager()
    mgr.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
    mgr.set_extension_enabled_immediate("isaacsim.sensors.rtx", True)


def _find_first_robot_root(stage):
    candidates = []
    for prim in Usd.PrimRange(stage.GetPseudoRoot()):
        path = str(prim.GetPath())
        if "/env_0/Robot" in path or path.endswith("/Robot"):
            candidates.append(path)
    if not candidates:
        raise RuntimeError("Could not find robot root in stage.")
    candidates.sort(key=len)
    return candidates[0]


def _find_link_by_keywords(stage, root_path: str, keywords):
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Root prim not found: {root_path}")
    # Preserve keyword priority: choose the first keyword that matches anything.
    for keyword in keywords:
        hits = []
        for prim in Usd.PrimRange(root):
            path = str(prim.GetPath())
            if keyword in path.lower():
                hits.append(path)
        if hits:
            hits.sort(key=len)
            return hits[0]
    return root_path


def _is_valid_odom_chassis_prim(prim) -> bool:
    return bool(prim and prim.IsValid() and (UsdPhysics.RigidBodyAPI(prim) or UsdPhysics.ArticulationRootAPI(prim)))


def _find_odom_chassis_prim(stage, robot_root: str) -> str:
    root = stage.GetPrimAtPath(robot_root)
    if _is_valid_odom_chassis_prim(root):
        return robot_root

    preferred = []
    fallback = []
    for prim in Usd.PrimRange(root):
        if not _is_valid_odom_chassis_prim(prim):
            continue
        path = str(prim.GetPath())
        fallback.append(path)
        low = path.lower()
        if any(k in low for k in ("pelvis", "base_link", "base", "torso", "trunk", "root")):
            preferred.append(path)

    if preferred:
        preferred.sort(key=len)
        return preferred[0]
    if fallback:
        fallback.sort(key=len)
        return fallback[0]

    print(f"[WARN] No valid rigid-body/articulation prim found under {robot_root}; odom may fail.")
    return robot_root


def _set_local_pose(stage, prim_path: str, translation_xyz, rotation_rpy_deg):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    # Remove existing xform properties to avoid precision/type conflicts and ensure deterministic pose.
    for op_name in (
        "xformOp:translate",
        "xformOp:rotateXYZ",
        "xformOp:rotateX",
        "xformOp:rotateY",
        "xformOp:rotateZ",
        "xformOp:orient",
        "xformOp:transform",
        "xformOp:scale",
        "xformOp:translate:pivot",
        "xformOp:invert:translate:pivot",
    ):
        if prim.HasProperty(op_name):
            prim.RemoveProperty(op_name)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    rotate_op = xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble)
    t_xyz = tuple(map(float, translation_xyz))
    r_rpy = tuple(map(float, rotation_rpy_deg))
    translate_op.Set(Gf.Vec3d(*t_xyz))
    rotate_op.Set(Gf.Vec3d(*r_rpy))


def _frame_skip_from_target_hz(target_hz: float, base_hz: float) -> int:
    safe_target_hz = max(1.0e-3, float(target_hz))
    safe_base_hz = max(1.0e-3, float(base_hz))
    ratio = int(round(safe_base_hz / safe_target_hz))
    return max(0, ratio - 1)


def _gate_step_from_target_hz(target_hz: float, base_hz: float) -> int:
    safe_target_hz = max(1.0e-3, float(target_hz))
    safe_base_hz = max(1.0e-3, float(base_hz))
    return max(1, int(round(safe_base_hz / safe_target_hz)))


def _camera_resolution_4_3():
    width = max(1, int(args_cli.camera_width))
    requested_height = max(1, int(args_cli.camera_height))
    enforced_height = int(round(width * 3.0 / 4.0))
    if requested_height != enforced_height:
        print(f"[WARN] camera_height={requested_height} overridden to {enforced_height} to enforce 4:3.")
    return width, enforced_height


def _configure_camera_intrinsics(stage, camera_prim_path: str):
    prim = stage.GetPrimAtPath(camera_prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Camera prim not found: {camera_prim_path}")

    cam = UsdGeom.Camera(prim)
    width, height = _camera_resolution_4_3()
    aspect = float(width) / float(height)
    hfov_deg = float(max(1.0, min(179.0, args_cli.camera_hfov_deg)))

    # USD camera model uses aperture/focal length in mm-equivalent units.
    horizontal_aperture = 20.955
    vertical_aperture = horizontal_aperture / aspect
    focal_length = horizontal_aperture / (2.0 * math.tan(math.radians(hfov_deg) * 0.5))

    cam.GetHorizontalApertureAttr().Set(float(horizontal_aperture))
    cam.GetVerticalApertureAttr().Set(float(vertical_aperture))
    cam.GetFocalLengthAttr().Set(float(focal_length))
    cam.GetProjectionAttr().Set("perspective")
    near_clip = max(0.001, float(args_cli.camera_near_clip))
    far_clip = max(near_clip + 0.1, float(args_cli.camera_far_clip))
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(float(near_clip), float(far_clip)))
    print(
        f"[INFO] Camera intrinsics set: resolution={width}x{height}, hfov_deg={hfov_deg}, "
        f"focal_length={focal_length:.4f}, aperture=({horizontal_aperture:.4f},{vertical_aperture:.4f}), "
        f"clip=({near_clip:.4f},{far_clip:.1f})"
    )


def _create_camera(stage, parent_link: str):
    camera_mount_path = f"{parent_link}/CameraMount"
    if stage.GetPrimAtPath(camera_mount_path).IsValid():
        stage.RemovePrim(camera_mount_path)
    UsdGeom.Xform.Define(stage, camera_mount_path)
    _set_local_pose(stage, camera_mount_path, CAMERA_LOCAL_T, CAMERA_AIM_RPY_DEG)

    camera_roll_path = f"{camera_mount_path}/CameraRoll"
    UsdGeom.Xform.Define(stage, camera_roll_path)
    _set_local_pose(stage, camera_roll_path, (0.0, 0.0, 0.0), (0.0, 0.0, float(args_cli.camera_roll_deg)))

    camera_prim_path = f"{camera_roll_path}/Camera"
    UsdGeom.Camera.Define(stage, camera_prim_path)
    _configure_camera_intrinsics(stage, camera_prim_path)
    print(
        f"[INFO] Camera mounted at {camera_prim_path} (parent={parent_link}, local_t={CAMERA_LOCAL_T}, "
        f"aim_rpy={CAMERA_AIM_RPY_DEG}, roll_deg={args_cli.camera_roll_deg})"
    )
    return camera_prim_path


def _create_lidar(stage, parent_link: str, lidar_config: str):
    lidar_mount_path = f"{parent_link}/LidarMount"
    if stage.GetPrimAtPath(lidar_mount_path).IsValid():
        stage.RemovePrim(lidar_mount_path)
    UsdGeom.Xform.Define(stage, lidar_mount_path)
    _set_local_pose(stage, lidar_mount_path, LIDAR_LOCAL_T, LIDAR_LOCAL_RPY_DEG)

    lidar_prim_name_path = f"{lidar_mount_path}/Lidar"
    resolved_lidar_config = _resolve_lidar_config(lidar_config)
    config_name = resolved_lidar_config
    variant_name = None
    if resolved_lidar_config.startswith(("OS0_", "OS1_", "OS2_")):
        config_name = resolved_lidar_config.split("_", 1)[0]
        variant_name = resolved_lidar_config

    create_kwargs = dict(
        path="Lidar",
        parent=lidar_mount_path,
        config=config_name,
        translation=Gf.Vec3d(0.0, 0.0, 0.0),
        orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
        visibility=False,
    )
    if variant_name is not None:
        create_kwargs["variant"] = variant_name

    cmd_result = omni.kit.commands.execute(
        "IsaacSensorCreateRtxLidar",
        **create_kwargs,
    )
    lidar_sensor_path = _resolve_created_lidar_prim(stage, lidar_mount_path, lidar_prim_name_path, cmd_result)
    # Force native world-frame point output from RTX lidar so ROS point cloud can be published directly in odom.
    lidar_prim = stage.GetPrimAtPath(lidar_sensor_path)
    if lidar_prim and lidar_prim.IsValid():
        frame_ref_attr = lidar_prim.GetAttribute("omni:sensor:Core:outputFrameOfReference")
        if frame_ref_attr and frame_ref_attr.IsValid():
            frame_ref_attr.Set("WORLD")
            print("[INFO] Lidar outputFrameOfReference set to WORLD.")
        else:
            print("[WARN] Lidar prim has no outputFrameOfReference attribute; cannot enforce WORLD output frame.")
    print(
        f"[INFO] Lidar mounted at {lidar_sensor_path} (parent={parent_link}, local_t={LIDAR_LOCAL_T}, local_rpy={LIDAR_LOCAL_RPY_DEG}, config={config_name}, variant={variant_name})"
    )
    return lidar_sensor_path


def _resolve_lidar_config(lidar_config: str) -> str:
    config = lidar_config.strip()
    if not config:
        return "OS1_REV6_32ch10hz1024res"

    normalized = config.rstrip("/\\")
    if "/" in normalized or "\\" in normalized or normalized.endswith((".json", ".usd", ".usda")):
        token = os.path.splitext(os.path.basename(normalized))[0]
        print(f"[INFO] Normalized lidar config '{lidar_config}' -> '{token}'")
        return token
    return normalized


def _prim_path_from_command_result(result):
    if result is None:
        return None
    if isinstance(result, str):
        return result
    if hasattr(result, "GetPath"):
        try:
            return str(result.GetPath())
        except Exception:
            pass
    if isinstance(result, tuple) or isinstance(result, list):
        for item in result:
            path = _prim_path_from_command_result(item)
            if path:
                return path
    return None


def _resolve_created_lidar_prim(stage, parent_link: str, mount_path: str, cmd_result):
    candidates = []
    result_path = _prim_path_from_command_result(cmd_result)
    if result_path:
        candidates.append(result_path)
    candidates.append(mount_path)

    parent = stage.GetPrimAtPath(parent_link)
    if parent and parent.IsValid():
        for prim in Usd.PrimRange(parent):
            prim_path = str(prim.GetPath())
            if "lidar" in prim_path.lower() and prim.GetTypeName() in ("OmniLidar", "Camera"):
                candidates.append(prim_path)
        for prim in Usd.PrimRange(parent):
            if prim.GetTypeName() == "OmniLidar":
                candidates.append(str(prim.GetPath()))

    seen = set()
    fallback_path = None
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            if prim.GetTypeName() in ("OmniLidar", "Camera"):
                return path
            if fallback_path is None:
                fallback_path = path

    if fallback_path is not None:
        mount_prim = stage.GetPrimAtPath(mount_path)
        if mount_prim and mount_prim.IsValid():
            for prim in Usd.PrimRange(mount_prim):
                if prim.GetTypeName() == "OmniLidar":
                    return str(prim.GetPath())
        raise RuntimeError(
            f"Lidar prim resolved to non-sensor path '{fallback_path}'. "
            f"No valid OmniLidar found under mount '{mount_path}'."
        )

    raise RuntimeError(f"Failed to resolve created lidar prim under parent: {parent_link}")


def _build_ros2_sensor_graph(
    camera_prim_path: str,
    lidar_prim_path: str,
    odom_chassis_prim: str,
    lidar_gate_step: int,
    odom_gate_step: int,
):
    stage = omni.usd.get_context().get_stage()
    graph_prim = stage.GetPrimAtPath("/World/ROS2Sensors")
    if graph_prim and graph_prim.IsValid():
        stage.RemovePrim("/World/ROS2Sensors")
    create_nodes = [
        ("tick", "omni.graph.action.OnPlaybackTick"),
        ("ctx", "isaacsim.ros2.bridge.ROS2Context"),
        ("time", "isaacsim.core.nodes.IsaacReadSystemTime"),
        ("clock_pub", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ("lidar_gate", "isaacsim.core.nodes.IsaacSimulationGate"),
        ("odom_gate", "isaacsim.core.nodes.IsaacSimulationGate"),
    ]
    connect_nodes = []
    set_values = []

    create_nodes += [
        ("cam_rp", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
        ("cam_pub", "isaacsim.ros2.bridge.ROS2CameraHelper"),
        ("lidar_rp", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
        ("lidar_pub", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
        ("odom_compute", "isaacsim.core.nodes.IsaacComputeOdometry"),
        ("odom_pub", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
        ("odom_tf", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
    ]
    connect_nodes += [
        ("tick.outputs:tick", "clock_pub.inputs:execIn"),
        ("ctx.outputs:context", "clock_pub.inputs:context"),
        ("time.outputs:systemTime", "clock_pub.inputs:timeStamp"),
        ("tick.outputs:tick", "cam_rp.inputs:execIn"),
        ("cam_rp.outputs:execOut", "cam_pub.inputs:execIn"),
        ("cam_rp.outputs:renderProductPath", "cam_pub.inputs:renderProductPath"),
        ("ctx.outputs:context", "cam_pub.inputs:context"),
        ("tick.outputs:tick", "lidar_gate.inputs:execIn"),
        ("lidar_gate.outputs:execOut", "lidar_rp.inputs:execIn"),
        ("lidar_rp.outputs:execOut", "lidar_pub.inputs:execIn"),
        ("lidar_rp.outputs:renderProductPath", "lidar_pub.inputs:renderProductPath"),
        ("ctx.outputs:context", "lidar_pub.inputs:context"),
        ("tick.outputs:tick", "odom_gate.inputs:execIn"),
        ("odom_gate.outputs:execOut", "odom_compute.inputs:execIn"),
        ("odom_compute.outputs:execOut", "odom_pub.inputs:execIn"),
        ("odom_compute.outputs:execOut", "odom_tf.inputs:execIn"),
        ("ctx.outputs:context", "odom_pub.inputs:context"),
        ("ctx.outputs:context", "odom_tf.inputs:context"),
        ("odom_compute.outputs:position", "odom_pub.inputs:position"),
        ("odom_compute.outputs:orientation", "odom_pub.inputs:orientation"),
        ("odom_compute.outputs:linearVelocity", "odom_pub.inputs:linearVelocity"),
        ("odom_compute.outputs:angularVelocity", "odom_pub.inputs:angularVelocity"),
        ("odom_compute.outputs:position", "odom_tf.inputs:translation"),
        ("odom_compute.outputs:orientation", "odom_tf.inputs:rotation"),
        ("time.outputs:systemTime", "odom_pub.inputs:timeStamp"),
        ("time.outputs:systemTime", "odom_tf.inputs:timeStamp"),
    ]
    set_values += [
        ("lidar_gate.inputs:step", int(lidar_gate_step)),
        ("odom_gate.inputs:step", int(odom_gate_step)),
        ("cam_rp.inputs:cameraPrim", camera_prim_path),
        ("cam_rp.inputs:width", _camera_resolution_4_3()[0]),
        ("cam_rp.inputs:height", _camera_resolution_4_3()[1]),
        ("cam_rp.inputs:enabled", True),
        ("cam_pub.inputs:type", "rgb"),
        ("cam_pub.inputs:topicName", args_cli.camera_topic),
        ("cam_pub.inputs:frameId", args_cli.camera_frame),
        ("cam_pub.inputs:enabled", True),
        ("cam_pub.inputs:frameSkipCount", args_cli.camera_frame_skip),
        ("cam_pub.inputs:queueSize", 10),
        ("cam_pub.inputs:useSystemTime", True),
        ("lidar_rp.inputs:cameraPrim", lidar_prim_path),
        ("lidar_rp.inputs:enabled", True),
        ("lidar_pub.inputs:type", "point_cloud"),
        ("lidar_pub.inputs:topicName", args_cli.lidar_topic),
        ("lidar_pub.inputs:frameId", args_cli.odom_frame),
        ("lidar_pub.inputs:enabled", True),
        ("lidar_pub.inputs:frameSkipCount", 0),
        ("lidar_pub.inputs:queueSize", 10),
        ("lidar_pub.inputs:useSystemTime", True),
        ("lidar_pub.inputs:fullScan", True),
        ("odom_compute.inputs:chassisPrim", odom_chassis_prim),
        ("odom_pub.inputs:topicName", args_cli.odom_topic),
        ("odom_pub.inputs:chassisFrameId", args_cli.base_frame),
        ("odom_pub.inputs:odomFrameId", args_cli.odom_frame),
        ("odom_pub.inputs:queueSize", 10),
        ("odom_pub.inputs:publishRawVelocities", False),
        ("odom_tf.inputs:parentFrameId", args_cli.odom_frame),
        ("odom_tf.inputs:childFrameId", args_cli.base_frame),
        ("odom_tf.inputs:queueSize", 10),
    ]

    og.Controller.edit(
        {"graph_path": "/World/ROS2Sensors", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: create_nodes,
            og.Controller.Keys.CONNECT: connect_nodes,
            og.Controller.Keys.SET_VALUES: set_values,
        },
    )


def _ensure_scene_colliders(stage) -> tuple[str | None, int, int, int]:
    """Best-effort collision fallback for imported USD meshes."""
    root_prim = None
    root_path = None
    for candidate in TERRAIN_COLLISION_ROOTS:
        prim = stage.GetPrimAtPath(candidate)
        if prim and prim.IsValid():
            root_prim = prim
            root_path = candidate
            break

    if root_prim is None:
        print(f"[WARN] No terrain root found in {TERRAIN_COLLISION_ROOTS}; skip collider fallback.")
        return None, 0, 0, 0

    mesh_count = 0
    collider_applied = 0
    apply_failed = 0
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_count += 1

        # Referenced prototype prims are read-only; skip safely.
        if prim.IsInPrototype() or prim.IsInstanceProxy():
            continue

        try:
            collision_api = UsdPhysics.CollisionAPI(prim)
            if not collision_api:
                collision_api = UsdPhysics.CollisionAPI.Apply(prim)
                collider_applied += 1
            collision_api.GetCollisionEnabledAttr().Set(True)
        except Exception:
            apply_failed += 1

    print(
        f"[INFO] Collider fallback on {root_path}: meshes={mesh_count}, "
        f"new_colliders={collider_applied}, failed={apply_failed}"
    )
    return root_path, mesh_count, collider_applied, apply_failed


def _read_world_pose(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    xform = UsdGeom.Xformable(prim)
    world_tf = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_tf.Orthonormalize()
    pos = world_tf.ExtractTranslation()
    quat = world_tf.ExtractRotationQuat()
    translation = (float(pos[0]), float(pos[1]), float(pos[2]))
    quat_wxyz = (float(quat.real), float(quat.imaginary[0]), float(quat.imaginary[1]), float(quat.imaginary[2]))
    return translation, quat_wxyz


def _invert_pose(translation_xyz, quat_wxyz):
    tx, ty, tz = map(float, translation_xyz)
    qw, qx, qy, qz = map(float, quat_wxyz)
    q_norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if q_norm < 1.0e-9:
        return (-tx, -ty, -tz), (1.0, 0.0, 0.0, 0.0)
    qw /= q_norm
    qx /= q_norm
    qy /= q_norm
    qz /= q_norm

    # Rotation matrix for q(w,x,y,z).
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)
    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qx * qw)
    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    # Inverse transform: R^T and -R^T * t.
    itx = -(r00 * tx + r10 * ty + r20 * tz)
    ity = -(r01 * tx + r11 * ty + r21 * tz)
    itz = -(r02 * tx + r12 * ty + r22 * tz)
    inv_q = (qw, -qx, -qy, -qz)
    return (itx, ity, itz), inv_q


def _compose_pose(pose_ab, pose_bc):
    """Compose transforms T_ac = T_ab * T_bc.

    Pose format: (translation_xyz, quat_wxyz), with p_a = R_ab * p_b + t_ab.
    """
    (tabx, taby, tabz), (qaw, qax, qay, qaz) = pose_ab
    (tbcx, tbcy, tbcz), (qbw, qbx, qby, qbz) = pose_bc

    # Rotation matrix from q_ab.
    r00 = 1.0 - 2.0 * (qay * qay + qaz * qaz)
    r01 = 2.0 * (qax * qay - qaz * qaw)
    r02 = 2.0 * (qax * qaz + qay * qaw)
    r10 = 2.0 * (qax * qay + qaz * qaw)
    r11 = 1.0 - 2.0 * (qax * qax + qaz * qaz)
    r12 = 2.0 * (qay * qaz - qax * qaw)
    r20 = 2.0 * (qax * qaz - qay * qaw)
    r21 = 2.0 * (qay * qaz + qax * qaw)
    r22 = 1.0 - 2.0 * (qax * qax + qay * qay)

    tacx = tabx + (r00 * tbcx + r01 * tbcy + r02 * tbcz)
    tacy = taby + (r10 * tbcx + r11 * tbcy + r12 * tbcz)
    tacz = tabz + (r20 * tbcx + r21 * tbcy + r22 * tbcz)

    # Quaternion multiply q_ac = q_ab * q_bc (wxyz).
    qcw = qaw * qbw - qax * qbx - qay * qby - qaz * qbz
    qcx = qaw * qbx + qax * qbw + qay * qbz - qaz * qby
    qcy = qaw * qby - qax * qbz + qay * qbw + qaz * qbx
    qcz = qaw * qbz + qax * qby - qay * qbx + qaz * qbw
    qn = math.sqrt(qcw * qcw + qcx * qcx + qcy * qcy + qcz * qcz)
    if qn > 1.0e-9:
        qcw, qcx, qcy, qcz = qcw / qn, qcx / qn, qcy / qn, qcz / qn
    else:
        qcw, qcx, qcy, qcz = 1.0, 0.0, 0.0, 0.0

    return (tacx, tacy, tacz), (qcw, qcx, qcy, qcz)


def _quat_wxyz_from_rpy_deg(rotation_rpy_deg):
    roll = math.radians(float(rotation_rpy_deg[0]))
    pitch = math.radians(float(rotation_rpy_deg[1]))
    yaw = math.radians(float(rotation_rpy_deg[2]))

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm < 1.0e-9:
        return (1.0, 0.0, 0.0, 0.0)
    return (qw / norm, qx / norm, qy / norm, qz / norm)


def main():
    """Main function."""
    camera_width, camera_height = _camera_resolution_4_3()
    print(
        f"[INFO] run_env active: camera_aim_rpy={CAMERA_AIM_RPY_DEG}, camera_roll_deg={args_cli.camera_roll_deg}, "
        f"camera_hfov_deg={args_cli.camera_hfov_deg}, camera_resolution={camera_width}x{camera_height}, "
        f"camera_topic={args_cli.camera_topic}, lidar_topic={args_cli.lidar_topic}, lidar_config={args_cli.lidar_config}, "
        f"lidar_hz={args_cli.lidar_hz}, odom_hz={args_cli.odom_hz}"
    )
    # load the trained jit policy
    policy_path = os.path.abspath(args_cli.checkpoint)
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    policy = torch.jit.load(file, map_location=args_cli.device)

    # setup environment
    env_cfg = H1RoughEnvCfg_PLAY()
    # Use full-collision H1 for deployment; H1_MINIMAL can pass through nearby objects.
    env_cfg.scene.robot = H1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
    )
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.physx.enable_ccd = True
    if env_cfg.scene.robot.spawn.articulation_props is not None:
        env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8
        env_cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 4
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False
    # Keep command term active but stop random standing/heading behavior.
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
    sim_step_hz = 1.0 / (float(env_cfg.sim.dt) * float(env_cfg.decimation))
    lidar_gate_step = _gate_step_from_target_hz(args_cli.lidar_hz, sim_step_hz)
    odom_gate_step = _gate_step_from_target_hz(args_cli.odom_hz, sim_step_hz)
    print(
        f"[INFO] Sim step rate={sim_step_hz:.2f} Hz, lidar_gate_step={lidar_gate_step}, "
        f"odom_gate_step={odom_gate_step}"
    )

    # create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    command_term = env.command_manager.get_term("base_velocity")

    cmd_vel_sub = CmdVelSubscriber(args_cli.cmd_vel_topic, args_cli.cmd_vel_timeout, args_cli.odom_topic)
    _enable_ros2_sensor_extensions()

    # run inference with the policy
    obs, _ = env.reset()
    stage = omni.usd.get_context().get_stage()
    robot_root = _find_first_robot_root(stage)
    odom_chassis_prim = _find_odom_chassis_prim(stage, robot_root)
    print(f"[INFO] odom_chassis_prim={odom_chassis_prim}")
    _ensure_scene_colliders(stage)

    camera_parent = _find_link_by_keywords(stage, robot_root, CAMERA_PARENT_KEYWORDS)
    print(f"[INFO] camera_parent={camera_parent}")
    camera_prim_path = _create_camera(stage, camera_parent)

    lidar_parent = _find_link_by_keywords(stage, robot_root, LIDAR_PARENT_KEYWORDS)
    print(f"[INFO] lidar_parent={lidar_parent}")
    lidar_prim_path = _create_lidar(stage, lidar_parent, args_cli.lidar_config)
    print(f"[INFO] base_lidar_tf enabled: {args_cli.base_frame} -> {args_cli.lidar_frame}, prim={lidar_prim_path}")
    print(f"[INFO] Lidar point cloud publish mode: native WORLD data with ROS frame_id='{args_cli.odom_frame}'.")

    _build_ros2_sensor_graph(
        camera_prim_path,
        lidar_prim_path,
        odom_chassis_prim,
        lidar_gate_step=lidar_gate_step,
        odom_gate_step=odom_gate_step,
    )

    default_base_to_lidar = (tuple(map(float, LIDAR_LOCAL_T)), _quat_wxyz_from_rpy_deg(LIDAR_LOCAL_RPY_DEG))
    last_base_to_lidar = default_base_to_lidar

    with torch.inference_mode():
        while simulation_app.is_running():
            _apply_cmd_vel(command_term, cmd_vel_sub.get())
            base_world_pose = _read_world_pose(stage, odom_chassis_prim)
            lidar_world_pose = _read_world_pose(stage, lidar_prim_path)
            if base_world_pose is not None and lidar_world_pose is not None:
                last_base_to_lidar = _compose_pose(
                    _invert_pose(base_world_pose[0], base_world_pose[1]),
                    lidar_world_pose,
                )
            cmd_vel_sub.publish_transform(
                args_cli.base_frame,
                args_cli.lidar_frame,
                last_base_to_lidar[0],
                last_base_to_lidar[1],
            )
            action = policy(obs["policy"])
            obs, _, _, _, _ = env.step(action)

    cmd_vel_sub.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
