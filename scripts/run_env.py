# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal fixed-layout H1 policy runner with ROS2 sensors.

Usage:
    python scripts/run_env_simple.py --checkpoint /path/to/policy.pt
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Minimal H1 runner (fixed scene/robot/topic/frame setup).")
parser.add_argument("--checkpoint", type=str, default="weight/policy.pt", help="Path to jit policy checkpoint.")
parser.add_argument("--camera_width", type=int, default=1600, help="Camera width in pixels.")
parser.add_argument("--camera_height", type=int, default=1200, help="Camera height in pixels.")
parser.add_argument("--camera_hfov_deg", type=float, default=100.0, help="Camera horizontal FOV in degrees.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.device = "cuda:0"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import io
import json
import math
import os
import threading
import time

import omni
import omni.graph.core as og
import omni.kit.app
import omni.kit.commands
import omni.usd
import torch

from pxr import Gf, Usd, UsdGeom

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import H1_CFG
from factory_sim_h1.tasks.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY

# Fixed scene/robot paths
ROBOT_ROOT = "/World/envs/env_0/Robot"
ODOM_CHASSIS_PRIM = "/World/envs/env_0/Robot/torso_link"
CAMERA_PARENT_PRIM = "/World/envs/env_0/Robot/torso_link"
LIDAR_PARENT_PRIM = "/World/envs/env_0/Robot/torso_link"

# Fixed ROS2 endpoints
CMD_VEL_TOPIC = "/cmd_vel"
CMD_VEL_TIMEOUT_S = 0.5
ENV_CTRL_TOPIC = "/isaacsim/env_ctrl"
CAMERA_TOPIC = "/isaacsim/camera_torso"
CAMERA_DEPTH_TOPIC = "/isaacsim/depth_torso"
CAMERA_FRAME = "camera_torso_frame"
LIDAR_TOPIC = "/isaacsim/lidar"
# Lidar data are configured as WORLD coordinates and published with odom frame id.
LIDAR_FRAME = "lidar"
ODOM_TOPIC = "/isaacsim/Odometry"
ODOM_FRAME = "odom"
BASE_FRAME = "torso_link"

# Fixed sensor configs
CAMERA_LOCAL_T = (0.12, 0.0, 0.18)
CAMERA_AIM_RPY_DEG = (0.0, -90.0, 0.0)
CAMERA_ROLL_DEG = -90.0
CAMERA_NEAR_CLIP = 0.01
CAMERA_FAR_CLIP = 300.0

LIDAR_CONFIG = "OS1_REV6_32ch10hz1024res"
LIDAR_LOCAL_T = (0.05, 0.0, 0.22)
LIDAR_LOCAL_RPY_DEG = (0.0, 0.0, 0.0)

# Fixed publication frequencies
LIDAR_HZ = 10.0
ODOM_HZ = 50.0


class CmdVelSubscriber:
    """ROS2 /cmd_vel subscriber with background spinning."""

    def __init__(self, topic: str, timeout_s: float):
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

        self._node = rclpy.create_node("h1_ros_bridge_listener")
        self._sub = self._node.create_subscription(Twist, topic, self._callback, 10)
        from std_msgs.msg import String

        self._env_sub = self._node.create_subscription(String, ENV_CTRL_TOPIC, self._env_ctrl_callback, 10)
        self._env_cmd_queue = []

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
                print("[INFO] TFMessage fallback enabled.")
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

    @staticmethod
    def _normalize_json_text(text: str) -> str:
        return (
            text.replace("“", '"')
            .replace("”", '"')
            .replace("‘", '"')
            .replace("’", '"')
            .replace("：", ":")
            .replace("，", ",")
            .replace("｛", "{")
            .replace("｝", "}")
        )

    def _env_ctrl_callback(self, msg):
        payload_raw = str(getattr(msg, "data", "")).strip()
        if not payload_raw:
            return

        try:
            payload = json.loads(self._normalize_json_text(payload_raw))
        except Exception as exc:
            print(f"[WARN] {ENV_CTRL_TOPIC} invalid json: {exc}. payload={payload_raw!r}")
            return

        if not isinstance(payload, dict):
            print(f"[WARN] {ENV_CTRL_TOPIC} payload must be object: {payload!r}")
            return

        cmd = payload.get("cmd")
        if not isinstance(cmd, str) or not cmd.strip():
            print(f"[WARN] {ENV_CTRL_TOPIC} missing valid cmd field: {payload!r}")
            return

        param = payload.get("param", {})
        if not isinstance(param, dict):
            print(f"[WARN] {ENV_CTRL_TOPIC} param must be object, got {type(param).__name__}. Use empty object.")
            param = {}

        with self._lock:
            self._env_cmd_queue.append({"cmd": cmd.strip(), "param": param})
            if len(self._env_cmd_queue) > 32:
                self._env_cmd_queue = self._env_cmd_queue[-32:]

    def get(self) -> tuple[float, float, float]:
        with self._lock:
            cmd = self._cmd
            stamp = self._stamp
        if self._timeout_s > 0.0 and (time.monotonic() - stamp) > self._timeout_s:
            return (0.0, 0.0, 0.0)
        return cmd

    def pop_env_commands(self):
        with self._lock:
            if not self._env_cmd_queue:
                return []
            queued = self._env_cmd_queue
            self._env_cmd_queue = []
        return queued

    def publish_transform(self, parent_frame: str, child_frame: str, translation_xyz, quat_wxyz):
        if (self._tf_broadcaster is None and self._tf_pub is None) or self._tf_msg_type is None:
            return

        msg = self._tf_msg_type()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = parent_frame
        msg.child_frame_id = child_frame
        msg.transform.translation.x = float(translation_xyz[0])
        msg.transform.translation.y = float(translation_xyz[1])
        msg.transform.translation.z = float(translation_xyz[2])

        qw, qx, qy, qz = map(float, quat_wxyz)
        n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if n < 1.0e-9:
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        else:
            qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n

        msg.transform.rotation.x = qx
        msg.transform.rotation.y = qy
        msg.transform.rotation.z = qz
        msg.transform.rotation.w = qw

        if self._tf_broadcaster is not None:
            self._tf_broadcaster.sendTransform(msg)
            return

        if self._tf_pub is not None and self._tf_array_msg_type is not None:
            tf_msg = self._tf_array_msg_type()
            tf_msg.transforms = [msg]
            self._tf_pub.publish(tf_msg)

    def close(self):
        self._stop_evt.set()
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)
        self._node.destroy_subscription(self._sub)
        self._node.destroy_subscription(self._env_sub)
        self._executor.remove_node(self._node)
        self._node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()


def _enable_ros2_sensor_extensions():
    mgr = omni.kit.app.get_app().get_extension_manager()
    mgr.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
    mgr.set_extension_enabled_immediate("isaacsim.sensors.rtx", True)


def _apply_cmd_vel(command_term, cmd_vel: tuple[float, float, float]):
    command_term.vel_command_b[:, 0] = cmd_vel[0]
    command_term.vel_command_b[:, 1] = cmd_vel[1]
    command_term.vel_command_b[:, 2] = cmd_vel[2]
    if hasattr(command_term, "is_standing_env"):
        command_term.is_standing_env[:] = False
    if hasattr(command_term, "is_heading_env"):
        command_term.is_heading_env[:] = False


def _set_local_pose(stage, prim_path: str, translation_xyz, rotation_rpy_deg):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    t = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    r = xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble)
    t.Set(Gf.Vec3d(*tuple(map(float, translation_xyz))))
    r.Set(Gf.Vec3d(*tuple(map(float, rotation_rpy_deg))))


def _camera_resolution():
    return max(1, int(args_cli.camera_width)), max(1, int(args_cli.camera_height))


def _configure_camera_intrinsics(stage, camera_prim_path: str):
    prim = stage.GetPrimAtPath(camera_prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Camera prim not found: {camera_prim_path}")

    cam = UsdGeom.Camera(prim)
    width, height = _camera_resolution()
    hfov_deg = float(max(1.0, min(179.0, args_cli.camera_hfov_deg)))
    aspect = float(width) / float(height)

    horizontal_aperture = 20.955
    vertical_aperture = horizontal_aperture / aspect
    focal_length = horizontal_aperture / (2.0 * math.tan(math.radians(hfov_deg) * 0.5))

    cam.GetHorizontalApertureAttr().Set(float(horizontal_aperture))
    cam.GetVerticalApertureAttr().Set(float(vertical_aperture))
    cam.GetFocalLengthAttr().Set(float(focal_length))
    cam.GetProjectionAttr().Set("perspective")

    near_clip = max(0.001, float(CAMERA_NEAR_CLIP))
    far_clip = max(near_clip + 0.1, float(CAMERA_FAR_CLIP))
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(float(near_clip), float(far_clip)))


def _create_camera(stage):
    mount = f"{CAMERA_PARENT_PRIM}/CameraMount"
    roll = f"{mount}/CameraRoll"
    cam_path = f"{roll}/Camera"

    if stage.GetPrimAtPath(mount).IsValid():
        stage.RemovePrim(mount)

    UsdGeom.Xform.Define(stage, mount)
    _set_local_pose(stage, mount, CAMERA_LOCAL_T, CAMERA_AIM_RPY_DEG)

    UsdGeom.Xform.Define(stage, roll)
    _set_local_pose(stage, roll, (0.0, 0.0, 0.0), (0.0, 0.0, float(CAMERA_ROLL_DEG)))

    UsdGeom.Camera.Define(stage, cam_path)
    _configure_camera_intrinsics(stage, cam_path)
    return cam_path


def _resolve_created_lidar_prim(stage, mount_path: str, cmd_result):
    candidates = []
    if isinstance(cmd_result, tuple) and len(cmd_result) >= 2 and hasattr(cmd_result[1], "GetPath"):
        candidates.append(str(cmd_result[1].GetPath()))
    candidates.append(f"{mount_path}/Lidar")

    for p in candidates:
        prim = stage.GetPrimAtPath(p)
        if prim and prim.IsValid() and prim.GetTypeName() in ("OmniLidar", "Camera"):
            return p

    mount_prim = stage.GetPrimAtPath(mount_path)
    if mount_prim and mount_prim.IsValid():
        for prim in Usd.PrimRange(mount_prim):
            if prim.GetTypeName() == "OmniLidar":
                return str(prim.GetPath())

    raise RuntimeError(f"Failed to resolve lidar prim under {mount_path}")


def _create_lidar(stage):
    mount = f"{LIDAR_PARENT_PRIM}/LidarMount"
    if stage.GetPrimAtPath(mount).IsValid():
        stage.RemovePrim(mount)

    UsdGeom.Xform.Define(stage, mount)
    _set_local_pose(stage, mount, LIDAR_LOCAL_T, LIDAR_LOCAL_RPY_DEG)

    config_name = LIDAR_CONFIG
    variant_name = None
    if LIDAR_CONFIG.startswith(("OS0_", "OS1_", "OS2_")):
        config_name = LIDAR_CONFIG.split("_", 1)[0]
        variant_name = LIDAR_CONFIG

    kwargs = dict(
        path="Lidar",
        parent=mount,
        config=config_name,
        translation=Gf.Vec3d(0.0, 0.0, 0.0),
        orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
        visibility=False,
    )
    if variant_name is not None:
        kwargs["variant"] = variant_name

    cmd_result = omni.kit.commands.execute("IsaacSensorCreateRtxLidar", **kwargs)
    lidar_path = _resolve_created_lidar_prim(stage, mount, cmd_result)

    lidar_prim = stage.GetPrimAtPath(lidar_path)
    if lidar_prim and lidar_prim.IsValid():
        frame_ref_attr = lidar_prim.GetAttribute("omni:sensor:Core:outputFrameOfReference")
        if frame_ref_attr and frame_ref_attr.IsValid():
            frame_ref_attr.Set("WORLD")
            print("[INFO] Lidar outputFrameOfReference set to WORLD.")

    return lidar_path


def _gate_step_from_target_hz(target_hz: float, base_hz: float) -> int:
    return max(1, int(round(max(1.0e-3, float(base_hz)) / max(1.0e-3, float(target_hz)))))


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

    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1.0e-9:
        return (1.0, 0.0, 0.0, 0.0)
    return (qw / n, qx / n, qy / n, qz / n)


def _quat_wxyz_from_yaw(yaw_rad: float):
    half = 0.5 * float(yaw_rad)
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _yaw_from_quat_wxyz(quat_wxyz):
    qw, qx, qy, qz = tuple(map(float, quat_wxyz))
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _safe_float(value, default_value: float) -> float:
    if value is None:
        return float(default_value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default_value)


def _reset_h1_to_pose(env: ManagerBasedRLEnv, robot, default_pose: dict, param: dict):
    obs, _ = env.reset()

    target_x = _safe_float(param.get("x"), default_pose["x"])
    target_y = _safe_float(param.get("y"), default_pose["y"])
    target_z = _safe_float(param.get("z"), default_pose["z"])
    target_yaw = _safe_float(param.get("yaw"), default_pose["yaw"])

    root_state = robot.data.default_root_state.clone()
    root_state[0, 0] = target_x
    root_state[0, 1] = target_y
    root_state[0, 2] = target_z
    qw, qx, qy, qz = _quat_wxyz_from_yaw(target_yaw)
    root_state[0, 3] = qw
    root_state[0, 4] = qx
    root_state[0, 5] = qy
    root_state[0, 6] = qz
    root_state[0, 7:13] = 0.0
    robot.write_root_state_to_sim(root_state[0:1], env_ids=[0])

    print(
        f"[INFO] reset_h1 applied: x={target_x:.3f}, y={target_y:.3f}, z={target_z:.3f}, yaw={target_yaw:.3f} rad"
    )
    return obs


def _build_ros2_sensor_graph(camera_prim_path: str, lidar_prim_path: str, sim_step_hz: float):
    lidar_gate_step = _gate_step_from_target_hz(LIDAR_HZ, sim_step_hz)
    odom_gate_step = _gate_step_from_target_hz(ODOM_HZ, sim_step_hz)

    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath("/World/ROS2Sensors").IsValid():
        stage.RemovePrim("/World/ROS2Sensors")

    width, height = _camera_resolution()

    og.Controller.edit(
        {"graph_path": "/World/ROS2Sensors", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("tick", "omni.graph.action.OnPlaybackTick"),
                ("ctx", "isaacsim.ros2.bridge.ROS2Context"),
                ("time", "isaacsim.core.nodes.IsaacReadSystemTime"),
                ("clock_pub", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("lidar_gate", "isaacsim.core.nodes.IsaacSimulationGate"),
                ("odom_gate", "isaacsim.core.nodes.IsaacSimulationGate"),
                ("cam_rp", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("cam_pub", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("cam_depth_pub", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("lidar_rp", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("lidar_pub", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
                ("odom_compute", "isaacsim.core.nodes.IsaacComputeOdometry"),
                ("odom_world_pose", "isaacsim.core.nodes.IsaacReadWorldPose"),
                ("odom_pub", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                ("odom_tf", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
            ],
            og.Controller.Keys.CONNECT: [
                ("tick.outputs:tick", "clock_pub.inputs:execIn"),
                ("ctx.outputs:context", "clock_pub.inputs:context"),
                ("time.outputs:systemTime", "clock_pub.inputs:timeStamp"),
                ("tick.outputs:tick", "cam_rp.inputs:execIn"),
                ("cam_rp.outputs:execOut", "cam_pub.inputs:execIn"),
                ("cam_rp.outputs:renderProductPath", "cam_pub.inputs:renderProductPath"),
                ("ctx.outputs:context", "cam_pub.inputs:context"),
                ("cam_rp.outputs:execOut", "cam_depth_pub.inputs:execIn"),
                ("cam_rp.outputs:renderProductPath", "cam_depth_pub.inputs:renderProductPath"),
                ("ctx.outputs:context", "cam_depth_pub.inputs:context"),
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
                ("odom_world_pose.outputs:translation", "odom_pub.inputs:position"),
                ("odom_world_pose.outputs:orientation", "odom_pub.inputs:orientation"),
                ("odom_compute.outputs:linearVelocity", "odom_pub.inputs:linearVelocity"),
                ("odom_compute.outputs:angularVelocity", "odom_pub.inputs:angularVelocity"),
                ("odom_world_pose.outputs:translation", "odom_tf.inputs:translation"),
                ("odom_world_pose.outputs:orientation", "odom_tf.inputs:rotation"),
                ("time.outputs:systemTime", "odom_pub.inputs:timeStamp"),
                ("time.outputs:systemTime", "odom_tf.inputs:timeStamp"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("lidar_gate.inputs:step", int(lidar_gate_step)),
                ("odom_gate.inputs:step", int(odom_gate_step)),
                ("cam_rp.inputs:cameraPrim", camera_prim_path),
                ("cam_rp.inputs:width", width),
                ("cam_rp.inputs:height", height),
                ("cam_rp.inputs:enabled", True),
                ("cam_pub.inputs:type", "rgb"),
                ("cam_pub.inputs:topicName", CAMERA_TOPIC),
                ("cam_pub.inputs:frameId", CAMERA_FRAME),
                ("cam_pub.inputs:enabled", True),
                ("cam_pub.inputs:frameSkipCount", 0),
                ("cam_pub.inputs:queueSize", 10),
                ("cam_pub.inputs:useSystemTime", True),
                ("cam_depth_pub.inputs:type", "depth"),
                ("cam_depth_pub.inputs:topicName", CAMERA_DEPTH_TOPIC),
                ("cam_depth_pub.inputs:frameId", CAMERA_FRAME),
                ("cam_depth_pub.inputs:enabled", True),
                ("cam_depth_pub.inputs:frameSkipCount", 0),
                ("cam_depth_pub.inputs:queueSize", 10),
                ("cam_depth_pub.inputs:useSystemTime", True),
                ("lidar_rp.inputs:cameraPrim", lidar_prim_path),
                ("lidar_rp.inputs:enabled", True),
                ("lidar_pub.inputs:type", "point_cloud"),
                ("lidar_pub.inputs:topicName", LIDAR_TOPIC),
                ("lidar_pub.inputs:frameId", ODOM_FRAME),
                ("lidar_pub.inputs:enabled", True),
                ("lidar_pub.inputs:frameSkipCount", 0),
                ("lidar_pub.inputs:queueSize", 10),
                ("lidar_pub.inputs:useSystemTime", True),
                ("lidar_pub.inputs:fullScan", True),
                ("odom_compute.inputs:chassisPrim", ODOM_CHASSIS_PRIM),
                ("odom_world_pose.inputs:prim", ODOM_CHASSIS_PRIM),
                ("odom_pub.inputs:topicName", ODOM_TOPIC),
                ("odom_pub.inputs:chassisFrameId", BASE_FRAME),
                ("odom_pub.inputs:odomFrameId", ODOM_FRAME),
                ("odom_pub.inputs:queueSize", 10),
                ("odom_pub.inputs:publishRawVelocities", False),
                ("odom_tf.inputs:parentFrameId", ODOM_FRAME),
                ("odom_tf.inputs:childFrameId", BASE_FRAME),
                ("odom_tf.inputs:queueSize", 10),
            ],
        },
    )

    print(
        f"[INFO] ROS2 graph ready: camera_rgb={CAMERA_TOPIC}, camera_depth={CAMERA_DEPTH_TOPIC}, "
        f"lidar={LIDAR_TOPIC}, odom={ODOM_TOPIC}, "
        f"lidar_hz={LIDAR_HZ}, odom_hz={ODOM_HZ}"
    )


def _validate_fixed_paths(stage):
    for p in (ROBOT_ROOT, ODOM_CHASSIS_PRIM, CAMERA_PARENT_PRIM, LIDAR_PARENT_PRIM):
        if not stage.GetPrimAtPath(p).IsValid():
            raise RuntimeError(f"Required fixed prim path not found: {p}")


def main():
    print(
        f"[INFO] run_env active: checkpoint={args_cli.checkpoint}, device={args_cli.device}, "
        f"camera={args_cli.camera_width}x{args_cli.camera_height}@{args_cli.camera_hfov_deg}deg"
    )

    policy_path = os.path.abspath(args_cli.checkpoint)
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    policy = torch.jit.load(file, map_location=args_cli.device)

    env_cfg = H1RoughEnvCfg_PLAY()
    env_cfg.scene.robot = H1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd",
    )
    env_cfg.sim.device = args_cli.device
    env_cfg.sim.physx.enable_ccd = True
    if env_cfg.scene.robot.spawn.articulation_props is not None:
        env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 8
        env_cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 4

    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
    # Keep IsaacLab base-contact termination for built-in auto reset on fall.
    # Disable time-out and random respawn drift for a stable single-robot setup.
    env_cfg.episode_length_s = 1.0e9
    if getattr(env_cfg, "terminations", None) is not None:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
    if getattr(env_cfg, "events", None) is not None and getattr(env_cfg.events, "reset_base", None) is not None:
        env_cfg.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

    sim_step_hz = 1.0 / (float(env_cfg.sim.dt) * float(env_cfg.decimation))

    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    command_term = env.command_manager.get_term("base_velocity")
    cmd_vel_sub = CmdVelSubscriber(CMD_VEL_TOPIC, CMD_VEL_TIMEOUT_S)

    _enable_ros2_sensor_extensions()

    obs, _ = env.reset()
    default_root_state = robot.data.default_root_state[0].clone()
    default_pose = {
        "x": float(default_root_state[0].item()),
        "y": float(default_root_state[1].item()),
        "z": float(default_root_state[2].item()),
        "yaw": _yaw_from_quat_wxyz(default_root_state[3:7].tolist()),
    }
    stage = omni.usd.get_context().get_stage()
    _validate_fixed_paths(stage)

    camera_prim_path = _create_camera(stage)
    lidar_prim_path = _create_lidar(stage)
    _build_ros2_sensor_graph(camera_prim_path, lidar_prim_path, sim_step_hz)

    base_to_lidar_t = tuple(map(float, LIDAR_LOCAL_T))
    base_to_lidar_q = _quat_wxyz_from_rpy_deg(LIDAR_LOCAL_RPY_DEG)

    print(
        f"[INFO] Fixed paths: robot={ROBOT_ROOT}, base={ODOM_CHASSIS_PRIM}, "
        f"camera_parent={CAMERA_PARENT_PRIM}, lidar_parent={LIDAR_PARENT_PRIM}"
    )
    print(f"[INFO] Env control topic enabled: {ENV_CTRL_TOPIC} (cmd=reset_h1)")

    with torch.inference_mode():
        while simulation_app.is_running():
            for ctrl in cmd_vel_sub.pop_env_commands():
                cmd = ctrl.get("cmd")
                param = ctrl.get("param", {})
                if cmd == "reset_h1":
                    if not isinstance(param, dict):
                        param = {}
                    obs = _reset_h1_to_pose(env, robot, default_pose, param)
                else:
                    print(f"[WARN] Unknown env_ctrl cmd: {cmd!r}")

            _apply_cmd_vel(command_term, cmd_vel_sub.get())
            cmd_vel_sub.publish_transform(BASE_FRAME, LIDAR_FRAME, base_to_lidar_t, base_to_lidar_q)
            action = policy(obs["policy"])
            obs, _, _, _, _ = env.step(action)

    cmd_vel_sub.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
