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

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import io
import os
import threading
import time

import torch

import omni

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from factory_sim_h1.tasks.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY


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

        self._node = rclpy.create_node("h1_cmd_vel_listener")
        self._sub = self._node.create_subscription(Twist, topic, self._callback, 10)
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


def _apply_cmd_vel(command_term, cmd_vel: tuple[float, float, float]):
    command_term.vel_command_b[:, 0] = cmd_vel[0]
    command_term.vel_command_b[:, 1] = cmd_vel[1]
    command_term.vel_command_b[:, 2] = cmd_vel[2]
    if hasattr(command_term, "is_standing_env"):
        command_term.is_standing_env[:] = False
    if hasattr(command_term, "is_heading_env"):
        command_term.is_heading_env[:] = False


def main():
    """Main function."""
    # load the trained jit policy
    policy_path = os.path.abspath(args_cli.checkpoint)
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    policy = torch.jit.load(file, map_location=args_cli.device)

    # setup environment
    env_cfg = H1RoughEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
    )
    env_cfg.sim.device = args_cli.device
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False
    # Keep command term active but stop random standing/heading behavior.
    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)

    # create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    command_term = env.command_manager.get_term("base_velocity")

    cmd_vel_sub = CmdVelSubscriber(args_cli.cmd_vel_topic, args_cli.cmd_vel_timeout)

    # run inference with the policy
    obs, _ = env.reset()
    with torch.inference_mode():
        while simulation_app.is_running():
            _apply_cmd_vel(command_term, cmd_vel_sub.get())
            action = policy(obs["policy"])
            obs, _, _, _, _ = env.step(action)

    cmd_vel_sub.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
