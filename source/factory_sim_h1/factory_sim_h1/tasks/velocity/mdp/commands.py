# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass


class H1UniformVelocityCommand(UniformVelocityCommand):
    """Velocity command term with configurable debug marker position offset."""

    cfg: H1UniformVelocityCommandCfg

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        if not self.robot.is_initialized:
            return
        # place markers with a configurable offset so arrows are easier to see
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 0] += float(self.cfg.marker_pos_offset[0])
        base_pos_w[:, 1] += float(self.cfg.marker_pos_offset[1])
        base_pos_w[:, 2] += float(self.cfg.marker_pos_offset[2])

        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])

        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)


@configclass
class H1UniformVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for :class:`H1UniformVelocityCommand`."""

    class_type: type = H1UniformVelocityCommand
    marker_pos_offset: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Position offset applied to both goal/current velocity arrows in world frame."""
