#!/usr/bin/env python3

import copy
import importlib
import threading

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node


TRUE_STRINGS = {'1', 'true', 'reached', 'arrived', 'success', 'done', 'ok'}


def _load_msg_class(type_name: str):
    parts = type_name.split('/')
    if len(parts) == 3 and parts[1] == 'msg':
        package_name, _, message_name = parts
    elif len(parts) == 2:
        package_name, message_name = parts
    else:
        raise ValueError(f'Invalid message type: {type_name}')

    module = importlib.import_module(f'{package_name}.msg')
    return getattr(module, message_name)


class GoalReplayNode(Node):
    def __init__(self):
        super().__init__('goal_point_replay')

        self.declare_parameter('goal_topic', '/goal_point')
        self.declare_parameter('goal_type', 'geometry_msgs/msg/PointStamped')
        self.declare_parameter('status_topic', '/far_reach_goal_status')
        self.declare_parameter('status_type', 'std_msgs/msg/Bool')
        self.declare_parameter('buffer_size', 4)
        self.declare_parameter('publish_hz', 1.0)
        self.declare_parameter('arrive_delay', 1.0)
        self.declare_parameter('loop_delay', 5.0)

        self.goal_topic = self.get_parameter('goal_topic').value
        self.goal_type = self.get_parameter('goal_type').value
        self.status_topic = self.get_parameter('status_topic').value
        self.status_type = self.get_parameter('status_type').value
        self.buffer_size = int(self.get_parameter('buffer_size').value)
        self.publish_hz = float(self.get_parameter('publish_hz').value)
        self.arrive_delay = float(self.get_parameter('arrive_delay').value)
        self.loop_delay = float(self.get_parameter('loop_delay').value)

        goal_msg_class = _load_msg_class(self.goal_type)
        status_msg_class = _load_msg_class(self.status_type)

        self.lock = threading.Lock()
        self.goal_buffer = []
        self.current_index = 0
        self.ready = False
        self.goal_reached = False
        self.wait_until = None
        self.pending_restart = False
        self.trigger = False

        self.goal_publisher = self.create_publisher(goal_msg_class, self.goal_topic, 10)
        self.goal_subscriber = self.create_subscription(
            goal_msg_class,
            self.goal_topic,
            self._goal_callback,
            10,
        )
        self.status_subscriber = self.create_subscription(
            status_msg_class,
            self.status_topic,
            self._status_callback,
            10,
        )

        period = 1.0 / self.publish_hz if self.publish_hz > 0.0 else 1.0
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f'Waiting for first {self.buffer_size} points from {self.goal_topic}'
        )

    def _goal_callback(self, msg):
        if self.trigger:
            self.trigger = False
            return
        self.trigger = True
        with self.lock:
            if len(self.goal_buffer) >= self.buffer_size:
                return

            self.goal_buffer.append(copy.deepcopy(msg))
            self.get_logger().info(
                f'Stored goal {len(self.goal_buffer)}/{self.buffer_size}'
            )

            if len(self.goal_buffer) == self.buffer_size:
                self.ready = True
                self.current_index = 0
                self.goal_reached = False
                self.wait_until = None
                self.pending_restart = False
                self.get_logger().info('Captured all points, replay started')

    def _status_callback(self, msg):
        status = self._status_to_bool(msg)
        if not status:
            return

        with self.lock:
            self.goal_reached = True

    def _status_to_bool(self, msg) -> bool:
        for field_name in ('data', 'reached', 'arrived', 'success', 'done', 'status'):
            if hasattr(msg, field_name):
                return self._coerce_bool(getattr(msg, field_name))
        return self._coerce_bool(msg)

    @staticmethod
    def _coerce_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in TRUE_STRINGS
        return bool(value)

    def _publish_current_goal(self):
        goal = copy.deepcopy(self.goal_buffer[self.current_index])
        self.goal_publisher.publish(goal)

    def _on_timer(self):
        with self.lock:
            if not self.ready:
                return

            now = self.get_clock().now()

            if self.wait_until is not None:
                if now >= self.wait_until:
                    self.wait_until = None
                    self.goal_reached = False
                    if self.pending_restart:
                        self.pending_restart = False
                        self.current_index = 0
                    else:
                        self.current_index += 1
                else:
                    return

            if self.goal_reached:
                if self.current_index == len(self.goal_buffer) - 1:
                    self.get_logger().info(
                        f'Reached {self.buffer_size} goals, restart in {self.loop_delay:.1f} seconds'
                    )
                    self.pending_restart = True
                    self.wait_until = now + Duration(seconds=self.loop_delay)
                else:
                    self.wait_until = now + Duration(seconds=self.arrive_delay)
                return

            self._publish_current_goal()


def main(args=None):
    rclpy.init(args=args)
    node = GoalReplayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
