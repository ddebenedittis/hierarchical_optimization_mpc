#!/usr/bin/env python3

from time import sleep

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile


class DiffDrivePublisher(Node):
    def __init__(self):
        super().__init__('diff_drive_publisher')

        qos_profile = QoSProfile(depth=10)
        self.publisher_ = self.create_publisher(
            TwistStamped, '/diff_drive_base_controller/cmd_vel', qos_profile
        )

        # Desired circle parameters
        self.radius = 5.0  # meters
        self.linear_speed = 0.2  # m/s
        self.angular_speed = self.linear_speed / self.radius  # rad/s

        # Publish command every 10 ms (100 Hz)
        self.timer = self.create_timer(0.01, self.publish_command)
        self.get_logger().info(
            f'DiffDrivePublisher started: moving in a circle of radius {self.radius} m.'
        )

    def publish_command(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = self.linear_speed
        msg.twist.angular.z = self.angular_speed
        self.publisher_.publish(msg)

    def stop_robot(self):
        """Publish a zero velocity command before shutting down."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = 0.0
        msg.twist.angular.z = 0.0
        self.publisher_.publish(msg)
        self.get_logger().info('Published stop command.')
        sleep(0.1)


def main(args=None):
    rclpy.init(args=args)
    node = DiffDrivePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down node...')
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
