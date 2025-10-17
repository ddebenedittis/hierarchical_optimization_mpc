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

        # 50 ms timer (20 Hz)
        timer_period = 0.05
        self.timer = self.create_timer(timer_period, self.publish_command)

    def publish_command(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'  # optional, good practice
        msg.twist.linear.x = -8.0  # Forward speed (m/s)
        msg.twist.angular.z = 1.0  # Angular speed (rad/s)

        self.publisher_.publish(msg)
        self.get_logger().info(
            f'Publishing: linear.x={msg.twist.linear.x:.2f}, angular.z={msg.twist.angular.z:.2f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = DiffDrivePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
