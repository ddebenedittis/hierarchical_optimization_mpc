import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class DiffDrivePublisher(Node):
    def __init__(self):
        super().__init__(
            'multi_robot_command_relay',
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )

        self.n_nodes = self.get_parameter('N_AGENTS').value  # total number of agents

        self.namespaces = [f'robot_{i+1}' for i in range(self.n_nodes)]
        self.step = 0
        # Subscriptions
        self.subscriptions_list = {}
        for j in range(self.n_nodes):
            topic_name = f'/command_node_{j}'
            self.subscriptions_list[j] = self.create_subscription(
                Float32MultiArray,
                topic_name,
                lambda msg, node=j: self.listener_callback_command(msg, node),
                20,
            )
            self.get_logger().info(f'Subscribed to {topic_name}')

        # Publishers
        self.publishers_list = {}
        for ns in self.namespaces:
            topic_name = f'/{ns}/diff_drive_base_controller/cmd_vel'
            self.publishers_list[ns] = self.create_publisher(TwistStamped, topic_name, 10)
            self.get_logger().info(f'Publisher created for {topic_name}')
        self.published_command = self.create_publisher(Float32MultiArray, '/step_published', 10)

        self.get_logger().info('DiffDrivePublisher node started.')

        self.received_command = {j: [] for j in range(self.n_nodes)}

    def listener_callback_command(self, msg, node):
        """Store message for a given namespace"""
        self.received_command[node] = msg.data
        self.get_logger().info(f'Received command from robot_{node}: {msg.data}')

        # Check if we have received messages from all robots
        if all(self.received_command[j] for j in range(self.n_nodes)):
            self.publish_all()

    def publish_all(self):
        """Publish commands to all robots once all have sent input"""
        for j, data in self.received_command.items():
            ns = f'robot_{j+1}'
            if data is None or len(data) < 2:
                self.get_logger().warn(f'Skipping {ns}, invalid data: {data}')
                continue
            if self.step != data[0]:
                self.get_logger().warn(
                    f'Difference between step of algorithm and publisher for {ns}'
                )

            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.twist.linear.x = float(data[1])
            msg.twist.angular.z = float(data[2])

            self.publishers_list[ns].publish(msg)

            self.get_logger().info(
                f'Published time-command {data[0]} to {ns}/diff_drive_base_controller/cmd_vel: {data[1], data[2]}'
            )

        # Reset received messages to wait for the next synchronized batch
        for j in range(self.n_nodes):
            self.received_command[j] = None
        msg = Float32MultiArray()
        msg.data = [float(self.step)]
        self.published_command.publish(msg)
        self.step += 1


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
