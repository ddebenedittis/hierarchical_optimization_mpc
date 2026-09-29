#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point, Pose, PoseArray
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion


class LaserScanRepublisher(Node):
    def __init__(self):
        super().__init__('agent')
        # Subscribe to original scan topic
        self.subscriber = self.create_subscription(
            LaserScan,
            '/robot_1/scan',  # input topic
            self.scan_callback,
            10,  # queue size
        )

        self.subscription = self.create_subscription(
            ModelStates,
            '/model_states',  # topic name
            self.states_callback,
            20,
        )
        # Publisher to /robot_1/scan
        self.publisher = self.create_publisher(
            PoseArray,
            '/robot_1/detected_objects',  # output topic
            10,
        )

        # Sensor configuration parameters
        self.gap_threshold = 5  # samples separating objects
        self.range_min = 0.3
        self.range_max = 8.0
        self.cylinder_radius = 0.5  # meters
        self.s = np.zeros(3)  # robot state: x, y, yaw

        self.get_logger().info('LaserScan Republisher started.')

    def states_callback(self, msg):
        """Extract the pose from gazebo"""
        # Extract position from Gazebo
        for n, name in enumerate(msg.name):
            if name == f'/robot_1':
                x = msg.pose[n].position.x  # msg.pose.pose.position.x
                y = msg.pose[n].position.y  # msg.pose.pose.position.y

                # Extract orientation (quaternion -> yaw)
                q = msg.pose[n].orientation  # msg.pose.pose.orientation
                roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
                self.s = np.array([x, y, yaw])  # self state

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges)

        # Replace invalid or out-of-range values with NaN
        ranges[(ranges < self.range_min) | (ranges > self.range_max)] = np.nan

        # Identify valid indices (where the LiDAR sees something)
        valid_indices = np.where(~np.isnan(ranges))[0]

        if len(valid_indices) == 0:
            self.get_logger().info('No objects detected.')
            return

        # Group detections separated by >= gap_threshold samples
        object_groups = []
        current_group = [valid_indices[0]]

        for idx in valid_indices[1:]:
            if idx - current_group[-1] <= self.gap_threshold:
                current_group.append(idx)
            else:
                object_groups.append(current_group)
                current_group = [idx]
        object_groups.append(current_group)

        # Compute each object's average range and angle
        objects_local = []
        for group in object_groups:
            group_ranges = ranges[group]
            if np.all(np.isnan(group_ranges)):
                continue

            # Minimum range (closest point)
            min_idx_in_group = group[np.nanargmin(group_ranges)]
            min_range = ranges[min_idx_in_group]

            # Mean angle for this object (its approximate direction)
            group_angles = msg.angle_min + np.array(group) * msg.angle_increment
            mean_angle = np.mean(group_angles)

            objects_local.append((min_range, mean_angle))

        if not objects_local:
            self.get_logger().info('No valid object clusters found.')
            return

        # --- Convert each detected object's closest point to world coordinates ---
        x_r, y_r, yaw_r = self.s  # Robot's current position and orientation
        poses = PoseArray()
        poses.header = msg.header  # copy time and frame info
        poses.header.frame_id = 'my_world'

        for i, (r, mean_ang) in enumerate(objects_local):
            # Object center (1 m further along the beam)
            range_to_center = r + self.cylinder_radius
            # Position in robot frame
            x_local = range_to_center * math.cos(mean_ang)
            y_local = range_to_center * math.sin(mean_ang)
            # Transform to world frame
            x_world = x_r + x_local * math.cos(yaw_r) - y_local * math.sin(yaw_r)
            y_world = y_r + x_local * math.sin(yaw_r) + y_local * math.cos(yaw_r)

            self.objects_global = [x_world, y_world]
            # Save as Pose (only position is relevant)
            pose = Pose()
            pose.position.x = x_world
            pose.position.y = y_world
            pose.position.z = 0.0
            poses.poses.append(pose)

        # Publish all detected objects
        self.publisher.publish(poses)
        self.get_logger().info(
            f'Published {len(poses.poses)} detected objects \n at robot position ({x_world:.2f}, {y_world:.2f}).'
        )


def main(args=None):
    rclpy.init(args=args)
    node = LaserScanRepublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
