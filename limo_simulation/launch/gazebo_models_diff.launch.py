import os

import xacro
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # generate coordinates of an square with center in the origin
    a = 5

    P = [[-a, a, 0], [a, a, 0], [a, -a, 0], [-a, -a, 0]]

    # Constants for paths to different files and folders
    robotXacroName = 'limo_four_diff'
    name_package = 'limo_simulation'
    modelFileRelativePath = 'model/limo_four_diff.xacro'
    worldFileRelativePath = 'model/empty_world.world'

    pathModelFile = os.path.join(get_package_share_path(name_package), modelFileRelativePath)
    pathWorldFile = os.path.join(get_package_share_path(name_package), worldFileRelativePath)
    robotDescription = xacro.process_file(pathModelFile).toxml()

    gazebo_rosPakageLaunch = PythonLaunchDescriptionSource(
        os.path.join(get_package_share_path('gazebo_ros'), 'launch', 'gazebo.launch.py')
    )
    gazeboLaunch = IncludeLaunchDescription(
        gazebo_rosPakageLaunch, launch_arguments={'world': pathWorldFile}.items()
    )

    spawnRobots = []
    robotsStatePub = []
    robotsStateBrod = []
    robotsControllers = []
    publishers = []
    for i in range(2):
        robot_name = f'robot_{i+1}'
        robotDescription = xacro.process_file(
            pathModelFile, mappings={'robot_name': robot_name, 'tf_prefix': robot_name}
        ).toxml()

        # Node to publish the state of the robot to tf
        robotStatePubNode = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            namespace=robot_name,
            output='screen',
            parameters=[{'robot_description': robotDescription, 'use_sim_time': True}],
        )

        robotsStatePub.append(robotStatePubNode)

        # Node to spawn the robot in gazebo
        spawnModelNode = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name=f'spawn_entity_{robot_name}',
            arguments=[
                '-topic',
                'robot_description',
                '-entity',
                robot_name,
                '-x',
                str(P[i][0]),
                '-y',
                str(P[i][1]),
                '-z',
                str(P[i][2]),
                '-Y',
                '0.00',
            ],
            namespace=robot_name,
            output='screen',
        )
        spawnRobots.append(spawnModelNode)

        broadcaster_namespace = f'joint_state_broadcaster_{i+1}'
        controller_namespace = f'diff_drive_base_controller_{i+1}'
        load_joint_state_broadcaster = Node(
            package='controller_manager',
            executable='spawner',
            name=broadcaster_namespace,
            namespace=robot_name,
            arguments=[
                'joint_state_broadcaster',
                '--controller-manager',
                'controller_manager',
            ],
            parameters=[{'use_sim_time': True}],
            output='screen',
        )
        robotsStateBrod.append(load_joint_state_broadcaster)

        load_diff_drive_base_controller = Node(
            package='controller_manager',
            executable='spawner',
            name=controller_namespace,
            namespace=robot_name,
            arguments=[
                'diff_drive_base_controller',
                '--controller-manager',
                'controller_manager',
            ],
            parameters=[{'use_sim_time': True}],
            output='screen',
        )
        robotsControllers.append(load_diff_drive_base_controller)

        # 🚀 Add your diff_drive_publisher node here
        publishers.append(
            Node(
                package='limo_simulation',
                executable='diff_drive_publisher',
                name=f'diff_drive_publisher_{i}',
                namespace=robot_name,
                output='screen',
                parameters=[{'use_sim_time': True}],
            )
        )

    # load_joint_state_broadcaster = Node(
    #         package="controller_manager",
    #         executable="spawner",
    #         name='joint_state_broadcaster',
    #         namespace='limo_four_diff',
    #         arguments=['joint_state_broadcaster', '--controller-manager', 'controller_manager',],
    #         parameters=[{'use_sim_time':True}],
    #         output="screen",
    #     )
    # load_diff_drive_base_controller = Node(
    #     package="controller_manager",
    #     executable="spawner",
    #     name='diff_drive_base_controller',
    #     namespace='limo_four_diff',
    #     arguments=['diff_drive_base_controller', '--controller-manager', 'controller_manager',],
    #     parameters=[{'use_sim_time':True}],
    #     output="screen",
    # )

    # spwnModelNode = Node(
    #     package='gazebo_ros',
    #     executable='spawn_entity.py',
    #     arguments=['-topic', 'robot_description', '-entity', robotXacroName],
    #     output='screen',
    # )
    # robotStatePubNode = Node(
    #     package='robot_state_publisher',
    #     executable='robot_state_publisher',
    #     name='robot_state_publisher',
    #     output='screen',
    #     parameters=[{'robot_description': robotDescription, 'use_sim_time': True}],
    # )

    # load_joint_state_broadcaster = ExecuteProcess(
    #     cmd=[
    #         'ros2',
    #         'control',
    #         'load_controller',
    #         '--set-state',
    #         'active',
    #         'joint_state_broadcaster',
    #     ],
    #     output='screen',
    # )
    # load_diff_drive_base_controller = ExecuteProcess(
    #     cmd=[
    #         'ros2',
    #         'control',
    #         'load_controller',
    #         '--set-state',
    #         'active',
    #         'diff_drive_base_controller',
    #     ],
    #     output='screen',
    # )

    LaunchDescriptionObject = LaunchDescription()
    LaunchDescriptionObject.add_action(gazeboLaunch)
    for i in range(2):
        LaunchDescriptionObject.add_action(spawnRobots[i])
        LaunchDescriptionObject.add_action(robotsStatePub[i])
        LaunchDescriptionObject.add_action(robotsStateBrod[i])
        LaunchDescriptionObject.add_action(robotsControllers[i])
        LaunchDescriptionObject.add_action(publishers[i])
    # LaunchDescriptionObject.add_action(spwnModelNode)
    # LaunchDescriptionObject.add_action(robotStatePubNode)
    # LaunchDescriptionObject.add_action(load_joint_state_broadcaster)
    # LaunchDescriptionObject.add_action(load_diff_drive_base_controller)

    return LaunchDescriptionObject
