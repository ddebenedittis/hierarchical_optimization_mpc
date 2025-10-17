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
    name_package = 'limo_simulation'
    modelFileRelativePath = 'model/limo_four_diff.xacro'
    worldFileRelativePath = 'model/empty_world.world'

    pathModelFile = os.path.join(get_package_share_path(name_package), modelFileRelativePath)
    pathWorldFile = os.path.join(get_package_share_path(name_package), worldFileRelativePath)
    # robotDescription = xacro.process_file(pathModelFile).toxml()

    gazebo_rosPakageLaunch = PythonLaunchDescriptionSource(
        os.path.join(get_package_share_path('gazebo_ros'), 'launch', 'gazebo.launch.py')
    )
    gazeboLaunch = IncludeLaunchDescription(
        gazebo_rosPakageLaunch, launch_arguments={'world': pathWorldFile}.items()
    )
    """# Path to gazebo_ros package
    gazebo_ros_pkg = FindPackageShare('gazebo_ros').find('gazebo_ros')
    world_path = os.path.join(gazebo_ros_pkg, 'worlds', 'empty.world')

    # Include Gazebo launch file
    gazebo_ros_launch = PythonLaunchDescriptionSource(
        os.path.join(gazebo_ros_pkg, 'launch', 'gazebo.launch.py')
    )

    gazeboLaunch = IncludeLaunchDescription(
        gazebo_ros_launch,
        launch_arguments={'world': world_path}.items()
    )

    # Set environment variables for GPU rendering
    gpu_env = [
        SetEnvironmentVariable('__NV_PRIME_RENDER_OFFLOAD','1'),
        SetEnvironmentVariable('_GLX_VENDOR_LIBRARY_NAME', 'nvidia'),
    ]
    gazeboLaunch = gpu_env + [gazeboLaunch]"""

    spawnRobots = []
    robotsStatePub = []
    for i in range(1):
        robot_name = f'robot_{i+1}'
        robotDescription = xacro.process_file(
            pathModelFile, mappings={'robot_name': robot_name, 'tf_prefix': robot_name}
        ).toxml()

        load_joint_state_broadcaster = ExecuteProcess(
            cmd=[
                'ros2',
                'control',
                'load_controller',
                '--set-state',
                'active',
                'joint_state_broadcaster',
            ],
            output='screen',
        )
        load_diff_drive_base_controller = ExecuteProcess(
            cmd=[
                'ros2',
                'control',
                'load_controller',
                '--set-state',
                'active',
                'diff_drive_base_controller',
            ],
            output='screen',
        )

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

        # Node to publish the state of the robot to tf
        robotStatePubNode = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name=f'robot_state_publisher',
            namespace=robot_name,
            output='screen',
            parameters=[{'robot_description': robotDescription, 'use_sim_time': True}],
        )
        robotsStatePub.append(robotStatePubNode)
    # spwnModelNode = Node(package='gazebo_ros',
    #                      executable='spawn_entity.py',
    #                      arguments=['-topic', 'robot_description', '-entity', robotXacroName],
    #                      output='screen'
    # )
    # robotStatePubNode = Node(
    #     package='robot_state_publisher',
    #     executable='robot_state_publisher',
    #     name='robot_state_publisher',
    #     output='screen',
    #     parameters=[{'robot_description': robotDescription, 'use_sim_time': True}],
    # )

    LaunchDescriptionObject = LaunchDescription()
    LaunchDescriptionObject.add_action(gazeboLaunch)
    for i in range(1):
        LaunchDescriptionObject.add_action(spawnRobots[i])
        LaunchDescriptionObject.add_action(robotsStatePub[i])
    LaunchDescriptionObject.add_action(load_joint_state_broadcaster)
    LaunchDescriptionObject.add_action(load_diff_drive_base_controller)
    # LaunchDescriptionObject.add_action(spwnModelNode)
    # LaunchDescriptionObject.add_action(robotStatePubNode)

    return LaunchDescriptionObject
