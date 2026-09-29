import os

import xacro
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from scripts import GazeboRosPaths


def generate_launch_description():
    # generate coordinates of an square with center in the origin
    a = 1.5

    # P = [[0, 0, 0, 0], [a, a, 0, -3], [a, a, 0, -3], [-a, a, 0, 0]]
    P = [
        [-0.437, -0.618, 0, 0.63],
        [-0.582, 1.416, 0, -0.676],
        [1.852, 1.443, 0, -2.5],
        [1.95, -0.498, 0, 2.5],
    ]

    # Constants for paths to different files and folders
    robotXacroName = 'limo_four_diff'
    name_package = 'limo_simulation'
    modelFileRelativePath = 'model/limo_four_diff.xacro'
    worldFileRelativePath = 'world/my_world.world'

    pathModelFile = os.path.join(get_package_share_path(name_package), modelFileRelativePath)
    pathWorldFile = os.path.join(get_package_share_path(name_package), worldFileRelativePath)
    # robotDescription = xacro.process_file(pathModelFile).toxml()

    """model, plugin, media = GazeboRosPaths.get_paths()
    if 'GAZEBO_MODEL_PATH' in os.environ:
        model += os.pathsep + os.environ['GAZEBO_MODEL_PATH']
    if 'GAZEBO_PLUGIN_PATH' in os.environ:
        plugin += os.pathsep + os.environ['GAZEBO_PLUGIN_PATH']
    if 'GAZEBO_RESOURCE_PATH' in os.environ:
        media += os.pathsep + os.environ['GAZEBO_RESOURCE_PATH']

    gazebo_config_file_path = os.path.join(
        get_package_share_path('limo_simulation'),
        'config',
        'gazebo_params.yaml',
    )
    
    gazebo_server = ExecuteProcess(
        cmd=[
            [
                'ros2 launch gazebo_ros gzserver.launch.py verbose:=true pause:=false world:=',
                pathWorldFile,
                ' params_file:=',
                gazebo_config_file_path,
            ]
        ],
        additional_env={
            '__NV_PRIME_RENDER_OFFLOAD': '1',
            '__GLX_VENDOR_LIBRARY_NAME': 'nvidia',
            'GAZEBO_MODEL_PATH': model,
            'GAZEBO_PLUGIN_PATH': plugin,
            'GAZEBO_RESOURCE_PATH': media,
        },
        shell=True,
        output='screen',
    )

    gazebo_client = ExecuteProcess(
        cmd=[['ros2 launch gazebo_ros gzclient.launch.py']],
        additional_env={'__NV_PRIME_RENDER_OFFLOAD': '1', '__GLX_VENDOR_LIBRARY_NAME': 'nvidia'},
        shell=True,
        output='screen',
    )        """

    gazebo_rosPakageLaunch = PythonLaunchDescriptionSource(
        os.path.join(get_package_share_path('gazebo_ros'), 'launch', 'gazebo.launch.py')
    )
    gazeboLaunch = IncludeLaunchDescription(
        gazebo_rosPakageLaunch,
        launch_arguments={'world': pathWorldFile}.items(),
    )

    spawnRobots = []
    robotsStatePub = []
    robotsStateBrod = []
    robotsControllers = []

    # publishers = []
    for i in range(4):
        robot_name = f'robot_{i+1}'
        robotDescription = xacro.process_file(
            pathModelFile, mappings={'robot_name': robot_name, 'namespace': robot_name}
        ).toxml()
        publisher_name = f'robot_state_publisher'

        # Node to publish the state of the robot to tf
        robotStatePubNode = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name=publisher_name,
            namespace=robot_name,
            output='screen',
            parameters=[{'robot_description': robotDescription, 'use_sim_time': True}],
        )

        robotsStatePub.append(robotStatePubNode)
        spawner_name = f'spawn_entity_{robot_name}'
        # Node to spawn the robot in gazebo
        spawnModelNode = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            name=spawner_name,
            arguments=[
                '-topic',
                f'/{robot_name}/robot_description',
                '-robot_namespace',
                robot_name,
                '-entity',
                f'/{robot_name}',
                '-x',
                str(P[i][0]),
                '-y',
                str(P[i][1]),
                '-z',
                str(P[i][2]),
                '-Y',
                str(P[i][3]),
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
                f'/{robot_name}/controller_manager',
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
                f'/{robot_name}/controller_manager',
            ],
            parameters=[{'use_sim_time': True}],
            output='screen',
        )
        robotsControllers.append(load_diff_drive_base_controller)

    LaunchDescriptionObject = LaunchDescription()

    # LaunchDescriptionObject.add_action(gazebo_server)
    # LaunchDescriptionObject.add_action(gazebo_client)

    LaunchDescriptionObject.add_action(gazeboLaunch)

    for i in range(4):
        LaunchDescriptionObject.add_action(spawnRobots[i])
        LaunchDescriptionObject.add_action(robotsStatePub[i])
        LaunchDescriptionObject.add_action(robotsStateBrod[i])
        LaunchDescriptionObject.add_action(robotsControllers[i])

    return LaunchDescriptionObject
