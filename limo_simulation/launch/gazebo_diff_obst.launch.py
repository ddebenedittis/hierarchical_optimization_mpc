import os

import xacro
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # generate coordinates of an square with center in the origin
    a = 2

    P = [[-a, -a, 0, 0], [-a, -a, 0, 1], [a, a, 0, 1], [-a, a, 0, 1]]

    # Constants for paths to different files and folders
    robotXacroName = 'limo_four_diff'
    name_package = 'limo_simulation'
    modelFileRelativePath = 'model/limo_four_diff.xacro'
    worldFileRelativePath = 'world/my_world.world'

    pathModelFile = os.path.join(get_package_share_path(name_package), modelFileRelativePath)
    pathWorldFile = os.path.join(get_package_share_path(name_package), worldFileRelativePath)
    # robotDescription = xacro.process_file(pathModelFile).toxml()

    gazebo_rosPakageLaunch = PythonLaunchDescriptionSource(
        os.path.join(get_package_share_path('gazebo_ros'), 'launch', 'gazebo.launch.py')
    )
    gazeboLaunch = IncludeLaunchDescription(
        gazebo_rosPakageLaunch, launch_arguments={'world': pathWorldFile}.items()
    )
    controller_params_file = os.path.join(
        get_package_share_path('limo_simulation'), 'config', 'diff_drive_controller.yaml'
    )
    # rviz_config_file = PathJoinSubstitution(
    #     [FindPackageShare("limo_simulation"), "rviz", "model_display.rviz"]
    # )

    spawnRobots = []
    robotsStatePub = []
    robotsStateBrod = []
    robotsControllers = []

    # publishers = []
    for i in range(1):
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

        # # diff_drive_publisher spawner
        # publishers.append(
        #     Node(
        #         package='limo_simulation',
        #         executable='diff_drive_publisher',
        #         name=f'diff_drive_publisher_{i}',
        #         namespace=robot_name,
        #         output='screen',
        #         parameters=[{'use_sim_time': True}],
        #     )
        # )

    """# load_joint_state_broadcaster = Node(
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
    robotDescription = xacro.process_file(
            pathModelFile, mappings={'robot_name': 'robot_1', 'tf_prefix': 'robot_1'}
        ).toxml()
    robotStatePubNode_1 = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace= 'robot_1',
        output='screen',
        parameters=[{'robot_description': robotDescription, 'use_sim_time': True}],
    )
    robotDescription = xacro.process_file(
            pathModelFile, mappings={'robot_name': 'robot_2', 'tf_prefix': 'robot_2'}
        ).toxml()
    robotStatePubNode_2 = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        namespace= 'robot_2',
        output='screen',
        parameters=[{'robot_description': robotDescription, 'use_sim_time': True}],
    )
    
    spwnModelNode_1 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic',
            '/robot_1/robot_description',
            '-robot_namespace',
            'robot_1',
            '-entity',
            'limo_diff_drive_1',
            '-x',
            str(P[0][0]),
            '-y',
            str(P[0][1]),
            '-z',
            str(P[0][2]),
            '-Y',
            str(P[0][3]),
        ],
        output='screen',
    )
    
    spwnModelNode_2 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic',
            '/robot_2/robot_description',
            '-robot_namespace',
            'robot_2',
            '-entity',
            'limo_diff_drive_2',
            '-x',
            str(P[1][0]),
            '-y',
            str(P[1][1]),
            '-z',
            str(P[1][2]),
            '-Y',
            str(P[1][3]),
        ],
        output='screen',
    )
    

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
    
    load_joint_state_broadcaster_r1 = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_state_broadcaster',
             '-c', '/robot_1/controller_manager'],
        output='screen'
    )
    
    load_diff_drive_base_controller_r1 = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'diff_drive_base_controller',
             '-c', '/robot_1/controller_manager'],
        output='screen'
    )
    load_joint_state_broadcaster_r2 = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_state_broadcaster',
             '-c', '/robot_2/controller_manager'],
        output='screen'
    )
    load_diff_drive_base_controller_r2 = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'diff_drive_base_controller',
             '-c', '/robot_2/controller_manager'],
        output='screen'
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
    )"""

    # rviz_node = Node(
    #     package="rviz2",
    #     executable="rviz2",
    #     name="rviz2",
    #     output="log",
    #     arguments=["-d", rviz_config_file],
    # )'''

    LaunchDescriptionObject = LaunchDescription()
    LaunchDescriptionObject.add_action(gazeboLaunch)
    for i in range(1):
        LaunchDescriptionObject.add_action(spawnRobots[i])
        LaunchDescriptionObject.add_action(robotsStatePub[i])
        LaunchDescriptionObject.add_action(robotsStateBrod[i])
        LaunchDescriptionObject.add_action(robotsControllers[i])
        # LaunchDescriptionObject.add_action(publishers[i])
    # LaunchDescriptionObject.add_action(spwnModelNode)
    # LaunchDescriptionObject.add_action(robotStatePubNode)
    # LaunchDescriptionObject.add_action(load_joint_state_broadcaster)
    # LaunchDescriptionObject.add_action(load_diff_drive_base_controller)

    return LaunchDescriptionObject
    # return LaunchDescription([
    #     RegisterEventHandler(
    #         event_handler=OnProcessExit(
    #               target_action=spwnModelNode_1,
    #               on_exit=[
    #                         load_joint_state_broadcaster_r1,
    #                         load_diff_drive_base_controller_r1
    #                       ],
    #         )
    #     ),
    #     RegisterEventHandler(
    #         event_handler=OnProcessExit(
    #               target_action=spwnModelNode_2,
    #               on_exit=[
    #                         load_joint_state_broadcaster_r2,
    #                         load_diff_drive_base_controller_r2
    #                       ],
    #         )
    #     ),
    #     gazeboLaunch,
    #     robotStatePubNode_1,
    #     robotStatePubNode_2,
    #     spwnModelNode_1,
    #     spwnModelNode_2,
    # ])
