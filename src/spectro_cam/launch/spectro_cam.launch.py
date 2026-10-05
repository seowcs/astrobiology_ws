import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _spectrum_node(context):
    parameters = [LaunchConfiguration('config').perform(context)]
    device = LaunchConfiguration('device').perform(context)
    if device:
        parameters.append({'device': device})
    return [
        Node(
            package='spectro_cam',
            executable='spectrum_node',
            name='spectrum_node',
            namespace=LaunchConfiguration('namespace'),
            parameters=parameters,
            output='screen',
        )
    ]


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('spectro_cam'), 'config', 'spectro.yaml'
    )
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default_config,
                              description='Parameter YAML file'),
        DeclareLaunchArgument('device', default_value='',
                              description='Override the camera device or test image/video file'),
        DeclareLaunchArgument('namespace', default_value='spectro'),
        OpaqueFunction(function=_spectrum_node),
    ])
