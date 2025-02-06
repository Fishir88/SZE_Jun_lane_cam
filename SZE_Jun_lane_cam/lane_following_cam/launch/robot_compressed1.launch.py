from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lane_following_cam',
            executable='lane_detect',
            output='screen',
            parameters=[{
                'raw_image': False, # True for raw image, False for compressed image
                'image_topic': 'image_raw/compressed'
            }],
        ),
    ])