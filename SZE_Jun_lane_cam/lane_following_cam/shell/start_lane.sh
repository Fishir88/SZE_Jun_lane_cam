#!/bin/bash

# START


screen -ls | grep driver
screen -ls | grep lane

if ! screen -ls | grep -q "lane"
then
    echo -e "\e[42mStart lane\e[0m"
    screen -m -d -S lane bash -c 'source ~/ros2_ws/install/setup.bash && ros2 launch lane_following_cam robot_compressed1.launch.py'
else
    echo -e "\e[41merror\e[0m lane already started"
fi