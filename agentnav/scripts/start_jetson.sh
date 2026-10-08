#!/usr/bin/env bash
# agentnav/scripts/start_jetson.sh
# Source ROS2 and launch the full robot stack on Jetson.
# Adjust ROS_DISTRO and launch file paths for your setup.
set -euo pipefail

ROS_DISTRO="${ROS_DISTRO:-humble}"
source "/opt/ros/$ROS_DISTRO/setup.bash"

echo "[jetson] Launching robot stack (ROS2 $ROS_DISTRO)..."
ros2 launch your_robot_bringup robot.launch.py
