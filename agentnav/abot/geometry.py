"""ABot camera and waypoint coordinate transforms."""

from __future__ import annotations

import math

import numpy as np

from agentnav.abot.types import CameraIntrinsics

#投影公式
def pixel_to_local_goal(
    u: int,
    v: int,
    depth_m: float,
    camera: CameraIntrinsics,
    stop_margin_m: float,
) -> np.ndarray:
    """Project front-camera z-depth to ABot local ``[forward, left]`` metres."""
    del v  # Vertical position is used by depth probing, not planar projection.
    z = float(depth_m)
    x_right = (float(u) - camera.cx) * z / camera.fx
    goal = np.array([z, -x_right], dtype=np.float64)
    distance = float(np.linalg.norm(goal))
    if distance <= stop_margin_m:
        return np.zeros(2, dtype=np.float64)
    return goal * ((distance - stop_margin_m) / distance)


def local_to_world(local_front_left: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    """Map robot-local [forward, left] into the observation's world pose."""
    local = np.asarray(local_front_left, dtype=np.float64)
    camera_xyz = np.array([local[0], local[1], 0.0], dtype=np.float64)
    pose = np.asarray(camera_to_world, dtype=np.float64)
    return pose[:3, :3] @ camera_xyz + pose[:3, 3]


def world_to_local(goal_world: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    pose = np.asarray(camera_to_world, dtype=np.float64)
    camera_xyz = pose[:3, :3].T @ (np.asarray(goal_world) - pose[:3, 3])
    return np.array([camera_xyz[0], camera_xyz[1]], dtype=np.float64)


def heading_from_pose(camera_to_world: np.ndarray) -> float:
    pose = np.asarray(camera_to_world, dtype=np.float64)
    return math.atan2(float(pose[1, 0]), float(pose[0, 0]))


def wrap_angle_rad(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
