# agentnav/drivers/perception.py
"""
pixel_to_pose — coordinate conversion tool.

Converts a pixel coordinate (u, v) from the current camera frame into a
navigable pose {x, y, theta} in the robot's coordinate frame.

Intended use (via skills/locate.md):
  1. robot_capture()         — agent sees the scene
  2. agent estimates (u, v)  — pixel position of the target in the image
  3. pixel_to_pose(u, v)     — converts to robot-frame pose
  4. s1_move(pose)           — executes navigation
"""
from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

DRIVER_META = {
    "triggers": ["pixel to pose", "locate", "convert coordinates", "where is", "target pose"],
    "safety_level": "safe",
    "phase": 2,
    "description": "Convert pixel coordinates from a camera frame to a robot-frame pose.",
}


def register(mcp: FastMCP, state, task_mgr, ros_client, meta=None) -> None:
    from agentnav.bridge_core.driver_meta import meta_suffix
    from agentnav.bridge_core.robot_state import NavState
    _sfx = meta_suffix(meta) if meta else ""

    def pixel_to_pose(u: int, v: int) -> dict:
        """
        Convert pixel coordinates (u, v) from the current camera frame into
        a navigable pose {x, y, theta} in the robot's coordinate frame.

        Call robot_capture() first, then visually estimate where the target
        is in the image (u = horizontal pixels from left, v = vertical pixels
        from top). Pass those coordinates here to get a pose for s1_move.

        Requires: depth camera publishing on /camera/depth, and ros_client
        running with valid camera intrinsics (CAMERA_FX/FY/CX/CY env vars).

        Args:
            u: horizontal pixel coordinate (0 = left edge of frame)
            v: vertical pixel coordinate (0 = top edge of frame)

        Returns:
            {"x": float, "y": float, "theta": float}
            Pose in the robot's base_link frame (metres / radians).
        """
        state.set_nav_state(NavState.PLANNING)
        try:
            pose = ros_client.pixel_to_pose(u, v)
            state.last_location = f"target at pixel ({u}, {v}) → {pose}"
            logger.info("pixel_to_pose(%d, %d) → %s", u, v, pose)
            return pose
        except RuntimeError as exc:
            return {"error": str(exc)}
        finally:
            state.set_nav_state(NavState.IDLE)

    mcp.tool(description=(pixel_to_pose.__doc__ or "").strip() + _sfx)(pixel_to_pose)
