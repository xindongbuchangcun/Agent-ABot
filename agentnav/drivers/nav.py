# agentnav/drivers/nav.py
"""
Navigation tools — Phase 3.

s1_move:     Send the robot to a pose via Nav2 (non-blocking → returns task_id).
task_status: Poll progress of a navigation task by task_id.
task_cancel: Cancel a running navigation task.
"""
from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

DRIVER_META = {
    "triggers": ["go to", "navigate to", "move to", "drive to", "proceed to", "head to"],
    "safety_level": "caution",
    "phase": 3,
    "description": "Send the robot to a position via Nav2 NavigateToPose.",
}


def register(mcp: FastMCP, state, task_mgr, ros_client, meta=None) -> None:
    from agentnav.bridge_core.driver_meta import meta_suffix
    from agentnav.bridge_core.robot_state import NavState
    from agentnav.bridge_core import server as _server

    _sfx = meta_suffix(meta) if meta else ""
    s1_mode = state.s1_mode

    # S1Client is a long-lived singleton owned by server.py. register() must
    # stay side-effect-free so hot-reload never spawns a second action client.
    s1 = _server.services.get("s1_client")

    # ── s1_move ───────────────────────────────────────────────────────────────

    async def s1_move(pose: dict) -> dict:
        """
        Send the robot to a pose using Nav2 NavigateToPose. Non-blocking.

        The pose is in the robot's current base_link frame (metres / radians):
          x     — forward distance from the robot's current position
          y     — lateral offset (positive = left, negative = right)
          theta — heading change in radians (positive = counter-clockwise)

        Typical workflow:
          1. robot_capture()            — see the scene
          2. pixel_to_pose(u, v)        — convert target pixel to pose
          3. s1_move(pose)              — start navigation, get task_id
          4. task_status(task_id) × N   — poll until arrived or failed
          5. robot_capture()            — visually confirm arrival

        Returns {"task_id": str, "status": "started"} immediately.
        Use task_status(task_id) to monitor progress (poll every 3–5 s).

        The robot clears any previous emergency stop flag before moving.
        If Nav2 is not running, returns {"error": "..."} instead.

        Args:
            pose: {"x": float, "y": float, "theta": float}
        """
        if not isinstance(pose, dict) or "x" not in pose:
            return {"error": 'pose must be {"x": float, "y": float, "theta": float}'}

        if s1 is None:
            return {
                "error": (
                    f"S1_MODE={s1_mode!r} is not yet implemented. "
                    "Only S1_MODE=nav2 is currently supported. "
                    "Set S1_MODE=nav2 and restart the bridge."
                )
            }

        if not s1.is_ready:
            return {
                "error": (
                    "Nav2 not connected. Is Nav2 running? "
                    "Try ros_list_nodes() to verify the navigation stack is up."
                )
            }

        state.clear_stop()
        state.set_nav_state(NavState.MOVING)

        # task_id_ref is populated after task_mgr.start() returns,
        # before the event loop runs navigate_to (no await in between).
        task_id_ref: list = [None]
        task_id = task_mgr.start(
            lambda: s1.navigate_to(pose, task_id_ref),
            instruction=f"s1_move to {pose}",
        )
        task_id_ref[0] = task_id

        logger.info("s1_move: task_id=%s  pose=%s", task_id, pose)
        return {"task_id": task_id, "status": "started"}

    # ── task_status ───────────────────────────────────────────────────────────

    def task_status(task_id: str) -> dict:
        """
        Poll the status of a navigation task started by s1_move.

        Response fields:
          status             — running | retrying | completed | failed | cancelled
          phase              — planning | moving | arrived | failed
          distance_to_goal_m — metres remaining (updated ~1 Hz from Nav2 feedback)
          elapsed_s          — seconds since the task started
          retries            — number of automatic retry attempts so far
          error              — present only if status is "failed"

        Poll every 3–5 seconds. Stop polling when status is one of:
          completed, failed, cancelled

        After "completed", call robot_capture() to visually confirm arrival.

        Args:
            task_id: string returned by s1_move
        """
        return task_mgr.get_status(task_id)

    # ── task_cancel ───────────────────────────────────────────────────────────

    def task_cancel(task_id: str) -> dict:
        """
        Cancel a running navigation task and stop the robot.

        Cancels the asyncio task, sends a cancel request to Nav2, and
        publishes zero velocity to /cmd_vel. Use this to abort a navigation
        mid-route (e.g., wrong destination, obstacle detected by agent).

        For immediate emergency stop (< 50 ms), prefer robot_stop() instead.

        Args:
            task_id: string returned by s1_move
        """
        cancelled = task_mgr.cancel(task_id)
        state.set_stop()
        ros_client.publish_stop()
        if cancelled:
            return {"status": "cancelled", "task_id": task_id}
        return {"status": "not_found_or_already_done", "task_id": task_id}

    mcp.tool(description=(s1_move.__doc__ or "").strip() + _sfx)(s1_move)
    mcp.tool(description=(task_status.__doc__ or "").strip() + _sfx)(task_status)
    mcp.tool(description=(task_cancel.__doc__ or "").strip() + _sfx)(task_cancel)
