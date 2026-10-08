# agentnav/core/s1_client.py
"""
Nav2 NavigateToPose action client — Phase 3.

Sends navigation goals to Nav2 and streams progress back via TaskManager.
Handles stop-flag cancellation and base_link → map frame conversion.

Architecture:
  - Background thread: rclpy node + rclpy.spin()
  - navigate_to() coroutine: asyncio, polls threading.Event for completion
  - ROS2 callbacks (on_feedback, on_result) run in the rclpy spin thread
  - threading.Event bridges rclpy callbacks → asyncio polling loop
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
from typing import TYPE_CHECKING, Optional

# Maximum time (seconds) to wait for a navigation goal to complete.
# Override with NAV_TIMEOUT_S env var (e.g. export NAV_TIMEOUT_S=60).
_NAV_TIMEOUT_S = float(os.environ.get("NAV_TIMEOUT_S", "120"))

if TYPE_CHECKING:
    from agentnav.bridge_core.robot_state import RobotState
    from agentnav.bridge_core.task_manager import TaskManager
    from agentnav.bridge_core.telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)


class S1Client:
    """
    Nav2 NavigateToPose action client.

    Usage (from nav driver):
        s1 = S1Client(state, task_mgr)
        s1.start()

        task_id_ref = [None]
        task_id = task_mgr.start(
            lambda: s1.navigate_to(pose, task_id_ref),
            instruction="go to chair",
        )
        task_id_ref[0] = task_id
    """

    def __init__(
        self,
        state: "RobotState",
        task_mgr: "TaskManager",
        notifier: "Optional[TelegramNotifier]" = None,
    ):
        self._state = state
        self._task_mgr = task_mgr
        self._notifier = notifier
        self._node = None
        self._action_client = None
        self._nav2_ready = threading.Event()
        self._started = False

    def start(self) -> None:
        """Start background thread: create rclpy node and connect to Nav2.

        No-op if S1_MODE != 'nav2' (caller should gate before constructing).
        """
        if self._started:
            return
        if self._state.s1_mode != "nav2":
            logger.warning(
                "S1Client.start() called with s1_mode=%r — skipping. "
                "This client only supports nav2.",
                self._state.s1_mode,
            )
            return
        self._started = True
        t = threading.Thread(target=self._spin, name="s1_client_spin", daemon=True)
        t.start()

    @property
    def is_ready(self) -> bool:
        """True once the navigate_to_pose action server is reachable."""
        return self._nav2_ready.is_set()

    # ── Background thread ─────────────────────────────────────────────────────

    def _spin(self) -> None:
        try:
            import rclpy
            from rclpy.action import ActionClient
            from nav2_msgs.action import NavigateToPose
        except ImportError as exc:
            logger.error(
                "S1Client: rclpy/nav2_msgs not available: %s\n"
                "Source /opt/ros/humble/setup.bash before starting the bridge.",
                exc,
            )
            return

        if not rclpy.ok():
            rclpy.init()

        node = rclpy.create_node("agentnav_s1_client")
        self._node = node
        self._action_client = ActionClient(node, NavigateToPose, "navigate_to_pose")

        logger.info("S1Client: waiting for navigate_to_pose action server (30 s)...")
        if self._action_client.wait_for_server(timeout_sec=30.0):
            self._nav2_ready.set()
            logger.info("S1Client: connected to navigate_to_pose")
        else:
            logger.warning(
                "S1Client: navigate_to_pose not available after 30 s. "
                "Start Nav2 and verify with ros_list_nodes()."
            )

        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()

    # ── Navigation coroutine ──────────────────────────────────────────────────

    async def navigate_to(self, pose: dict, task_id_ref: list) -> None:
        """
        Send a NavigateToPose goal and wait until arrived, failed, or stopped.

        Args:
            pose:        {"x", "y", "theta"} in robot base_link frame (metres / rad).
                         Sent as-is to Nav2 with frame_id="base_link" — Nav2 resolves
                         the transform via TF (AMCL-corrected map→base_link).
            task_id_ref: Single-element list; [task_id] is populated by s1_move
                         after task_mgr.start() returns — guaranteed before the
                         event loop runs this coroutine.

        Raises:
            RuntimeError:        Nav2 rejected the goal or navigation failed.
            asyncio.CancelledError: robot_stop() was called during navigation.
        """
        if not self._nav2_ready.is_set():
            raise RuntimeError(
                "Nav2 action server (navigate_to_pose) not connected. "
                "Is Nav2 running? Run ros_list_nodes() to check."
            )

        task_id: Optional[str] = task_id_ref[0]
        notifier = self._notifier
        t_start = time.monotonic()

        x = float(pose["x"])
        y = float(pose.get("y", 0.0))
        theta = float(pose.get("theta", 0.0))
        logger.info(
            "S1Client: navigate_to base_link=(%.3f, %.3f, %.3f)",
            x, y, theta,
        )

        if notifier:
            notifier.reset()

        from nav2_msgs.action import NavigateToPose

        goal = NavigateToPose.Goal()
        # Send in base_link frame — Nav2 transforms to map via TF (AMCL-corrected).
        # This is more accurate than manually converting with drifting /odom.
        goal.pose.header.frame_id = "base_link"
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(theta / 2)
        goal.pose.pose.orientation.w = math.cos(theta / 2)

        # ── Threading bridge: rclpy callbacks → asyncio polling ───────────────
        goal_event = threading.Event()    # set when goal is accepted/rejected
        result_event = threading.Event()  # set when navigation finishes
        goal_handle_ref: list = [None]
        error_ref: list = [None]

        def on_goal_response(future):
            gh = future.result()
            if not gh.accepted:
                error_ref[0] = "Nav2 rejected the goal (plan failed or invalid pose)"
                if notifier:
                    notifier.send("Navigation failed: goal rejected by Nav2.", force=True)
                goal_event.set()
                result_event.set()
                return
            goal_handle_ref[0] = gh
            if task_id:
                self._task_mgr.update(task_id, phase="moving")
            if notifier:
                notifier.send(
                    f"Navigation started.\nGoal: ({x:.2f} m forward, {y:.2f} m lateral)",
                    force=True,
                )
            goal_event.set()
            result_future = gh.get_result_async()
            result_future.add_done_callback(on_result)

        def on_feedback(feedback_msg):
            dist = round(getattr(feedback_msg.feedback, "distance_remaining", 0.0), 2)
            if task_id:
                self._task_mgr.update(task_id, distance_to_goal_m=dist)
            if notifier:
                elapsed = int(time.monotonic() - t_start)
                notifier.send(f"Navigating...\nDistance: {dist} m remaining | {elapsed} s")

        def on_result(future):
            result = future.result()
            try:
                from action_msgs.msg import GoalStatus
                if result.status != GoalStatus.STATUS_SUCCEEDED:
                    error_ref[0] = (
                        f"Navigation failed (Nav2 status={result.status}). "
                        "Try robot_capture() to see current position and retry."
                    )
            except Exception as exc:
                error_ref[0] = str(exc)
            result_event.set()

        # ── Send goal (safe to call from any thread) ──────────────────────────
        if task_id:
            self._task_mgr.update(task_id, phase="planning")

        goal_future = self._action_client.send_goal_async(
            goal, feedback_callback=on_feedback
        )
        goal_future.add_done_callback(on_goal_response)

        # ── Wait for goal acceptance ──────────────────────────────────────────
        while not goal_event.is_set():
            if self._state.should_stop:
                if notifier:
                    notifier.send("Navigation cancelled.", force=True)
                raise asyncio.CancelledError("robot_stop() called before goal accepted")
            await asyncio.sleep(0.2)

        if error_ref[0] and goal_handle_ref[0] is None:
            raise RuntimeError(error_ref[0])

        # ── Wait for navigation to complete ───────────────────────────────────
        while not result_event.is_set():
            if self._state.should_stop:
                gh = goal_handle_ref[0]
                if gh is not None:
                    gh.cancel_goal_async()
                if notifier:
                    notifier.send("Navigation cancelled.", force=True)
                raise asyncio.CancelledError("robot_stop() called during navigation")
            if time.monotonic() - t_start > _NAV_TIMEOUT_S:
                gh = goal_handle_ref[0]
                if gh is not None:
                    gh.cancel_goal_async()
                msg = (
                    f"Navigation timed out after {_NAV_TIMEOUT_S:.0f} s. "
                    "The goal may be unreachable or Nav2 may be stuck. "
                    "Call robot_capture() to reassess the scene."
                )
                if notifier:
                    notifier.send(f"Navigation timed out.\n{msg}", force=True)
                raise RuntimeError(msg)
            await asyncio.sleep(0.5)

        if error_ref[0]:
            if notifier:
                notifier.send(f"Navigation failed.\n{error_ref[0]}", force=True)
            raise RuntimeError(error_ref[0])

        elapsed = int(time.monotonic() - t_start)
        if task_id:
            self._task_mgr.update(task_id, phase="arrived", distance_to_goal_m=0.0)
        if notifier:
            notifier.send(f"Arrived.\nTotal time: {elapsed} s", force=True)
        logger.info("S1Client: task %s arrived at goal (%d s)", task_id, elapsed)
