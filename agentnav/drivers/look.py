# agentnav/drivers/look.py
"""
Perception tools — Phase 2.

robot_capture: return the current camera frame as MCP ImageContent.
               The agent's native vision perceives it directly.

robot_scan:    rotate to multiple angles and return one frame per angle.
               The agent analyses all frames to determine the best direction.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

logger = logging.getLogger(__name__)

_CAPTURE_LOG_DIR  = os.environ.get("CAPTURE_LOG_DIR",  str(Path.home() / ".agentnav" / "captures"))
_CAPTURE_MAX_FILES = int(os.environ.get("CAPTURE_MAX_FILES", "500"))


def _evict_old_captures(log_dir: Path) -> None:
    """Delete oldest .jpg files when the directory exceeds _CAPTURE_MAX_FILES."""
    files = sorted(log_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
    for old in files[: max(0, len(files) - _CAPTURE_MAX_FILES)]:
        try:
            old.unlink()
        except OSError:
            pass


def _save_frame(frame: bytes) -> None:
    try:
        log_dir = Path(_CAPTURE_LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3] + ".jpg"
        (log_dir / filename).write_bytes(frame)
        _evict_old_captures(log_dir)
    except Exception as e:
        logger.warning("capture log write failed: %s", e)

DRIVER_META = {
    "triggers": ["look", "see", "capture", "scan", "what do you see", "describe", "find"],
    "safety_level": "safe",
    "phase": 2,
    "description": "Capture camera frames for agent visual perception.",
}


def register(mcp: FastMCP, state, task_mgr, ros_client, meta=None) -> None:
    from agentnav.bridge_core.driver_meta import meta_suffix
    from agentnav.bridge_core.robot_state import NavState
    _sfx = meta_suffix(meta) if meta else ""

    def robot_capture() -> ImageContent | str:
        """
        Capture the current camera frame and return it as an image.

        The agent can directly perceive and describe what it sees — no
        external vision model is needed.

        Use this to:
        - Understand the scene before navigating
        - Estimate pixel coordinates of a target (then call pixel_to_pose)
        - Confirm arrival after a navigation task completes

        Returns the latest frame from the configured color camera topic as a JPEG image.
        """
        state.set_nav_state(NavState.LOOKING)
        try:
            frame, depth = state.pop_frame()
            if frame is None:
                return "No camera frame available. Check that /camera/image_raw is publishing."
            # Save depth snapshot so pixel_to_pose() uses the depth that matches
            # this exact frame, not whatever arrives later.
            state.captured_depth = depth
            _save_frame(frame)
            return ImageContent(
                type="image",
                data=base64.b64encode(frame).decode(),
                mimeType="image/jpeg",
            )
        finally:
            state.set_nav_state(NavState.IDLE)

    async def _wait_fresh_frame(seq_before: int, timeout_s: float = 2.0) -> None:
        """Block until a new RGB frame arrives after rotation (frame_seq changes)."""
        t0 = time.monotonic()
        while True:
            with state._frame_lock:
                current_seq = state.frame_seq
            if current_seq != seq_before:
                return
            if time.monotonic() - t0 > timeout_s:
                logger.warning("Timed out waiting for fresh frame after rotation")
                return
            await asyncio.sleep(0.05)

    async def robot_scan(angles: list[int] | None = None) -> list:
        """
        Rotate the robot to multiple angles and capture one frame at each.

        Default angles: [0, 90, 180, 270] degrees.
        Returns a list of images (one per angle) for the agent to analyse.

        Use this when the target is not visible in the current view.
        After scanning, estimate which frame shows the best direction,
        then use pixel_to_pose to compute a goal pose toward it.
        """
        angles = angles or [0, 90, 180, 270]
        results: list = []
        for angle in angles:
            results.append(TextContent(type="text", text=f"--- {angle}° ---"))
            with state._frame_lock:
                seq_before = state.frame_seq
            await asyncio.to_thread(ros_client.rotate_to, angle)
            # Wait for a frame that arrived AFTER rotation completed — the buffer
            # may still hold a frame captured mid-rotation.
            await _wait_fresh_frame(seq_before)
            results.append(robot_capture())
        return results

    mcp.tool(description=(robot_capture.__doc__ or "").strip() + _sfx)(robot_capture)
    mcp.tool(description=(robot_scan.__doc__ or "").strip() + _sfx)(robot_scan)
