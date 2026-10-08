# agentnav/bridge_core/robot_state.py
"""
Shared navigation state — thread-safe.
All drivers and low-level clients synchronize through this object.
"""
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class NavState(Enum):
    IDLE     = "idle"
    LOOKING  = "looking"   # agent is capturing / perceiving
    PLANNING = "planning"  # pixel_to_pose in progress
    MOVING   = "moving"    # S1 is executing motion
    ARRIVED  = "arrived"
    FAILED   = "failed"


@dataclass
class RobotState:
    # ── Navigation state machine ────────────────────────────────────────────
    nav_state: NavState = NavState.IDLE
    _state_lock: threading.Lock = field(default_factory=threading.Lock)

    # ── Latest perception (written by robot_capture / pixel_to_pose) ─────────
    last_scene: str = ""
    last_location: str = ""  # "target at pixel (u, v) → {x, y, theta}"

    # ── Odometry + power (pushed by ros_client) ──────────────────────────────
    pose: dict = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "theta": 0.0})
    velocity: dict = field(default_factory=lambda: {"v": 0.0, "w": 0.0})
    battery_pct: int = -1       # -1 = not yet received from /PowerVoltage
    battery_voltage: float = 0.0  # raw voltage from /PowerVoltage

    # ── Latest camera frame + depth (pushed by ros_client) ──────────────────
    latest_frame: Optional[bytes] = None
    latest_depth: Optional[object] = None  # numpy array
    frame_seq: int = 0                      # incremented on every push_frame
    _frame_lock: threading.Lock = field(default_factory=threading.Lock)

    # ── Depth snapshot captured atomically with the last robot_capture() ─────
    # pixel_to_pose() uses this instead of re-reading latest_depth, ensuring
    # the depth matches the exact frame the agent analysed.
    captured_depth: Optional[object] = None  # numpy array, mm uint16

    # ── Emergency stop flag (written by robot_stop, polled by ros_client) ───
    _stop_flag: bool = False

    # ── S1 connection parameters ─────────────────────────────────────────────
    s1_mode: str = "nav2"       # nav2 (default) | navdp
    s1_checkpoint: str = ""     # only used when s1_mode=navdp

    # ── State machine helpers ────────────────────────────────────────────────
    def set_nav_state(self, state: NavState) -> None:
        with self._state_lock:
            self.nav_state = state

    def get_nav_state(self) -> NavState:
        with self._state_lock:
            return self.nav_state

    # ── Stop flag ────────────────────────────────────────────────────────────
    def set_stop(self) -> None:
        self._stop_flag = True

    def clear_stop(self) -> None:
        self._stop_flag = False

    @property
    def should_stop(self) -> bool:
        return self._stop_flag

    # ── Camera frame ─────────────────────────────────────────────────────────
    def push_frame(self, frame: bytes) -> None:
        """Update latest RGB frame. Called from ros_client color callback."""
        with self._frame_lock:
            self.latest_frame = frame
            self.frame_seq += 1

    def push_depth(self, depth) -> None:
        """Update latest depth array (uint16 mm). Called from ros_client depth callback."""
        with self._frame_lock:
            self.latest_depth = depth

    def pop_frame(self) -> tuple[Optional[bytes], Optional[object]]:
        with self._frame_lock:
            return self.latest_frame, self.latest_depth
