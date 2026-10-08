"""Shared state and records for the ABot AgentNav backend."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class NavMode(str, Enum):
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    RECOVERY = "RECOVERY"
    TERMINATED = "TERMINATED"
    FAILED = "FAILED"
    SYSTEM_ERROR = "SYSTEM_ERROR"


@dataclass
class RuntimeState:
    """Episode-scoped control state owned by the POI agent."""

    mode: NavMode = NavMode.PLANNING
    transition_reason: str = "episode_start"
    terminated: bool = False
    vlm_step_count: int = 0
    executor_step_count: int = 0
    status_call_steps: list[int] = field(default_factory=list)
    progress_positions: list[np.ndarray] = field(default_factory=list)
    last_progress_step: int | None = None
    no_progress_steps: int = 0
    stop_reason: str = ""
    scan_active: bool = False
    scan_completed: bool = False
    scan_direction: str = "left"
    scan_accumulated_deg: float = 0.0
    scan_views_checked: int = 0
    scan_rotation_failures: int = 0
    recovery_scan_cycles: int = 0
    last_scan_start_position: np.ndarray | None = None
    focused_reacquire_active: bool = False
    focused_reacquire_attempted: bool = False
    focused_reacquire_stage: int = 0
    reacquire_goal_world: np.ndarray | None = None
    semantic_relocalization_required: bool = False
    semantic_verification_failures: int = 0
    consecutive_blocked_navigation: int = 0

    def reset(self) -> None:
        fresh = RuntimeState()
        self.__dict__.update(fresh.__dict__)


class TaskStatus(str, Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    MIDPOINT_REACHED = "MIDPOINT_REACHED"
    GOAL_REACHED = "GOAL_REACHED"
    INVALID_GOAL = "INVALID_GOAL"
    BLOCKED = "BLOCKED"
    STUCK = "STUCK"
    TIMEOUT = "TIMEOUT"
    FAILED = "FAILED"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    CANCELLED = "CANCELLED"

    @property
    def is_goal_failure(self) -> bool:
        return self in {
            TaskStatus.INVALID_GOAL,
            TaskStatus.BLOCKED,
            TaskStatus.STUCK,
            TaskStatus.TIMEOUT,
            TaskStatus.FAILED,
        }

    @property
    def is_terminal(self) -> bool:
        return self not in {TaskStatus.IDLE, TaskStatus.RUNNING}


@dataclass(frozen=True)
class AgentSafeObservation:
    """The complete and intentionally small high-level policy input."""

    poi_name: str
    front_rgb: Any
    step_count: int
    mode: NavMode
    memory_summary: dict[str, Any]
    transition_reason: str = ""
    scan_state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int = 720
    height: int = 640
    fx: float = 252.075
    fy: float = 252.075
    cx: float = 360.0
    cy: float = 320.0
    extrinsic_height: float = 0.65


@dataclass(frozen=True)
class PixelProposal:
    candidate_id: str
    u: int
    v: int
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "view": "front",
            "u": self.u,
            "v": self.v,
            "visual_reason": self.reason,
        }


@dataclass(frozen=True)
class PixelMeasurement:
    proposal: PixelProposal
    depth_m: float | None
    depth_mad_m: float | None
    valid_depth_ratio: float
    local_goal: np.ndarray
    depth_reliable: bool
    corridor_safe: bool | None
    safety_debug: dict[str, Any]

    @property
    def reachable(self) -> bool:
        return (
            self.depth_reliable
            and self.corridor_safe is not False
            and float(np.linalg.norm(self.local_goal)) > 0.0
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.proposal.as_dict(),
            "depth_m": self.depth_m,
            "depth_mad_m": self.depth_mad_m,
            "valid_depth_ratio": self.valid_depth_ratio,
            "local_goal_front_left_m": self.local_goal.tolist(),
            "depth_reliable": self.depth_reliable,
            "corridor_safe": self.corridor_safe,
            "safety_debug": self.safety_debug,
            "reachable": self.reachable,
        }


@dataclass
class NavigationTask:
    task_id: str
    task_type: str
    status: TaskStatus
    phase: str
    created_step: int
    goal_world: np.ndarray | None = None
    semantic_goal_world: np.ndarray | None = None
    goal_local_initial: np.ndarray | None = None
    goal_heading_rad: float | None = None
    turn_direction: str = ""
    turn_requested_angle_deg: float | None = None
    heading_before_rad: float | None = None
    heading_after_rad: float | None = None
    actual_delta_heading_rad: float | None = None
    frame_before: dict[str, Any] | None = None
    frame_after: dict[str, Any] | None = None
    frame_change_mae_0_255: float | None = None
    turn_frame_before_sample: np.ndarray | None = field(default=None, repr=False)
    semantic_anchor: str = ""
    navigation_anchor: str = ""
    selected_pixel: tuple[int, int] | None = None
    is_midpoint_task: bool = False
    long_horizon_preview_allowed: bool = False
    steps_elapsed: int = 0
    distance_to_local_goal: float = 0.0
    initial_distance: float = 0.0
    progress: float = 0.0
    failure_reason: str = ""
    last_status_step: int | None = None
    status_checks: int = 0
    distance_history: list[float] = field(default_factory=list)
    position_history: list[np.ndarray] = field(default_factory=list)

    def public_dict(self) -> dict[str, Any]:
        result = {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "task_status": self.status.value,
            "task_phase": self.phase,
            "steps_elapsed": self.steps_elapsed,
            "distance_to_local_goal": round(self.distance_to_local_goal, 4),
            "progress": round(self.progress, 4),
            "failure_reason": self.failure_reason,
            "is_midpoint_task": self.is_midpoint_task,
            "semantic_anchor": self.semantic_anchor,
            "navigation_anchor": self.navigation_anchor,
            "selected_pixel": list(self.selected_pixel) if self.selected_pixel else None,
        }
        if self.task_type == "turn":
            result.update(
                {
                    "turn_direction": self.turn_direction,
                    "turn_requested_angle_deg": self.turn_requested_angle_deg,
                    "heading_before_deg": (
                        round(math.degrees(self.heading_before_rad), 4)
                        if self.heading_before_rad is not None
                        else None
                    ),
                    "heading_after_deg": (
                        round(math.degrees(self.heading_after_rad), 4)
                        if self.heading_after_rad is not None
                        else None
                    ),
                    "actual_delta_heading_deg": (
                        round(math.degrees(self.actual_delta_heading_rad), 4)
                        if self.actual_delta_heading_rad is not None
                        else None
                    ),
                    "frame_before": self.frame_before,
                    "frame_after": self.frame_after,
                    "frame_change_mae_0_255": (
                        round(self.frame_change_mae_0_255, 4)
                        if self.frame_change_mae_0_255 is not None
                        else None
                    ),
                }
            )
        return result


@dataclass
class EpisodeMemory:
    poi_name: str = ""
    visited_positions: list[np.ndarray] = field(default_factory=list)
    semantic_observations: list[str] = field(default_factory=list)
    selected_pixels: list[dict[str, Any]] = field(default_factory=list)
    goals: list[dict[str, Any]] = field(default_factory=list)
    failed_goals: list[np.ndarray] = field(default_factory=list)
    unconfirmed_goals: list[dict[str, Any]] = field(default_factory=list)
    planning_cycle_index: int = 0
    failed_directions: list[str] = field(default_factory=list)
    exploration_history: list[str] = field(default_factory=list)
    seen_signs: list[str] = field(default_factory=list)
    seen_landmarks: list[str] = field(default_factory=list)
    vlm_calls: list[dict[str, Any]] = field(default_factory=list)
    task_history: list[dict[str, Any]] = field(default_factory=list)
    facts: list[dict[str, str]] = field(default_factory=list)
    failure_count: int = 0

    def reset(self) -> None:
        fresh = EpisodeMemory()
        self.__dict__.update(fresh.__dict__)

    def begin_planning_cycle(self) -> None:
        self.planning_cycle_index += 1
        self.unconfirmed_goals = [
            item
            for item in self.unconfirmed_goals
            if int(item["expires_after_planning_cycle"]) >= self.planning_cycle_index
        ]

    def record_unconfirmed_goal(
        self,
        task: NavigationTask,
        reason: str,
        verification_step: int,
        ttl_planning_cycles: int = 3,
        target_visible: bool = False,
    ) -> None:
        if task.goal_world is None or task.task_type != "navigation":
            return
        ttl = max(1, int(ttl_planning_cycles))
        self.unconfirmed_goals.append(
            {
                "world_goal": np.asarray(task.goal_world, dtype=np.float64).copy(),
                "semantic_anchor": task.semantic_anchor,
                "navigation_anchor": task.navigation_anchor,
                "source_pixel": list(task.selected_pixel) if task.selected_pixel else None,
                "source_frame_id": task.created_step,
                "verification_step": int(verification_step),
                "verification_reason": str(reason)[:500],
                "target_visible": bool(target_visible),
                "expires_after_planning_cycle": self.planning_cycle_index + ttl,
            }
        )
        self.unconfirmed_goals = self.unconfirmed_goals[-12:]

    def active_unconfirmed_goals(self) -> list[dict[str, Any]]:
        return [
            item
            for item in self.unconfirmed_goals
            if int(item["expires_after_planning_cycle"]) >= self.planning_cycle_index
        ]

    def high_level_summary(self) -> dict[str, Any]:
        """Return memory without world coordinates or evaluator-only values."""
        recent_actions = []
        for call in self.vlm_calls[-6:]:
            action = str(call.get("terminal_action") or "")
            if action in {"RETURN_TO_PLANNING", "TERMINATE"}:
                continue
            if action:
                recent_actions.append(action)

        recent_failures = [
            {
                "task_type": task.get("task_type"),
                "status": task.get("task_status"),
                "navigation_anchor": task.get("navigation_anchor"),
                "failure_reason": task.get("failure_reason"),
            }
            for task in self.task_history
            if task.get("task_status")
            in {
                TaskStatus.INVALID_GOAL.value,
                TaskStatus.BLOCKED.value,
                TaskStatus.STUCK.value,
                TaskStatus.TIMEOUT.value,
                TaskStatus.FAILED.value,
            }
        ][-6:]

        return {
            "poi_name": self.poi_name,
            "last_action": recent_actions[-1] if recent_actions else None,
            "recent_actions": recent_actions,
            "recent_failures": recent_failures,
            "recent_verification_failures": [
                {
                    "confirmed": item.get("target_visible", False),
                    "semantic_anchor": item["semantic_anchor"],
                    "navigation_anchor": item["navigation_anchor"],
                    "source_frame_id": item["source_frame_id"],
                    "verification_step": item["verification_step"],
                    "reason": item["verification_reason"],
                    "remaining_planning_cycles": max(
                        0,
                        int(item["expires_after_planning_cycle"])
                        - self.planning_cycle_index
                        + 1,
                    ),
                    "instruction": (
                        "Do not immediately return to the same approach point "
                        "unless new visual evidence justifies it."
                    ),
                }
                for item in self.active_unconfirmed_goals()[-6:]
            ],
            "visited_state_count": len(self.visited_positions),
            "recent_semantic_observations": self.semantic_observations[-6:],
            # Pixel coordinates belong only to their source RGB frame. Keep
            # semantic history, but never send stale (u, v) values back to the
            # VLM where they can be copied into a new observation.
            "previous_navigation_anchors": [
                {
                    "source_frame_id": item.get("step"),
                    "semantic_anchor": item.get("semantic_anchor", ""),
                    "navigation_anchor": item.get("navigation_anchor", ""),
                }
                for item in self.selected_pixels[-6:]
            ],
            "previous_goals": [
                {
                    "task_type": item.get("task_type"),
                    "navigation_anchor": item.get("navigation_anchor"),
                    "is_midpoint_task": item.get("is_midpoint_task", False),
                    "outcome": item.get("outcome", ""),
                }
                for item in self.goals[-6:]
            ],
            "failed_directions": self.failed_directions[-6:],
            "exploration_history": self.exploration_history[-6:],
            "seen_signs": self.seen_signs[-10:],
            "seen_landmarks": self.seen_landmarks[-10:],
            "recent_tasks": [
                {
                    "task_type": item.get("task_type"),
                    "status": item.get("task_status"),
                    "failure_reason": item.get("failure_reason", ""),
                    "navigation_anchor": item.get("navigation_anchor", ""),
                    "is_midpoint_task": item.get("is_midpoint_task", False),
                }
                for item in self.task_history[-6:]
            ],
            "facts": self.facts[-10:],
            "failure_count": self.failure_count,
        }
