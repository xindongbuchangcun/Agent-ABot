"""S1-like multi-step task execution for the ABot simulator backend."""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from agentnav.abot.geometry import (
    heading_from_pose,
    local_to_world,
    world_to_local,
    wrap_angle_rad,
)
from agentnav.abot.harness import PixelHarness
from agentnav.abot.observation import front_rgb
from agentnav.abot.types import EpisodeMemory, NavigationTask, PixelMeasurement, TaskStatus


class ExecutorSystemError(RuntimeError):
    """Infrastructure failure that semantic replanning cannot repair."""


@dataclass(frozen=True)
class ExecutorConfig:
    max_semantic_goal_distance: float = 4.0
    midpoint_execution_distance: float = 6.0
    exploration_execution_distance: float = 2.0
    exploration_endpoint_extension_m: float = 0.25
    max_step_length: float = 0.35
    min_clearance: float = 0.35
    max_safe_obstacle_risk: float = 0.5
    enable_long_horizon_preview: bool = False
    long_horizon_preview_m: float = 2.0
    long_horizon_reaction_m: float = 1.2
    local_goal_tolerance: float = 0.15
    goal_tolerance_epsilon: float = 1e-3
    turn_tolerance_deg: float = 5.0
    max_turn_step_deg: float = 45.0
    max_task_steps: int = 30
    stuck_window: int = 3
    stuck_progress_threshold: float = 0.08
    stuck_movement_epsilon: float = 0.05
    failed_goal_radius_m: float = 0.45
    unconfirmed_goal_radius_m: float = 0.75


class ABotS1Executor:
    """Track one agent-created local task across multiple ABot environment steps."""

    def __init__(
        self,
        harness: PixelHarness,
        memory: EpisodeMemory,
        config: ExecutorConfig | None = None,
        secondary_harness: PixelHarness | None = None,
    ) -> None:
        self.harness = harness
        self.secondary_harness = secondary_harness
        self.memory = memory
        self.config = config or ExecutorConfig()
        self.active_task: NavigationTask | None = None
        self.last_plan_debug: dict[str, Any] = {}
        self.total_status_calls = 0
        self._blocked_regions: list[tuple[np.ndarray, np.ndarray]] = []
        self._failed_navigation_pixels: set[tuple[int, int, int]] = set()
        self._last_supported_position: np.ndarray | None = None
        self._retreat_origins: list[np.ndarray] = []
        self._preview_task_id: str | None = None
        self._preferred_detour_side = 0
        self._detour_hold_until = 0
        self._direct_clear_streak = 0
        self._secondary_disabled_for_episode = False
        self._secondary_depth_for_plan: np.ndarray | None = None

    def reset(self) -> None:
        self.active_task = None
        self.last_plan_debug = {}
        self.total_status_calls = 0
        self._blocked_regions = []
        self._failed_navigation_pixels = set()
        self._last_supported_position = None
        self._retreat_origins = []
        self._preview_task_id = None
        self._preferred_detour_side = 0
        self._detour_hold_until = 0
        self._direct_clear_streak = 0
        self._secondary_disabled_for_episode = False
        self._secondary_depth_for_plan = None

    def has_active_task(self) -> bool:
        return self.active_task is not None

    #任务构造
    def create_navigation_task(
        self,
        observation: Any,
        measurement: PixelMeasurement | None,
        semantic_anchor: str,
        navigation_anchor: str,
        exploration: bool = False,
    ) -> NavigationTask:
        step = int(observation.step_count)
        if measurement is None or not measurement.reachable:
            task = NavigationTask(
                task_id=self._task_id(),
                task_type="navigation",
                status=TaskStatus.INVALID_GOAL,
                phase="rejected",
                created_step=step,
                semantic_anchor=semantic_anchor,
                navigation_anchor=navigation_anchor,
                selected_pixel=(
                    (measurement.proposal.u, measurement.proposal.v)
                    if measurement is not None
                    else None
                ),
                failure_reason=(
                    measurement.safety_debug.get("reason", "unreachable_pixel")
                    if measurement is not None
                    else "missing_pixel_measurement"
                ),
            )
            self.active_task = task
            return task

        #3.5/局部
        original = np.asarray(measurement.local_goal, dtype=np.float64)
        original_distance = float(np.linalg.norm(original))
        if exploration and original_distance > 1e-6:
            # A ground pixel marks a route, not a semantic stopping point.
            # Continue a little past it; the executor still checks every step.
            original = original * (
                1.0 + self.config.exploration_endpoint_extension_m / original_distance
            )
            original_distance = float(np.linalg.norm(original))
        is_midpoint = (
            original_distance > self.config.max_semantic_goal_distance
            or exploration and original_distance > self.config.exploration_execution_distance
        )
        task_distance = (
            min(self.config.midpoint_execution_distance, original_distance)
            if is_midpoint
            else original_distance
        )
        if exploration:
            # Reobserve after a short exploratory move instead of committing
            # to a distant pixel whose destination may be a dead end.
            task_distance = min(task_distance, self.config.exploration_execution_distance)
        local_goal = original * (task_distance / original_distance)
        goal_world = local_to_world(local_goal, observation.rotation)
        # An exploration route has no confirmed POI position. Its endpoint
        # must not become the bearing for focused semantic reacquisition.
        semantic_goal_world = (
            None if exploration else local_to_world(original, observation.rotation)
        )
        def reject_prepared_goal(reason: str) -> NavigationTask:
            task = NavigationTask(
                task_id=self._task_id(),
                task_type="navigation",
                status=TaskStatus.INVALID_GOAL,
                phase="rejected",
                created_step=step,
                goal_world=goal_world,
                semantic_goal_world=semantic_goal_world,
                goal_local_initial=local_goal,
                semantic_anchor=semantic_anchor,
                navigation_anchor=navigation_anchor,
                selected_pixel=(measurement.proposal.u, measurement.proposal.v),
                is_midpoint_task=is_midpoint,
                failure_reason=reason,
            )
            self.active_task = task
            return task

        if self.candidate_recently_blocked(observation, measurement):
            return reject_prepared_goal("goal_matches_recent_blocked_region")
        if any(
            float(np.linalg.norm(goal_world[:2] - failed[:2]))
            < self.config.failed_goal_radius_m
            for failed in self.memory.failed_goals
        ):
            return reject_prepared_goal("goal_matches_failed_region")

        matching_unconfirmed = next(
            (
                item
                for item in reversed(self.memory.active_unconfirmed_goals())
                if float(
                    np.linalg.norm(
                        goal_world[:2]
                        - np.asarray(item["world_goal"], dtype=np.float64)[:2]
                    )
                )
                < self.config.unconfirmed_goal_radius_m
            ),
            None,
        )
        if matching_unconfirmed is not None:
            return reject_prepared_goal("goal_matches_recent_unconfirmed_region")

        task = NavigationTask(
            task_id=self._task_id(),
            task_type="navigation",
            status=TaskStatus.RUNNING,
            phase="moving",
            created_step=step,
            goal_world=goal_world,
            semantic_goal_world=semantic_goal_world,
            goal_local_initial=local_goal,
            semantic_anchor=semantic_anchor,
            navigation_anchor=navigation_anchor,
            selected_pixel=(measurement.proposal.u, measurement.proposal.v),
            is_midpoint_task=is_midpoint,
            long_horizon_preview_allowed=(
                measurement.safety_debug.get("reason")
                != "far_depth_bearing_fallback"
            ),
            distance_to_local_goal=task_distance,
            initial_distance=task_distance,
        )
        self.active_task = task
        self.memory.selected_pixels.append(
            {
                "step": step,
                "u": measurement.proposal.u,
                "v": measurement.proposal.v,
                "semantic_anchor": semantic_anchor,
                "navigation_anchor": navigation_anchor,
            }
        )
        self.memory.goals.append(
            {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "navigation_anchor": navigation_anchor,
                "is_midpoint_task": is_midpoint,
                "outcome": "created",
            }
        )
        return task

    def create_depth_retreat_task(self, observation: Any) -> NavigationTask | None:
        """Return to the last pose from which a depth-supported step was issued."""
        previous = self._last_supported_position
        if previous is None:
            return None
        pose = np.asarray(observation.rotation, dtype=np.float64)
        current = pose[:3, 3].copy()
        distance = float(np.linalg.norm((previous - current)[:2]))
        if not 0.2 <= distance <= 1.0:
            return None
        if any(float(np.linalg.norm((origin - current)[:2])) < 1.0
               for origin in self._retreat_origins):
            return None
        self._retreat_origins.append(current)
        local_goal = world_to_local(previous, pose)
        task = NavigationTask(
            task_id=self._task_id(), task_type="depth_retreat",
            status=TaskStatus.RUNNING, phase="aligning",
            created_step=int(observation.step_count),
            goal_world=previous.copy(), goal_local_initial=local_goal,
            navigation_anchor="last depth-supported position",
            distance_to_local_goal=distance, initial_distance=distance,
        )
        self.active_task = task
        return task

    def candidate_recently_blocked(
        self, observation: Any, measurement: PixelMeasurement
    ) -> bool:
        if not measurement.reachable:
            return False
        # Pixel numbers are meaningful only in the frame in which they were
        # selected. A fresh image can legitimately place a new goal at the
        # same numerical coordinate as an old failed pixel.
        if (
            int(observation.step_count),
            measurement.proposal.u,
            measurement.proposal.v,
        ) in self._failed_navigation_pixels:
            return True
        local = np.asarray(measurement.local_goal, dtype=np.float64)
        distance = float(np.linalg.norm(local))
        if distance <= 1e-6:
            return False
        goal_distance = (
            min(self.config.midpoint_execution_distance, distance)
            if distance > self.config.max_semantic_goal_distance
            else distance
        )
        goal_world = local_to_world(local * (goal_distance / distance), observation.rotation)
        position = np.asarray(observation.rotation, dtype=np.float64)[:2, 3]
        for blocked, origin in self._blocked_regions:
            if (
                float(np.linalg.norm(position - origin)) >= 1.0
                or float(np.linalg.norm(goal_world[:2] - blocked[:2])) >= 0.65
            ):
                continue
            old_ray = blocked[:2] - origin
            new_ray = goal_world[:2] - origin
            norms = float(np.linalg.norm(old_ray) * np.linalg.norm(new_ray))
            if norms <= 1e-6:
                return True
            cosine = float(np.dot(old_ray, new_ray) / norms)
            if cosine >= math.cos(math.radians(25.0)):
                return True
        return False

    def create_turn_task(
        self,
        observation: Any,
        direction: str,
        degrees: float,
        reason: str,
    ) -> NavigationTask:
        direction = direction.lower()
        if direction not in {"left", "right"}:
            raise ValueError("turn direction must be left or right")
        signed = abs(float(degrees)) * (1.0 if direction == "left" else -1.0)
        current_heading = heading_from_pose(observation.rotation)
        frame_before, frame_before_sample = self._turn_frame_snapshot(observation)
        task = NavigationTask(
            task_id=self._task_id(),
            task_type="turn",
            status=TaskStatus.RUNNING,
            phase="turning",
            created_step=int(observation.step_count),
            goal_heading_rad=wrap_angle_rad(current_heading + math.radians(signed)),
            turn_direction=direction,
            turn_requested_angle_deg=abs(float(degrees)),
            heading_before_rad=current_heading,
            heading_after_rad=current_heading,
            actual_delta_heading_rad=0.0,
            frame_before=frame_before,
            frame_after=frame_before.copy(),
            frame_change_mae_0_255=0.0,
            turn_frame_before_sample=frame_before_sample,
            navigation_anchor=f"turn_{direction}_{abs(float(degrees)):.1f}_deg",
            semantic_anchor=reason,
            initial_distance=abs(math.radians(signed)),
            distance_to_local_goal=abs(math.radians(signed)),
        )
        self.active_task = task
        self.memory.exploration_history.append(
            f"step {observation.step_count}: TURN {direction} {abs(float(degrees)):.1f}"
        )
        return task

    def task_status(self, observation: Any) -> TaskStatus:
        task = self.active_task
        if task is None:
            return TaskStatus.IDLE
        step = int(observation.step_count)
        if task.last_status_step == step:
            raise RuntimeError(
                f"task_status({task.task_id}) called more than once for ABot step {step}"
            )
        task.last_status_step = step
        task.status_checks += 1
        task.steps_elapsed += 1
        self.total_status_calls += 1

        if task.status.is_terminal:
            return task.status

        pose = np.asarray(observation.rotation, dtype=np.float64)
        position = pose[:3, 3].copy()
        task.position_history.append(position)
        self.memory.visited_positions.append(position)

        if task.task_type == "turn":
            assert task.goal_heading_rad is not None
            current_heading = heading_from_pose(pose)
            task.heading_after_rad = current_heading
            if task.heading_before_rad is not None:
                task.actual_delta_heading_rad = wrap_angle_rad(
                    current_heading - task.heading_before_rad
                )
            frame_after, frame_after_sample = self._turn_frame_snapshot(observation)
            task.frame_after = frame_after
            if task.turn_frame_before_sample is not None:
                task.frame_change_mae_0_255 = float(
                    np.mean(
                        np.abs(
                            frame_after_sample.astype(np.float32)
                            - task.turn_frame_before_sample.astype(np.float32)
                        )
                    )
                )
            remaining = abs(
                wrap_angle_rad(task.goal_heading_rad - current_heading)
            )
            task.distance_to_local_goal = remaining
            task.progress = max(0.0, task.initial_distance - remaining)
            if remaining <= math.radians(self.config.turn_tolerance_deg):
                task.status = TaskStatus.GOAL_REACHED
                task.phase = "arrived"
            elif task.steps_elapsed > self.config.max_task_steps:
                return self._fail(task, TaskStatus.TIMEOUT, "max_task_steps_exceeded")
            return task.status

        if task.goal_world is None:
            return self._fail(task, TaskStatus.SYSTEM_ERROR, "missing_goal_world")
        local_goal = world_to_local(task.goal_world, pose)
        remaining = float(np.linalg.norm(local_goal))
        task.distance_to_local_goal = remaining
        task.progress = max(0.0, task.initial_distance - remaining)
        task.distance_history.append(remaining)
        if remaining <= (
            self.config.local_goal_tolerance
            + self.config.goal_tolerance_epsilon
        ):
            task.status = (
                TaskStatus.MIDPOINT_REACHED
                if task.is_midpoint_task
                else TaskStatus.GOAL_REACHED
            )
            task.phase = "arrived"
            return task.status
        if task.phase != "aligning" and self._is_stuck(task):
            return self._fail(task, TaskStatus.STUCK, "no_local_goal_progress")
        if task.steps_elapsed > self.config.max_task_steps:
            progress_window = 8
            recent_progress = (
                task.distance_history[-progress_window] - task.distance_history[-1]
                if len(task.distance_history) >= progress_window else 0.0
            )
            if (
                task.steps_elapsed > 2 * self.config.max_task_steps
                or recent_progress < self.config.stuck_progress_threshold
            ):
                return self._fail(task, TaskStatus.TIMEOUT, "max_task_steps_exceeded")
        return TaskStatus.RUNNING

    def step(self, observation: Any):
        """Generate one short ABot action after this step's status check."""
        from abotn_evaluator.interface.point_goal import WaypointPrediction

        task = self.active_task
        if task is None:
            raise RuntimeError("executor.step called without an active task")
        step = int(observation.step_count)
        if task.last_status_step != step:
            raise RuntimeError("executor.step requires task_status exactly once first")
        if task.status is not TaskStatus.RUNNING:
            raise RuntimeError(f"executor.step called for terminal status {task.status.value}")

        if task.task_type == "turn":
            assert task.goal_heading_rad is not None
            delta = wrap_angle_rad(
                task.goal_heading_rad - heading_from_pose(observation.rotation)
            )
            command = float(
                np.clip(
                    delta,
                    -math.radians(self.config.max_turn_step_deg),
                    math.radians(self.config.max_turn_step_deg),
                )
            )
            direction = np.array([[math.cos(command), math.sin(command)]], dtype=np.float32)
            self.last_plan_debug = {
                "type": "turn",
                "remaining_deg": math.degrees(delta),
                "command_deg": math.degrees(command),
                "turn_requested_angle_deg": task.turn_requested_angle_deg,
                "heading_before_deg": (
                    math.degrees(task.heading_before_rad)
                    if task.heading_before_rad is not None
                    else None
                ),
                "heading_after_deg": math.degrees(
                    heading_from_pose(observation.rotation)
                ),
                "actual_delta_heading_deg": (
                    math.degrees(task.actual_delta_heading_rad)
                    if task.actual_delta_heading_rad is not None
                    else None
                ),
            }
            return WaypointPrediction(
                waypoint=np.zeros((1, 2), dtype=np.float32),
                directions=direction,
                arrive=False,
                confidence=1.0,
            )

        try:
            depth = self.harness.current_depth(observation)
            secondary_depth = None
            secondary_error = None
            if self.secondary_harness is not None and not self._secondary_disabled_for_episode:
                try:
                    secondary_depth = self.secondary_harness.current_depth(observation)
                except Exception as exc:
                    self._secondary_disabled_for_episode = True
                    secondary_error = f"{type(exc).__name__}: {exc}"[:300]
            local_goal = world_to_local(task.goal_world, observation.rotation)
            self._secondary_depth_for_plan = secondary_depth
            waypoint, candidates = self._plan_local_waypoint(depth, local_goal)
        except Exception as exc:
            self._fail(task, TaskStatus.SYSTEM_ERROR, f"executor_internal_error: {exc}")
            raise ExecutorSystemError(str(exc)) from exc
        finally:
            self._secondary_depth_for_plan = None
        self.last_plan_debug = {
            "type": task.task_type,
            "goal_local_front_left_m": local_goal.tolist(),
            "candidates": candidates,
            "selected_waypoint_front_left_m": waypoint.tolist() if waypoint is not None else None,
        }
        if secondary_error is not None:
            self.last_plan_debug["secondary_depth_error"] = secondary_error
        if waypoint is None:
            if candidates and candidates[0].get("reason") == "goal_outside_front_depth_view":
                bearing_rad = math.atan2(float(local_goal[1]), float(local_goal[0]))
                command_rad = float(np.clip(
                    bearing_rad,
                    -math.radians(self.config.max_turn_step_deg),
                    math.radians(self.config.max_turn_step_deg),
                ))
                task.phase = "aligning"
                self.last_plan_debug["type"] = "navigation_alignment"
                self.last_plan_debug["command_deg"] = math.degrees(command_rad)
                return WaypointPrediction(
                    waypoint=np.zeros((1, 2), dtype=np.float32),
                    directions=np.array([[
                        math.cos(command_rad), math.sin(command_rad)
                    ]], dtype=np.float32),
                    arrive=False,
                    confidence=1.0,
                )
            self._fail(task, TaskStatus.BLOCKED, "no_safe_local_candidate")
            return WaypointPrediction(
                waypoint=np.zeros((1, 2), dtype=np.float32),
                arrive=False,
                confidence=0.0,
            )
        waypoint = np.asarray(waypoint, dtype=np.float32)
        waypoint_norm = float(np.linalg.norm(waypoint))
        if not np.isfinite(waypoint_norm) or waypoint_norm <= 1e-6:
            self._fail(task, TaskStatus.SYSTEM_ERROR, "invalid_navigation_waypoint")
            raise ExecutorSystemError("navigation waypoint must be finite and non-zero")
        task.phase = "moving"
        if task.task_type == "navigation":
            self._last_supported_position = np.asarray(
                observation.rotation, dtype=np.float64
            )[:3, 3].copy()
        # The evaluator maps an API waypoint [a, b] to local [b, -a]
        # before applying the current pose. Its directions are applied without
        # that swap. Convert the position once at this boundary; both the
        # internal goal and output heading remain [forward, left].
        api_waypoint = np.array([-waypoint[1], waypoint[0]], dtype=np.float32)
        direction = waypoint / waypoint_norm
        self.last_plan_debug["selected_direction_front_left"] = direction.tolist()
        self.last_plan_debug["selected_api_waypoint"] = api_waypoint.tolist()
        return WaypointPrediction(
            waypoint=api_waypoint.reshape(1, 2),
            directions=direction.reshape(1, 2),
            arrive=False,
            confidence=1.0,
        )

    def finish_current_task(self) -> NavigationTask | None:
        task = self.active_task
        if task is None:
            return None
        self._preview_task_id = None
        self._preferred_detour_side = 0
        self._detour_hold_until = 0
        self._direct_clear_streak = 0
        if task.status.is_goal_failure and task.task_type != "depth_retreat":
            self.memory.failure_count += 1
            if (
                task.task_type == "navigation"
                and task.selected_pixel is not None
                and task.status in {TaskStatus.BLOCKED, TaskStatus.STUCK, TaskStatus.TIMEOUT}
            ):
                self._failed_navigation_pixels.add(
                    (task.created_step, *task.selected_pixel)
                )
            if (
                task.status is TaskStatus.BLOCKED
                and task.goal_world is not None
                and task.position_history
            ):
                self._blocked_regions.append((
                    task.goal_world.copy(), task.position_history[-1][:2].copy()
                ))
            if (
                task.goal_world is not None
                and task.status in {TaskStatus.STUCK, TaskStatus.TIMEOUT, TaskStatus.FAILED}
                and task.failure_reason != "goal_matches_recent_unconfirmed_region"
            ):
                self.memory.failed_goals.append(task.goal_world.copy())
            self.memory.failed_directions.append(task.navigation_anchor)
        self.memory.task_history.append(task.public_dict())
        for goal in reversed(self.memory.goals):
            if goal.get("task_id") == task.task_id:
                goal["outcome"] = task.status.value
                break
        self.active_task = None
        return task

    def cancel(self, reason: str = "cancelled") -> None:
        if self.active_task is not None:
            self.active_task.status = TaskStatus.CANCELLED
            self.active_task.phase = "failed"
            self.active_task.failure_reason = reason

    def mark_failed(self, status: TaskStatus, reason: str) -> None:
        if self.active_task is None:
            raise RuntimeError("cannot fail a missing task")
        self._fail(self.active_task, status, reason)

    def _plan_local_waypoint(
        self,
        depth: np.ndarray,
        local_goal: np.ndarray,
        secondary_depth: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, list[dict[str, Any]]]:
        if secondary_depth is None:
            secondary_depth = self._secondary_depth_for_plan
        distance = float(np.linalg.norm(local_goal))
        if distance <= 1e-6:
            return np.zeros(2, dtype=np.float64), []
        desired_angle = math.atan2(float(local_goal[1]), float(local_goal[0]))
        camera = getattr(self.harness, "camera", None)
        half_fov = (
            math.atan2(camera.width / 2.0, camera.fx)
            if camera is not None
            else math.radians(55.0)
        )
        if abs(desired_angle) > half_fov:
            return None, [{
                "safe": False,
                "reason": "goal_outside_front_depth_view",
                "goal_bearing_deg": math.degrees(desired_angle),
                "front_half_fov_deg": math.degrees(half_fov),
            }]
        task = self.active_task
        if task is not None and task.task_id != self._preview_task_id:
            self._preview_task_id = task.task_id
            self._preferred_detour_side = 0
            self._detour_hold_until = 0
            self._direct_clear_streak = 0
        horizon = min(self.config.long_horizon_preview_m, distance)
        preview_blocked = False
        direct_preview: tuple[bool, dict[str, Any]] | None = None
        if (
            self.config.enable_long_horizon_preview
            and (task is None or task.long_horizon_preview_allowed)
            and horizon >= 1.0
        ):
            direct_ray = np.array([
                math.cos(desired_angle) * horizon,
                math.sin(desired_angle) * horizon,
            ])
            direct_preview = self.harness.depth_corridor_is_safe(
                depth, direct_ray, max_lookahead_m=horizon
            )
            direct_safe, direct_debug = direct_preview
            preview_blocked = (
                not direct_safe
                and direct_debug.get("reason") == "depth_obstacle"
                and direct_debug.get("nearest_obstacle_m") is not None
                # Distant monocular depth has enough uncertainty to steer us
                # away from a useful target before the obstacle is relevant.
                and float(direct_debug["nearest_obstacle_m"])
                <= min(horizon, self.config.long_horizon_reaction_m)
            )
            if direct_safe:
                self._direct_clear_streak += 1
                if self._direct_clear_streak >= 2:
                    self._preferred_detour_side = 0
            else:
                self._direct_clear_streak = 0
            if task is not None and task.steps_elapsed > self._detour_hold_until:
                self._preferred_detour_side = 0
        distance_to_tolerance = max(
            0.0, distance - self.config.local_goal_tolerance
        )
        step_length = min(self.config.max_step_length, distance_to_tolerance)
        # The final step may be shorter than 5 cm. Keep the progress gate
        # proportional to that step so a safe approach can reach tolerance.
        min_progress = min(0.05, 0.5 * step_length)
        candidates = []
        selected = None
        selected_record = None
        best_score = -math.inf
        offsets_deg = (0.0, 25.0, -25.0, 45.0, -45.0, 65.0, -65.0)
        if preview_blocked:
            offsets_deg += (15.0, -15.0, 30.0, -30.0)
        for offset_deg in offsets_deg:
            angle = desired_angle + math.radians(offset_deg)
            waypoint = np.array(
                [math.cos(angle) * step_length, math.sin(angle) * step_length],
                dtype=np.float64,
            )
            if abs(angle) > half_fov:
                candidates.append({
                    "offset_deg": offset_deg,
                    "waypoint_front_left_m": waypoint.tolist(),
                    "safe": False,
                    "progress_projection_m": float(np.dot(waypoint, local_goal / distance)),
                    "obstacle_risk": 1.0,
                    "score": -1.0,
                    "safety": {"reason": "candidate_outside_front_depth_view"},
                })
                continue
            safe, debug = self.harness.depth_corridor_is_safe(
                depth, waypoint, max_lookahead_m=distance
            )
            secondary_guard = None
            if (
                safe and secondary_depth is not None
                and self.secondary_harness is not None
                and not self._secondary_disabled_for_episode
            ):
                try:
                    _, secondary_debug = self.secondary_harness.depth_corridor_is_safe(
                        secondary_depth, waypoint, max_lookahead_m=distance
                    )
                    primary_scale = debug.get("depth_scale_correction")
                    secondary_scale = secondary_debug.get("depth_scale_correction")
                    scale_ratio = None
                    if (
                        primary_scale is not None and secondary_scale is not None
                        and float(primary_scale) > 0 and float(secondary_scale) > 0
                    ):
                        scale_ratio = float(primary_scale) / float(secondary_scale)
                    scale_uncertain = bool(
                        scale_ratio is not None
                        and (scale_ratio > 1.6 or scale_ratio < 1.0 / 1.6)
                    )
                    nearest = secondary_debug.get("nearest_obstacle_m")
                    strong_obstacle = (
                        secondary_debug.get("reason") == "depth_obstacle"
                        and int(secondary_debug.get("blocking_points", 0)) >= 20
                        and int(secondary_debug.get("blocking_columns", 0)) >= 4
                        and nearest is not None
                        and np.isfinite(float(nearest))
                    )
                    if strong_obstacle:
                        nearest = float(nearest)
                        if nearest <= step_length + 0.05:
                            safe = False
                            status = "blocked"
                        elif nearest <= self.secondary_harness.depth_corridor_lookahead_m:
                            status = "short_step"
                        else:
                            status = "distant"
                        secondary_guard = {
                            "status": status,
                            "nearest_obstacle_m": round(nearest, 3),
                            "blocking_points": secondary_debug["blocking_points"],
                            "blocking_columns": secondary_debug["blocking_columns"],
                        }
                    if scale_uncertain:
                        # Both estimators may call a projected obstacle floor.
                        # Disagreement is uncertainty, so reobserve after a
                        # shorter step instead of declaring a hard obstacle.
                        secondary_guard = {
                            **(secondary_guard or {"status": "scale_disagreement"}),
                            "scale_ratio": round(scale_ratio, 3),
                            "scale_disagreement": True,
                        }
                except Exception as exc:
                    self._secondary_disabled_for_episode = True
                    secondary_guard = {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}"[:300],
                    }
            progress_projection = float(np.dot(waypoint, local_goal / distance))
            obstacle_risk = float(debug.get("obstacle_risk", 0.0 if safe else 1.0))
            # Prefer useful forward progress, while penalizing both depth risk
            # and unnecessary steering.  A corridor can be technically clear
            # below the hard blocking-point threshold but still be too risky.
            score = (
                2.0 * progress_projection
                - 0.75 * obstacle_risk
                - 0.15 * abs(offset_deg) / 65.0
            )
            preview_debug = None
            preview_adjustment = 0.0
            if preview_blocked and safe:
                if offset_deg == 0.0 and direct_preview is not None:
                    preview_safe, preview_debug = direct_preview
                else:
                    preview_ray = np.array([
                        math.cos(angle) * horizon,
                        math.sin(angle) * horizon,
                    ])
                    preview_safe, preview_debug = self.harness.depth_corridor_is_safe(
                        depth, preview_ray, max_lookahead_m=horizon
                    )
                if preview_safe:
                    preview_adjustment = 0.20
                elif preview_debug.get("reason") == "depth_obstacle":
                    nearest = float(preview_debug.get("nearest_obstacle_m") or 0.0)
                    preview_adjustment = -0.20 * max(0.0, horizon - nearest) / horizon
                else:
                    preview_adjustment = -0.05
                side = int(np.sign(offset_deg))
                if (
                    task is not None
                    and task.steps_elapsed <= self._detour_hold_until
                    and self._preferred_detour_side
                    and side
                ):
                    preview_adjustment += (
                        0.08 if side == self._preferred_detour_side else -0.18
                    )
                score += preview_adjustment
            record = {
                "offset_deg": offset_deg,
                "waypoint_front_left_m": waypoint.tolist(),
                "safe": safe,
                "progress_projection_m": progress_projection,
                "obstacle_risk": obstacle_risk,
                "score": score,
                "safety": debug,
            }
            if secondary_guard is not None:
                record["secondary_depth_guard"] = secondary_guard
            if preview_debug is not None:
                record["long_horizon_preview"] = {
                    "horizon_m": round(horizon, 3),
                    "safe": preview_safe,
                    "reason": preview_debug.get("reason"),
                    "nearest_obstacle_m": preview_debug.get("nearest_obstacle_m"),
                    "score_adjustment": round(preview_adjustment, 4),
                    "preferred_side": self._preferred_detour_side,
                }
            candidates.append(record)
            eligible = (
                safe
                and obstacle_risk <= self.config.max_safe_obstacle_risk
                and progress_projection > min_progress
            )
            if eligible and score > best_score:
                selected = waypoint
                selected_record = record
                best_score = score
        if preview_blocked and selected_record is not None and task is not None:
            chosen_preview = selected_record.get("long_horizon_preview") or {}
            chosen_side = int(np.sign(selected_record["offset_deg"]))
            if chosen_side and chosen_preview.get("safe"):
                self._preferred_detour_side = chosen_side
                self._detour_hold_until = task.steps_elapsed + 5
        if (
            selected is not None and selected_record is not None
            and (
                (selected_record.get("secondary_depth_guard") or {}).get("status")
                == "short_step"
                or (selected_record.get("secondary_depth_guard") or {}).get(
                    "scale_disagreement", False
                )
            )
        ):
            selected = selected * 0.5
            selected_record["secondary_step_limited_m"] = round(
                float(np.linalg.norm(selected)), 3
            )
        if selected is not None and candidates:
            direct_safety = candidates[0].get("safety", {})
            nearest = direct_safety.get("nearest_obstacle_m")
            if (
                direct_safety.get("reason") == "depth_obstacle"
                and nearest is not None
                and float(nearest) <= self.config.min_clearance
            ):
                # A side corridor can look clear while the robot passes very
                # close to the frontal obstacle. Reobserve halfway through
                # the detour instead of committing the full 0.35 m step.
                selected = selected * 0.5
                candidates[0]["detour_step_limited_m"] = round(float(np.linalg.norm(selected)), 3)
        return selected, candidates

    @staticmethod
    def _turn_frame_snapshot(
        observation: Any,
    ) -> tuple[dict[str, Any], np.ndarray]:
        image = np.asarray(front_rgb(observation), dtype=np.uint8)
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        image = np.ascontiguousarray(image[:, :, :3])
        height, width = image.shape[:2]
        y_stride = max(1, height // 64)
        x_stride = max(1, width // 64)
        sample = np.ascontiguousarray(
            image[::y_stride, ::x_stride][:64, :64]
        )
        step = int(observation.step_count)
        return (
            {
                "step": step,
                "render_name": f"{step}_front.jpg",
                "shape": list(image.shape),
                "sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            },
            sample,
        )

    def _is_stuck(self, task: NavigationTask) -> bool:
        window = self.config.stuck_window
        if len(task.distance_history) < window or len(task.position_history) < window:
            return False
        progress = task.distance_history[-window] - task.distance_history[-1]
        movement = float(
            np.linalg.norm(
                task.position_history[-1][:2] - task.position_history[-window][:2]
            )
        )
        return (
            progress < self.config.stuck_progress_threshold
            and movement < self.config.stuck_movement_epsilon
        )

    @staticmethod
    def _fail(task: NavigationTask, status: TaskStatus, reason: str) -> TaskStatus:
        task.status = status
        task.phase = "failed"
        task.failure_reason = reason
        return status

    @staticmethod
    def _task_id() -> str:
        return uuid.uuid4().hex[:8]
