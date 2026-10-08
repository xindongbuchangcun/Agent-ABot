"""POI agent that combines nanobot reasoning with a persistent S1 executor."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from abotn_evaluator.interface.poi_goal import BasePoiGoalAgent
from abotn_evaluator.interface.point_goal import WaypointPrediction

from agentnav.abot.depth import (
    Metric3DDepthEstimator, ObservationDepthCache, UniDepthV2DepthEstimator,
)
from agentnav.abot.executor import ABotS1Executor, ExecutorConfig, ExecutorSystemError
from agentnav.abot.geometry import world_to_local
from agentnav.abot.harness import PixelHarness
from agentnav.abot.high_level import NanobotPoiPlanner
from agentnav.abot.observation import to_agent_safe_observation
from agentnav.abot.types import (
    CameraIntrinsics,
    EpisodeMemory,
    NavMode,
    RuntimeState,
    TaskStatus,
)


class AgentNavPoiGoalAgent(BasePoiGoalAgent):
    """AgentNav high-level planning plus old-Harness metric-depth execution."""

    def __init__(
        self,
        workspace: str = "agentnav/abot/workspace",
        api_base: str = "http://localhost:8000/v1",
        model: str = "qwen3-vl-4b-instruct",
        metric3d_source: str = "/home/lifan/Benchmark/models/depth_sources/Metric3D",
        metric3d_checkpoint: str = "/home/lifan/Benchmark/models/Metric3D-ViT-Small/metric_depth_vit_small_800k.pth",
        depth_device: str = "cuda:1",
        enable_secondary_depth_guard: bool = False,
        unidepth_source: str = "/home/lifan/Benchmark/models/depth_sources/UniDepth",
        unidepth_checkpoint: str = "/home/lifan/Benchmark/models/unidepth-v2-vits14",
        log_dir: str = "outputs/abot_agentnav_logs",
        camera_width: int = 720,
        camera_height: int = 640,
        camera_fx: float = 252.075,
        camera_fy: float = 252.075,
        camera_height_m: float = 0.65,
        robot_radius_m: float = 0.10,
        depth_safety_margin_m: float = 0.10,
        min_obstacle_height_m: float = 0.08,
        stop_margin_m: float = 0.15,
        max_reliable_depth_m: float = 30.0,
        depth_patch_size: int = 5,
        depth_valid_ratio_threshold: float = 0.6,
        query_budget: int = 8,
        duplicate_query_budget: int = 1,
        max_semantic_goal_distance: float = 4.0,
        midpoint_execution_distance: float = 6.0,
        exploration_execution_distance: float = 2.0,
        max_step_length: float = 0.35,
        min_clearance: float = 0.35,
        max_safe_obstacle_risk: float = 0.5,
        enable_long_horizon_preview: bool = False,
        long_horizon_preview_m: float = 2.0,
        long_horizon_reaction_m: float = 1.2,
        local_goal_tolerance: float = 0.15,
        depth_corridor_lookahead_m: float = 0.70,
        unconfirmed_goal_radius_m: float = 0.75,
        unconfirmed_goal_ttl_planning_cycles: int = 3,
        max_task_steps: int = 30,
        stuck_window: int = 3,
        stuck_progress_threshold: float = 0.08,
        verify_terminate_distance_m: float = 2.0,
        max_tool_iterations: int = 40,
        arrive_threshold: float | None = None,
        planner: Any | None = None,
        depth_estimator: Any | None = None,
        secondary_depth_estimator: Any | None = None,
        max_no_progress_steps: int = 24,
        progress_radius_m: float = 0.5,
        vlm_max_tokens: int = 1024,
        vlm_request_timeout_s: float = 120.0,
        vlm_request_recovery_attempts: int = 2,
        vlm_context_window_tokens: int = 8192,
        vlm_context_safety_margin_tokens: int = 64,
        vlm_min_completion_tokens: int = 128,
        vlm_image_scale: float = 2.0,
        scan_direction: str = "left",
        scan_increment_deg: float = 45.0,
        max_scan_rotation_failures: int = 2,
        max_recovery_scan_cycles: int = 1,
        max_planning_cycles_per_episode: int = 40,
    ) -> None:
        # Runner compatibility only. The evaluator owns success thresholds;
        # the high-level policy never receives or uses this value.
        del arrive_threshold
        self.max_no_progress_steps = int(max_no_progress_steps)
        self.progress_radius_m = float(progress_radius_m)
        if self.max_no_progress_steps < 1:
            raise ValueError("max_no_progress_steps must be positive")
        if not np.isfinite(self.progress_radius_m) or self.progress_radius_m <= 0:
            raise ValueError("progress_radius_m must be positive and finite")
        self.scan_direction = str(scan_direction).lower()
        self.scan_increment_deg = float(scan_increment_deg)
        if self.scan_direction not in {"left", "right"}:
            raise ValueError("scan_direction must be left or right")
        if not np.isfinite(self.scan_increment_deg) or not 10.0 <= self.scan_increment_deg <= 90.0:
            raise ValueError("scan_increment_deg must be between 10 and 90 degrees")
        self.max_scan_rotation_failures = int(max_scan_rotation_failures)
        if self.max_scan_rotation_failures < 1:
            raise ValueError("max_scan_rotation_failures must be positive")
        self.max_recovery_scan_cycles = int(max_recovery_scan_cycles)
        if self.max_recovery_scan_cycles < 0:
            raise ValueError("max_recovery_scan_cycles must be non-negative")
        self.max_planning_cycles_per_episode = int(max_planning_cycles_per_episode)
        if self.max_planning_cycles_per_episode < 1:
            raise ValueError("max_planning_cycles_per_episode must be positive")
        self.memory = EpisodeMemory()
        camera = CameraIntrinsics(
            camera_width,
            camera_height,
            camera_fx,
            camera_fy,
            camera_width / 2.0,
            camera_height / 2.0,
            camera_height_m,
        )
        estimator = depth_estimator or Metric3DDepthEstimator(
            metric3d_source, metric3d_checkpoint, depth_device
        )
        self.harness = PixelHarness(
            ObservationDepthCache(estimator),
            camera,
            stop_margin_m=stop_margin_m,
            robot_radius_m=robot_radius_m,
            depth_safety_margin_m=depth_safety_margin_m,
            min_obstacle_height_m=min_obstacle_height_m,
            max_reliable_depth_m=max_reliable_depth_m,
            depth_patch_size=depth_patch_size,
            depth_valid_ratio_threshold=depth_valid_ratio_threshold,
            depth_corridor_lookahead_m=depth_corridor_lookahead_m,
            query_budget=query_budget,
            duplicate_query_budget=duplicate_query_budget,
        )
        self.secondary_harness = None
        if enable_secondary_depth_guard:
            secondary_estimator = secondary_depth_estimator or UniDepthV2DepthEstimator(
                unidepth_source, unidepth_checkpoint, camera, depth_device
            )
            self.secondary_harness = PixelHarness(
                ObservationDepthCache(secondary_estimator),
                camera,
                robot_radius_m=robot_radius_m,
                depth_safety_margin_m=depth_safety_margin_m,
                min_obstacle_height_m=min_obstacle_height_m,
                depth_corridor_lookahead_m=depth_corridor_lookahead_m,
            )
        self.executor = ABotS1Executor(
            self.harness,
            self.memory,
            ExecutorConfig(
                max_semantic_goal_distance=max_semantic_goal_distance,
                midpoint_execution_distance=midpoint_execution_distance,
                exploration_execution_distance=exploration_execution_distance,
                max_step_length=max_step_length,
                min_clearance=min_clearance,
                max_safe_obstacle_risk=max_safe_obstacle_risk,
                enable_long_horizon_preview=enable_long_horizon_preview,
                long_horizon_preview_m=long_horizon_preview_m,
                long_horizon_reaction_m=long_horizon_reaction_m,
                local_goal_tolerance=local_goal_tolerance,
                unconfirmed_goal_radius_m=unconfirmed_goal_radius_m,
                max_task_steps=max_task_steps,
                stuck_window=stuck_window,
                stuck_progress_threshold=stuck_progress_threshold,
            ),
            secondary_harness=self.secondary_harness,
        )
        self.unconfirmed_goal_ttl_planning_cycles = max(
            1, int(unconfirmed_goal_ttl_planning_cycles)
        )
        self.verify_terminate_distance_m = float(verify_terminate_distance_m)
        if (
            not np.isfinite(self.verify_terminate_distance_m)
            or self.verify_terminate_distance_m <= 0
        ):
            raise ValueError(
                "verify_terminate_distance_m must be a positive finite value"
            )
        self.planner = planner or NanobotPoiPlanner(
            workspace,
            api_base,
            model,
            max_tool_iterations=max_tool_iterations,
            max_tokens=vlm_max_tokens,
            request_timeout_s=vlm_request_timeout_s,
            request_recovery_attempts=vlm_request_recovery_attempts,
            context_window_tokens=vlm_context_window_tokens,
            context_safety_margin_tokens=vlm_context_safety_margin_tokens,
            min_completion_tokens=vlm_min_completion_tokens,
            image_scale=vlm_image_scale,
        )
        self.log_dir = Path(log_dir)
        run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
        self.log_dir = self.log_dir / run_id
        self.log_dir.mkdir(parents=True, exist_ok=False)
        self.episode_index = -1
        self.runtime = RuntimeState()
        self.reset()

    def reset(self) -> None:
        self.memory.reset()
        self.harness.reset()
        if self.secondary_harness is not None:
            self.secondary_harness.reset()
        self.executor.reset()
        self.episode_index += 1
        self.runtime.reset()
        self.runtime.scan_direction = self.scan_direction
        self.log_path = self.log_dir / f"episode_{self.episode_index:06d}.jsonl"

    def predict(self, observation: Any) -> WaypointPrediction:
        self.memory.poi_name = str(observation.poi_name)
        trace: dict[str, Any] = {
            "step": int(observation.step_count),
            "mode_before": self.mode.value,
            "transition_reason": self.transition_reason,
            "gt_fields_sent_to_vlm": False,
            "gt_depth_used": False,
            "occupancy_map_used": False,
        }
        # Evaluator-only visualization metadata. It is never included in
        # AgentSafeObservation or in a VLM message.
        pose = np.asarray(observation.rotation, dtype=np.float64)
        trace["visual_pose_world_xy_m"] = pose[:2, 3].tolist()
        trace["visual_heading_deg"] = float(
            np.degrees(np.arctan2(pose[1, 0], pose[0, 0]))
        )
        if self.terminated:
            return self._stop_prediction(trace)

        self._record_spatial_progress(observation, trace)

        finished = None
        if self.executor.has_active_task():
            status = self.executor.task_status(observation)
            self.status_call_steps.append(int(observation.step_count))
            trace["task_status"] = status.value
            trace["task"] = self.executor.active_task.public_dict()
            if status is TaskStatus.RUNNING:
                if (
                    not self.runtime.scan_active
                    and not self.runtime.focused_reacquire_active
                    and self.runtime.no_progress_steps >= self.max_no_progress_steps
                ):
                    return self._stop_stalled(trace)
                self.mode = NavMode.EXECUTING
                try:
                    prediction = self.executor.step(observation)
                except ExecutorSystemError as exc:
                    self.mode = NavMode.SYSTEM_ERROR
                    finished = self.executor.finish_current_task()
                    trace["task_status"] = TaskStatus.SYSTEM_ERROR.value
                    trace["finished_task"] = finished.public_dict() if finished else None
                    trace["system_error"] = str(exc)
                    trace["vlm_called"] = False
                    self._log(trace)
                    raise
                self.executor_step_count += 1
                trace["executor_plan"] = self.executor.last_plan_debug
                trace["vlm_called"] = False
                prediction.extra["agentnav_trace"] = trace
                self._log(trace)
                return prediction

            finished = self.executor.finish_current_task()
            trace["finished_task"] = finished.public_dict() if finished else None
            if status is TaskStatus.SYSTEM_ERROR:
                self.mode = NavMode.SYSTEM_ERROR
                self._log(trace)
                raise ExecutorSystemError(finished.failure_reason if finished else "system error")
            if status is TaskStatus.MIDPOINT_REACHED:
                self.mode = NavMode.PLANNING
                self.transition_reason = TaskStatus.MIDPOINT_REACHED.value
                if finished is not None:
                    self._arm_focused_reacquisition(finished)
                    self.harness.invalidate_frame_queries(finished.created_step)
            elif status is TaskStatus.GOAL_REACHED:
                if finished is None:
                    self.mode = NavMode.SYSTEM_ERROR
                    self._log(trace)
                    raise ExecutorSystemError("completed task record is missing")
                if finished.task_type == "turn":
                    if self.runtime.focused_reacquire_active:
                        self.runtime.focused_reacquire_active = False
                        self.mode = NavMode.PLANNING
                        self.transition_reason = "FOCUSED_REACQUIRE_VIEW_READY"
                    elif self.runtime.scan_active:
                        actual_deg = abs(
                            float(np.degrees(finished.actual_delta_heading_rad or 0.0))
                        )
                        self.runtime.scan_accumulated_deg = min(
                            360.0, self.runtime.scan_accumulated_deg + actual_deg
                        )
                        self.runtime.scan_views_checked += 1
                        self.runtime.scan_rotation_failures = 0
                        scan_complete = self.runtime.scan_accumulated_deg >= (
                            360.0 - 0.5
                        )
                        self.runtime.scan_completed = scan_complete
                        self.runtime.scan_active = not scan_complete
                        self.mode = NavMode.PLANNING
                        self.transition_reason = (
                            "SCAN_360_COMPLETED_NO_TARGET"
                            if scan_complete
                            else "SCAN_VIEW_READY"
                        )
                    else:
                        raise ExecutorSystemError(
                            "turn task completed outside scan or focused reacquisition"
                        )
                    self.harness.invalidate_frame_queries(finished.created_step)
                elif finished.task_type == "depth_retreat":
                    self.mode = NavMode.PLANNING
                    self.transition_reason = "DEPTH_RETREAT_REACHED"
                    self.runtime.semantic_relocalization_required = False
                    self.harness.invalidate_frame_queries(finished.created_step)
                elif finished.task_type == "navigation":
                    self.mode = NavMode.VERIFYING
                    self.transition_reason = TaskStatus.GOAL_REACHED.value
                else:
                    self.mode = NavMode.SYSTEM_ERROR
                    self._log(trace)
                    raise ExecutorSystemError(
                        f"unsupported completed task type: {finished.task_type}"
                    )
            else:
                if (
                    finished is not None
                    and finished.task_type == "turn"
                    and self.runtime.focused_reacquire_active
                ):
                    self.runtime.focused_reacquire_active = False
                    trace["focused_reacquire_rotation_failure"] = {
                        "task_status": status.value,
                        "failure_reason": finished.failure_reason,
                    }
                elif (
                    finished is not None
                    and finished.task_type == "turn"
                    and self.runtime.scan_active
                ):
                    self.runtime.scan_rotation_failures += 1
                    trace["scan_rotation_failure"] = {
                        "count": self.runtime.scan_rotation_failures,
                        "limit": self.max_scan_rotation_failures,
                        "task_status": status.value,
                        "failure_reason": finished.failure_reason,
                    }
                    if (
                        self.runtime.scan_rotation_failures
                        >= self.max_scan_rotation_failures
                    ):
                        self.mode = NavMode.FAILED
                        self.terminated = True
                        self.transition_reason = "SCAN_ROTATION_FAILED"
                        self.runtime.stop_reason = "scan_rotation_failed"
                        trace.update(
                            stop_reason=self.runtime.stop_reason,
                            mode_after=self.mode.value,
                            vlm_called=False,
                        )
                        prediction = self._stop_prediction(trace)
                        self._log(trace)
                        return prediction
                if finished is not None and finished.task_type == "depth_retreat":
                    self.runtime.semantic_relocalization_required = True
                if finished is not None and finished.task_type == "navigation":
                    if status is TaskStatus.BLOCKED:
                        self.runtime.consecutive_blocked_navigation += 1
                        sparse = self._plan_lacks_depth_evidence(self.executor.last_plan_debug)
                        if sparse:
                            retreat = self.executor.create_depth_retreat_task(observation)
                            if retreat is not None:
                                self._arm_focused_reacquisition(finished)
                                self.mode = NavMode.EXECUTING
                                self.transition_reason = "DEPTH_RETREAT"
                                trace["depth_retreat_reason"] = "missing_depth_evidence"
                                trace["depth_retreat_task"] = retreat.public_dict()
                                prediction = self._prediction(np.zeros(2), False, trace)
                                self._log(trace)
                                return prediction
                            # New pixels cannot restore absent near-depth evidence.
                            self.runtime.semantic_relocalization_required = True
                    else:
                        self.runtime.consecutive_blocked_navigation = 0
                    self._arm_focused_reacquisition(finished)
                self.mode = NavMode.RECOVERY
                self.transition_reason = status.value
            trace["transition_reason"] = self.transition_reason

        # Allow the final visual verification before declaring a stall.
        if (
            self.mode is not NavMode.VERIFYING
            and not self.runtime.scan_active
            and not self.runtime.focused_reacquire_active
            and not self.runtime.scan_completed
            and self.runtime.no_progress_steps >= self.max_no_progress_steps
        ):
            return self._stop_stalled(trace)

        if self.mode in {NavMode.PLANNING, NavMode.RECOVERY}:
            if self.vlm_step_count >= self.max_planning_cycles_per_episode:
                return self._stop_planning_budget(trace)
            self.memory.begin_planning_cycle()
        safe = to_agent_safe_observation(
            observation,
            self.mode,
            self.memory,
            self.transition_reason,
            self._scan_state(),
        )
        try:
            decision = self.planner.decide(
                safe, observation, self.harness, self.executor, self.memory
            )
        except Exception as exc:
            self.mode = NavMode.SYSTEM_ERROR
            trace["vlm_called"] = True
            error = str(exc) or type(exc).__name__
            trace["system_error"] = f"high_level_error: {error}"
            self._log(trace)
            raise ExecutorSystemError(error) from exc
        self.vlm_step_count += 1
        action = decision.terminal["action"]
        terminal_task = decision.terminal.get("task") or {}
        self.memory.vlm_calls.append(
            {
                "step": int(observation.step_count),
                "mode": self.mode.value,
                "latency_s": decision.latency_s,
                "usage": decision.usage,
                "terminal_action": action,
            }
        )
        trace.update(
            {
                "vlm_called": True,
                "vlm": {
                    "safe_input": {
                        "poi_name": safe.poi_name,
                        "step_count": safe.step_count,
                        "mode": safe.mode.value,
                        "memory": safe.memory_summary,
                        "scan_state": safe.scan_state,
                    },
                    "raw_responses": decision.raw_responses,
                    "tool_trace": decision.tool_trace,
                    "terminal": decision.terminal,
                    "latency_s": decision.latency_s,
                    "usage": decision.usage,
                },
            }
        )
        if action == "TERMINATE":
            if self.mode is not NavMode.VERIFYING:
                raise ExecutorSystemError("TERMINATE is only valid in VERIFYING mode")
            if not decision.terminal.get("target_visible", False):
                raise ExecutorSystemError("TERMINATE requires confirmed POI visibility")
            try:
                distance_to_goal_m = float(observation.distance_to_goal)
            except (AttributeError, TypeError, ValueError):
                distance_to_goal_m = float("nan")
            distance_valid = bool(np.isfinite(distance_to_goal_m))
            within_terminate_distance = bool(
                distance_valid
                and distance_to_goal_m <= self.verify_terminate_distance_m
            )
            trace["verification_gate"] = {
                "target_visible": True,
                "distance_to_goal_m": distance_to_goal_m if distance_valid else None,
                "terminate_distance_threshold_m": self.verify_terminate_distance_m,
                "within_terminate_distance": within_terminate_distance,
            }
            if within_terminate_distance:
                self.terminated = True
                self.mode = NavMode.TERMINATED
                prediction = self._prediction(np.zeros(2), True, trace)
            else:
                continuation = decision.continuation_measurement
                if continuation is not None and continuation.reachable:
                    task = self.executor.create_navigation_task(
                        observation,
                        continuation,
                        self.memory.poi_name,
                        continuation.proposal.reason
                        or "visible POI continuation",
                    )
                    decision.terminal["continuation_task"] = task.public_dict()
                    trace["vlm"]["terminal"] = decision.terminal
                    if task.status is TaskStatus.RUNNING:
                        self.mode = NavMode.EXECUTING
                        self.transition_reason = (
                            "POI_VISIBLE_CONTINUE_APPROACH"
                        )
                    else:
                        self.executor.finish_current_task()
                        self.mode = NavMode.RECOVERY
                        self.transition_reason = task.status.value
                else:
                    self.mode = NavMode.RECOVERY
                    self.transition_reason = (
                        "POI_VISIBLE_BUT_NO_CONTINUATION_CANDIDATE"
                    )
                prediction = self._prediction(np.zeros(2), False, trace)
        elif action == "RETURN_TO_PLANNING":
            if self.mode is not NavMode.VERIFYING:
                raise ExecutorSystemError(
                    "RETURN_TO_PLANNING is only valid in VERIFYING mode"
                )
            if finished is not None:
                self.memory.record_unconfirmed_goal(
                    finished,
                    str(
                        decision.terminal.get(
                            "reason", "POI was not visually confirmed after arrival"
                        )
                    ),
                    int(observation.step_count),
                    self.unconfirmed_goal_ttl_planning_cycles,
                )
            self.runtime.semantic_verification_failures += 1
            # The arrived-at local goal was not the POI. Reinspect this new
            # view before spending environment steps on a scan, and do not
            # steer back toward the disproved goal's saved world bearing.
            self.runtime.semantic_relocalization_required = False
            self.runtime.reacquire_goal_world = None
            self.runtime.focused_reacquire_active = False
            self.runtime.focused_reacquire_attempted = True
            self.mode = NavMode.RECOVERY
            self.transition_reason = "POI_NOT_CONFIRMED"
            prediction = self._prediction(np.zeros(2), False, trace)
        elif action == "SCAN_360":
            if self.mode not in {NavMode.PLANNING, NavMode.RECOVERY}:
                raise ExecutorSystemError(
                    "SCAN_360 is only valid in PLANNING or RECOVERY mode"
                )
            if self.runtime.scan_completed:
                raise ExecutorSystemError("SCAN_360 cannot restart after a completed scan")
            if not self.runtime.scan_active:
                focused_task = (
                    None
                    if self.runtime.semantic_relocalization_required
                    else self._try_focused_reacquisition(observation, decision)
                )
                if focused_task is not None:
                    decision.terminal["task"] = focused_task.public_dict()
                    decision.terminal["execution_mode"] = "FOCUSED_REACQUIRE"
                    trace["vlm"]["terminal"] = decision.terminal
                    self.mode = NavMode.EXECUTING
                    self.transition_reason = "FOCUSED_REACQUIRE"
                    prediction = self._execute_new_turn_task(observation, trace)
                    self._log(trace)
                    return prediction
                position = np.asarray(observation.rotation, dtype=np.float64)[:2, 3]
                previous_scan = self.runtime.last_scan_start_position
                relocation_m = max(1.0, 2.0 * self.progress_radius_m)
                if previous_scan is not None and float(
                    np.linalg.norm(position - previous_scan)
                ) >= relocation_m:
                    # A full scan is informative again after meaningful
                    # translation, but small shifts cannot refresh the budget.
                    self.runtime.recovery_scan_cycles = 0
                if self.mode is NavMode.RECOVERY:
                    if self.runtime.recovery_scan_cycles >= self.max_recovery_scan_cycles:
                        trace["recovery_scan_limit"] = {
                            "completed_cycles": self.runtime.recovery_scan_cycles,
                            "max_cycles": self.max_recovery_scan_cycles,
                            "distance_since_last_scan_m": (
                                None if previous_scan is None else
                                float(np.linalg.norm(position - previous_scan))
                            ),
                            "required_relocation_m": relocation_m,
                        }
                        return self._stop_stalled(trace, "recovery_scan_limit_exhausted")
                    self.runtime.recovery_scan_cycles += 1
                self.runtime.last_scan_start_position = position.copy()
                self.runtime.scan_active = True
                self.runtime.semantic_relocalization_required = False
                self.runtime.consecutive_blocked_navigation = 0
                self.runtime.scan_accumulated_deg = 0.0
                self.runtime.scan_views_checked = 1  # The current view was just checked.
            scan_task = self.executor.create_turn_task(
                observation,
                self.runtime.scan_direction,
                min(
                    self.scan_increment_deg,
                    max(0.1, 360.0 - self.runtime.scan_accumulated_deg),
                ),
                str(decision.terminal.get("reason", "named POI not visible")),
            )
            decision.terminal["task"] = scan_task.public_dict()
            self.mode = NavMode.EXECUTING
            self.transition_reason = action
            prediction = self._execute_new_turn_task(observation, trace)
        elif action == "SEARCH_EXHAUSTED":
            if not self.runtime.scan_completed:
                raise ExecutorSystemError("SEARCH_EXHAUSTED requires a completed 360-degree scan")
            self.mode = NavMode.FAILED
            self.terminated = True
            self.transition_reason = "SCAN_360_TARGET_NOT_FOUND"
            self.runtime.stop_reason = "scan_360_target_not_found"
            trace.update(
                stop_reason=self.runtime.stop_reason,
                mode_after=self.mode.value,
            )
            prediction = self._stop_prediction(trace)
        elif action in {"SET_NAVIGATION_GOAL", "SET_EXPLORATION_GOAL"}:
            if self.mode not in {NavMode.PLANNING, NavMode.RECOVERY}:
                raise ExecutorSystemError(
                    f"{action} is only valid in PLANNING or RECOVERY mode"
                )
            task = self.executor.active_task
            if task is None:
                raise ExecutorSystemError(f"{action} did not create a task")
            if task.status is not TaskStatus.RUNNING:
                self.executor.finish_current_task()
                self.mode = NavMode.RECOVERY
                self.transition_reason = task.status.value
                prediction = self._prediction(np.zeros(2), False, trace)
                self._log(trace)
                return prediction
            # A reachable target anchor means the visual search succeeded.
            # Scan steps intentionally contain no translation, so begin the
            # translation-progress budget from this decision rather than from
            # the start of the scan.
            if (
                (self.runtime.scan_active or self.runtime.scan_completed)
                and self.runtime.recovery_scan_cycles == 0
            ):
                self.runtime.last_progress_step = int(observation.step_count)
                self.runtime.no_progress_steps = 0
            self.runtime.scan_active = False
            self.runtime.scan_completed = False
            self.runtime.scan_accumulated_deg = 0.0
            self.runtime.scan_views_checked = 0
            self.runtime.scan_rotation_failures = 0
            self.runtime.focused_reacquire_active = False
            self.runtime.focused_reacquire_attempted = False
            self.runtime.focused_reacquire_stage = 0
            self.runtime.reacquire_goal_world = None
            self.mode = NavMode.EXECUTING
            self.transition_reason = action
            prediction = self._prediction(np.zeros(2), False, trace)
        else:
            raise ExecutorSystemError(f"unsupported high-level action: {action}")
        self._log(trace)
        return prediction

    def _execute_new_turn_task(
        self,
        observation: Any,
        trace: dict[str, Any],
    ) -> WaypointPrediction:
        """Emit the first turn command in the same environment step as SCAN_360."""
        task = self.executor.active_task
        if task is None or task.task_type != "turn":
            raise ExecutorSystemError("new turn task is missing")
        status = self.executor.task_status(observation)
        self.status_call_steps.append(int(observation.step_count))
        prefix = "new_" if "task_status" in trace else ""
        trace[f"{prefix}task_status"] = status.value
        trace[f"{prefix}task"] = task.public_dict()
        trace["immediate_task_start"] = True
        if status is not TaskStatus.RUNNING:
            return self._prediction(np.zeros(2), False, trace)
        try:
            prediction = self.executor.step(observation)
        except ExecutorSystemError:
            self.mode = NavMode.SYSTEM_ERROR
            raise
        self.executor_step_count += 1
        trace["executor_plan"] = self.executor.last_plan_debug
        prediction.extra["agentnav_trace"] = trace
        return prediction

    @staticmethod
    def _plan_lacks_depth_evidence(plan: dict[str, Any]) -> bool:
        reasons = [
            candidate.get("safety", {}).get("reason")
            for candidate in plan.get("candidates", [])
            if candidate.get("safety", {}).get("reason")
            != "candidate_outside_front_depth_view"
        ]
        return bool(reasons) and all(reason in {
            "insufficient_corridor_evidence",
            "insufficient_ground_plane_support",
            "no_valid_depth",
            "missing_depth",
        } for reason in reasons)

    def _arm_focused_reacquisition(self, task: Any) -> None:
        goal = getattr(task, "semantic_goal_world", None)
        if goal is None:
            return
        goal = np.asarray(goal, dtype=np.float64)
        if goal.shape != (3,) or not np.all(np.isfinite(goal)):
            return
        self.runtime.reacquire_goal_world = goal.copy()
        self.runtime.focused_reacquire_active = False
        self.runtime.focused_reacquire_attempted = False
        self.runtime.focused_reacquire_stage = 0

    def _try_focused_reacquisition(
        self,
        observation: Any,
        decision: Any,
    ) -> Any | None:
        del decision
        state = self.runtime
        if state.focused_reacquire_attempted or state.reacquire_goal_world is None:
            return None
        local = world_to_local(state.reacquire_goal_world, observation.rotation)
        distance = float(np.linalg.norm(local))
        if not np.isfinite(distance) or distance <= 0.20:
            state.focused_reacquire_attempted = True
            return None

        base_delta = math.atan2(float(local[1]), float(local[0]))
        offsets_deg = (0.0, 45.0, -45.0)
        tolerance = math.radians(self.executor.config.turn_tolerance_deg)
        while state.focused_reacquire_stage < len(offsets_deg):
            offset_deg = offsets_deg[state.focused_reacquire_stage]
            state.focused_reacquire_stage += 1
            delta = (
                base_delta + math.radians(offset_deg) + math.pi
            ) % (2.0 * math.pi) - math.pi
            if abs(delta) <= tolerance:
                continue
            direction = "left" if delta > 0.0 else "right"
            task = self.executor.create_turn_task(
                observation,
                direction,
                abs(math.degrees(delta)),
                (
                    "focused reacquisition near the latest visible POI "
                    f"bearing ({offset_deg:+.0f} deg)"
                ),
            )
            state.focused_reacquire_active = True
            state.focused_reacquire_attempted = True
            return task

        state.focused_reacquire_attempted = True
        return None

    def _scan_state(self) -> dict[str, Any]:
        return {
            "active": self.runtime.scan_active,
            "completed": self.runtime.scan_completed,
            "direction": self.runtime.scan_direction,
            "increment_deg": self.scan_increment_deg,
            "accumulated_deg": round(self.runtime.scan_accumulated_deg, 4),
            "views_checked": self.runtime.scan_views_checked,
            "rotation_failures": self.runtime.scan_rotation_failures,
            "recovery_cycles": self.runtime.recovery_scan_cycles,
            "max_recovery_cycles": self.max_recovery_scan_cycles,
            "focused_reacquire_active": self.runtime.focused_reacquire_active,
            "focused_reacquire_attempted": self.runtime.focused_reacquire_attempted,
            "focused_reacquire_stage": self.runtime.focused_reacquire_stage,
            "has_reacquire_bearing": self.runtime.reacquire_goal_world is not None,
            "semantic_relocalization_required": self.runtime.semantic_relocalization_required,
            "semantic_verification_failures": self.runtime.semantic_verification_failures,
            "consecutive_blocked_navigation": self.runtime.consecutive_blocked_navigation,
        }

    def _record_spatial_progress(self, observation: Any, trace: dict[str, Any]) -> None:
        """Only reaching a new XY region resets the cross-task stall budget."""
        position = np.asarray(observation.rotation, dtype=np.float64)[:2, 3].copy()
        if not np.all(np.isfinite(position)):
            raise ExecutorSystemError("non-finite robot position")
        step = int(observation.step_count)
        state = self.runtime
        reached_new_region = not state.progress_positions or all(
            float(np.linalg.norm(position - previous)) >= self.progress_radius_m
            for previous in state.progress_positions
        )
        if reached_new_region:
            state.progress_positions.append(position)
            state.last_progress_step = step
        state.no_progress_steps = step - state.last_progress_step
        trace["progress_guard"] = {
            "no_progress_steps": state.no_progress_steps,
            "max_no_progress_steps": self.max_no_progress_steps,
            "visited_region_count": len(state.progress_positions),
            "progress_radius_m": self.progress_radius_m,
        }

    def _stop_prediction(self, trace: dict[str, Any]) -> WaypointPrediction:
        # ABot exposes only arrive as a stop signal. The adapter records stalled
        # stops as failures, independently of the evaluator's distance check.
        prediction = self._prediction(np.zeros(2), True, trace)
        if self.runtime.stop_reason:
            prediction.confidence = 0.0
            prediction.extra["stop_reason"] = self.runtime.stop_reason
        return prediction

    def _stop_stalled(
        self, trace: dict[str, Any], reason: str = "no_new_position_region"
    ) -> WaypointPrediction:
        if self.executor.has_active_task():
            self.executor.mark_failed(TaskStatus.STUCK, reason)
            finished = self.executor.finish_current_task()
            trace["finished_task"] = finished.public_dict()
            trace["task_status"] = TaskStatus.STUCK.value
        self.mode = NavMode.FAILED
        self.terminated = True
        self.transition_reason = reason
        self.runtime.stop_reason = reason
        trace.update(stop_reason=reason, mode_after=self.mode.value, vlm_called=False)
        self._log(trace)
        return self._stop_prediction(trace)

    def _stop_planning_budget(self, trace: dict[str, Any]) -> WaypointPrediction:
        """Bound repeated perception/replanning even when small moves occur."""
        reason = "planning_cycle_budget_exhausted"
        if self.executor.has_active_task():
            self.executor.mark_failed(TaskStatus.STUCK, reason)
            finished = self.executor.finish_current_task()
            trace["finished_task"] = finished.public_dict()
            trace["task_status"] = TaskStatus.STUCK.value
        self.mode = NavMode.FAILED
        self.terminated = True
        self.transition_reason = reason
        self.runtime.stop_reason = reason
        trace.update(
            stop_reason=reason,
            mode_after=self.mode.value,
            vlm_called=False,
            planning_cycle_guard={
                "used": self.vlm_step_count,
                "limit": self.max_planning_cycles_per_episode,
            },
        )
        self._log(trace)
        return self._stop_prediction(trace)

    def architecture_metrics(self) -> dict[str, Any]:
        tasks = self.memory.task_history.copy()
        if self.executor.active_task is not None:
            tasks.append(self.executor.active_task.public_dict())
        terminal_tasks = [
            task
            for task in tasks
            if task["task_status"] not in {TaskStatus.IDLE.value, TaskStatus.RUNNING.value}
        ]
        success_statuses = {
            TaskStatus.GOAL_REACHED.value,
            TaskStatus.MIDPOINT_REACHED.value,
        }
        failure_statuses = {
            TaskStatus.INVALID_GOAL.value,
            TaskStatus.BLOCKED.value,
            TaskStatus.STUCK.value,
            TaskStatus.TIMEOUT.value,
            TaskStatus.FAILED.value,
        }
        status_counts = {
            status.value: sum(task["task_status"] == status.value for task in tasks)
            for status in TaskStatus
            if status not in {TaskStatus.IDLE, TaskStatus.RUNNING}
        }
        terminal_actions = [call.get("terminal_action") for call in self.memory.vlm_calls]
        task_steps = [int(task.get("steps_elapsed", 0)) for task in terminal_tasks]
        denominator = max(1, len(terminal_tasks))
        recovered_failures = self._recovered_failures(tasks)
        goal_failure_count = sum(
            task["task_status"] in failure_statuses for task in terminal_tasks
        )
        return {
            "agentnav_vlm_calls": self.vlm_step_count,
            "agentnav_executor_steps": self.executor_step_count,
            "agentnav_steps_per_vlm_call": self.executor_step_count
            / max(1, self.vlm_step_count),
            "agentnav_status_calls": self.executor.total_status_calls,
            "agentnav_depth_inferences": self.harness.depth_cache.inference_count,
            "agentnav_depth_query_count": self.harness.total_query_count,
            "agentnav_navigation_task_count": sum(
                action in {"SET_NAVIGATION_GOAL", "SET_EXPLORATION_GOAL"}
                for action in terminal_actions
            ),
            "agentnav_exploration_goal_count": terminal_actions.count("SET_EXPLORATION_GOAL"),
            "agentnav_scan_request_count": terminal_actions.count("SCAN_360"),
            "agentnav_turn_task_count": sum(
                task.get("task_type") == "turn" for task in tasks
            ),
            "agentnav_midpoint_task_count": sum(
                bool(task.get("is_midpoint_task")) for task in tasks
            ),
            "agentnav_executor_success_rate": sum(
                task["task_status"] in success_statuses for task in terminal_tasks
            )
            / denominator,
            "agentnav_average_task_length": sum(task_steps) / max(1, len(task_steps)),
            "agentnav_replan_count": max(0, self.vlm_step_count - 1),
            "agentnav_semantic_verification_count": sum(
                call.get("mode") == NavMode.VERIFYING.value for call in self.memory.vlm_calls
            ),
            "agentnav_failure_recovery_success_count": recovered_failures,
            "agentnav_failure_recovery_success_rate": recovered_failures
            / max(1, goal_failure_count),
            **{
                f"agentnav_{status.lower()}_rate": count / denominator
                for status, count in status_counts.items()
                if status in failure_statuses or status == TaskStatus.SYSTEM_ERROR.value
            },
            "agentnav_failure_count": self.memory.failure_count,
            "agentnav_stop_reason": self.runtime.stop_reason,
            "agentnav_no_progress_steps": self.runtime.no_progress_steps,
            "agentnav_gt_depth_used": False,
            "agentnav_occupancy_map_used": False,
            "agentnav_single_front_rgb": True,
        }

    @staticmethod
    def _recovered_failures(tasks: list[dict[str, Any]]) -> int:
        pending_failure = False
        recovered = 0
        for task in tasks:
            status = task["task_status"]
            if status in {
                TaskStatus.INVALID_GOAL.value,
                TaskStatus.BLOCKED.value,
                TaskStatus.STUCK.value,
                TaskStatus.TIMEOUT.value,
                TaskStatus.FAILED.value,
            }:
                pending_failure = True
            elif pending_failure and status in {
                TaskStatus.GOAL_REACHED.value,
                TaskStatus.MIDPOINT_REACHED.value,
            }:
                recovered += 1
                pending_failure = False
        return recovered

    @staticmethod
    def _prediction(waypoint: np.ndarray, arrive: bool, trace: dict[str, Any]) -> WaypointPrediction:
        return WaypointPrediction(
            waypoint=np.asarray(waypoint, dtype=np.float32).reshape(1, 2),
            arrive=arrive,
            confidence=1.0 if arrive else 0.0,
            extra={"agentnav_trace": trace},
        )

    def _log(self, record: dict[str, Any]) -> None:
        task = self.executor.active_task
        if task is not None and task.goal_world is not None:
            record["visual_active_goal_world_xy_m"] = np.asarray(
                task.goal_world, dtype=np.float64
            )[:2].tolist()
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=self._json_default) + "\n")

    @property
    def mode(self) -> NavMode:
        return self.runtime.mode

    @mode.setter
    def mode(self, value: NavMode) -> None:
        self.runtime.mode = value

    @property
    def transition_reason(self) -> str:
        return self.runtime.transition_reason

    @transition_reason.setter
    def transition_reason(self, value: str) -> None:
        self.runtime.transition_reason = value

    @property
    def terminated(self) -> bool:
        return self.runtime.terminated

    @terminated.setter
    def terminated(self, value: bool) -> None:
        self.runtime.terminated = value

    @property
    def vlm_step_count(self) -> int:
        return self.runtime.vlm_step_count

    @vlm_step_count.setter
    def vlm_step_count(self, value: int) -> None:
        self.runtime.vlm_step_count = value

    @property
    def executor_step_count(self) -> int:
        return self.runtime.executor_step_count

    @executor_step_count.setter
    def executor_step_count(self, value: int) -> None:
        self.runtime.executor_step_count = value

    @property
    def status_call_steps(self) -> list[int]:
        return self.runtime.status_call_steps

    @status_call_steps.setter
    def status_call_steps(self, value: list[int]) -> None:
        self.runtime.status_call_steps = value

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return str(value)
