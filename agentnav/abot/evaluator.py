"""ABot POI evaluator adapter that renders one front RGB camera only."""

from __future__ import annotations

import os
from typing import Any

from abotn_evaluator.poi_goal.evaluator import PoiGoalEvaluator

from agentnav.abot.executor import ExecutorSystemError


class AgentNavPoiGoalEvaluator(PoiGoalEvaluator):
    """Keep official POI metrics while exposing only one rendered front view."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.renderer.num_views = 1
        self.renderer.yaw_offsets = [0]
        self.renderer.view_names = ["front"]
        self.task_system_attempts = max(
            1, min(3, int(os.environ.get("AGENTNAV_TASK_SYSTEM_ATTEMPTS", "2")))
        )

    def _build_poi_observation(self, *args: Any, **kwargs: Any):
        observation = super()._build_poi_observation(*args, **kwargs)
        short_memory = kwargs.get("short_memory")
        current_images = short_memory.get_current_images() if short_memory else []
        if not current_images:
            raise RuntimeError("single-front renderer returned no image")
        # Parent _build_observation maps a lone image to the first hard-coded
        # key (left) and inserts a zero-valued front placeholder. Bypass that
        # mapping and use the renderer's only current image directly.
        observation.images = {"front": current_images[0]}
        observation.history_images = None
        observation.history_poses = None
        observation.occ_map = None
        observation.height_map = None
        observation.meta_data = None
        return observation

    @staticmethod
    def _save_task_video(agent: Any, short_memory: Any, task_dir: str, result: dict) -> None:
        # Visualization is an output artifact; a codec or partial render
        # failure must not change the official evaluator result.
        try:
            from agentnav.abot.visualize import render_task_video

            video = render_task_video(
                task_dir,
                agent.log_path,
                result,
                final_pose=short_memory.get_last_pose(),
                camera=agent.harness.camera,
            )
            print(f"[AgentNav] task video: {video}")
        except Exception as exc:
            print(f"[AgentNav] task video unavailable for {result.get('task_id')}: {exc}")

    def _evaluate_task(
        self,
        agent: Any,
        episode: Any,
        task: Any,
        short_memory: Any,
        episode_dir: str,
    ) -> dict:
        # This adapter renders exactly one front image per frame. ShortMemory
        # defaults to three views and would otherwise treat previous front
        # frames as the current [left, front, right] triplet, feeding stale
        # images to the agent while the pose keeps advancing.
        short_memory.num_current_views = 1
        short_memory.reorder_views = False
        system_error = None
        task_system_attempts = max(1, int(getattr(self, "task_system_attempts", 2)))
        for attempt in range(1, task_system_attempts + 1):
            try:
                result = super()._evaluate_task(
                    agent, episode, task, short_memory, episode_dir
                )
                break
            except ExecutorSystemError as exc:
                system_error = exc
                print(
                    f"[AgentNav][system retry] task={task.task_id} "
                    f"attempt={attempt}/{task_system_attempts}: {exc}"
                )
        else:
            poi_name = getattr(task, "goal_label", None) or "unknown"
            result = self._make_error_result(episode, task)
            result.update(
                status="agent_system_error",
                target_label=poi_name,
                gt_taget_instance=f"poi_goal:{poi_name}",
                system_error=str(system_error),
                spl=0.0,
            )
            result["metrics"] = {
                "poi_name": poi_name,
                "agent_system_error": True,
                "system_attempts": task_system_attempts,
            }
            task_dir = os.path.join(episode_dir, task.task_id)
            self._save_task_result(result, task_dir)
            self._save_task_video(agent, short_memory, task_dir, result)
            return result

        # arrive=True is also the only stop signal for an exhausted exploration.
        # Never count that stop as a successful POI confirmation, even nearby.
        stop_reason = getattr(getattr(agent, "runtime", None), "stop_reason", "")
        if stop_reason:
            status = (
                "search_exhausted"
                if stop_reason == "scan_360_target_not_found"
                else "stalled"
            )
            result.update(
                status=status, success=False, oracle_success=False,
                spl=0.0, stop_reason=stop_reason,
            )
        metrics = result.setdefault("metrics", {})
        if hasattr(agent, "architecture_metrics"):
            metrics.update(agent.architecture_metrics())
        task_dir = os.path.join(episode_dir, task.task_id)
        self._save_task_result(result, task_dir)
        self._save_task_video(agent, short_memory, task_dir, result)
        return result
