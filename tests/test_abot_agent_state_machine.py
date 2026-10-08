from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from agentnav.abot.depth import DepthPrediction
from agentnav.abot.high_level import HighLevelDecision
from agentnav.abot.poi_agent import AgentNavPoiGoalAgent
from agentnav.abot.types import NavMode, PixelMeasurement, PixelProposal, TaskStatus


class FakeDepth:
    def predict(self, rgb):
        return DepthPrediction(np.full((640, 720), 2.0, dtype=np.float32))


class ScanPlanner:
    def __init__(self):
        self.calls = 0
        self.scan_states = []

    def decide(self, safe, observation, harness, executor, memory):
        self.calls += 1
        self.scan_states.append(safe.scan_state.copy())
        if safe.mode is NavMode.VERIFYING:
            return HighLevelDecision(
                {
                    "action": "RETURN_TO_PLANNING",
                    "target_visible": False,
                    "reason": "POI is not confirmed",
                },
                [],
                [],
                0.01,
                {},
            )
        return HighLevelDecision(
            {"action": "SCAN_360", "reason": "target absent"}, [], [], 0.01, {}
        )


def obs(step, heading_degrees=0.0):
    heading = np.deg2rad(heading_degrees)
    rotation = np.eye(4)
    rotation[:2, :2] = [
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading), np.cos(heading)],
    ]
    return SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library",
        step_count=step,
        rotation=rotation,
        target_position=np.array([50.0, 50.0]),
        distance_to_goal=70.0,
        occ_map=np.ones((4, 4)),
    )


def start_navigation_task(agent, observation):
    measurement = PixelMeasurement(
        proposal=PixelProposal("P0", 360, 400, "visible POI entrance"),
        depth_m=10.0,
        depth_mad_m=0.01,
        valid_depth_ratio=1.0,
        local_goal=np.array([10.0, 0.0]),
        depth_reliable=True,
        corridor_safe=True,
        safety_debug={"reason": "clear"},
    )
    task = agent.executor.create_navigation_task(
        observation,
        measurement,
        semantic_anchor="library",
        navigation_anchor="entrance ground",
    )
    assert task.status is TaskStatus.RUNNING
    agent.mode = NavMode.EXECUTING
    agent.executor._plan_local_waypoint = lambda depth, goal: (
        np.array([0.5, 0.0]),
        [],
    )


def test_vlm_is_not_called_while_task_is_running(tmp_path):
    planner = ScanPlanner()
    agent = AgentNavPoiGoalAgent(
        planner=planner, depth_estimator=FakeDepth(), log_dir=str(tmp_path)
    )
    first = agent.predict(obs(0))
    assert planner.calls == 1
    assert not first.arrive
    second = agent.predict(obs(1))
    assert planner.calls == 1
    assert not second.arrive
    assert second.extra["agentnav_trace"]["vlm_called"] is False
    assert agent.executor.total_status_calls == 2


def test_scan_uses_fixed_direction_and_accumulates_actual_rotation(tmp_path):
    planner = ScanPlanner()
    agent = AgentNavPoiGoalAgent(
        planner=planner,
        depth_estimator=FakeDepth(),
        log_dir=str(tmp_path),
        scan_direction="left",
        scan_increment_deg=45,
    )
    agent.predict(obs(0))
    assert agent.executor.active_task.turn_direction == "left"
    assert agent.executor.active_task.turn_requested_angle_deg == 45
    agent.predict(obs(1))
    replanning = agent.predict(obs(2, heading_degrees=45.0))
    assert replanning.extra["agentnav_trace"]["vlm"]["safe_input"]["mode"] == (
        NavMode.PLANNING.value
    )
    assert replanning.extra["agentnav_trace"]["transition_reason"] == "SCAN_VIEW_READY"
    assert agent.runtime.scan_accumulated_deg == 45
    assert agent.executor.active_task.turn_direction == "left"
    assert planner.scan_states[-1]["direction"] == "left"


def test_scan_is_not_interrupted_by_translation_stall_guard(tmp_path):
    planner = ScanPlanner()
    agent = AgentNavPoiGoalAgent(
        planner=planner, depth_estimator=FakeDepth(), log_dir=str(tmp_path),
        max_no_progress_steps=2,
    )
    for step in range(5):
        prediction = agent.predict(obs(step, step * 45))
        assert not prediction.arrive
        assert agent.runtime.scan_active
        assert not agent.runtime.stop_reason
    assert agent.runtime.no_progress_steps == 4
    assert planner.calls == 5

    agent.reset()
    assert agent.runtime.no_progress_steps == 0
    assert not agent.runtime.progress_positions
    assert not agent.runtime.stop_reason
    assert not agent.predict(obs(0)).arrive


def test_new_positions_reset_budget_but_revisiting_positions_does_not(tmp_path):
    agent = AgentNavPoiGoalAgent(
        planner=ScanPlanner(), depth_estimator=FakeDepth(), log_dir=str(tmp_path),
        max_no_progress_steps=4, progress_radius_m=0.5, stuck_window=100,
    )
    start_navigation_task(agent, obs(0))
    # New positions at steps 0 and 2, followed by oscillation between them.
    for step, x in enumerate([0.0, 0.2, 0.6, 0.0, 0.6, 0.0, 0.6]):
        current = obs(step)
        current.rotation[0, 3] = x
        prediction = agent.predict(current)
        assert prediction.arrive is (step == 6)
    assert agent.runtime.last_progress_step == 2
    assert agent.runtime.no_progress_steps == 4


def test_stall_interrupts_running_task_without_waiting_for_task_timeout(tmp_path):
    agent = AgentNavPoiGoalAgent(
        planner=ScanPlanner(), depth_estimator=FakeDepth(), log_dir=str(tmp_path),
        max_no_progress_steps=2, stuck_window=100,
    )
    start_navigation_task(agent, obs(0))
    agent.predict(obs(0))
    agent.predict(obs(1))
    stopped = agent.predict(obs(2))  # Pose never follows the waypoint commands.
    assert stopped.arrive
    assert agent.memory.task_history[-1]["task_status"] == TaskStatus.STUCK.value
    assert agent.memory.task_history[-1]["failure_reason"] == "no_new_position_region"
    assert agent.executor.active_task is None


def test_scan_finishes_one_direction_full_circle_and_stops(tmp_path):
    class FullScanPlanner(ScanPlanner):
        def decide(self, safe, observation, harness, executor, memory):
            self.calls += 1
            self.scan_states.append(safe.scan_state.copy())
            action = "SEARCH_EXHAUSTED" if safe.scan_state["completed"] else "SCAN_360"
            return HighLevelDecision(
                {"action": action, "reason": "target absent"}, [], [], 0.01, {}
            )

    planner = FullScanPlanner()
    agent = AgentNavPoiGoalAgent(
        planner=planner,
        depth_estimator=FakeDepth(),
        log_dir=str(tmp_path),
        scan_direction="left",
        scan_increment_deg=45,
        max_no_progress_steps=30,
    )
    assert not agent.predict(obs(0, 0)).arrive
    for index in range(1, 8):
        prediction = agent.predict(obs(index, index * 45))
        assert not prediction.arrive
        assert agent.executor.active_task.turn_direction == "left"
    stopped = agent.predict(obs(8, 360))
    assert stopped.arrive
    assert stopped.extra["stop_reason"] == "scan_360_target_not_found"
    assert agent.mode is NavMode.FAILED
    assert agent.runtime.scan_completed
    assert agent.runtime.scan_accumulated_deg == pytest.approx(360)
    assert planner.scan_states[-1]["completed"] is True
    assert all(
        task["turn_direction"] == "left"
        for task in agent.memory.task_history
        if task["task_type"] == "turn"
    )



def test_scan_stops_when_rotation_commands_do_not_change_heading(tmp_path):
    planner = ScanPlanner()
    agent = AgentNavPoiGoalAgent(
        planner=planner,
        depth_estimator=FakeDepth(),
        log_dir=str(tmp_path),
        max_task_steps=1,
        max_no_progress_steps=2,
        max_scan_rotation_failures=2,
    )
    assert not agent.predict(obs(0)).arrive
    assert not agent.predict(obs(1)).arrive
    stopped = agent.predict(obs(2))  # Both retries emitted commands immediately.
    assert stopped.arrive
    assert stopped.confidence == 0.0
    assert stopped.extra["stop_reason"] == "scan_rotation_failed"
    assert agent.mode is NavMode.FAILED
    assert agent.runtime.scan_rotation_failures == 2
    assert planner.calls == 2


def test_recovery_scan_cannot_repeat_without_new_spatial_progress(tmp_path):
    planner = ScanPlanner()
    agent = AgentNavPoiGoalAgent(
        planner=planner,
        depth_estimator=FakeDepth(),
        log_dir=str(tmp_path),
        max_recovery_scan_cycles=1,
    )
    agent.mode = NavMode.RECOVERY
    agent.runtime.recovery_scan_cycles = 1

    stopped = agent.predict(obs(0))

    assert stopped.arrive
    assert stopped.confidence == 0.0
    assert stopped.extra["stop_reason"] == "recovery_scan_limit_exhausted"
    assert agent.mode is NavMode.FAILED
    assert agent.executor.active_task is None


def test_recovery_scan_reopens_after_meaningful_translation(tmp_path):
    agent = AgentNavPoiGoalAgent(
        planner=ScanPlanner(),
        depth_estimator=FakeDepth(),
        log_dir=str(tmp_path),
        max_recovery_scan_cycles=1,
    )
    agent.mode = NavMode.RECOVERY
    agent.runtime.recovery_scan_cycles = 1
    agent.runtime.last_scan_start_position = np.zeros(2)
    moved = obs(0)
    moved.rotation[0, 3] = 1.1

    prediction = agent.predict(moved)

    assert not prediction.arrive
    assert agent.runtime.scan_active
    assert agent.runtime.recovery_scan_cycles == 1
    assert np.allclose(agent.runtime.last_scan_start_position, [1.1, 0.0])


def test_episode_planning_cycle_budget_stops_replanning_loop(tmp_path):
    planner = ScanPlanner()
    agent = AgentNavPoiGoalAgent(
        planner=planner,
        depth_estimator=FakeDepth(),
        log_dir=str(tmp_path),
        max_planning_cycles_per_episode=2,
        max_no_progress_steps=30,
    )

    assert not agent.predict(obs(0, 0)).arrive
    assert not agent.predict(obs(1, 45)).arrive
    stopped = agent.predict(obs(2, 90))

    assert stopped.arrive
    assert stopped.confidence == 0.0
    assert stopped.extra["stop_reason"] == "planning_cycle_budget_exhausted"
    assert planner.calls == 2
    assert agent.mode is NavMode.FAILED


def test_new_spatial_region_does_not_restore_episode_recovery_scan_budget(tmp_path):
    agent = AgentNavPoiGoalAgent(
        planner=ScanPlanner(),
        depth_estimator=FakeDepth(),
        log_dir=str(tmp_path),
        progress_radius_m=0.5,
    )
    agent.runtime.recovery_scan_cycles = 1
    agent._record_spatial_progress(obs(0), {})
    assert agent.runtime.recovery_scan_cycles == 1

    moved = obs(1)
    moved.rotation[0, 3] = 0.6
    agent._record_spatial_progress(moved, {})
    assert agent.runtime.recovery_scan_cycles == 1


def test_exploration_goal_enters_persistent_execution_and_clears_scan(tmp_path):
    class ExplorationPlanner(ScanPlanner):
        def decide(self, safe, observation, harness, executor, memory):
            measurement = PixelMeasurement(
                proposal=PixelProposal("P0", 360, 400, "open storefront route"),
                depth_m=4.0,
                depth_mad_m=0.01,
                valid_depth_ratio=1.0,
                local_goal=np.array([4.0, 0.0]),
                depth_reliable=True,
                corridor_safe=True,
                safety_debug={"reason": "clear"},
            )
            task = executor.create_navigation_task(
                observation,
                measurement,
                semantic_anchor="storefront cluster",
                navigation_anchor="open route",
            )
            return HighLevelDecision(
                {"action": "SET_EXPLORATION_GOAL", "task": task.public_dict()},
                [],
                [],
                0.01,
                {},
            )

    agent = AgentNavPoiGoalAgent(
        planner=ExplorationPlanner(),
        depth_estimator=FakeDepth(),
        log_dir=str(tmp_path),
    )
    agent.runtime.scan_completed = True
    agent.runtime.scan_accumulated_deg = 360.0
    agent.runtime.scan_views_checked = 9

    prediction = agent.predict(obs(9, 360))

    assert not prediction.arrive
    assert agent.mode is NavMode.EXECUTING
    assert agent.executor.has_active_task()
    assert agent.transition_reason == "SET_EXPLORATION_GOAL"
    assert not agent.runtime.scan_completed
    assert agent.runtime.scan_accumulated_deg == 0.0
    assert agent.architecture_metrics()["agentnav_exploration_goal_count"] == 1

def test_sparse_block_requires_viewpoint_change_but_real_obstacle_does_not():
    sparse = {"candidates": [
        {"safety": {"reason": "insufficient_corridor_evidence"}},
        {"safety": {"reason": "candidate_outside_front_depth_view"}},
        {"safety": {"reason": "insufficient_corridor_evidence"}},
    ]}
    obstacle = {"candidates": [
        {"safety": {"reason": "insufficient_corridor_evidence"}},
        {"safety": {"reason": "depth_obstacle"}},
    ]}

    assert AgentNavPoiGoalAgent._plan_lacks_depth_evidence(sparse)
    assert not AgentNavPoiGoalAgent._plan_lacks_depth_evidence(obstacle)
    assert not AgentNavPoiGoalAgent._plan_lacks_depth_evidence({"candidates": []})



def test_rejected_navigation_task_enters_recovery_without_clearing_scan(tmp_path):
    class RejectedPlanner:
        def decide(self, safe, observation, harness, executor, memory):
            measurement = PixelMeasurement(
                proposal=PixelProposal("P0", 360, 400, "failed target anchor"),
                depth_m=None, depth_mad_m=None, valid_depth_ratio=0.0,
                local_goal=np.array([1.0, 0.0]), depth_reliable=False,
                corridor_safe=False, safety_debug={"reason": "depth_obstacle"},
            )
            task = executor.create_navigation_task(
                observation, measurement, "library", "entrance"
            )
            assert task.status is TaskStatus.INVALID_GOAL
            return HighLevelDecision(
                {"action": "SET_NAVIGATION_GOAL", "task": task.public_dict()},
                [], [], 0.01, {},
            )

    agent = AgentNavPoiGoalAgent(
        planner=RejectedPlanner(), depth_estimator=FakeDepth(), log_dir=str(tmp_path)
    )
    agent.runtime.scan_completed = True
    agent.runtime.scan_accumulated_deg = 360.0
    prediction = agent.predict(obs(9))

    assert not prediction.arrive
    assert agent.mode is NavMode.RECOVERY
    assert agent.transition_reason == TaskStatus.INVALID_GOAL.value
    assert agent.runtime.scan_completed
    assert agent.executor.active_task is None
    assert agent.memory.failure_count == 1
