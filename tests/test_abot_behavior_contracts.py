from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from agentnav.abot.depth import DepthPrediction
from agentnav.abot.executor import ExecutorSystemError
from agentnav.abot.geometry import (
    local_to_world,
    pixel_to_local_goal,
)
from agentnav.abot.high_level import HighLevelDecision
from agentnav.abot.poi_agent import AgentNavPoiGoalAgent
from agentnav.abot.types import (
    CameraIntrinsics,
    NavMode,
    PixelMeasurement,
    PixelProposal,
    RuntimeState,
    TaskStatus,
)


class FakeDepth:
    def __init__(self):
        self.calls = 0

    def predict(self, rgb):
        self.calls += 1
        return DepthPrediction(np.full((640, 720), 5.0, dtype=np.float32))


class DeterministicHarness:
    def __init__(self, blocked=False, raises=False):
        self.blocked = blocked
        self.raises = raises
        self.invalidated = []
        self.total_query_count = 0
        self.depth_cache = SimpleNamespace(inference_count=0)

    def reset(self):
        self.invalidated.clear()

    def current_depth(self, observation):
        if self.raises:
            raise RuntimeError("depth backend unavailable")
        return np.ones((8, 8), dtype=np.float32)

    def depth_corridor_is_safe(self, depth, waypoint, max_lookahead_m=None):
        return (not self.blocked), {"reason": "blocked" if self.blocked else "clear"}

    def invalidate_frame_queries(self, frame_id):
        self.invalidated.append(frame_id)


def observation(
    step,
    forward=0.0,
    yaw_deg=0.0,
    color=None,
    world_x=0.0,
    distance_to_goal=123.0,
):
    angle = np.deg2rad(yaw_deg)
    pose = np.eye(4)
    pose[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    pose[0, 3] = world_x + forward * np.cos(angle)
    pose[1, 3] = forward * np.sin(angle)
    return SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), color or (step, 0, 0))},
        poi_name="library",
        step_count=step,
        rotation=pose,
        target_position=np.array([99.0, 99.0]),
        distance_to_goal=distance_to_goal,
        occ_map=np.ones((4, 4)),
        height_map=np.ones((4, 4)),
    )


def measurement(distance, pixel=(360, 400), reachable=True):
    return PixelMeasurement(
        PixelProposal(f"P{pixel[0]}", *pixel, "test anchor"),
        distance if reachable else None,
        0.01 if reachable else None,
        1.0 if reachable else 0.0,
        np.array([distance, 0.0]) if reachable else np.zeros(2),
        reachable,
        reachable,
        {"reason": "clear" if reachable else "unsafe"},
    )


def goal_action(distance, pixel=(360, 400), reachable=True):
    def action(safe, observation, harness, executor, memory):
        task = executor.create_navigation_task(
            observation,
            measurement(distance, pixel, reachable),
            "library sign",
            "free ground",
        )
        return HighLevelDecision(
            {"action": "SET_NAVIGATION_GOAL", "task": task.public_dict()},
            [],
            [],
            0.0,
            {},
        )

    return action


def scan_action():
    def action(safe, observation, harness, executor, memory):
        return HighLevelDecision(
            {"action": "SCAN_360", "reason": "target absent"}, [], [], 0.0, {}
        )

    return action


def terminate_action(safe, observation, harness, executor, memory):
    return HighLevelDecision(
        {"action": "TERMINATE", "target_visible": True}, [], [], 0.0, {}
    )


def reject_verification_action(
    safe, observation, harness, executor, memory
):
    return HighLevelDecision(
        {
            "action": "RETURN_TO_PLANNING",
            "target_visible": False,
            "reason": "POI is not visually confirmed",
        },
        [],
        [],
        0.0,
        {},
    )


class ScriptedPlanner:
    def __init__(self, actions):
        self.actions = actions
        self.calls = 0
        self.scan_states = []
        self.front_images = []
        self.modes = []
        self.memory_summaries = []

    def decide(self, safe, observation, harness, executor, memory):
        self.scan_states.append(safe.scan_state.copy())
        self.front_images.append(safe.front_rgb)
        self.modes.append(safe.mode)
        self.memory_summaries.append(safe.memory_summary)
        action = self.actions[self.calls]
        self.calls += 1
        return action(safe, observation, harness, executor, memory)


class FailingPlanner:
    def decide(self, *args, **kwargs):
        raise ConnectionError("vLLM unavailable")


def make_agent(tmp_path, planner, harness=None, max_task_steps=12):
    agent = AgentNavPoiGoalAgent(
        planner=planner,
        depth_estimator=FakeDepth(),
        log_dir=str(tmp_path),
        max_task_steps=max_task_steps,
    )
    if harness is not None:
        agent.harness = harness
        agent.executor.harness = harness
    return agent


def test_runtime_state_is_explicit_and_resettable():
    state = RuntimeState(mode=NavMode.EXECUTING, vlm_step_count=3)
    state.status_call_steps.append(2)
    state.reset()
    assert state.mode is NavMode.PLANNING
    assert state.vlm_step_count == 0
    assert state.status_call_steps == []


def test_pixel_geometry_has_correct_left_sign_and_world_mapping():
    camera = CameraIntrinsics()
    left = pixel_to_local_goal(260, 400, 2.0, camera, 0.0)
    right = pixel_to_local_goal(460, 400, 2.0, camera, 0.0)
    assert left[0] == pytest.approx(2.0)
    assert left[1] > 0.0
    assert right[1] < 0.0
    assert np.allclose(local_to_world(left, np.eye(4)), [left[0], left[1], 0.0])


def test_local_waypoint_stops_at_goal_tolerance_boundary(tmp_path):
    agent = make_agent(tmp_path, ScriptedPlanner([scan_action()]), DeterministicHarness())
    remaining = 0.47335806

    waypoint, candidates = agent.executor._plan_local_waypoint(
        np.ones((8, 8), dtype=np.float32),
        np.array([remaining, 0.0]),
    )

    expected = min(
        agent.executor.config.max_step_length,
        remaining - agent.executor.config.local_goal_tolerance,
    )
    assert waypoint is not None
    assert np.linalg.norm(waypoint) == pytest.approx(expected)
    assert all(
        np.linalg.norm(candidate["waypoint_front_left_m"]) == pytest.approx(expected)
        for candidate in candidates
    )


def test_local_waypoint_requires_fresh_view_for_goal_behind_camera(tmp_path):
    agent = make_agent(tmp_path, ScriptedPlanner([scan_action()]), DeterministicHarness())

    waypoint, candidates = agent.executor._plan_local_waypoint(
        np.ones((8, 8), dtype=np.float32),
        np.array([-0.04, -0.67]),
    )

    assert waypoint is None
    assert candidates[0]["reason"] == "goal_outside_front_depth_view"


def test_navigation_arrival_tolerates_floating_point_noise(tmp_path):
    agent = make_agent(
        tmp_path, ScriptedPlanner([scan_action()]), DeterministicHarness()
    )
    current = observation(0)
    tolerance = agent.executor.config.local_goal_tolerance
    measurement = PixelMeasurement(
        PixelProposal("P0", 360, 400, "boundary"),
        tolerance,
        0.0,
        1.0,
        np.array([tolerance + 1e-9, 0.0]),
        True,
        True,
        {"reason": "clear"},
    )
    task = agent.executor.create_navigation_task(
        current, measurement, "target", "boundary"
    )

    assert agent.executor.task_status(current) is TaskStatus.GOAL_REACHED
    assert task.phase == "arrived"


def test_normal_task_persists_across_steps_and_verifies_semantically(tmp_path):
    planner = ScriptedPlanner([goal_action(2.4), terminate_action])
    agent = make_agent(tmp_path, planner, DeterministicHarness())
    created = agent.predict(observation(0))
    task_id = agent.executor.active_task.task_id
    assert created.extra["agentnav_trace"]["vlm_called"] is True

    for step, forward in [(1, 0.0), (2, 0.8), (3, 1.6)]:
        prediction = agent.predict(observation(step, forward))
        trace = prediction.extra["agentnav_trace"]
        assert trace["task"]["task_id"] == task_id
        assert trace["task_status"] == TaskStatus.RUNNING.value
        assert trace["vlm_called"] is False

    final = agent.predict(observation(4, 2.4, distance_to_goal=1.5))
    assert final.arrive
    assert planner.calls == 2
    assert planner.modes == [NavMode.PLANNING, NavMode.VERIFYING]
    assert planner.front_images[0] is not planner.front_images[1]
    assert planner.front_images[1].getpixel((0, 0)) == (4, 0, 0)
    assert agent.status_call_steps == [1, 2, 3, 4]
    assert agent.executor_step_count == 3
    assert agent.architecture_metrics()["agentnav_steps_per_vlm_call"] > 1.0


def test_midpoint_replans_from_new_rgb_and_invalidates_source_frame(tmp_path):
    planner = ScriptedPlanner([goal_action(8.0, (300, 400)), goal_action(2.0, (420, 400))])
    harness = DeterministicHarness()
    agent = make_agent(tmp_path, planner, harness)
    agent.predict(observation(0, color="red"))
    first = agent.executor.active_task
    assert first.is_midpoint_task
    assert np.linalg.norm(first.goal_local_initial) == pytest.approx(6.0)

    for step, forward in [(1, 0.0), (2, 1.5), (3, 3.0), (4, 4.5)]:
        trace = agent.predict(observation(step, forward)).extra["agentnav_trace"]
        assert trace["task_status"] == TaskStatus.RUNNING.value
        assert trace["vlm_called"] is False

    agent.predict(observation(5, 6.0, color="blue"))
    second = agent.executor.active_task
    assert planner.calls == 2
    assert planner.modes == [NavMode.PLANNING, NavMode.PLANNING]
    assert agent.memory.task_history[-1]["task_status"] == TaskStatus.MIDPOINT_REACHED.value
    assert harness.invalidated == [0]
    assert second.task_id != first.task_id
    assert second.selected_pixel == (420, 400)
    assert planner.front_images[0] is not planner.front_images[1]


def test_invalid_goal_is_exposed_then_replanned(tmp_path):
    planner = ScriptedPlanner([goal_action(2.0, reachable=False), scan_action()])
    agent = make_agent(tmp_path, planner, DeterministicHarness())
    agent.predict(observation(0))
    prediction = agent.predict(observation(1))
    trace = prediction.extra["agentnav_trace"]
    assert trace["task_status"] == TaskStatus.INVALID_GOAL.value
    assert trace["vlm_called"] is True
    assert planner.calls == 2
    assert agent.memory.failure_count == 1


def test_blocked_is_reported_on_next_status_before_replan(tmp_path):
    planner = ScriptedPlanner([goal_action(2.0), scan_action()])
    agent = make_agent(tmp_path, planner, DeterministicHarness(blocked=True))
    agent.predict(observation(0))
    blocked_action = agent.predict(observation(1))
    assert planner.calls == 1
    assert len(blocked_action.extra["agentnav_trace"]["executor_plan"]["candidates"]) == 7
    assert agent.executor.active_task.status is TaskStatus.BLOCKED

    replanned = agent.predict(observation(2))
    assert replanned.extra["agentnav_trace"]["task_status"] == TaskStatus.BLOCKED.value
    assert replanned.extra["agentnav_trace"]["vlm_called"] is True
    terminal = replanned.extra["agentnav_trace"]["vlm"]["terminal"]
    assert terminal["action"] == "SCAN_360"
    assert terminal["task"]["task_type"] == "turn"
    assert planner.calls == 2


def test_blocked_region_rejects_repeat_and_allows_distinct_candidate(tmp_path):
    planner = ScriptedPlanner(
        [goal_action(2.0), goal_action(2.0), goal_action(3.0)]
    )
    agent = make_agent(tmp_path, planner, DeterministicHarness(blocked=True))

    agent.predict(observation(0))
    agent.predict(observation(1))
    agent.predict(observation(2))
    assert agent.runtime.consecutive_blocked_navigation == 1

    agent.predict(observation(3))
    replanned = agent.predict(observation(4))

    assert any(
        task["failure_reason"] == "goal_matches_recent_blocked_region"
        for task in agent.memory.task_history
    )
    assert planner.scan_states[-1]["semantic_relocalization_required"] is False
    # The second blocked navigation is still consecutive until a move succeeds.
    assert planner.scan_states[-1]["consecutive_blocked_navigation"] == 1
    assert planner.calls == 3
    assert agent.executor.active_task is not None
    assert agent.executor.active_task.status in {TaskStatus.RUNNING, TaskStatus.BLOCKED}
    assert not agent.runtime.scan_active


def test_blocked_navigation_uses_focused_reacquisition_before_full_scan(tmp_path):
    planner = ScriptedPlanner([goal_action(2.0), scan_action()])
    agent = make_agent(tmp_path, planner, DeterministicHarness(blocked=True))
    agent.predict(observation(0))
    agent.predict(observation(1))

    focused = agent.predict(observation(2, yaw_deg=90))
    terminal = focused.extra["agentnav_trace"]["vlm"]["terminal"]

    assert terminal["action"] == "SCAN_360"
    assert terminal["execution_mode"] == "FOCUSED_REACQUIRE"
    assert terminal["task"]["task_type"] == "turn"
    assert terminal["task"]["turn_direction"] == "right"
    assert terminal["task"]["turn_requested_angle_deg"] == pytest.approx(90.0)
    assert agent.runtime.focused_reacquire_active
    assert agent.runtime.focused_reacquire_attempted
    assert agent.runtime.focused_reacquire_stage == 1
    assert not agent.runtime.scan_active
    assert agent.runtime.recovery_scan_cycles == 0


def test_stuck_and_timeout_are_exposed_before_replan(tmp_path):
    stuck_planner = ScriptedPlanner([goal_action(2.0), scan_action()])
    stuck = make_agent(tmp_path / "stuck", stuck_planner, DeterministicHarness())
    stuck.predict(observation(0))
    stuck.predict(observation(1))
    stuck.predict(observation(2))
    trace = stuck.predict(observation(3)).extra["agentnav_trace"]
    assert trace["task_status"] == TaskStatus.STUCK.value
    assert trace["vlm_called"] is True

    timeout_planner = ScriptedPlanner([goal_action(2.0), scan_action()])
    timeout = make_agent(
        tmp_path / "timeout", timeout_planner, DeterministicHarness(), max_task_steps=2
    )
    timeout.predict(observation(0))
    timeout.predict(observation(1))
    timeout.predict(observation(2))
    trace = timeout.predict(observation(3)).extra["agentnav_trace"]
    assert trace["task_status"] == TaskStatus.TIMEOUT.value
    assert trace["vlm_called"] is True


def test_system_errors_finish_task_and_never_semantically_replan(tmp_path):
    planner = ScriptedPlanner([goal_action(2.0)])
    agent = make_agent(tmp_path / "executor", planner, DeterministicHarness(raises=True))
    agent.predict(observation(0))
    with pytest.raises(ExecutorSystemError, match="depth backend unavailable"):
        agent.predict(observation(1))
    assert planner.calls == 1
    assert agent.mode is NavMode.SYSTEM_ERROR
    assert agent.executor.active_task is None
    assert agent.memory.task_history[-1]["task_status"] == TaskStatus.SYSTEM_ERROR.value

    high_level = make_agent(
        tmp_path / "high_level", FailingPlanner(), DeterministicHarness()
    )
    with pytest.raises(ExecutorSystemError, match="vLLM unavailable"):
        high_level.predict(observation(0))
    assert high_level.mode is NavMode.SYSTEM_ERROR
    assert high_level.executor.active_task is None


def test_failures_can_request_python_controlled_scan(tmp_path):
    planner = ScriptedPlanner(
        [goal_action(2.0, reachable=False), goal_action(2.0, reachable=False), scan_action()]
    )
    agent = make_agent(tmp_path, planner, DeterministicHarness())
    agent.predict(observation(0))
    agent.predict(observation(1))
    agent.predict(observation(2))
    assert agent.executor.active_task.task_type == "turn"
    assert agent.executor.active_task.turn_direction == "left"
    assert agent.executor.active_task.turn_requested_angle_deg == 45


def test_scan_turn_is_persistent_and_uses_new_front_rgb(tmp_path):
    planner = ScriptedPlanner([scan_action(), scan_action()])
    agent = make_agent(tmp_path, planner, DeterministicHarness())
    agent.predict(observation(0, color="red"))
    running = agent.predict(observation(1, color="green"))
    assert running.extra["agentnav_trace"]["vlm_called"] is False
    replanned = agent.predict(
        observation(2, yaw_deg=45, color="blue", distance_to_goal=1.5)
    )
    assert not replanned.arrive
    assert planner.calls == 2
    assert planner.modes == [NavMode.PLANNING, NavMode.PLANNING]
    trace = replanned.extra["agentnav_trace"]
    assert trace["transition_reason"] == "SCAN_VIEW_READY"
    assert trace["vlm"]["terminal"]["action"] == "SCAN_360"
    assert planner.front_images[0] is not planner.front_images[1]
    turn_audit = agent.memory.task_history[-1]
    assert turn_audit["turn_requested_angle_deg"] == pytest.approx(45.0)
    assert turn_audit["heading_before_deg"] == pytest.approx(0.0)
    assert turn_audit["heading_after_deg"] == pytest.approx(45.0)
    assert turn_audit["actual_delta_heading_deg"] == pytest.approx(45.0)
    assert turn_audit["frame_before"]["render_name"] == "0_front.jpg"
    assert turn_audit["frame_after"]["render_name"] == "2_front.jpg"
    assert turn_audit["frame_change_mae_0_255"] > 0.0

@pytest.mark.parametrize("distance_to_goal", [2.0001, float("nan")])
def test_visual_confirmation_cannot_terminate_outside_harness_distance_gate(
    tmp_path, distance_to_goal
):
    planner = ScriptedPlanner([goal_action(2.0), terminate_action])
    agent = make_agent(tmp_path, planner, DeterministicHarness())
    agent.predict(observation(0, color="red"))
    agent.predict(observation(1, forward=0.0, color="green"))
    agent.predict(observation(2, forward=0.8, color="yellow"))
    agent.predict(observation(3, forward=1.6, color="purple"))

    rejected = agent.predict(
        observation(
            4,
            forward=2.0,
            color="blue",
            distance_to_goal=distance_to_goal,
        )
    )

    assert not rejected.arrive
    assert agent.mode is NavMode.RECOVERY
    trace = rejected.extra["agentnav_trace"]
    assert trace["vlm"]["terminal"]["action"] == "TERMINATE"
    assert trace["verification_gate"]["target_visible"] is True
    assert trace["verification_gate"]["within_terminate_distance"] is False
    assert trace["verification_gate"]["terminate_distance_threshold_m"] == 2.0
    # A locally reached waypoint is valid progress when the POI remains
    # visible but is still too far. It must not poison the same approach as a
    # failed semantic goal.
    assert agent.memory.active_unconfirmed_goals() == []
    retry = agent.executor.create_navigation_task(
        observation(5, forward=1.8), measurement(0.2), "library", "same approach"
    )
    assert retry.status is TaskStatus.RUNNING


def test_visible_but_far_verification_immediately_creates_continuation_task(
    tmp_path,
):
    def terminate_with_continuation(
        safe, observation, harness, executor, memory
    ):
        return HighLevelDecision(
            {"action": "TERMINATE", "target_visible": True},
            [],
            [],
            0.0,
            {},
            continuation_measurement=measurement(3.0, (420, 400)),
        )

    planner = ScriptedPlanner(
        [goal_action(2.0, (360, 400)), terminate_with_continuation]
    )
    agent = make_agent(tmp_path, planner, DeterministicHarness())
    agent.predict(observation(0))
    agent.predict(observation(1, forward=0.0))
    agent.predict(observation(2, forward=0.8))
    agent.predict(observation(3, forward=1.6))

    continued = agent.predict(
        observation(4, forward=2.0, distance_to_goal=8.0)
    )

    assert not continued.arrive
    assert agent.mode is NavMode.EXECUTING
    assert agent.transition_reason == "POI_VISIBLE_CONTINUE_APPROACH"
    assert agent.executor.active_task is not None
    assert agent.executor.active_task.created_step == 4
    assert agent.executor.active_task.selected_pixel == (420, 400)
    assert agent.memory.active_unconfirmed_goals() == []
    terminal = continued.extra["agentnav_trace"]["vlm"]["terminal"]
    assert terminal["continuation_task"]["task_status"] == TaskStatus.RUNNING.value


def test_final_verification_can_succeed_at_stall_budget_boundary(tmp_path):
    planner = ScriptedPlanner([goal_action(0.1), terminate_action])
    agent = make_agent(tmp_path, planner, DeterministicHarness())
    agent.max_no_progress_steps = 1
    agent.predict(observation(0))
    prediction = agent.predict(observation(1, distance_to_goal=1.0))
    assert prediction.arrive
    assert agent.mode is NavMode.TERMINATED
    assert not agent.runtime.stop_reason
    assert planner.modes == [NavMode.PLANNING, NavMode.VERIFYING]

def test_scan_progress_reaches_each_new_planning_view(tmp_path):
    planner = ScriptedPlanner([scan_action(), scan_action(), scan_action()])
    agent = make_agent(tmp_path, planner, DeterministicHarness())

    agent.predict(observation(0, yaw_deg=0, color="red"))
    agent.predict(observation(1, yaw_deg=0, color="green"))
    first_replan = agent.predict(observation(2, yaw_deg=45, color="blue"))
    assert first_replan.extra["agentnav_trace"]["vlm"]["terminal"]["action"] == "SCAN_360"

    agent.predict(observation(3, yaw_deg=45, color="yellow"))
    second_replan = agent.predict(observation(4, yaw_deg=90, color="white"))
    assert second_replan.extra["agentnav_trace"]["vlm"]["terminal"]["action"] == "SCAN_360"

    assert planner.modes == [
        NavMode.PLANNING,
        NavMode.PLANNING,
        NavMode.PLANNING,
    ]
    assert [image.getpixel((0, 0)) for image in planner.front_images] == [
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 255),
    ]
    assert planner.scan_states[0]["views_checked"] == 0
    assert planner.scan_states[1]["views_checked"] == 2
    assert planner.scan_states[2]["views_checked"] == 3
    assert [state["direction"] for state in planner.scan_states] == ["left"] * 3

def test_goal_reached_verifies_then_replans_or_terminates(tmp_path):
    planner = ScriptedPlanner(
        [
            scan_action(),
            goal_action(2.0),
            terminate_action,
        ]
    )
    agent = make_agent(tmp_path, planner, DeterministicHarness())

    created_turn = agent.predict(observation(0, yaw_deg=0, color="red"))
    assert created_turn.extra["agentnav_trace"]["vlm"]["terminal"]["action"] == "SCAN_360"

    turn_running = agent.predict(observation(1, yaw_deg=0, color="green"))
    assert turn_running.extra["agentnav_trace"]["task_status"] == TaskStatus.RUNNING.value
    assert turn_running.extra["agentnav_trace"]["vlm_called"] is False

    replanned = agent.predict(observation(2, yaw_deg=45, color="blue"))
    replan_trace = replanned.extra["agentnav_trace"]
    assert replan_trace["task_status"] == TaskStatus.GOAL_REACHED.value
    assert replan_trace["finished_task"]["task_type"] == "turn"
    assert replan_trace["vlm"]["safe_input"]["mode"] == NavMode.PLANNING.value
    assert replan_trace["vlm"]["terminal"]["action"] == "SET_NAVIGATION_GOAL"

    for step, travel in [(3, 0.0), (4, 0.8), (5, 1.6)]:
        moving = agent.predict(
            observation(
                step,
                yaw_deg=45,
                color="purple",
                forward=travel,
            )
        )
        assert moving.extra["agentnav_trace"]["task_status"] == TaskStatus.RUNNING.value
        assert moving.extra["agentnav_trace"]["vlm_called"] is False

    verified = agent.predict(
        observation(
            6,
            yaw_deg=45,
            color="white",
            forward=2.0,
            distance_to_goal=1.5,
        )
    )
    verify_trace = verified.extra["agentnav_trace"]
    assert verified.arrive
    assert verify_trace["task_status"] == TaskStatus.GOAL_REACHED.value
    assert verify_trace["finished_task"]["task_type"] == "navigation"
    assert verify_trace["vlm"]["safe_input"]["mode"] == NavMode.VERIFYING.value
    assert verify_trace["vlm"]["terminal"]["action"] == "TERMINATE"
    assert planner.modes == [
        NavMode.PLANNING,
        NavMode.PLANNING,
        NavMode.VERIFYING,
    ]
    assert [image.getpixel((0, 0)) for image in planner.front_images] == [
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 255),
    ]

def test_navigation_verify_false_records_unconfirmed_goal(tmp_path):
    planner = ScriptedPlanner(
        [goal_action(2.0), reject_verification_action,
         goal_action(2.0, pixel=(420, 400))]
    )
    agent = make_agent(tmp_path, planner, DeterministicHarness())

    agent.predict(observation(0, color="red"))
    for step, forward in [(1, 0.0), (2, 0.8), (3, 1.6)]:
        running = agent.predict(
            observation(step, forward=forward, color="purple")
        )
        assert running.extra["agentnav_trace"]["task_status"] == (
            TaskStatus.RUNNING.value
        )

    rejected = agent.predict(
        observation(4, forward=2.0, color="blue")
    )
    trace = rejected.extra["agentnav_trace"]
    assert trace["vlm"]["safe_input"]["mode"] == NavMode.VERIFYING.value
    assert trace["vlm"]["terminal"]["action"] == "RETURN_TO_PLANNING"
    assert agent.mode is NavMode.RECOVERY
    assert not agent.runtime.semantic_relocalization_required
    assert agent.runtime.semantic_verification_failures == 1
    assert agent.runtime.reacquire_goal_world is None
    assert not agent.runtime.focused_reacquire_active
    assert len(agent.memory.unconfirmed_goals) == 1
    summary = agent.memory.high_level_summary()
    recent = summary["recent_verification_failures"][0]
    assert recent["reason"] == "POI is not visually confirmed"
    assert recent["semantic_anchor"] == "library sign"
    assert "source_pixel" not in recent
    assert "world_goal" not in repr(summary)

    fresh = agent.predict(observation(5, forward=2.0, color="green"))
    assert fresh.extra["agentnav_trace"]["vlm"]["safe_input"]["mode"] == "RECOVERY"
    assert fresh.extra["agentnav_trace"]["vlm"]["terminal"]["action"] == "SET_NAVIGATION_GOAL"
    assert agent.executor.active_task.selected_pixel == (420, 400)
    assert not agent.runtime.scan_active


def test_episode_reset_clears_runtime_memory_and_depth_state(tmp_path):
    planner = ScriptedPlanner([scan_action()])
    agent = make_agent(tmp_path, planner, DeterministicHarness())
    agent.predict(observation(0))
    agent.memory.failed_directions.append("front")
    agent.reset()
    assert agent.mode is NavMode.PLANNING
    assert agent.memory.poi_name == ""
    assert agent.memory.failed_directions == []
    assert agent.executor.active_task is None
    assert agent.status_call_steps == []
