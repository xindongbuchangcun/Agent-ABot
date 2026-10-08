from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from agentnav.abot.depth import DepthPrediction, ObservationDepthCache
from agentnav.abot.executor import ABotS1Executor, ExecutorConfig
from agentnav.abot.geometry import local_to_world, world_to_local
from agentnav.abot.observation import to_agent_safe_observation
from agentnav.abot.types import (
    CameraIntrinsics,
    EpisodeMemory,
    NavMode,
    PixelMeasurement,
    PixelProposal,
    TaskStatus,
)


class FakeDepth:
    def __init__(self, value=2.0):
        self.calls = 0
        self.value = value

    def predict(self, rgb):
        self.calls += 1
        return DepthPrediction(np.full((640, 720), self.value, dtype=np.float32))


def observation(step=0, pose=None, color="gray"):
    return SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), color)},
        poi_name="library",
        step_count=step,
        rotation=np.eye(4) if pose is None else pose,
        target_position=np.array([99.0, 99.0]),
        distance_to_goal=123.0,
        occ_map=np.ones((10, 10)),
        meta_data={"secret": True},
    )


def heading_pose(degrees):
    heading = np.deg2rad(degrees)
    pose = np.eye(4)
    pose[:2, :2] = [
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading), np.cos(heading)],
    ]
    return pose


def test_safe_observation_whitelists_policy_fields():
    safe = to_agent_safe_observation(observation(), NavMode.PLANNING, EpisodeMemory())
    assert set(vars(safe)) == {
        "poi_name",
        "front_rgb",
        "step_count",
        "mode",
        "memory_summary",
        "transition_reason",
        "scan_state",
    }
    text = repr(safe.memory_summary)
    assert "target_position" not in text
    assert "distance_to_goal" not in text
    assert "occ_map" not in text


def test_world_local_round_trip_matches_abot_convention():
    pose = np.eye(4)
    pose[:3, 3] = [4.0, -3.0, 0.65]
    local = np.array([2.3, -0.7])
    assert np.allclose(world_to_local(local_to_world(local, pose), pose), local)


def test_depth_cache_runs_once_per_frame():
    estimator = FakeDepth()
    cache = ObservationDepthCache(estimator)
    rgb = observation().images["front"]
    assert cache.get(3, rgb) is cache.get(3, rgb)
    assert estimator.calls == 1
    cache.get(4, rgb)
    assert estimator.calls == 2


class AlwaysSafeHarness:
    def __init__(self):
        self.last_max_lookahead_m = None

    def depth_corridor_is_safe(self, depth, waypoint, max_lookahead_m=None):
        self.last_max_lookahead_m = max_lookahead_m
        return True, {"reason": "clear_depth_corridor", "obstacle_risk": 0.0}

    def current_depth(self, observation):
        return np.ones((16, 16), dtype=np.float32)


def measurement(distance=6.0):
    return PixelMeasurement(
        PixelProposal("P0", 360, 400, "path"),
        distance,
        0.0,
        1.0,
        np.array([distance, 0.0]),
        True,
        True,
        {"reason": "clear_depth_corridor"},
    )


def test_remote_goal_is_truncated_to_midpoint():
    memory = EpisodeMemory()
    executor = ABotS1Executor(AlwaysSafeHarness(), memory)
    task = executor.create_navigation_task(observation(), measurement(8.0), "door", "ground")
    assert task.is_midpoint_task
    assert np.linalg.norm(task.goal_local_initial) == pytest.approx(6.0)


def test_exploration_extends_ground_route_without_changing_semantic_goal():
    ordinary = ABotS1Executor(AlwaysSafeHarness(), EpisodeMemory())
    normal_task = ordinary.create_navigation_task(
        observation(), measurement(1.0), "library", "entrance"
    )
    explorer = ABotS1Executor(AlwaysSafeHarness(), EpisodeMemory())
    explore_task = explorer.create_navigation_task(
        observation(), measurement(1.0), "post-scan exploration", "open ground",
        exploration=True,
    )
    assert normal_task.initial_distance == pytest.approx(1.0)
    assert explore_task.initial_distance == pytest.approx(1.25)
    assert np.linalg.norm(explore_task.goal_local_initial) == pytest.approx(1.25)
    assert normal_task.semantic_goal_world is not None
    assert explore_task.semantic_goal_world is None


def test_distant_exploration_reobserves_after_short_segment():
    executor = ABotS1Executor(
        AlwaysSafeHarness(), EpisodeMemory(),
        ExecutorConfig(enable_long_horizon_preview=True),
    )
    task = executor.create_navigation_task(
        observation(), measurement(6.0), "post-scan exploration", "open ground",
        exploration=True,
    )
    assert task.is_midpoint_task
    assert task.initial_distance == pytest.approx(2.0)
    assert task.long_horizon_preview_allowed
    assert task.semantic_goal_world is None


def test_failed_pixel_number_is_only_blocked_in_its_original_frame():
    executor = ABotS1Executor(AlwaysSafeHarness(), EpisodeMemory())
    first = executor.create_navigation_task(
        observation(), measurement(2.0), "library", "entrance"
    )
    executor.mark_failed(TaskStatus.BLOCKED, "no_safe_local_candidate")
    executor.finish_current_task()
    assert executor.candidate_recently_blocked(observation(), measurement(2.0))
    moved_pose = np.eye(4)
    moved_pose[0, 3] = 2.0
    later = observation(step=8, pose=moved_pose)
    assert not executor.candidate_recently_blocked(later, measurement(2.0))
    other = replace(
        measurement(2.0),
        proposal=PixelProposal("P1", 361, 400, "new current-frame pixel"),
    )
    assert not executor.candidate_recently_blocked(later, other)
    assert first.selected_pixel == (360, 400)


def test_blocked_motion_allows_a_fresh_alternative_from_the_same_viewpoint():
    executor = ABotS1Executor(AlwaysSafeHarness(), EpisodeMemory())
    executor.create_navigation_task(
        observation(), measurement(8.0), "library", "entrance"
    )
    current = observation(step=1)
    assert executor.task_status(current) is TaskStatus.RUNNING
    executor._plan_local_waypoint = lambda depth, goal: (
        None, [{"offset_deg": 0.0, "safe": False}]
    )
    executor.step(current)
    assert executor.active_task.status is TaskStatus.BLOCKED
    executor.finish_current_task()

    alternative = replace(
        measurement(2.0),
        proposal=PixelProposal("P1", 361, 400, "fresh current-frame pixel"),
    )
    assert not executor.candidate_recently_blocked(observation(step=2), alternative)
    assert not executor.candidate_recently_blocked(
        observation(step=3, pose=heading_pose(90)), alternative
    )


def test_blocked_region_rejects_same_bearing_but_allows_side_route():
    executor = ABotS1Executor(AlwaysSafeHarness(), EpisodeMemory())
    executor._blocked_regions.append((np.array([0.7, 0.0, 0.0]), np.zeros(2)))
    straight = measurement(0.7)
    side = replace(
        straight,
        proposal=PixelProposal("P1", 390, 400, "new side route"),
        local_goal=np.array([0.7 * np.cos(np.deg2rad(30)),
                             0.7 * np.sin(np.deg2rad(30))]),
    )
    current = observation(step=2)
    assert executor.candidate_recently_blocked(current, straight)
    assert not executor.candidate_recently_blocked(current, side)


def test_near_frontal_obstacle_halves_side_detour_step():
    class NarrowPassageHarness(AlwaysSafeHarness):
        def depth_corridor_is_safe(self, depth, waypoint, max_lookahead_m=None):
            angle = np.degrees(np.arctan2(waypoint[1], waypoint[0]))
            if 20.0 < angle < 30.0:
                return True, {"reason": "clear_depth_corridor", "obstacle_risk": 0.0}
            return False, {
                "reason": "depth_obstacle", "nearest_obstacle_m": 0.3,
                "obstacle_risk": 1.0,
            }

    executor = ABotS1Executor(NarrowPassageHarness(), EpisodeMemory())
    waypoint, candidates = executor._plan_local_waypoint(
        np.ones((16, 16), dtype=np.float32), np.array([2.0, 0.0])
    )
    assert np.linalg.norm(waypoint) == pytest.approx(0.175)
    assert candidates[0]["detour_step_limited_m"] == pytest.approx(0.175)


def test_depth_retreat_uses_last_supported_pose_once_per_region():
    executor = ABotS1Executor(AlwaysSafeHarness(), EpisodeMemory())
    executor.create_navigation_task(
        observation(), measurement(2.0), "library", "entrance"
    )
    assert executor.task_status(observation(step=1)) is TaskStatus.RUNNING
    executor.step(observation(step=1))
    moved_pose = np.eye(4)
    moved_pose[0, 3] = 0.35
    moved = observation(step=2, pose=moved_pose)
    assert executor.task_status(moved) is TaskStatus.RUNNING
    executor._plan_local_waypoint = lambda depth, goal: (
        None, [{"safety": {"reason": "insufficient_corridor_evidence"}}]
    )
    executor.step(moved)
    assert executor.active_task.status is TaskStatus.BLOCKED
    executor.finish_current_task()
    retreat = executor.create_depth_retreat_task(observation(step=3, pose=moved_pose))
    assert retreat is not None
    assert retreat.task_type == "depth_retreat"
    assert np.allclose(retreat.goal_world, np.zeros(3))
    assert executor.create_depth_retreat_task(observation(step=4, pose=moved_pose)) is None


def test_task_status_may_be_polled_once_per_environment_step():
    executor = ABotS1Executor(AlwaysSafeHarness(), EpisodeMemory())
    executor.create_turn_task(observation(), "left", 90, "explore")
    assert executor.task_status(observation(step=1)) is TaskStatus.RUNNING
    with pytest.raises(RuntimeError, match="more than once"):
        executor.task_status(observation(step=1))


def test_progressing_navigation_can_pass_soft_timeout_but_not_hard_timeout():
    executor = ABotS1Executor(
        AlwaysSafeHarness(), EpisodeMemory(), ExecutorConfig(max_task_steps=30)
    )
    executor.create_navigation_task(observation(), measurement(8.0), "library", "entrance")
    for step in range(1, 62):
        pose = np.eye(4)
        pose[0, 3] = 0.04 * step
        status = executor.task_status(observation(step=step, pose=pose))
        assert status is (TaskStatus.RUNNING if step <= 60 else TaskStatus.TIMEOUT)


def test_navigation_without_recent_goal_progress_times_out_at_soft_limit():
    executor = ABotS1Executor(
        AlwaysSafeHarness(), EpisodeMemory(), ExecutorConfig(max_task_steps=30)
    )
    executor.create_navigation_task(observation(), measurement(8.0), "library", "entrance")
    for step in range(1, 32):
        pose = np.eye(4)
        pose[1, 3] = 0.06 * step
        status = executor.task_status(observation(step=step, pose=pose))
    assert status is TaskStatus.TIMEOUT


def test_goal_reached_on_timeout_boundary_is_not_marked_timeout():
    executor = ABotS1Executor(
        AlwaysSafeHarness(), EpisodeMemory(), ExecutorConfig(max_task_steps=1)
    )
    executor.create_navigation_task(observation(), measurement(2.0), "library", "entrance")
    assert executor.task_status(observation(step=1)) is TaskStatus.RUNNING
    pose = np.eye(4)
    pose[0, 3] = 2.0
    assert executor.task_status(observation(step=2, pose=pose)) is TaskStatus.GOAL_REACHED


def test_turn_completion_uses_actual_heading_and_records_frame_change():
    executor = ABotS1Executor(AlwaysSafeHarness(), EpisodeMemory())
    task = executor.create_turn_task(
        observation(color="black"), "left", 45, "inspect"
    )

    first = observation(step=1, pose=heading_pose(0), color="gray")
    assert executor.task_status(first) is TaskStatus.RUNNING
    command = executor.step(first)
    assert np.rad2deg(np.arctan2(command.directions[0, 1], command.directions[0, 0])) == pytest.approx(45.0)

    small_turn = observation(step=2, pose=heading_pose(3), color="darkgray")
    assert executor.task_status(small_turn) is TaskStatus.RUNNING
    assert task.public_dict()["actual_delta_heading_deg"] == pytest.approx(3.0)

    completed = observation(step=3, pose=heading_pose(45), color="white")
    assert executor.task_status(completed) is TaskStatus.GOAL_REACHED
    audit = task.public_dict()
    assert audit["turn_requested_angle_deg"] == pytest.approx(45.0)
    assert audit["heading_before_deg"] == pytest.approx(0.0)
    assert audit["heading_after_deg"] == pytest.approx(45.0)
    assert audit["actual_delta_heading_deg"] == pytest.approx(45.0)
    assert audit["frame_before"]["render_name"] == "0_front.jpg"
    assert audit["frame_after"]["render_name"] == "3_front.jpg"
    assert audit["frame_before"]["sha256"] != audit["frame_after"]["sha256"]
    assert audit["frame_change_mae_0_255"] > 0.0


def test_navigation_prediction_returns_unit_direction_for_resulting_pose():
    executor = ABotS1Executor(AlwaysSafeHarness(), EpisodeMemory())
    diagonal = PixelMeasurement(
        PixelProposal("P0", 300, 400, "diagonal approach"),
        2.0,
        0.0,
        1.0,
        np.array([1.6, 1.2]),
        True,
        True,
        {"reason": "clear_depth_corridor"},
    )
    executor.create_navigation_task(
        observation(step=0), diagonal, "library", "left entrance"
    )
    current = observation(step=1)

    assert executor.task_status(current) is TaskStatus.RUNNING
    prediction = executor.step(current)

    api_waypoint = prediction.waypoint[0]
    expected = np.array([api_waypoint[1], -api_waypoint[0]])
    expected /= np.linalg.norm(expected)
    assert prediction.directions is not None
    assert prediction.directions.shape == (1, 2)
    assert np.linalg.norm(prediction.directions[0]) == pytest.approx(1.0)
    assert np.allclose(prediction.directions[0], expected)
    assert np.allclose(
        executor.last_plan_debug["selected_direction_front_left"], expected
    )
    from abotn_evaluator.point_goal.evaluator import PointGoalEvaluator

    memory = SimpleNamespace(get_last_pose=lambda: current.rotation)
    next_pose = PointGoalEvaluator.get_pred_poses(
        prediction.waypoint, prediction.directions, memory
    )[0]
    expected_position = local_to_world(
        expected * np.linalg.norm(api_waypoint), current.rotation
    )
    assert np.allclose(next_pose[:3, 3], expected_position, atol=1e-6)
    assert np.allclose(next_pose[:2, 0], expected, atol=1e-6)



def test_navigation_aligns_to_goal_outside_front_view_before_moving():
    executor = ABotS1Executor(AlwaysSafeHarness(), EpisodeMemory())
    side_goal = PixelMeasurement(
        PixelProposal("P0", 300, 400, "side entrance"),
        4.0, 0.0, 1.0, np.array([0.0, 4.0]), True, True,
        {"reason": "clear_depth_corridor"},
    )
    task = executor.create_navigation_task(observation(), side_goal, "library", "entrance")

    first = observation(step=1)
    assert executor.task_status(first) is TaskStatus.RUNNING
    command = executor.step(first)
    assert np.allclose(command.waypoint, 0.0)
    assert task.phase == "aligning"
    assert executor.last_plan_debug["type"] == "navigation_alignment"
    assert np.rad2deg(np.arctan2(command.directions[0, 1], command.directions[0, 0])) == pytest.approx(45.0)

    second = observation(step=2, pose=heading_pose(45))
    assert executor.task_status(second) is TaskStatus.RUNNING
    command = executor.step(second)
    assert np.linalg.norm(command.waypoint) > 0.0
    assert task.phase == "moving"


def test_local_planner_rejects_high_risk_clear_corridor_and_uses_detour():
    class RiskHarness(AlwaysSafeHarness):
        def depth_corridor_is_safe(self, depth, waypoint, max_lookahead_m=None):
            angle_deg = np.rad2deg(np.arctan2(waypoint[1], waypoint[0]))
            risk = 0.75 if abs(angle_deg) < 1.0 else 0.0
            return True, {
                "reason": "clear_depth_corridor",
                "obstacle_risk": risk,
            }

    executor = ABotS1Executor(RiskHarness(), EpisodeMemory())
    waypoint, candidates = executor._plan_local_waypoint(
        np.ones((16, 16), dtype=np.float32), np.array([2.0, 0.0])
    )

    assert candidates[0]["obstacle_risk"] == pytest.approx(0.75)
    assert waypoint is not None
    assert abs(np.rad2deg(np.arctan2(waypoint[1], waypoint[0]))) == pytest.approx(25.0)


def test_secondary_depth_guard_limits_a_clear_primary_step_and_detours_when_close():
    class SecondaryHarness(AlwaysSafeHarness):
        depth_corridor_lookahead_m = 0.7

        def __init__(self, nearest):
            super().__init__()
            self.nearest = nearest

        def depth_corridor_is_safe(self, depth, waypoint, max_lookahead_m=None):
            angle = abs(np.rad2deg(np.arctan2(waypoint[1], waypoint[0])))
            if angle < 1.0:
                return False, {
                    "reason": "depth_obstacle", "blocking_points": 100,
                    "blocking_columns": 20, "nearest_obstacle_m": self.nearest,
                }
            return True, {"reason": "clear_depth_corridor", "blocking_points": 0}

    depth = np.ones((16, 16), dtype=np.float32)
    cautious = ABotS1Executor(
        AlwaysSafeHarness(), EpisodeMemory(),
        secondary_harness=SecondaryHarness(0.6),
    )
    short, records = cautious._plan_local_waypoint(depth, np.array([2.0, 0.0]), depth)
    assert np.allclose(short, [0.175, 0.0])
    assert records[0]["secondary_depth_guard"]["status"] == "short_step"

    close = ABotS1Executor(
        AlwaysSafeHarness(), EpisodeMemory(),
        secondary_harness=SecondaryHarness(0.3),
    )
    detour, records = close._plan_local_waypoint(depth, np.array([2.0, 0.0]), depth)
    assert records[0]["secondary_depth_guard"]["status"] == "blocked"
    assert records[0]["safe"] is False
    assert detour is not None
    assert abs(np.rad2deg(np.arctan2(detour[1], detour[0]))) == pytest.approx(25.0)


def test_depth_scale_disagreement_shortens_step_without_blocking():
    class ScaleHarness(AlwaysSafeHarness):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def depth_corridor_is_safe(self, depth, waypoint, max_lookahead_m=None):
            return True, {
                "reason": "clear_depth_corridor",
                "obstacle_risk": 0.0,
                "depth_scale_correction": self.scale,
            }

    depth = np.ones((16, 16), dtype=np.float32)
    uncertain = ABotS1Executor(
        ScaleHarness(0.5), EpisodeMemory(),
        secondary_harness=ScaleHarness(0.9),
    )
    waypoint, records = uncertain._plan_local_waypoint(
        depth, np.array([2.0, 0.0]), depth
    )
    assert np.linalg.norm(waypoint) == pytest.approx(0.175)
    assert records[0]["secondary_depth_guard"]["scale_disagreement"]
    assert records[0]["safe"] is True

    agreeing = ABotS1Executor(
        ScaleHarness(0.8), EpisodeMemory(),
        secondary_harness=ScaleHarness(0.9),
    )
    waypoint, _ = agreeing._plan_local_waypoint(
        depth, np.array([2.0, 0.0]), depth
    )
    assert np.linalg.norm(waypoint) == pytest.approx(0.35)


def test_optional_long_horizon_preview_prefers_a_clear_side_without_weakening_short_step_safety():
    class DistantObstacleHarness(AlwaysSafeHarness):
        def depth_corridor_is_safe(self, depth, waypoint, max_lookahead_m=None):
            distance = float(np.linalg.norm(waypoint))
            angle = np.rad2deg(np.arctan2(waypoint[1], waypoint[0]))
            if distance > 1.0 and not 15.0 < angle < 35.0:
                return False, {
                    "reason": "depth_obstacle", "nearest_obstacle_m": 1.2,
                    "obstacle_risk": 1.0,
                }
            return True, {"reason": "clear_depth_corridor", "obstacle_risk": 0.0}

    depth = np.ones((16, 16), dtype=np.float32)
    goal = np.array([3.0, 0.0])
    baseline = ABotS1Executor(DistantObstacleHarness(), EpisodeMemory())
    baseline_waypoint, _ = baseline._plan_local_waypoint(depth, goal)
    preview = ABotS1Executor(
        DistantObstacleHarness(), EpisodeMemory(),
        ExecutorConfig(enable_long_horizon_preview=True),
    )
    preview_waypoint, candidates = preview._plan_local_waypoint(depth, goal)

    assert np.allclose(baseline_waypoint, [0.35, 0.0])
    assert preview_waypoint is not None and preview_waypoint[1] > 0.0
    assert np.linalg.norm(preview_waypoint) == pytest.approx(0.35)
    assert candidates[0]["long_horizon_preview"]["reason"] == "depth_obstacle"
    assert next(item for item in candidates if item["offset_deg"] == 25.0)["safe"]


def test_long_horizon_preview_does_not_steer_for_distant_depth_noise():
    class FarObstacleHarness(AlwaysSafeHarness):
        def depth_corridor_is_safe(self, depth, waypoint, max_lookahead_m=None):
            if float(np.linalg.norm(waypoint)) > 1.0:
                return False, {
                    "reason": "depth_obstacle", "nearest_obstacle_m": 1.9,
                    "obstacle_risk": 1.0,
                }
            return True, {"reason": "clear_depth_corridor", "obstacle_risk": 0.0}

    executor = ABotS1Executor(
        FarObstacleHarness(), EpisodeMemory(),
        ExecutorConfig(enable_long_horizon_preview=True),
    )
    waypoint, candidates = executor._plan_local_waypoint(
        np.ones((16, 16), dtype=np.float32), np.array([3.0, 0.0])
    )

    assert np.allclose(waypoint, [0.35, 0.0])
    assert len(candidates) == 7


def test_preview_skips_navigation_from_bounded_far_depth_fallback():
    class ObstacleHarness(AlwaysSafeHarness):
        def depth_corridor_is_safe(self, depth, waypoint, max_lookahead_m=None):
            if float(np.linalg.norm(waypoint)) > 1.0:
                return False, {
                    "reason": "depth_obstacle", "nearest_obstacle_m": 0.9,
                    "obstacle_risk": 1.0,
                }
            return True, {"reason": "clear_depth_corridor", "obstacle_risk": 0.0}

    executor = ABotS1Executor(
        ObstacleHarness(), EpisodeMemory(),
        ExecutorConfig(enable_long_horizon_preview=True),
    )
    far_measurement = replace(
        measurement(6.0),
        safety_debug={"reason": "far_depth_bearing_fallback"},
    )
    task = executor.create_navigation_task(
        observation(), far_measurement, "library", "sign"
    )
    waypoint, candidates = executor._plan_local_waypoint(
        np.ones((16, 16), dtype=np.float32), np.array([3.0, 0.0])
    )

    assert task.long_horizon_preview_allowed is False
    assert np.allclose(waypoint, [0.35, 0.0])
    assert len(candidates) == 7



def test_local_planner_accepts_safe_corridor_at_risk_limit():
    class BorderlineHarness(AlwaysSafeHarness):
        def depth_corridor_is_safe(self, depth, waypoint, max_lookahead_m=None):
            return True, {
                "reason": "clear_depth_corridor",
                "obstacle_risk": 0.5,
            }

    executor = ABotS1Executor(BorderlineHarness(), EpisodeMemory())
    waypoint, candidates = executor._plan_local_waypoint(
        np.ones((16, 16), dtype=np.float32), np.array([2.0, 0.0])
    )

    assert candidates[0]["safe"] is True
    assert candidates[0]["obstacle_risk"] == pytest.approx(0.5)
    assert waypoint is not None
    assert np.allclose(waypoint, [0.35, 0.0])


@pytest.mark.parametrize("distance", [0.158, 0.190])
def test_local_planner_can_finish_a_safe_short_step(distance):
    executor = ABotS1Executor(AlwaysSafeHarness(), EpisodeMemory())
    waypoint, candidates = executor._plan_local_waypoint(
        np.ones((16, 16), dtype=np.float32), np.array([distance, 0.0])
    )

    assert candidates[0]["safe"] is True
    assert waypoint is not None
    assert np.allclose(waypoint, [distance - executor.config.local_goal_tolerance, 0.0])


def test_local_planner_still_rejects_unsafe_short_step():
    class BlockedHarness(AlwaysSafeHarness):
        def depth_corridor_is_safe(self, depth, waypoint, max_lookahead_m=None):
            return False, {"reason": "depth_obstacle", "obstacle_risk": 1.0}

    executor = ABotS1Executor(BlockedHarness(), EpisodeMemory())
    waypoint, candidates = executor._plan_local_waypoint(
        np.ones((16, 16), dtype=np.float32), np.array([0.18, 0.0])
    )

    assert waypoint is None
    assert all(not candidate["safe"] for candidate in candidates)



def test_local_planner_rejects_clear_candidate_outside_depth_fov():
    executor = ABotS1Executor(AlwaysSafeHarness(), EpisodeMemory())
    waypoint, candidates = executor._plan_local_waypoint(
        np.ones((16, 16), dtype=np.float32), np.array([2.0, 0.0])
    )

    assert waypoint is not None
    assert candidates[-2]["offset_deg"] == pytest.approx(65.0)
    assert candidates[-2]["safe"] is False
    assert candidates[-2]["safety"]["reason"] == "candidate_outside_front_depth_view"
    assert candidates[-1]["safe"] is False


def test_endpoint_distance_caps_corridor_lookahead():
    harness = AlwaysSafeHarness()
    executor = ABotS1Executor(harness, EpisodeMemory())
    executor.create_navigation_task(
        observation(step=0), measurement(0.4), "library", "near entrance"
    )
    current = observation(step=1)

    assert executor.task_status(current) is TaskStatus.RUNNING
    executor.step(current)

    assert harness.last_max_lookahead_m == pytest.approx(0.4)


def test_midpoint_task_preserves_full_semantic_goal_bearing():
    executor = ABotS1Executor(AlwaysSafeHarness(), EpisodeMemory())
    task = executor.create_navigation_task(
        observation(step=0), measurement(8.0), "library", "entrance"
    )

    assert task.is_midpoint_task
    assert np.allclose(task.goal_world[:2], [6.0, 0.0])
    assert np.allclose(task.semantic_goal_world[:2], [8.0, 0.0])


def test_unreachable_measurement_becomes_invalid_goal():
    bad = PixelMeasurement(
        PixelProposal("P0", 360, 400),
        None,
        None,
        0.0,
        np.zeros(2),
        False,
        False,
        {"reason": "missing_depth"},
    )
    executor = ABotS1Executor(AlwaysSafeHarness(), EpisodeMemory())
    task = executor.create_navigation_task(observation(), bad, "door", "ground")
    assert task.status is TaskStatus.INVALID_GOAL
    assert task.failure_reason == "missing_depth"


def test_harness_query_count_is_episode_scoped():
    from agentnav.abot.harness import PixelHarness

    harness = PixelHarness(
        ObservationDepthCache(FakeDepth()),
        CameraIntrinsics(),
        depth_min_corridor_points=1,
    )
    harness.query_candidates(
        observation(),
        [PixelProposal("P0", 360, 400), PixelProposal("P1", 380, 400)],
    )
    assert harness.total_query_count == 2
    harness.reset()
    assert harness.total_query_count == 0


def test_harness_rejects_only_extreme_image_corner_anchors():
    from agentnav.abot.harness import PixelHarness

    harness = PixelHarness(
        ObservationDepthCache(FakeDepth(2.0)), CameraIntrinsics()
    )
    corner, nearby = harness.query_candidates(
        observation(),
        [
            PixelProposal("corner", 719, 639, "sign in upper right"),
            PixelProposal("nearby", 710, 630, "fresh nearby ground"),
        ],
    )
    assert not corner.reachable
    assert corner.safety_debug["reason"] == "image_corner_anchor_unreliable"
    assert nearby.reachable


def test_upper_and_lower_image_pixels_can_be_navigation_candidates():
    from agentnav.abot.harness import PixelHarness

    harness = PixelHarness(
        ObservationDepthCache(FakeDepth(2.0)),
        CameraIntrinsics(),
    )
    harness.depth_corridor_is_safe = lambda depth, waypoint: (
        True,
        {"reason": "clear_depth_corridor"},
    )
    measurements = harness.query_candidates(
        observation(),
        [PixelProposal("sign", 360, 80), PixelProposal("road", 360, 500)],
    )
    assert [item.depth_reliable for item in measurements] == [True, True]
    assert [item.reachable for item in measurements] == [True, True]



def scaled_floor_depth(scale_error=1.6):
    camera = CameraIntrinsics()
    depth = np.zeros((camera.height, camera.width), dtype=np.float32)
    rows = np.arange(camera.height, dtype=np.float64)
    floor_rows = rows > camera.cy
    true_floor_depth = (
        camera.extrinsic_height
        * camera.fy
        / (rows[floor_rows] - camera.cy)
    )
    depth[floor_rows, :] = true_floor_depth[:, None] / scale_error
    return depth


class ArrayDepth:
    def __init__(self, depth):
        self.depth = np.asarray(depth, dtype=np.float32)

    def predict(self, rgb):
        return DepthPrediction(self.depth.copy())


def test_ground_fit_corrects_target_depth_and_defers_route_check():
    from agentnav.abot.harness import PixelHarness

    camera = CameraIntrinsics()
    depth = scaled_floor_depth(scale_error=1.6)
    harness = PixelHarness(ObservationDepthCache(ArrayDepth(depth)), camera)
    harness.depth_corridor_is_safe = lambda *args: (_ for _ in ()).throw(
        AssertionError("target creation must not run corridor safety")
    )

    measurement = harness.query_candidates(
        observation(), [PixelProposal("floor", 360, 500)]
    )[0]

    expected_depth = camera.extrinsic_height * camera.fy / (500 - camera.cy)
    assert measurement.depth_m == pytest.approx(expected_depth, rel=0.03)
    assert measurement.corridor_safe is None
    assert measurement.reachable
    assert measurement.safety_debug["reason"] == (
        "route_check_deferred_to_executor"
    )
    calibration = measurement.safety_debug["ground_calibration"]
    assert calibration["ground_plane_found"] is True
    assert calibration["depth_scale_correction"] == pytest.approx(1.6, rel=0.03)


def test_ground_fit_rejects_implausibly_large_monocular_scale():
    from agentnav.abot.harness import PixelHarness

    depth = scaled_floor_depth(scale_error=1.9)
    harness = PixelHarness(
        ObservationDepthCache(ArrayDepth(depth)), CameraIntrinsics()
    )

    safe, debug = harness.depth_corridor_is_safe(
        depth, np.array([0.5, 0.0])
    )

    assert not safe
    assert debug["ground_plane_found"] is False
    assert debug["reason"] == "insufficient_ground_plane_support"


def test_narrow_two_column_obstacle_blocks_corridor():
    from agentnav.abot.harness import PixelHarness

    depth = scaled_floor_depth(scale_error=1.6)
    depth[400:412, 356:364] = 0.35 / 1.6
    harness = PixelHarness(
        ObservationDepthCache(ArrayDepth(depth)), CameraIntrinsics()
    )

    safe, debug = harness.depth_corridor_is_safe(
        depth, np.array([0.5, 0.0])
    )

    assert not safe
    assert debug["reason"] == "depth_obstacle"
    assert debug["blocking_points"] >= 4
    assert debug["blocking_columns"] >= 2


def test_ground_fit_keeps_scaled_floor_clear_and_detects_real_obstacle():
    from agentnav.abot.harness import PixelHarness

    depth = scaled_floor_depth(scale_error=1.6)
    harness = PixelHarness(
        ObservationDepthCache(ArrayDepth(depth)), CameraIntrinsics()
    )
    # At 0.5 m the lower camera sees enough of the fitted floor to support
    # a clear-corridor decision.
    waypoint = np.array([0.5, 0.0])

    floor_safe, floor_debug = harness.depth_corridor_is_safe(depth, waypoint)

    obstacle_depth = depth.copy()
    obstacle_depth[330:520, 320:400] = 0.35 / 1.6
    obstacle_safe, obstacle_debug = harness.depth_corridor_is_safe(
        obstacle_depth, waypoint
    )

    assert floor_safe
    assert floor_debug["ground_plane_found"] is True
    assert floor_debug["depth_scale_correction"] == pytest.approx(1.6, rel=0.03)
    assert floor_debug["blocking_points"] == 0
    assert not obstacle_safe
    assert obstacle_debug["reason"] == "depth_obstacle"
    assert obstacle_debug["blocking_columns"] >= 4
    assert obstacle_debug["nearest_obstacle_m"] == pytest.approx(0.35, abs=0.01)


def test_supported_floor_inside_near_body_image_band_is_clear():
    from agentnav.abot.harness import PixelHarness

    depth = scaled_floor_depth(scale_error=1.6)
    harness = PixelHarness(
        ObservationDepthCache(ArrayDepth(depth)), CameraIntrinsics()
    )
    safe, debug = harness.depth_corridor_is_safe(depth, np.array([1.0, 0.0]))

    assert safe
    assert debug["ground_plane_found"] is True
    assert debug["corridor_points"] >= 8
    assert debug["near_body_blocking_points"] == 0
    assert debug["blocking_points"] == 0



def test_near_goal_uses_farther_evidence_but_only_checks_endpoint_for_obstacles():
    from agentnav.abot.harness import PixelHarness

    depth = scaled_floor_depth(scale_error=1.6)
    harness = PixelHarness(
        ObservationDepthCache(ArrayDepth(depth)), CameraIntrinsics()
    )
    short_step = np.array([0.1, 0.0])
    safe, debug = harness.depth_corridor_is_safe(
        depth, short_step, max_lookahead_m=0.2
    )

    assert safe
    assert debug["corridor_lookahead_m"] == pytest.approx(0.2)
    assert debug["evidence_lookahead_m"] == pytest.approx(0.7)
    assert debug["corridor_points"] < 8
    assert debug["evidence_points"] >= 8

    obstacle = depth.copy()
    obstacle[400:412, 356:364] = 0.15 / 1.6
    blocked, blocked_debug = harness.depth_corridor_is_safe(
        obstacle, short_step, max_lookahead_m=0.2
    )
    assert not blocked
    assert blocked_debug["reason"] == "depth_obstacle"


def test_lookahead_supplies_evidence_but_truly_sparse_corridor_is_not_clear():
    from agentnav.abot.harness import PixelHarness

    depth = scaled_floor_depth(scale_error=1.6)
    lookahead = PixelHarness(
        ObservationDepthCache(ArrayDepth(depth)), CameraIntrinsics()
    )
    short_sight = PixelHarness(
        ObservationDepthCache(ArrayDepth(depth)),
        CameraIntrinsics(),
        depth_corridor_lookahead_m=0.35,
    )

    lookahead_safe, lookahead_debug = lookahead.depth_corridor_is_safe(
        depth, np.array([0.35, 0.0])
    )
    sparse_safe, sparse_debug = short_sight.depth_corridor_is_safe(
        depth, np.array([0.35, 0.0])
    )

    assert lookahead_safe
    assert lookahead_debug["corridor_lookahead_m"] == pytest.approx(0.7)
    assert lookahead_debug["corridor_evidence_sparse"] is False
    assert not sparse_safe
    assert sparse_debug["corridor_evidence_sparse"] is True
    assert sparse_debug["reason"] == "insufficient_corridor_evidence"
    assert sparse_debug["obstacle_risk"] == 1.0


def tilted_scaled_floor_depth(
    scale_error=1.7,
    slope_x=0.10,
    slope_z=0.08,
):
    camera = CameraIntrinsics()
    v, u = np.indices((camera.height, camera.width), dtype=np.float64)
    ray_x = (u - camera.cx) / camera.fx
    ray_y = (v - camera.cy) / camera.fy
    normalizer = np.sqrt(slope_x**2 + slope_z**2 + 1.0)
    intercept = camera.extrinsic_height * normalizer
    denominator = ray_y - slope_x * ray_x - slope_z
    true_depth = np.zeros_like(denominator)
    visible = denominator > 0.03
    true_depth[visible] = intercept / denominator[visible]
    return (true_depth / scale_error).astype(np.float32)


def test_3d_ground_plane_handles_tilt_and_keeps_floor_out_of_obstacles():
    from agentnav.abot.harness import PixelHarness

    depth = tilted_scaled_floor_depth()
    harness = PixelHarness(
        ObservationDepthCache(ArrayDepth(depth)), CameraIntrinsics()
    )

    safe, debug = harness.depth_corridor_is_safe(
        depth, np.array([0.5, 0.0])
    )

    assert safe
    assert debug["ground_model"] == "robust_3d_plane"
    assert debug["ground_plane_found"] is True
    assert debug["depth_scale_correction"] == pytest.approx(1.7, rel=0.03)
    assert debug["ground_tilt_deg"] > 5.0
    assert debug["ground_column_coverage_ratio"] > 0.9
    assert debug["ground_row_span_ratio"] > 0.8
    assert debug["ground_bottom_inlier_points"] > 100
    assert debug["blocking_points"] == 0


def test_out_of_bounds_pixels_are_rejected_before_depth_query():
    from agentnav.abot.harness import PixelHarness

    harness = PixelHarness(ObservationDepthCache(FakeDepth()), CameraIntrinsics())
    proposals = harness.normalize_proposals(
        [
            {"u": 360, "v": 120, "reason": "valid"},
            {"u": -1, "v": 120, "reason": "left of image"},
            {"u": 360, "v": 640, "reason": "below image"},
        ]
    )
    assert [(item.u, item.v) for item in proposals] == [(360, 120)]


def test_invalid_depth_is_rejected_and_far_stable_depth_uses_bearing_fallback():
    from agentnav.abot.harness import PixelHarness

    invalid = PixelHarness(
        ObservationDepthCache(FakeDepth(np.nan)), CameraIntrinsics()
    ).query_candidates(observation(), [PixelProposal("invalid", 360, 80)])[0]
    saturated = PixelHarness(
        ObservationDepthCache(FakeDepth(48.5)),
        CameraIntrinsics(),
        max_reliable_depth_m=30.0,
    ).query_candidates(observation(), [PixelProposal("far", 360, 500)])[0]

    assert not invalid.depth_reliable and not invalid.reachable
    assert invalid.safety_debug["reason"] == "invalid_depth"
    assert saturated.depth_reliable and saturated.reachable
    assert saturated.safety_debug["reason"] == "far_depth_bearing_fallback"
    assert saturated.safety_debug["projection_depth_m"] == 12.0


def test_failed_pixel_region_is_rejected_within_the_same_frame():
    from agentnav.abot.harness import PixelHarness

    harness = PixelHarness(
        ObservationDepthCache(FakeDepth(np.nan)),
        CameraIntrinsics(),
        failed_pixel_radius=24,
    )
    first = harness.query_candidates(
        observation(), [PixelProposal("first", 360, 80)]
    )[0]
    repeated_region = harness.query_candidates(
        observation(), [PixelProposal("nearby", 370, 90)]
    )[0]

    assert first.safety_debug["reason"] == "invalid_depth"
    assert repeated_region.safety_debug["reason"] == "already_failed_pixel_region"
    assert not repeated_region.reachable


def test_same_numeric_pixel_is_remeasured_in_a_new_observation():
    from agentnav.abot.harness import PixelHarness

    harness = PixelHarness(
        ObservationDepthCache(FakeDepth(2.0)), CameraIntrinsics()
    )
    first = harness.query_candidates(
        observation(step=0), [PixelProposal("first", 250, 200)]
    )[0]
    remeasured = harness.query_candidates(
        observation(step=1), [PixelProposal("fresh-frame", 250, 200)]
    )[0]

    assert first.reachable
    assert remeasured.reachable
    assert remeasured.safety_debug["reason"] == "route_check_deferred_to_executor"


def test_memory_compacts_turn_audit_before_sending_it_to_vlm():
    memory = EpisodeMemory()
    memory.task_history.append(
        {
            "task_type": "turn",
            "task_status": "GOAL_REACHED",
            "failure_reason": "",
            "navigation_anchor": "turn_left_45_deg",
            "is_midpoint_task": False,
            "frame_before": {"sha256": "a" * 64, "shape": [640, 720, 3]},
            "frame_after": {"sha256": "b" * 64, "shape": [640, 720, 3]},
            "turn_requested_angle_deg": 45.0,
            "heading_before_deg": 0.0,
            "heading_after_deg": 45.0,
        }
    )

    recent = memory.high_level_summary()["recent_tasks"]

    assert recent == [
        {
            "task_type": "turn",
            "status": "GOAL_REACHED",
            "failure_reason": "",
            "navigation_anchor": "turn_left_45_deg",
            "is_midpoint_task": False,
        }
    ]
    assert "frame_before" not in repr(recent)
    assert "sha256" not in repr(recent)

def test_memory_does_not_expose_stale_pixel_coordinates_to_vlm():
    memory = EpisodeMemory()
    memory.selected_pixels.append(
        {
            "step": 3,
            "u": 250,
            "v": 200,
            "semantic_anchor": "shop sign",
            "navigation_anchor": "entrance",
        }
    )

    summary = memory.high_level_summary()

    assert "previous_selected_pixels" not in summary
    assert summary["previous_navigation_anchors"] == [
        {
            "source_frame_id": 3,
            "semantic_anchor": "shop sign",
            "navigation_anchor": "entrance",
        }
    ]
    assert '"u"' not in repr(summary)
    assert '"v"' not in repr(summary)


def test_unconfirmed_goal_is_temporarily_blocked_and_expires():
    memory = EpisodeMemory()
    memory.begin_planning_cycle()
    executor = ABotS1Executor(AlwaysSafeHarness(), memory)

    reached = executor.create_navigation_task(
        observation(step=0),
        measurement(2.0),
        "red storefront",
        "front approach",
    )
    reached.status = TaskStatus.GOAL_REACHED
    reached.phase = "arrived"
    memory.record_unconfirmed_goal(
        reached,
        "POI not visible after arrival",
        verification_step=4,
        ttl_planning_cycles=3,
    )
    executor.finish_current_task()

    memory.begin_planning_cycle()
    repeated = executor.create_navigation_task(
        observation(step=5),
        measurement(2.0),
        "red storefront",
        "same approach",
    )
    assert repeated.status is TaskStatus.INVALID_GOAL
    assert repeated.failure_reason == "goal_matches_recent_unconfirmed_region"
    executor.finish_current_task()
    assert memory.failed_goals == []

    summary = memory.high_level_summary()
    verification = summary["recent_verification_failures"][0]
    assert verification["reason"] == "POI not visible after arrival"
    assert verification["semantic_anchor"] == "red storefront"
    assert "source_pixel" not in verification
    assert verification["remaining_planning_cycles"] == 3
    assert "world_goal" not in repr(summary)

    memory.begin_planning_cycle()
    memory.begin_planning_cycle()
    memory.begin_planning_cycle()
    allowed = executor.create_navigation_task(
        observation(step=8),
        measurement(2.0),
        "red storefront",
        "revisited after TTL",
    )
    assert allowed.status is TaskStatus.RUNNING


def test_corridor_reuses_latest_reliable_ground_scale_when_floor_fit_drops():
    from agentnav.abot.harness import PixelHarness

    harness = PixelHarness(
        ObservationDepthCache(FakeDepth(2.0)), CameraIntrinsics()
    )
    harness._last_reliable_ground_scale = 1.0
    harness._estimate_ground_plane = lambda depth: (
        None,
        {
            "ground_model": "robust_3d_plane",
            "ground_plane_found": False,
        },
    )

    _, debug = harness.depth_corridor_is_safe(
        np.full((640, 720), 2.0, dtype=np.float32),
        np.array([0.2, 0.0]),
    )

    assert debug["ground_model"] == "cached_scale_horizontal_plane"
    assert debug["ground_scale_reused"] is True
    assert debug["depth_scale_correction"] == 1.0
