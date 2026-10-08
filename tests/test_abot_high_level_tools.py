from types import SimpleNamespace

import asyncio
import pytest

import numpy as np
from PIL import Image

from agentnav.abot.high_level import NanobotPoiPlanner
from agentnav.abot.executor import ABotS1Executor
from agentnav.abot.observation import to_agent_safe_observation
from agentnav.abot.types import EpisodeMemory, NavMode, PixelMeasurement, PixelProposal, TaskStatus
from nanobot.providers.base import LLMResponse, ToolCallRequest


class ScriptedProvider:
    def __init__(self):
        self.index = 0
        self.last_tools = []
        self.last_messages = []
        self.calls = [
            ("QUERY_DEPTH", {"points": [{"view": "front", "u": 360, "v": 120, "reason": "store sign"}]}),
            ("SET_NAVIGATION_GOAL", {"candidate_id": "P0", "semantic_anchor": "library sign", "navigation_anchor": "approach"}),
        ]

    async def chat_with_retry(self, **kwargs):
        self.last_tools = kwargs["tools"]
        self.last_messages = kwargs["messages"]
        name, arguments = self.calls[self.index]
        self.index += 1
        return LLMResponse(
            content="",
            tool_calls=[ToolCallRequest(str(self.index), name, arguments)],
            finish_reason="stop",
            usage={},
        )


class Harness:
    camera = SimpleNamespace(width=720, height=640, cy=320.0)

    def normalize_proposals(self, points):
        return [PixelProposal("P0", points[0]["u"], points[0]["v"], "ground")]

    def query_candidates(self, observation, proposals):
        return [PixelMeasurement(proposals[0], 2.0, 0.01, 1.0, np.array([1.6, 0.0]), True, True, {"reason": "clear"})]


class Task:
    status = TaskStatus.RUNNING

    def public_dict(self):
        return {"task_id": "test", "task_status": "RUNNING"}


class Executor:
    def __init__(self):
        self.created = 0

    def create_navigation_task(self, *args, **kwargs):
        self.created += 1
        return Task()

    def create_turn_task(self, *args):
        return Task()


class UnreachableHarness(Harness):
    def query_candidates(self, observation, proposals):
        return [
            PixelMeasurement(
                proposals[0],
                2.0,
                0.01,
                1.0,
                np.zeros(2),
                True,
                False,
                {"reason": "depth_obstacle"},
            )
        ]


class SilentProvider:
    def __init__(self):
        self.calls = 0

    async def chat_with_retry(self, **kwargs):
        self.calls += 1
        return LLMResponse(
            content="",
            tool_calls=[],
            finish_reason="stop",
            usage={},
        )


class VerificationProvider:
    def __init__(self, confirmed):
        self.confirmed = confirmed
        self.tool_names = set()

    async def chat_with_retry(self, **kwargs):
        self.tool_names = {
            tool["function"]["name"] for tool in kwargs["tools"]
        }
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    "verify-1",
                    "VERIFY_POI",
                    {
                        "confirmed": self.confirmed,
                        "reason": (
                            "named POI is visible"
                            if self.confirmed
                            else "named POI is not visible"
                        ),
                        **(
                            {
                                "approach_u": 410,
                                "approach_v": 380,
                                "approach_reason": "fresh entrance approach",
                            }
                            if self.confirmed
                            else {}
                        ),
                    },
                )
            ],
            finish_reason="stop",
            usage={},
        )


class UnreachableProvider:
    def __init__(self):
        self.index = 0
        self.tool_names = []
        self.calls = [
            ("QUERY_DEPTH", {"points": [{"view": "front", "u": 360, "v": 400, "reason": "ground"}]}),
            ("SCAN_360", {"reason": "target is not visible"}),
        ]

    async def chat_with_retry(self, **kwargs):
        self.tool_names.append({tool["function"]["name"] for tool in kwargs["tools"]})
        name, arguments = self.calls[self.index]
        self.index += 1
        return LLMResponse(
            content="",
            tool_calls=[ToolCallRequest(str(self.index), name, arguments)],
            finish_reason="stop",
            usage={},
        )


def test_set_goal_rejects_a_different_named_poi(tmp_path):
    provider = ScriptedProvider()
    provider.calls = [
        provider.calls[0],
        ("SET_NAVIGATION_GOAL", {
            "candidate_id": "P0", "semantic_anchor": "YESFASHION",
            "navigation_anchor": "YESFASHION entrance",
        }),
        ("SCAN_360", {"reason": "named target is not visible"}),
    ]
    planner = NanobotPoiPlanner(str(tmp_path), provider=provider, max_tool_iterations=10)
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="便利蜂", step_count=0, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(observation, NavMode.PLANNING, memory)
    executor = Executor()

    decision = planner.decide(safe, observation, Harness(), executor, memory)

    assert executor.created == 0
    assert decision.terminal["action"] == "SCAN_360"
    rejected = next(item for item in decision.tool_trace if item["tool"] == "SET_NAVIGATION_GOAL")
    assert rejected["result"]["accepted"] is False
    assert rejected["result"]["named_poi"] == "便利蜂"


def test_blocked_recovery_rechecks_current_view_before_scanning(tmp_path):
    provider = ScriptedProvider()
    provider.calls = [
        ("SCAN_360", {"reason": "blocked"}),
        ("QUERY_DEPTH", {"points": [{
            "view": "front", "u": 420, "v": 440,
            "reason": "fresh ground beside visible library entrance",
        }]}),
        ("SET_NAVIGATION_GOAL", {
            "candidate_id": "P0", "semantic_anchor": "library",
            "navigation_anchor": "new entrance approach",
        }),
    ]
    planner = NanobotPoiPlanner(str(tmp_path), provider=provider, max_tool_iterations=10)
    current = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library", step_count=5, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(
        current, NavMode.RECOVERY, memory, "BLOCKED",
    )

    decision = planner.decide(safe, current, Harness(), Executor(), memory)

    assert decision.terminal["action"] == "SET_NAVIGATION_GOAL"
    assert decision.tool_trace[0]["tool"] == "SCAN_360"
    assert decision.tool_trace[0]["result"]["accepted"] is False
    assert decision.tool_trace[1]["tool"] == "QUERY_DEPTH"


def test_failed_arrival_reopens_current_view_navigation_tools(tmp_path):
    provider = ScriptedProvider()
    planner = NanobotPoiPlanner(str(tmp_path), provider=provider, max_tool_iterations=10)
    current = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library", step_count=5, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(
        current, NavMode.RECOVERY, memory, "POI_NOT_CONFIRMED",
        {"completed": False, "semantic_relocalization_required": False},
    )

    decision = planner.decide(safe, current, Harness(), Executor(), memory)

    assert decision.terminal["action"] == "SET_NAVIGATION_GOAL"
    assert decision.tool_trace[0]["tool"] == "QUERY_DEPTH"


def test_set_goal_rejects_candidate_whose_reason_says_target_is_absent(tmp_path):
    class ReasonPreservingHarness(Harness):
        def normalize_proposals(self, points):
            point = points[0]
            return [
                PixelProposal("P0", point["u"], point["v"], point["reason"])
            ]

    provider = ScriptedProvider()
    provider.calls = [
        ("QUERY_DEPTH", {"points": [{
            "view": "front", "u": 360, "v": 400,
            "reason": "target POI is not visible; initiate 360-degree scan",
        }]}),
        ("SET_NAVIGATION_GOAL", {
            "candidate_id": "P0", "semantic_anchor": "library",
            "navigation_anchor": "entrance",
        }),
        ("SCAN_360", {"reason": "named target is not visible"}),
    ]
    planner = NanobotPoiPlanner(
        str(tmp_path), provider=provider, max_tool_iterations=10
    )
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library", step_count=0, rotation=np.eye(4),
    )
    safe = to_agent_safe_observation(
        observation, NavMode.PLANNING, EpisodeMemory()
    )
    executor = Executor()
    decision = planner.decide(
        safe, observation, ReasonPreservingHarness(), executor, EpisodeMemory()
    )
    assert executor.created == 0
    assert decision.terminal["action"] == "SCAN_360"
    rejected = next(
        x for x in decision.tool_trace if x["tool"] == "SET_NAVIGATION_GOAL"
    )
    assert rejected["result"]["accepted"] is False


def test_target_name_allows_mixed_language_alias():
    assert NanobotPoiPlanner._anchor_matches_target(
        "PARISBAGUE TTE巴黎贝甜", "PARIS BAGUETTE sign"
    )
    assert not NanobotPoiPlanner._anchor_matches_target("豫石记羊汤", "望城烟")


def test_recovery_high_sign_query_includes_lower_current_frame_options(tmp_path):
    class CaptureHarness(Harness):
        def normalize_proposals(self, points):
            self.sensor_points = points
            return super().normalize_proposals(points)

    provider = ScriptedProvider()
    harness = CaptureHarness()
    planner = NanobotPoiPlanner(str(tmp_path), provider=provider, max_tool_iterations=10)
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library", step_count=4, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(
        observation, NavMode.RECOVERY, memory, transition_reason="BLOCKED"
    )

    planner.decide(safe, observation, harness, Executor(), memory)

    assert len(harness.sensor_points) == 5
    assert harness.sensor_points[0]["v"] == 60
    assert {point["v"] for point in harness.sensor_points[1:]} == {397, 435, 474}
    assert all(point["reason"].startswith("fresh current-frame")
               for point in harness.sensor_points[1:])


def test_initial_high_sign_query_includes_fresh_ground_options(tmp_path):
    class CaptureHarness(Harness):
        def normalize_proposals(self, points):
            self.sensor_points = points
            return super().normalize_proposals(points)

    harness = CaptureHarness()
    planner = NanobotPoiPlanner(
        str(tmp_path), provider=ScriptedProvider(), max_tool_iterations=10
    )
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library", step_count=0, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(
        observation, NavMode.PLANNING, memory, transition_reason="episode_start"
    )
    planner.decide(safe, observation, harness, Executor(), memory)
    assert len(harness.sensor_points) == 5
    assert {point["v"] for point in harness.sensor_points[1:]} == {397, 435, 474}


def test_set_goal_appears_only_after_current_depth_query(tmp_path):
    provider = ScriptedProvider()
    planner = NanobotPoiPlanner(str(tmp_path), provider=provider, max_tool_iterations=10)
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library",
        step_count=0,
        rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(observation, NavMode.PLANNING, memory)
    executor = Executor()
    decision = planner.decide(safe, observation, Harness(), executor, memory)
    assert [item["tool"] for item in decision.tool_trace] == [
        "QUERY_DEPTH",
        "SET_NAVIGATION_GOAL",
    ]
    assert executor.created == 1
    assert decision.terminal["action"] == "SET_NAVIGATION_GOAL"
    planning_tools = {
        tool["function"]["name"] for tool in provider.last_tools
    }
    assert "VERIFY_POI" not in planning_tools
    assert "TERMINATE" not in planning_tools
    assert "OBSERVE" not in planning_tools
    assert "RECALL" not in planning_tools
    assert "REMEMBER" not in planning_tools
    system_prompt = provider.last_messages[0]["content"]
    assert "high-level semantic planner" in system_prompt
    assert "# nanobot" not in system_prompt
    assert len(system_prompt) < 500


def test_source_named_skills_are_loaded_into_planning_prompt(tmp_path):
    for name in ("navigate", "locate", "explore"):
        skill_dir = tmp_path / "skills" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\n---\n{name.upper()}_GUIDANCE\n",
            encoding="utf-8",
        )
    provider = ScriptedProvider()
    planner = NanobotPoiPlanner(str(tmp_path), provider=provider, max_tool_iterations=10)
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library", step_count=0, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(observation, NavMode.PLANNING, memory)

    planner.decide(safe, observation, Harness(), Executor(), memory)

    system_prompt = provider.last_messages[0]["content"]
    for name in ("navigate", "locate", "explore"):
        assert f"{name.upper()}_GUIDANCE" in system_prompt
    assert "# nanobot" not in system_prompt


def test_verifying_exposes_only_verification_and_returns_expected_state(tmp_path):
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library",
        step_count=4,
        rotation=np.eye(4),
    )

    for confirmed, expected_action in [
        (False, "RETURN_TO_PLANNING"),
        (True, "TERMINATE"),
    ]:
        provider = VerificationProvider(confirmed)
        planner = NanobotPoiPlanner(
            str(tmp_path / str(confirmed)),
            provider=provider,
            max_tool_iterations=2,
        )
        memory = EpisodeMemory()
        safe = to_agent_safe_observation(
            observation, NavMode.VERIFYING, memory, "GOAL_REACHED"
        )
        decision = planner.decide(
            safe, observation, Harness(), Executor(), memory
        )

        assert provider.tool_names == {"VERIFY_POI"}
        assert [item["tool"] for item in decision.tool_trace] == ["VERIFY_POI"]
        assert decision.terminal["action"] == expected_action
        assert decision.terminal["target_visible"] is confirmed
        assert (decision.continuation_measurement is not None) is confirmed
        if confirmed:
            assert decision.continuation_measurement.proposal.u == 205
            assert decision.terminal["continuation_candidate"]["reachable"] is True


def test_magnified_vlm_coordinates_map_back_to_sensor_pixels(tmp_path):
    provider = ScriptedProvider()
    planner = NanobotPoiPlanner(
        str(tmp_path), provider=provider, max_tool_iterations=10, image_scale=2.0
    )
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library", step_count=0, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(observation, NavMode.PLANNING, memory)
    decision = planner.decide(
        safe, observation, Harness(), Executor(), memory
    )

    query = decision.tool_trace[0]
    assert query["arguments"]["points"][0]["u"] == 360
    assert query["arguments"]["points"][0]["v"] == 120
    measurement = query["result"]["measurements"][0]
    assert measurement["u"] == 180
    assert measurement["v"] == 60
    query_tool = next(
        tool for tool in provider.last_tools
        if tool["function"]["name"] == "QUERY_DEPTH"
    )
    point_schema = query_tool["function"]["parameters"]["properties"]["points"]["items"]["properties"]
    assert point_schema["u"]["maximum"] == 1439
    assert point_schema["v"]["maximum"] == 1279


def test_verifying_iteration_limit_falls_back_to_planning(tmp_path):
    provider = SilentProvider()
    planner = NanobotPoiPlanner(
        str(tmp_path), provider=provider, max_tool_iterations=2
    )
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library",
        step_count=4,
        rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(
        observation, NavMode.VERIFYING, memory, "GOAL_REACHED"
    )

    decision = planner.decide(
        safe, observation, Harness(), Executor(), memory
    )

    assert provider.calls == 1
    assert decision.terminal["action"] == "RETURN_TO_PLANNING"
    assert decision.terminal["target_visible"] is False
    assert decision.tool_trace[-1]["tool"] == (
        "PYTHON_VERIFICATION_FALLBACK"
    )


def test_single_tool_loop_default_matches_agentnav(tmp_path):
    planner = NanobotPoiPlanner(str(tmp_path), provider=ScriptedProvider())
    assert planner.max_tool_iterations == 40


def test_fallback_requests_scan_without_creating_a_turn_task(tmp_path):
    planner = NanobotPoiPlanner(
        str(tmp_path), provider=SilentProvider(), max_tool_iterations=1
    )
    memory = EpisodeMemory()
    executor = ABotS1Executor(Harness(), memory)
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library", step_count=0, rotation=np.eye(4),
    )
    safe = to_agent_safe_observation(observation, NavMode.PLANNING, memory)
    decision = planner.decide(safe, observation, Harness(), executor, memory)
    assert decision.terminal["action"] == "SCAN_360"
    assert executor.active_task is None


def test_hanging_provider_is_cancelled_by_request_deadline(tmp_path):
    class HangingProvider:
        cancelled = False

        async def chat_with_retry(self, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled = True

    provider = HangingProvider()
    planner = NanobotPoiPlanner(
        str(tmp_path), provider=provider, request_timeout_s=0.01
    )
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library", step_count=0, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(observation, NavMode.PLANNING, memory)
    with pytest.raises(TimeoutError, match="VLM request exceeded"):
        planner.decide(safe, observation, Harness(), Executor(), memory)
    assert provider.cancelled


def test_unreachable_depth_candidate_cannot_be_submitted(tmp_path):
    provider = UnreachableProvider()
    planner = NanobotPoiPlanner(str(tmp_path), provider=provider, max_tool_iterations=4)
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library",
        step_count=0,
        rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(observation, NavMode.PLANNING, memory)
    executor = Executor()

    decision = planner.decide(
        safe, observation, UnreachableHarness(), executor, memory
    )

    assert "SET_NAVIGATION_GOAL" not in provider.tool_names[0]
    assert "SCAN_360" in provider.tool_names[0]
    assert "TURN" not in provider.tool_names[0]
    assert len(provider.tool_names) == 1
    assert executor.created == 0
    assert decision.terminal["action"] == "SCAN_360"
    assert decision.terminal["fallback_reason"] == (
        "remaining_tool_iterations_threshold"
    )
    assert [item["tool"] for item in decision.tool_trace] == [
        "QUERY_DEPTH",
        "PYTHON_PROGRESS_FALLBACK",
    ]


def test_near_iteration_limit_does_not_select_a_candidate_in_python(tmp_path):
    provider = ScriptedProvider()
    planner = NanobotPoiPlanner(
        str(tmp_path), provider=provider, max_tool_iterations=4
    )
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library",
        step_count=0,
        rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(
        observation, NavMode.PLANNING, memory
    )
    executor = Executor()

    decision = planner.decide(
        safe, observation, Harness(), executor, memory
    )

    assert provider.index == 1
    assert executor.created == 0
    assert decision.terminal["action"] == "SCAN_360"
    assert decision.terminal["reachable_candidate_ids"] == ["P0"]
    assert decision.terminal["fallback_reason"] == (
        "remaining_tool_iterations_threshold"
    )
    assert [item["tool"] for item in decision.tool_trace] == [
        "QUERY_DEPTH",
        "PYTHON_PROGRESS_FALLBACK",
    ]


def test_context_overflow_retries_with_server_derived_completion_budget(tmp_path):
    class ContextLimitedProvider:
        def __init__(self):
            self.max_tokens = []

        async def chat_with_retry(self, **kwargs):
            self.max_tokens.append(kwargs["max_tokens"])
            if len(self.max_tokens) == 1:
                return LLMResponse(
                    content=(
                        "Error: This model's maximum context length is 8192 tokens. "
                        "However, you requested 8656 tokens (7632 in the messages, "
                        "1024 in the completion)."
                    ),
                    finish_reason="error",
                    usage={},
                )
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        "scan-1", "SCAN_360", {"reason": "target absent"}
                    )
                ],
                finish_reason="stop",
                usage={},
            )

    provider = ContextLimitedProvider()
    planner = NanobotPoiPlanner(str(tmp_path), provider=provider)
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library", step_count=0, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(observation, NavMode.PLANNING, memory)

    decision = planner.decide(
        safe, observation, Harness(), Executor(), memory
    )

    assert provider.max_tokens == [1024, 496]
    assert decision.terminal["action"] == "SCAN_360"


class TwoQueryProvider:
    def __init__(self):
        self.index = 0

    async def chat_with_retry(self, **kwargs):
        points = [
            {"view": "front", "u": 300 + 80 * self.index, "v": 200, "reason": f"anchor {self.index}"}
        ]
        self.index += 1
        return LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    f"query-{self.index}",
                    "QUERY_DEPTH",
                    {"points": points},
                )
            ],
            finish_reason="stop",
            usage={},
        )


class SequencedHarness(Harness):
    def __init__(self):
        self.calls = 0

    def query_candidates(self, observation, proposals):
        self.calls += 1
        if self.calls == 1:
            return [
                PixelMeasurement(
                    proposals[0],
                    4.0,
                    0.4,
                    0.75,
                    np.array([3.0, 0.0]),
                    True,
                    True,
                    {"reason": "clear"},
                )
            ]
        return [
            PixelMeasurement(
                proposals[0],
                2.0,
                0.01,
                1.0,
                np.array([1.6, 0.0]),
                True,
                True,
                {"reason": "clear"},
            )
        ]


def test_candidates_persist_without_python_selecting_between_queries(tmp_path):
    provider = TwoQueryProvider()
    planner = NanobotPoiPlanner(
        str(tmp_path), provider=provider, max_tool_iterations=5
    )
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library",
        step_count=2,
        rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(
        observation, NavMode.PLANNING, memory
    )
    executor = Executor()

    decision = planner.decide(
        safe, observation, SequencedHarness(), executor, memory
    )

    assert provider.index == 2
    assert executor.created == 0
    assert decision.terminal["action"] == "SCAN_360"
    assert decision.terminal["reachable_candidate_ids"] == ["P0", "P1"]
    query_results = [
        item["result"]["measurements"][0]
        for item in decision.tool_trace
        if item["tool"] == "QUERY_DEPTH"
    ]
    assert [item["candidate_id"] for item in query_results] == ["P0", "P1"]
    assert decision.tool_trace[-1]["tool"] == "PYTHON_PROGRESS_FALLBACK"


class ExhaustedHarness(Harness):
    def query_budget_exhausted(self, frame_id):
        return True


class ExhaustedUnreachableHarness(UnreachableHarness):
    def query_budget_exhausted(self, frame_id):
        return True


def test_depth_budget_exhaustion_requires_vlm_candidate_selection(tmp_path):
    provider = ScriptedProvider()
    planner = NanobotPoiPlanner(
        str(tmp_path), provider=provider, max_tool_iterations=40
    )
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library",
        step_count=7,
        rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(
        observation, NavMode.PLANNING, memory
    )
    executor = Executor()

    decision = planner.decide(
        safe, observation, ExhaustedHarness(), executor, memory
    )

    assert provider.index == 2
    assert executor.created == 1
    assert decision.terminal["action"] == "SET_NAVIGATION_GOAL"
    assert "fallback_reason" not in decision.terminal
    assert "QUERY_DEPTH" not in {
        tool["function"]["name"] for tool in provider.last_tools
    }
    assert "SET_NAVIGATION_GOAL" in {
        tool["function"]["name"] for tool in provider.last_tools
    }
    assert [item["tool"] for item in decision.tool_trace] == [
        "QUERY_DEPTH",
        "PYTHON_DEPTH_BUDGET_CLOSED",
        "SET_NAVIGATION_GOAL",
    ]


def test_depth_query_schema_accepts_pixels_across_the_full_image(tmp_path):
    provider = ScriptedProvider()
    planner = NanobotPoiPlanner(str(tmp_path), provider=provider, max_tool_iterations=4)
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library",
        step_count=0,
        rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(observation, NavMode.PLANNING, memory)

    planner.decide(safe, observation, Harness(), Executor(), memory)

    query_tool = next(
        tool
        for tool in provider.last_tools
        if tool["function"]["name"] == "QUERY_DEPTH"
    )
    point_schema = query_tool["function"]["parameters"]["properties"]["points"]["items"]
    assert point_schema["properties"]["u"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 1439,
    }
    assert point_schema["properties"]["v"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 1279,
    }


def test_scan_state_is_present_in_nanobot_messages(tmp_path):
    planner = NanobotPoiPlanner(str(tmp_path), provider=ScriptedProvider())
    memory = EpisodeMemory()
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library",
        step_count=6,
        rotation=np.eye(4),
    )
    safe = to_agent_safe_observation(
        observation,
        NavMode.PLANNING,
        memory,
        scan_state={
            "active": True,
            "completed": False,
            "direction": "left",
            "increment_deg": 45.0,
            "accumulated_deg": 90.0,
            "views_checked": 3,
        },
    )
    prompt = planner._decision_prompt(safe)
    messages = planner.context.build_messages([], prompt)
    user_context = messages[-1]["content"]

    assert '"direction": "left"' in user_context
    assert '"increment_deg": 45.0' in user_context
    assert '"accumulated_deg": 90.0' in user_context
    assert '"views_checked": 3' in user_context
    assert "Do not choose a turn direction or angle" in user_context


def test_request_cycle_retries_truncated_tool_json_before_any_tool_runs(tmp_path):
    class RecoveringProvider:
        def __init__(self):
            self.calls = 0

        async def chat_with_retry(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    content="Error: Error code: 400 - Invalid JSON: EOF while parsing a string",
                    finish_reason="error",
                    usage={},
                )
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        "scan-after-retry",
                        "SCAN_360",
                        {"reason": "target absent in current image"},
                    )
                ],
                finish_reason="stop",
                usage={},
            )

    provider = RecoveringProvider()
    planner = NanobotPoiPlanner(
        str(tmp_path),
        provider=provider,
        max_tool_iterations=4,
        request_recovery_attempts=2,
    )
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library",
        step_count=0,
        rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(observation, NavMode.PLANNING, memory)

    decision = planner.decide(
        safe, observation, Harness(), Executor(), memory
    )

    assert provider.calls == 2
    assert planner.call_count == 2
    assert decision.terminal["action"] == "SCAN_360"
    assert [item["tool"] for item in decision.tool_trace] == ["SCAN_360"]


def test_exhausted_truncated_json_falls_back_without_system_error(tmp_path):
    class AlwaysMalformedProvider:
        def __init__(self):
            self.calls = 0

        async def chat_with_retry(self, **kwargs):
            self.calls += 1
            return LLMResponse(
                content="Error: Error code: 400 - Invalid JSON: EOF while parsing a string",
                finish_reason="error",
                usage={},
            )

    provider = AlwaysMalformedProvider()
    planner = NanobotPoiPlanner(
        str(tmp_path),
        provider=provider,
        max_tool_iterations=4,
        request_recovery_attempts=2,
    )
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library",
        step_count=0,
        rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(observation, NavMode.PLANNING, memory)

    decision = planner.decide(
        safe, observation, Harness(), Executor(), memory
    )

    assert provider.calls == 2
    assert decision.terminal["action"] == "SCAN_360"
    assert decision.terminal["fallback_reason"] == "recoverable_vlm_format_error"
    assert decision.tool_trace[-1]["tool"] == "PYTHON_PROVIDER_ERROR_FALLBACK"


class CompletedScanProvider:
    def __init__(self, second_action="SET_EXPLORATION_GOAL"):
        self.index = 0
        self.tool_names = []
        self.calls = [
            (
                "QUERY_DEPTH",
                {
                    "u": 800,
                    "v": 900,
                    "reason": "open route toward storefronts",
                },
            ),
            (
                second_action,
                (
                    {
                        "candidate_id": "P0",
                        "semantic_anchor": "visible storefront cluster",
                        "navigation_anchor": "open route toward storefronts",
                    }
                    if second_action == "SET_EXPLORATION_GOAL"
                    else {"reason": "bounded checks found no reachable route"}
                ),
            ),
        ]

    async def chat_with_retry(self, **kwargs):
        self.tool_names.append(
            {tool["function"]["name"] for tool in kwargs["tools"]}
        )
        return LLMResponse(
            content=(
                '{"candidates":[{"u":800,"v":900,'
                '"reason":"open route toward storefronts"}]}'
            ),
            finish_reason="stop",
            usage={},
        )


def completed_scan_safe(observation, memory):
    return to_agent_safe_observation(
        observation,
        NavMode.PLANNING,
        memory,
        scan_state={
            "active": False,
            "completed": True,
            "direction": "left",
            "increment_deg": 45.0,
            "accumulated_deg": 360.0,
            "views_checked": 9,
        },
    )


def test_completed_scan_requires_safe_exploration_before_search_exhausted(tmp_path):
    provider = CompletedScanProvider()
    planner = NanobotPoiPlanner(
        str(tmp_path), provider=provider, max_tool_iterations=10
    )
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library",
        step_count=9,
        rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    executor = Executor()

    decision = planner.decide(
        completed_scan_safe(observation, memory),
        observation,
        Harness(),
        executor,
        memory,
    )

    assert provider.tool_names == [set()]
    assert decision.terminal["action"] == "SET_EXPLORATION_GOAL"
    assert executor.created == 1
    assert [item["tool"] for item in decision.tool_trace] == [
        "QUERY_DEPTH",
        "SET_EXPLORATION_GOAL",
    ]


def test_completed_scan_loads_locate_and_explore_skills(tmp_path):
    class RecordingProvider(CompletedScanProvider):
        async def chat_with_retry(self, **kwargs):
            self.system_prompt = kwargs["messages"][0]["content"]
            return await super().chat_with_retry(**kwargs)

    for name in ("navigate", "locate", "explore"):
        skill_dir = tmp_path / "skills" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\n---\n{name.upper()}_GUIDANCE\n",
            encoding="utf-8",
        )
    provider = RecordingProvider()
    planner = NanobotPoiPlanner(str(tmp_path), provider=provider, max_tool_iterations=10)
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library", step_count=9, rotation=np.eye(4),
    )
    memory = EpisodeMemory()

    planner.decide(
        completed_scan_safe(observation, memory), observation,
        Harness(), Executor(), memory,
    )

    assert "LOCATE_GUIDANCE" in provider.system_prompt
    assert "EXPLORE_GUIDANCE" in provider.system_prompt
    assert "NAVIGATE_GUIDANCE" not in provider.system_prompt


def test_completed_scan_exploration_prefers_clear_future_corridor(tmp_path):
    class TwoRouteProvider:
        async def chat_with_retry(self, **kwargs):
            return LLMResponse(
                content=(
                    '{"target_visible":false,"candidates":['
                    '{"u":600,"v":900,"reason":"left open ground"},'
                    '{"u":800,"v":900,"reason":"right open ground"}]}'
                ),
                finish_reason="stop", usage={},
            )

    class TwoRouteHarness(Harness):
        def normalize_proposals(self, points):
            return [
                PixelProposal(f"P{i}", point["u"], point["v"], point["reason"])
                for i, point in enumerate(points)
            ]

        def query_candidates(self, observation, proposals):
            return [
                PixelMeasurement(
                    proposal, 2.0, 0.01, 1.0,
                    np.array([2.0, -0.3 if i == 0 else 0.3]),
                    True, True, {"reason": "route_check_deferred_to_executor"},
                )
                for i, proposal in enumerate(proposals)
            ]

        def current_depth(self, observation):
            return np.ones((16, 16), dtype=np.float32)

        def depth_corridor_is_safe(self, depth, waypoint, max_lookahead_m=None):
            if waypoint[1] < 0:
                return False, {"reason": "depth_obstacle", "nearest_obstacle_m": 0.8}
            return True, {"reason": "clear_depth_corridor"}

    class CapturingExecutor(Executor):
        def create_navigation_task(self, observation, measurement, *args, **kwargs):
            self.selected_pixel = (measurement.proposal.u, measurement.proposal.v)
            return super().create_navigation_task(
                observation, measurement, *args, **kwargs
            )

    planner = NanobotPoiPlanner(str(tmp_path), provider=TwoRouteProvider())
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library", step_count=9, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    executor = CapturingExecutor()
    decision = planner.decide(
        completed_scan_safe(observation, memory), observation,
        TwoRouteHarness(), executor, memory,
    )

    assert decision.terminal["action"] == "SET_EXPLORATION_GOAL"
    assert executor.selected_pixel == (400, 450)
    assert sum(
        item["tool"] == "EXPLORATION_ROUTE_PREVIEW"
        for item in decision.tool_trace
    ) == 2


def test_completed_scan_rejects_no_motion_route(tmp_path):
    class NearHarness(Harness):
        def query_candidates(self, observation, proposals):
            return [PixelMeasurement(
                proposal, 1.0, 0.01, 1.0, np.array([0.14, 0.0]),
                True, True, {"reason": "clear"},
            ) for proposal in proposals]

    planner = NanobotPoiPlanner(
        str(tmp_path), provider=CompletedScanProvider(), max_tool_iterations=10
    )
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library", step_count=9, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    executor = Executor()
    decision = planner.decide(
        completed_scan_safe(observation, memory),
        observation, NearHarness(), executor, memory,
    )
    assert decision.terminal["action"] == "SEARCH_EXHAUSTED"
    assert executor.created == 0
    assert all(
        item["safety_debug"]["reason"] == "goal_already_within_local_tolerance"
        for item in decision.tool_trace[0]["result"]["measurements"]
    )


def test_completed_scan_fallback_uses_fresh_farther_ground_pixels(tmp_path):
    class EmptyRouteProvider:
        async def chat_with_retry(self, **kwargs):
            return LLMResponse(content='{"target_visible":false,"candidates":[]}',
                               finish_reason="stop", usage={})

    planner = NanobotPoiPlanner(str(tmp_path), provider=EmptyRouteProvider())
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library", step_count=9, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    memory.selected_pixels.append({"step": 9, "u": 360, "v": 384})
    decision = planner.decide(
        completed_scan_safe(observation, memory), observation, Harness(), Executor(), memory
    )

    points = decision.tool_trace[0]["arguments"]["points"]
    assert decision.terminal["action"] == "SET_EXPLORATION_GOAL"
    assert len(points) == 8
    assert len({(point["u"], point["v"]) for point in points}) == 8
    assert sum(point["v"] == min(item["v"] for item in points) for point in points) == 2
    assert len({point["v"] for point in points}) >= 2


def test_completed_scan_rechecks_visible_poi_before_exploration(tmp_path):
    class VisibleTargetProvider:
        async def chat_with_retry(self, **kwargs):
            return LLMResponse(
                content=(
                    '{"target_visible":true,"candidates":['
                    '{"u":800,"v":900,"reason":"library entrance"}]}'
                ),
                finish_reason="stop",
                usage={},
            )

    planner = NanobotPoiPlanner(
        str(tmp_path), provider=VisibleTargetProvider(), max_tool_iterations=10
    )
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library",
        step_count=9,
        rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    executor = Executor()

    decision = planner.decide(
        completed_scan_safe(observation, memory),
        observation,
        Harness(),
        executor,
        memory,
    )

    assert decision.terminal["action"] == "SET_NAVIGATION_GOAL"
    assert decision.terminal["target_visible"] is True
    assert executor.created == 1


def test_completed_scan_accepts_fresh_pixel_at_old_frame_coordinates(tmp_path):
    class VisibleTargetProvider:
        async def chat_with_retry(self, **kwargs):
            return LLMResponse(
                content=(
                    '{"target_visible":true,"candidates":['
                    '{"u":800,"v":900,"reason":"library entrance"}]}'
                ),
                finish_reason="stop",
                usage={},
            )

    planner = NanobotPoiPlanner(str(tmp_path), provider=VisibleTargetProvider())
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library", step_count=9, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    # This coordinate belonged to an earlier image, not the current frame.
    memory.selected_pixels.append({"step": 8, "u": 400, "v": 450})
    decision = planner.decide(
        completed_scan_safe(observation, memory), observation,
        Harness(), Executor(), memory,
    )

    assert decision.terminal["action"] == "SET_NAVIGATION_GOAL"
    assert decision.tool_trace[-2]["arguments"]["points"][0]["u"] == 800
    assert decision.tool_trace[-2]["arguments"]["points"][0]["v"] == 900


def test_completed_scan_recheck_can_confirm_target_after_missing_name(tmp_path):
    class RecheckProvider:
        def __init__(self):
            self.calls = 0

        async def chat_with_retry(self, **kwargs):
            self.calls += 1
            reason = "store entrance" if self.calls == 1 else "library entrance"
            return LLMResponse(
                content=(
                    '{"target_visible":true,"candidates":['
                    f'{{"u":800,"v":900,"reason":"{reason}"}}]}}'
                ),
                finish_reason="stop",
                usage={},
            )

    provider = RecheckProvider()
    planner = NanobotPoiPlanner(str(tmp_path), provider=provider)
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library", step_count=9, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    decision = planner.decide(
        completed_scan_safe(observation, memory), observation,
        Harness(), Executor(), memory,
    )

    assert provider.calls == 2
    assert decision.terminal["action"] == "SET_NAVIGATION_GOAL"
    assert any(
        item["tool"] == "POST_SCAN_UNGROUNDED_TARGET"
        for item in decision.tool_trace
    )


def test_search_exhausted_appears_only_after_exploration_checks_fail(tmp_path):
    provider = CompletedScanProvider(second_action="SEARCH_EXHAUSTED")
    planner = NanobotPoiPlanner(
        str(tmp_path), provider=provider, max_tool_iterations=10
    )
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library",
        step_count=9,
        rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    executor = Executor()

    decision = planner.decide(
        completed_scan_safe(observation, memory),
        observation,
        ExhaustedUnreachableHarness(),
        executor,
        memory,
    )

    assert provider.tool_names == [set()]
    assert decision.terminal["action"] == "SEARCH_EXHAUSTED"
    assert executor.created == 0


def test_semantic_relocalization_exposes_only_scan_tool(tmp_path):
    class ScanOnlyProvider:
        def __init__(self):
            self.tool_names = set()

        async def chat_with_retry(self, **kwargs):
            self.tool_names = {
                tool["function"]["name"] for tool in kwargs["tools"]
            }
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        "scan", "SCAN_360", {"reason": "fresh relocalization"}
                    )
                ],
                finish_reason="stop",
                usage={},
            )

    provider = ScanOnlyProvider()
    planner = NanobotPoiPlanner(str(tmp_path), provider=provider)
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library",
        step_count=5,
        rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(
        observation,
        NavMode.RECOVERY,
        memory,
        "POI_NOT_CONFIRMED",
        {"completed": False, "semantic_relocalization_required": True},
    )

    decision = planner.decide(safe, observation, Harness(), Executor(), memory)

    assert provider.tool_names == {"SCAN_360"}
    assert decision.terminal["action"] == "SCAN_360"


def test_upper_sign_ocr_rejects_only_readable_conflicting_chinese():
    assert NanobotPoiPlanner._upper_sign_text_mismatch("祥汇便利店", "新宿店")
    assert not NanobotPoiPlanner._upper_sign_text_mismatch("望京东店", "红")
    assert not NanobotPoiPlanner._upper_sign_text_mismatch("望京东店", "无")
    assert not NanobotPoiPlanner._upper_sign_text_mismatch("望京东店", "京东店")
    assert not NanobotPoiPlanner._upper_sign_text_mismatch("望京东店", "")
    assert NanobotPoiPlanner._upper_sign_text_mismatch("巴比手工鲜肉包", "潮知味")
    assert not NanobotPoiPlanner._upper_sign_text_mismatch("麦当劳", "麦当劳\n巴蜀小面")
    assert not NanobotPoiPlanner._upper_sign_text_mismatch("麦当劳", "McDonald's")
    assert not NanobotPoiPlanner._upper_sign_text_mismatch("PARISBAGUETTE巴黎贝甜", "巴黎贝甜")


@pytest.mark.parametrize(
    ("observed_sign", "expected_action"),
    [("贵州米粉", "SCAN_360"), ("布家班", "SET_NAVIGATION_GOAL")],
)
def test_mid_facade_pixel_is_checked_against_its_local_sign(
    tmp_path, observed_sign, expected_action
):
    class SignProvider(ScriptedProvider):
        async def chat_with_retry(self, **kwargs):
            if not kwargs["tools"]:
                return LLMResponse(
                    content=observed_sign, tool_calls=[], finish_reason="stop",
                    usage={},
                )
            return await super().chat_with_retry(**kwargs)

    provider = SignProvider()
    provider.calls = [
        ("QUERY_DEPTH", {"points": [{
            "view": "front", "u": 432, "v": 332,
            "reason": "claimed 布家班 storefront sign",
        }]}),
        ("SET_NAVIGATION_GOAL", {
            "candidate_id": "P0", "semantic_anchor": "布家班",
            "navigation_anchor": "entrance facade",
        }),
        ("SCAN_360", {"reason": "different sign at selected pixel"}),
        ("SCAN_360", {"reason": "target absent after recheck"}),
    ]
    planner = NanobotPoiPlanner(str(tmp_path), provider=provider, max_tool_iterations=10)
    current = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="布家班", step_count=0, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(
        current, NavMode.PLANNING, memory, transition_reason="episode_start"
    )
    executor = Executor()

    decision = planner.decide(safe, current, Harness(), executor, memory)

    assert decision.terminal["action"] == expected_action
    assert executor.created == (1 if observed_sign == "布家班" else 0)
    assert (tmp_path / "runtime_images/sign_0000_216_166.jpg").exists()
    with Image.open(tmp_path / "runtime_images/sign_0000_216_166.jpg") as crop:
        assert crop.size == (180, 240)


def test_mid_facade_sign_check_is_limited_to_initial_and_midpoint_views():
    measurement = PixelMeasurement(
        PixelProposal("P0", 216, 166, "storefront"), 2.0, 0.01, 1.0,
        np.array([1.6, 0.0]), True, True, {"reason": "clear"},
    )
    runtime = SimpleNamespace(
        harness=Harness(), safe=SimpleNamespace(
            poi_name="布家班", transition_reason="episode_start"
        ),
    )
    assert NanobotPoiPlanner._needs_upper_sign_check(runtime, measurement)
    runtime.safe.transition_reason = "MIDPOINT_REACHED"
    assert NanobotPoiPlanner._needs_upper_sign_check(runtime, measurement)
    runtime.safe.transition_reason = "SCAN_VIEW_READY"
    assert not NanobotPoiPlanner._needs_upper_sign_check(runtime, measurement)


def test_rejected_sign_pixel_does_not_consume_depth_budget_again(tmp_path):
    class CountingHarness(Harness):
        def __init__(self):
            self.queries = 0

        def query_candidates(self, observation, proposals):
            self.queries += 1
            return super().query_candidates(observation, proposals)

    class SignProvider(ScriptedProvider):
        async def chat_with_retry(self, **kwargs):
            if not kwargs["tools"]:
                return LLMResponse(
                    content="贵州米粉" if self.index < 3 else "布家班",
                    tool_calls=[], finish_reason="stop", usage={},
                )
            return await super().chat_with_retry(**kwargs)

    provider = SignProvider()
    provider.calls = [
        ("QUERY_DEPTH", {"points": [{"view": "front", "u": 432, "v": 332,
                                      "reason": "布家班 entrance"}]}),
        ("SET_NAVIGATION_GOAL", {"candidate_id": "P0", "semantic_anchor": "布家班",
                                 "navigation_anchor": "entrance"}),
        ("QUERY_DEPTH", {"points": [{"view": "front", "u": 432, "v": 332,
                                      "reason": "same point again"}]}),
        ("QUERY_DEPTH", {"points": [{"view": "front", "u": 500, "v": 332,
                                      "reason": "different storefront pixel"}]}),
        ("SET_NAVIGATION_GOAL", {"candidate_id": "P1", "semantic_anchor": "布家班",
                                 "navigation_anchor": "entrance"}),
    ]
    planner = NanobotPoiPlanner(str(tmp_path), provider=provider, max_tool_iterations=10)
    current = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="布家班", step_count=0, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(
        current, NavMode.PLANNING, memory, transition_reason="episode_start"
    )
    harness = CountingHarness()
    executor = Executor()

    decision = planner.decide(safe, current, harness, executor, memory)

    assert decision.terminal["action"] == "SET_NAVIGATION_GOAL"
    assert harness.queries == 2
    duplicate = [item for item in decision.tool_trace if item["tool"] == "QUERY_DEPTH"][1]
    assert duplicate["result"]["measurements"] == []
    assert duplicate["result"]["query_depth_budget_exhausted"] is False


def test_query_depth_rejects_goal_already_within_arrival_tolerance(tmp_path):
    class NearHarness(Harness):
        def query_candidates(self, observation, proposals):
            return [PixelMeasurement(
                proposals[0], 1.0, 0.01, 1.0, np.array([0.14, 0.0]),
                True, True, {"reason": "clear"},
            )]

    provider = ScriptedProvider()
    provider.calls = [
        ("QUERY_DEPTH", {"points": [{"view": "front", "u": 360, "v": 400, "reason": "ground"}]}),
        ("SCAN_360", {"reason": "no useful approach in current image"}),
    ]
    planner = NanobotPoiPlanner(str(tmp_path), provider=provider, max_tool_iterations=3)
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library", step_count=0, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(observation, NavMode.PLANNING, memory)
    executor = Executor()
    decision = planner.decide(safe, observation, NearHarness(), executor, memory)

    assert executor.created == 0
    assert decision.terminal["action"] == "SCAN_360"
    result = decision.tool_trace[0]["result"]["measurements"][0]
    assert result["reachable"] is False
    assert result["safety_debug"]["reason"] == "goal_already_within_local_tolerance"


def test_verify_does_not_offer_recently_blocked_continuation(tmp_path):
    class BlockedExecutor(Executor):
        def candidate_recently_blocked(self, observation, measurement):
            return True

    planner = NanobotPoiPlanner(
        str(tmp_path), provider=VerificationProvider(True), max_tool_iterations=2
    )
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library", step_count=4, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(observation, NavMode.VERIFYING, memory, "GOAL_REACHED")
    decision = planner.decide(safe, observation, Harness(), BlockedExecutor(), memory)

    assert decision.terminal["action"] == "TERMINATE"
    assert decision.continuation_measurement is None
    assert decision.terminal["continuation_candidate"] is None


def test_verify_prefers_current_ground_below_high_sign(tmp_path):
    class UpperSignProvider:
        async def chat_with_retry(self, **kwargs):
            return LLMResponse(
                content="",
                tool_calls=[ToolCallRequest("verify", "VERIFY_POI", {
                    "confirmed": True,
                    "reason": "named storefront sign visible",
                    "approach_u": 789,
                    "approach_v": 142,
                    "approach_reason": "high storefront sign",
                })],
                finish_reason="stop", usage={},
            )

    class MultiPointHarness(Harness):
        def normalize_proposals(self, points):
            return [PixelProposal(f"P{i}", point["u"], point["v"], point["reason"])
                    for i, point in enumerate(points)]

        def query_candidates(self, observation, proposals):
            return [PixelMeasurement(
                point, 19.0 if point.v < 320 else 4.0, 0.01, 1.0,
                np.array([19.0 if point.v < 320 else 4.0, 0.0]),
                True, True, {"reason": "clear"},
            ) for point in proposals]

    planner = NanobotPoiPlanner(str(tmp_path), provider=UpperSignProvider())
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="YESFASHION逸丝风尚", step_count=20, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(observation, NavMode.VERIFYING, memory, "GOAL_REACHED")
    decision = planner.decide(safe, observation, MultiPointHarness(), Executor(), memory)

    assert decision.terminal["action"] == "TERMINATE"
    assert decision.continuation_measurement is not None
    assert decision.continuation_measurement.proposal.v >= 0.6 * 640
    assert decision.continuation_measurement.depth_m == 4.0


class RejectFirstExecutor(Executor):
    def __init__(self):
        super().__init__()
        self.finished = 0

    def create_navigation_task(self, *args, **kwargs):
        self.created += 1
        rejected = self.created == 1
        return SimpleNamespace(
            status=TaskStatus.INVALID_GOAL if rejected else TaskStatus.RUNNING,
            failure_reason="goal_matches_failed_region" if rejected else "",
            public_dict=lambda: {
                "task_id": f"test-{self.created}",
                "task_status": "INVALID_GOAL" if rejected else "RUNNING",
            },
        )

    def finish_current_task(self):
        self.finished += 1


class TwoPointHarness(Harness):
    def normalize_proposals(self, points):
        return [PixelProposal(f"P{i}", item["u"], item["v"], item["reason"])
                for i, item in enumerate(points)]

    def query_candidates(self, observation, proposals):
        return [PixelMeasurement(
            point, 4.0, 0.01, 1.0, np.array([4.0 - i, 0.0]),
            True, True, {"reason": "clear"},
        ) for i, point in enumerate(proposals)]


def test_planner_retries_another_candidate_after_executor_rejects_goal(tmp_path):
    provider = ScriptedProvider()
    provider.calls = [
        ("QUERY_DEPTH", {"points": [
            {"view": "front", "u": 350, "v": 400, "reason": "library entrance"},
            {"view": "front", "u": 400, "v": 420, "reason": "library entrance ground"},
        ]}),
        ("SET_NAVIGATION_GOAL", {
            "candidate_id": "P0", "semantic_anchor": "library", "navigation_anchor": "entrance",
        }),
        ("SET_NAVIGATION_GOAL", {
            "candidate_id": "P1", "semantic_anchor": "library", "navigation_anchor": "ground",
        }),
    ]
    planner = NanobotPoiPlanner(str(tmp_path), provider=provider, max_tool_iterations=8)
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library", step_count=0, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    executor = RejectFirstExecutor()
    decision = planner.decide(
        to_agent_safe_observation(observation, NavMode.PLANNING, memory),
        observation, TwoPointHarness(), executor, memory,
    )
    assert decision.terminal["action"] == "SET_NAVIGATION_GOAL"
    assert executor.created == 2
    assert executor.finished == 1
    assert decision.tool_trace[1]["result"]["accepted"] is False


def test_completed_scan_tries_next_route_after_executor_rejection(tmp_path):
    class TwoRouteProvider:
        async def chat_with_retry(self, **kwargs):
            return LLMResponse(
                content='{"target_visible":false,"candidates":['
                        '{"u":800,"v":900,"reason":"open route A"},'
                        '{"u":900,"v":920,"reason":"open route B"}]}',
                finish_reason="stop", usage={},
            )

    planner = NanobotPoiPlanner(str(tmp_path), provider=TwoRouteProvider())
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library", step_count=9, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    executor = RejectFirstExecutor()
    decision = planner.decide(
        completed_scan_safe(observation, memory),
        observation, TwoPointHarness(), executor, memory,
    )
    assert decision.terminal["action"] == "SET_EXPLORATION_GOAL"
    assert decision.terminal["candidate_id"] == "P1"
    assert executor.created == 2
    assert executor.finished == 1
    assert [x["tool"] for x in decision.tool_trace] == [
        "QUERY_DEPTH", "POST_SCAN_REJECTED_GOAL", "SET_EXPLORATION_GOAL"
    ]


def test_completed_scan_rechecks_ungrounded_target_as_exploration(tmp_path):
    class UngroundedProvider:
        def __init__(self):
            self.calls = 0

        async def chat_with_retry(self, **kwargs):
            self.calls += 1
            content = (
                '{"target_visible":true,"candidates":['
                '{"u":400,"v":900,"reason":"ground near storefront entrance"}]}'
                if self.calls == 1 else
                '{"target_visible":false,"candidates":[[102,523],[157,512]]}'
            )
            return LLMResponse(content=content, finish_reason="stop", usage={})

    provider = UngroundedProvider()
    planner = NanobotPoiPlanner(str(tmp_path), provider=provider)
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="祥汇便利店", step_count=27, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    decision = planner.decide(
        completed_scan_safe(observation, memory),
        observation, Harness(), Executor(), memory,
    )
    assert provider.calls == 2
    assert decision.terminal["action"] == "SET_EXPLORATION_GOAL"
    assert decision.terminal["target_visible"] is False
    assert decision.tool_trace[0]["tool"] == "POST_SCAN_UNGROUNDED_TARGET"
    assert decision.tool_trace[1]["arguments"]["points"] == [
        {"view": "front", "u": 102, "v": 523,
         "reason": "fresh current-frame open exploration route"},
        {"view": "front", "u": 157, "v": 512,
         "reason": "fresh current-frame open exploration route"},
    ]


def test_exploration_ranking_prefers_central_route_over_tiny_depth_noise():
    side = PixelMeasurement(
        PixelProposal("P0", 60, 260, "open side route"),
        6.13, 0.006, 1.0, np.array([6.03, 7.18]), True, True,
        {"reason": "clear"},
    )
    central = PixelMeasurement(
        PixelProposal("P1", 200, 300, "open forward route"),
        4.75, 0.04, 1.0, np.array([4.63, 2.94]), True, True,
        {"reason": "clear"},
    )
    assert NanobotPoiPlanner._candidate_score(side) > NanobotPoiPlanner._candidate_score(central)
    assert NanobotPoiPlanner._exploration_candidate_score(central) > NanobotPoiPlanner._exploration_candidate_score(side)



def test_midpoint_high_sign_query_includes_lower_facade_ranges(tmp_path):
    class CaptureHarness(Harness):
        def normalize_proposals(self, points):
            self.sensor_points = points
            return super().normalize_proposals(points)

    provider = ScriptedProvider()
    harness = CaptureHarness()
    planner = NanobotPoiPlanner(str(tmp_path), provider=provider, max_tool_iterations=10)
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="library", step_count=18, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(
        observation, NavMode.PLANNING, memory, transition_reason="MIDPOINT_REACHED"
    )
    planner.decide(safe, observation, harness, Executor(), memory)
    assert len(harness.sensor_points) == 8
    assert [point["v"] for point in harness.sensor_points[1:4]] == [179, 256, 320]
    assert all("facade" in point["reason"] for point in harness.sensor_points[1:4])


def test_target_absent_reason_does_not_generate_navigable_facade_options(tmp_path):
    class CaptureHarness(Harness):
        def normalize_proposals(self, points):
            self.sensor_points = points
            return [PixelProposal("P0", points[0]["u"], points[0]["v"], points[0]["reason"])]

    provider = ScriptedProvider()
    provider.calls = [
        ("QUERY_DEPTH", {"points": [{
            "view": "front", "u": 620, "v": 380,
            "reason": "target POI BRANEW布瑞琳 is not visible; initiate scan",
        }]}),
        ("SET_NAVIGATION_GOAL", {
            "candidate_id": "P0", "semantic_anchor": "BRANEW布瑞琳",
            "navigation_anchor": "front entrance",
        }),
        ("SCAN_360", {"reason": "target not visible"}),
    ]
    harness = CaptureHarness()
    planner = NanobotPoiPlanner(str(tmp_path), provider=provider, max_tool_iterations=8)
    observation = SimpleNamespace(
        images={"front": Image.new("RGB", (720, 640), "gray")},
        poi_name="BRANEW布瑞琳", step_count=56, rotation=np.eye(4),
    )
    memory = EpisodeMemory()
    safe = to_agent_safe_observation(
        observation, NavMode.PLANNING, memory, transition_reason="MIDPOINT_REACHED"
    )
    executor = Executor()
    decision = planner.decide(safe, observation, harness, executor, memory)
    assert len(harness.sensor_points) == 1
    assert executor.created == 0
    assert decision.terminal["action"] == "SCAN_360"
