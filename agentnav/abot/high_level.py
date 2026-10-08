"""Nanobot-backed high-level POI reasoning over one front RGB image."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from nanobot.agent.tools.base import Tool

from agentnav.abot.types import (
    AgentSafeObservation,
    EpisodeMemory,
    NavMode,
    PixelMeasurement,
    TaskStatus,
)


@dataclass
class ToolRuntime:
    safe: AgentSafeObservation
    observation: Any
    harness: Any
    executor: Any
    memory: EpisodeMemory
    measurements: dict[str, PixelMeasurement] = field(default_factory=dict)
    depth_budget_exhausted: bool = False
    terminal: dict[str, Any] | None = None
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    continuation_measurement: PixelMeasurement | None = None
    rejected_sign_pixels: set[tuple[int, int]] = field(default_factory=set)
    local_recheck_needed: bool = False
    vision_width: int = 0
    vision_height: int = 0


class RuntimeTool(Tool):
    def __init__(
        self,
        runtime: ToolRuntime,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        self.runtime = runtime
        self._name = name
        self._description = description
        self._parameters = parameters
        self._handler = handler

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    async def execute(self, **kwargs: Any) -> str:
        result = self._handler(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        self.runtime.tool_trace.append(
            {"tool": self.name, "arguments": kwargs, "result": result}
        )
        return json.dumps(result, ensure_ascii=False)

@dataclass(frozen=True)
class HighLevelDecision:
    terminal: dict[str, Any]
    raw_responses: list[dict[str, Any]]
    tool_trace: list[dict[str, Any]]
    latency_s: float
    usage: dict[str, int]
    continuation_measurement: PixelMeasurement | None = None


class NanobotPoiPlanner:
    """Use AgentNav's nanobot provider, context builder, skills, and tools."""

    def __init__(
        self,
        workspace: str,
        api_base: str = "http://localhost:8000/v1",
        model: str = "qwen3-vl-4b-instruct",
        api_key: str = "no-key",
        max_tool_iterations: int = 40,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        provider: Any | None = None,
        request_timeout_s: float = 120.0,
        request_recovery_attempts: int = 2,
        context_window_tokens: int = 8192,
        context_safety_margin_tokens: int = 64,
        min_completion_tokens: int = 128,
        image_scale: float = 2.0,
    ) -> None:
        from nanobot.agent.context import ContextBuilder
        from nanobot.agent.skills import SkillsLoader
        from nanobot.providers.custom_provider import CustomProvider

        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.context = ContextBuilder(self.workspace)
        self.context.skills = SkillsLoader(
            self.workspace, builtin_skills_dir=self.workspace / ".no_builtin_skills"
        )
        self.provider = provider or CustomProvider(api_key, api_base, model)
        self.model = model
        self.max_tool_iterations = int(max_tool_iterations)
        if self.max_tool_iterations < 1:
            raise ValueError("max_tool_iterations must be positive")
        self.fallback_tool_iterations = min(
            3, max(1, self.max_tool_iterations - 1)
        )
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.request_timeout_s = float(request_timeout_s)
        if not 0.0 < self.request_timeout_s < float("inf"):
            raise ValueError("request_timeout_s must be positive and finite")
        self.request_recovery_attempts = int(request_recovery_attempts)
        if self.request_recovery_attempts < 1:
            raise ValueError("request_recovery_attempts must be positive")
        self.context_window_tokens = int(context_window_tokens)
        self.context_safety_margin_tokens = int(context_safety_margin_tokens)
        self.min_completion_tokens = int(min_completion_tokens)
        if self.context_window_tokens < 1:
            raise ValueError("context_window_tokens must be positive")
        if self.context_safety_margin_tokens < 0:
            raise ValueError("context_safety_margin_tokens must be non-negative")
        if not 1 <= self.min_completion_tokens <= self.max_tokens:
            raise ValueError("min_completion_tokens must be between 1 and max_tokens")
        self.image_scale = float(image_scale)
        if not 1.0 <= self.image_scale <= 3.0:
            raise ValueError("image_scale must be between 1 and 3")
        self._event_loop = asyncio.new_event_loop()
        self.call_count = 0
        self.total_latency_s = 0.0

    def decide(
        self,
        safe: AgentSafeObservation,
        observation: Any,
        harness: Any,
        executor: Any,
        memory: EpisodeMemory,
    ) -> HighLevelDecision:
        return self._event_loop.run_until_complete(
            self._decide(safe, observation, harness, executor, memory)
        )

    #规划入口
    async def _decide(
        self,
        safe: AgentSafeObservation,
        observation: Any,
        harness: Any,
        executor: Any,
        memory: EpisodeMemory,
    ) -> HighLevelDecision:
        from nanobot.agent.tools.registry import ToolRegistry

        runtime = ToolRuntime(safe, observation, harness, executor, memory)
        runtime.local_recheck_needed = (
            safe.mode is NavMode.RECOVERY
            and safe.transition_reason == TaskStatus.BLOCKED.value
        )
        image_path, vision_size = self._save_front_rgb(safe)
        runtime.vision_width, runtime.vision_height = vision_size
        registry = ToolRegistry()
        self._register_tools(registry, runtime)
        prompt = self._decision_prompt(safe)
        messages = self.context.build_messages([], prompt, media=[str(image_path)])
        # Replace the generic assistant identity and bootstrap instructions
        # with the closed navigation contract and only its named skills.
        skill_names = ["navigate", "locate"]
        if safe.mode is not NavMode.VERIFYING:
            skill_names.append("explore")
        skill_guidance = self.context.skills.load_skills_for_context(skill_names)
        messages[0] = {
            "role": "system",
            "content": self._planner_system_prompt() + (
                "\n\n" + skill_guidance if skill_guidance else ""
            ),
        }
        if safe.scan_state.get("completed"):
            return await self._decide_after_completed_scan(runtime, messages)
        raw_responses: list[dict[str, Any]] = []
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        started = time.perf_counter()

        for iteration in range(self.max_tool_iterations):
            remaining_tool_iterations = self.max_tool_iterations - iteration
            if remaining_tool_iterations <= self.fallback_tool_iterations:
                if safe.mode is NavMode.VERIFYING:
                    runtime.terminal = {
                        "action": "RETURN_TO_PLANNING",
                        "reason": "verification_tool_iterations_threshold",
                        "target_visible": False,
                    }
                    runtime.tool_trace.append(
                        {
                            "tool": "PYTHON_VERIFICATION_FALLBACK",
                            "arguments": {
                                "reason": "remaining_tool_iterations_threshold"
                            },
                            "result": runtime.terminal,
                        }
                    )
                elif safe.mode in {NavMode.PLANNING, NavMode.RECOVERY}:
                    self._force_progress_fallback(
                        runtime,
                        "remaining_tool_iterations_threshold",
                    )
                if runtime.terminal is not None:
                    break
            # VLM output consists of tool calls. Retry a completed provider
            # cycle once when no tool has run and the response is recoverable.
            tools = registry.get_definitions()
            self._validate_tool_definitions(tools)
            response = await self._chat_with_request_recovery(
                messages=messages,
                tools=tools,
            )
            for key in usage:
                usage[key] += int(response.usage.get(key, 0))
            raw_responses.append(
                {
                    "content": response.content,
                    "reasoning_content": response.reasoning_content,
                    "finish_reason": response.finish_reason,
                    "tool_calls": [
                        {"id": call.id, "name": call.name, "arguments": call.arguments}
                        for call in response.tool_calls
                    ],
                    "usage": response.usage,
                }
            )
            if response.finish_reason == "error":
                if self._request_error_is_format_error(response.content):
                    if safe.mode is NavMode.VERIFYING:
                        runtime.terminal = {
                            "action": "RETURN_TO_PLANNING",
                            "reason": "recoverable_vlm_format_error",
                            "target_visible": False,
                            "fallback_reason": "recoverable_vlm_format_error",
                        }
                    else:
                        self._force_progress_fallback(
                            runtime, "recoverable_vlm_format_error"
                        )
                    runtime.tool_trace.append(
                        {
                            "tool": "PYTHON_PROVIDER_ERROR_FALLBACK",
                            "arguments": {
                                "reason": "recoverable_vlm_format_error"
                            },
                            "result": runtime.terminal,
                        }
                    )
                    break
                raise RuntimeError(response.content or "VLM request failed")
            messages = self.context.add_assistant_message(
                messages,
                response.content,
                [call.to_openai_tool_call() for call in response.tool_calls],
                response.reasoning_content,
            )
            if not response.tool_calls:
                continue
            for call in response.tool_calls:
                result = await registry.execute(call.name, call.arguments)
                messages = self.context.add_tool_result(
                    messages, call.id, call.name, result
                )
                if (
                    runtime.depth_budget_exhausted
                    and runtime.terminal is None
                    and safe.mode in {NavMode.PLANNING, NavMode.RECOVERY}
                ):
                    # The VLM must choose among the measurements it just saw.
                    # Closing QUERY_DEPTH prevents another query without
                    # silently turning Python's score into a semantic choice.
                    registry.unregister("QUERY_DEPTH")
                    runtime.tool_trace.append(
                        {
                            "tool": "PYTHON_DEPTH_BUDGET_CLOSED",
                            "arguments": {},
                            "result": {
                                "query_depth_available": False,
                                "instruction": (
                                    "Select a reachable current candidate with "
                                    "SET_NAVIGATION_GOAL, or request scanning."
                                ),
                            },
                        }
                    )
                if runtime.terminal is not None:
                    break
            if runtime.terminal is not None:
                break

        latency = time.perf_counter() - started
        self.total_latency_s += latency
        if runtime.terminal is None:
            attempted = [item["tool"] for item in runtime.tool_trace]
            raise RuntimeError(
                "VLM exhausted tool rounds without a terminal action; "
                f"attempted_tools={attempted}"
            )
        return HighLevelDecision(
            runtime.terminal,
            raw_responses,
            runtime.tool_trace,
            latency,
            usage,
            runtime.continuation_measurement,
        )

    async def _decide_after_completed_scan(
        self,
        runtime: ToolRuntime,
        messages: list[dict[str, Any]],
    ) -> HighLevelDecision:
        """Choose one fresh post-scan route before declaring search exhausted."""
        started = time.perf_counter()
        guidance = self.context.skills.load_skills_for_context(["locate", "explore"])
        messages[0] = {
            "role": "system",
            "content": (
                "You are the robot's visual route assessor. Use only the current "
                "attached RGB image and leak-safe state. Return exactly one JSON "
                "object and no prose or markdown."
                + ("\n\n" + guidance if guidance else "")
            ),
        }
        messages.append({
            "role": "user",
            "content": (
                "The 360-degree scan is complete. Recheck the current RGB view. "
                "Return one JSON object only: target_visible is a boolean and "
                "candidates is a list of 2 to 4 fresh pixels with integer u, v, "
                "and reason. If the named POI is visible, choose approach pixels "
                "on walkable ground near its entrance, and name the POI in each "
                "candidate reason to explain which storefront the pixel belongs to. "
                "Otherwise you must propose "
                "open, walkable exploration-route pixels that can reveal new storefronts; "
                "return an empty list only when no walkable ground is visible. "
                "Do not reuse coordinates from an earlier observation."
            ),
        })
        response = await self._chat_with_request_recovery(messages=messages, tools=[])
        usage = {
            key: int(response.usage.get(key, 0))
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        raw_responses = [{
            "content": response.content,
            "reasoning_content": response.reasoning_content,
            "finish_reason": response.finish_reason,
            "tool_calls": [],
            "usage": response.usage,
        }]
        payload: dict[str, Any] = {}
        if response.finish_reason != "error":
            try:
                content = response.content or "{}"
                start, end = content.find("{"), content.rfind("}")
                if start >= 0 and end >= start:
                    content = content[start : end + 1]
                decoded = json.loads(content)
                if isinstance(decoded, dict):
                    payload = decoded
            except (TypeError, ValueError):
                payload = {}

        named_evidence = self._post_scan_has_named_candidate(
            payload, runtime.safe.poi_name
        )
        if payload.get("target_visible") is True and not named_evidence:
            runtime.tool_trace.append({
                "tool": "POST_SCAN_UNGROUNDED_TARGET",
                "arguments": {"claimed_visible": True},
                "result": {"reason": "no candidate identifies the named POI"},
            })
            messages.append({
                "role": "user",
                "content": (
                    f"The previous candidate reasons did not identify '{runtime.safe.poi_name}'. "
                    "Recheck the current image. If that named POI is visibly confirmed, "
                    "return target_visible=true with 2-4 fresh approach pixels; "
                    "each reason must name the POI and describe the visible link "
                    "between its storefront and the pixel. Otherwise return "
                    "target_visible=false with 2-4 fresh open-ground exploration "
                    "pixels leading toward other storefronts. Do not approach an "
                    "unverified storefront."
                ),
            })
            retry = await self._chat_with_request_recovery(messages=messages, tools=[])
            for key in usage:
                usage[key] += int(retry.usage.get(key, 0))
            raw_responses.append({
                "content": retry.content,
                "reasoning_content": retry.reasoning_content,
                "finish_reason": retry.finish_reason,
                "tool_calls": [],
                "usage": retry.usage,
            })
            try:
                retry_content = retry.content or "{}"
                first, last = retry_content.find("{"), retry_content.rfind("}")
                retry_payload = json.loads(retry_content[first:last + 1])
                payload = retry_payload if isinstance(retry_payload, dict) else {}
            except (TypeError, ValueError):
                payload = {}
            retry_named_evidence = self._post_scan_has_named_candidate(
                payload, runtime.safe.poi_name
            )
            payload["target_visible"] = (
                payload.get("target_visible") is True and retry_named_evidence
            )

        points = []
        raw_candidates = payload.get("candidates", [])
        if not isinstance(raw_candidates, list):
            raw_candidates = []
        for item in raw_candidates[:8]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                item = {
                    "u": item[0], "v": item[1],
                    "reason": "fresh current-frame open exploration route",
                }
            if not isinstance(item, dict):
                continue
            try:
                points.append({
                    "view": "front",
                    "u": int(item["u"]),
                    "v": int(item["v"]),
                    "reason": str(item.get("reason", "open exploration route"))[:500],
                })
            except (KeyError, TypeError, ValueError):
                continue

        previously_selected = {
            (int(item["u"]), int(item["v"]))
            for item in runtime.memory.selected_pixels
            if item.get("step") == runtime.safe.step_count
            and "u" in item and "v" in item
        }
        points = [
            point for point in points
            if (
                self._vision_point_to_sensor(runtime, point)["u"],
                self._vision_point_to_sensor(runtime, point)["v"],
            ) not in previously_selected
        ]
        if not points:
            # Search farther visible ground before nearby ground. These are
            # fresh current-frame candidates, still subject to depth checks.
            for v_fraction in (0.60, 0.67, 0.74):
                for u_fraction in (0.50, 0.32, 0.68):
                    point = {
                        "view": "front",
                        "u": round(runtime.vision_width * u_fraction),
                        "v": round(runtime.vision_height * v_fraction),
                        "reason": "post-scan current-frame ground route",
                    }
                    sensor = self._vision_point_to_sensor(runtime, point)
                    if (sensor["u"], sensor["v"]) not in previously_selected:
                        points.append(point)
                    if len(points) >= 8:
                        break
                if len(points) >= 8:
                    break

        reachable: list[PixelMeasurement] = []
        if points:
            sensor_points = [self._vision_point_to_sensor(runtime, point) for point in points]
            proposals = runtime.harness.normalize_proposals(sensor_points)
            measurements = runtime.harness.query_candidates(runtime.observation, proposals)
            normalized = []
            for index, measurement in enumerate(measurements):
                measurement = self._reject_nonmoving_goal(runtime, measurement)
                measurement = self._reject_recently_blocked(runtime, measurement)
                measurement = replace(
                    measurement,
                    proposal=replace(measurement.proposal, candidate_id=f"P{index}"),
                )
                runtime.measurements[measurement.proposal.candidate_id] = measurement
                normalized.append(measurement)
                if measurement.reachable:
                    reachable.append(measurement)
            runtime.tool_trace.append({
                "tool": "QUERY_DEPTH",
                "arguments": {"points": points},
                "result": {"measurements": [item.as_dict() for item in normalized]},
            })

        chosen: tuple[PixelMeasurement, Any, str] | None = None
        if reachable:
            target_visible = payload.get("target_visible") is True
            anchor = str(payload.get("semantic_anchor") or (
                runtime.safe.poi_name if target_visible else "post-scan exploration"
            ))[:500]
            scores = {
                item.proposal.candidate_id: (
                    self._candidate_score(item) if target_visible
                    else self._exploration_candidate_score(item)
                )
                for item in reachable
            }
            if not target_visible:
                get_depth = getattr(runtime.harness, "current_depth", None)
                if callable(get_depth):
                    depth = get_depth(runtime.observation)
                    for item in reachable:
                        distance = float(np.linalg.norm(item.local_goal))
                        if distance < 1.0:
                            continue
                        horizon = min(2.0, distance)
                        ray = item.local_goal * (horizon / distance)
                        clear, preview = runtime.harness.depth_corridor_is_safe(
                            depth, ray, max_lookahead_m=horizon
                        )
                        nearest = preview.get("nearest_obstacle_m")
                        if clear:
                            adjustment = 0.30
                        elif (
                            preview.get("reason") == "depth_obstacle"
                            and nearest is not None
                            and float(nearest) <= 1.2
                        ):
                            adjustment = -0.30
                        else:
                            adjustment = -0.10
                        scores[item.proposal.candidate_id] += adjustment
                        runtime.tool_trace.append({
                            "tool": "EXPLORATION_ROUTE_PREVIEW",
                            "arguments": {
                                "candidate_id": item.proposal.candidate_id,
                                "horizon_m": round(horizon, 3),
                            },
                            "result": {
                                "safe": clear,
                                "reason": preview.get("reason"),
                                "nearest_obstacle_m": nearest,
                                "score_adjustment": adjustment,
                            },
                        })
            for measurement in sorted(
                reachable,
                key=lambda item: scores[item.proposal.candidate_id],
                reverse=True,
            ):
                task = runtime.executor.create_navigation_task(
                    runtime.observation,
                    measurement,
                    anchor,
                    measurement.proposal.reason,
                    exploration=not target_visible,
                )
                if task.status is TaskStatus.RUNNING:
                    chosen = (measurement, task, anchor)
                    break
                runtime.executor.finish_current_task()
                runtime.tool_trace.append({
                    "tool": "POST_SCAN_REJECTED_GOAL",
                    "arguments": {"candidate_id": measurement.proposal.candidate_id},
                    "result": {
                        "status": task.status.value,
                        "reason": task.failure_reason,
                    },
                })

        if chosen is not None:
            measurement, task, anchor = chosen
            action = "SET_NAVIGATION_GOAL" if target_visible else "SET_EXPLORATION_GOAL"
            runtime.terminal = {
                "action": action,
                "task": task.public_dict(),
                "target_visible": target_visible,
                "candidate_id": measurement.proposal.candidate_id,
            }
            runtime.tool_trace.append({
                "tool": action,
                "arguments": {
                    "candidate_id": measurement.proposal.candidate_id,
                    "semantic_anchor": anchor,
                    "navigation_anchor": measurement.proposal.reason,
                },
                "result": runtime.terminal,
            })
        else:
            runtime.terminal = {
                "action": "SEARCH_EXHAUSTED",
                "reason": "completed scan and bounded fresh route checks found no reachable candidate",
            }

        latency = time.perf_counter() - started
        self.total_latency_s += latency
        return HighLevelDecision(
            runtime.terminal,
            raw_responses,
            runtime.tool_trace,
            latency,
            usage,
        )

    async def _chat_with_request_recovery(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        """Retry one failed provider cycle before failing the trajectory.

        ``chat_with_retry`` handles individual transient HTTP failures. This
        outer retry also covers malformed/truncated tool JSON returned as an
        HTTP 400. Retrying is safe here because no tool has executed yet.
        """
        response = None
        current_max_tokens = self.max_tokens
        ordinary_attempt = 0
        context_budget_adjusted = False
        while ordinary_attempt < self.request_recovery_attempts:
            try:
                response = await asyncio.wait_for(
                    self.provider.chat_with_retry(
                        messages=messages,
                        tools=tools,
                        model=self.model,
                        max_tokens=current_max_tokens,
                        temperature=self.temperature,
                        tool_choice=(
                            "required" if ordinary_attempt == 0 else "auto"
                        ),
                    ),
                    timeout=self.request_timeout_s,
                )
            except asyncio.TimeoutError as exc:
                ordinary_attempt += 1
                if ordinary_attempt >= self.request_recovery_attempts:
                    raise TimeoutError(
                        f"VLM request exceeded {self.request_timeout_s:g}s including retries"
                    ) from exc
                continue
            self.call_count += 1
            if response.finish_reason == "length" and not response.tool_calls:
                response.content = (
                    "Error: truncated tool response: completion length limit reached"
                )
                response.finish_reason = "error"
            if response.finish_reason != "error":
                return response
            adjusted_tokens = self._completion_budget_after_context_error(
                response.content,
                current_max_tokens=current_max_tokens,
            )
            if adjusted_tokens is not None and not context_budget_adjusted:
                current_max_tokens = adjusted_tokens
                context_budget_adjusted = True
                # A deterministic budget correction does not consume the
                # separate transient/malformed-response retry allowance.
                continue
            if not self._request_error_is_recoverable(response.content):
                return response
            ordinary_attempt += 1
            if ordinary_attempt < self.request_recovery_attempts:
                await asyncio.sleep(1.0)
        return response

    def _completion_budget_after_context_error(
        self,
        content: str | None,
        *,
        current_max_tokens: int,
    ) -> int | None:
        """Return a smaller completion budget reported safe by the server."""
        text = content or ""
        match = re.search(
            r"maximum context length is\s+(\d+)\s+tokens.*?"
            r"\((\d+)\s+in the messages,\s*(\d+)\s+in the completion\)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match is None:
            return None
        server_context = int(match.group(1))
        prompt_tokens = int(match.group(2))
        configured_context = min(server_context, self.context_window_tokens)
        available = configured_context - prompt_tokens - self.context_safety_margin_tokens
        if available < self.min_completion_tokens:
            return None
        adjusted = min(current_max_tokens - 1, available)
        return adjusted if adjusted >= self.min_completion_tokens else None

    @staticmethod
    def _request_error_is_format_error(content: str | None) -> bool:
        text = (content or "").lower()
        return any(
            marker in text
            for marker in (
                "invalid json",
                "eof while parsing",
                "truncated tool response",
                "tool arguments must decode",
            )
        )

    @classmethod
    def _request_error_is_recoverable(cls, content: str | None) -> bool:
        text = (content or "").lower()
        return cls._request_error_is_format_error(content) or any(
            marker in text
            for marker in (
                "error code: 429",
                "error code: 500",
                "error code: 502",
                "error code: 503",
                "error code: 504",
                "timed out",
                "timeout",
                "connection error",
                "temporarily unavailable",
            )
        )

    @staticmethod
    def _validate_tool_definitions(tools: list[dict[str, Any]]) -> None:
        json.dumps(tools, ensure_ascii=False, allow_nan=False)
        names: set[str] = set()
        for tool in tools:
            if tool.get("type") != "function" or not isinstance(
                tool.get("function"), dict
            ):
                raise ValueError("each VLM tool must be an OpenAI function tool")
            function = tool["function"]
            name = function.get("name")
            parameters = function.get("parameters")
            if not isinstance(name, str) or not name or name in names:
                raise ValueError(f"invalid or duplicate VLM tool name: {name!r}")
            if not isinstance(parameters, dict) or parameters.get("type") != "object":
                raise ValueError(f"tool {name} parameters must be a JSON object schema")
            names.add(name)

    def _register_tools(self, registry: Any, rt: ToolRuntime) -> None:
        def query_depth(points: list[dict[str, Any]]) -> dict[str, Any]:
            sensor_points = [
                self._vision_point_to_sensor(rt, point) for point in points
            ]
            sensor_points = [
                point for point in sensor_points
                if (point["u"], point["v"]) not in rt.rejected_sign_pixels
            ]
            if not sensor_points:
                return {
                    "measurements": [],
                    "query_depth_budget_exhausted": rt.depth_budget_exhausted,
                    "reason": (
                        "These pixels were already rejected by the current-frame "
                        "sign check. Inspect the image and choose a different "
                        "target storefront pixel."
                    ),
                }
            rt.local_recheck_needed = False
            # After a blocked move, one high sign pixel does not describe a
            # walkable approach. Probe fresh lower pixels in the same view so
            # the planner can choose a ground anchor without another scan.
            retry_view = (
                rt.safe.mode is NavMode.RECOVERY
                or rt.safe.transition_reason in {
                    "episode_start", "MIDPOINT_REACHED", "SCAN_VIEW_READY",
                    "FOCUSED_REACQUIRE_VIEW_READY",
                }
            )
            if (
                retry_view
                and len(sensor_points) <= 2
                and not any(
                    self._proposal_declares_target_absent(point.get("reason", ""))
                    for point in sensor_points
                )
                and any(
                    point["v"] < rt.harness.camera.height * 0.55
                    for point in sensor_points
                )
            ):
                anchor = min(sensor_points, key=lambda point: point["v"])
                width, height = rt.harness.camera.width, rt.harness.camera.height
                if rt.safe.transition_reason == "MIDPOINT_REACHED":
                    for v_fraction in (0.28, 0.40, 0.50):
                        sensor_points.append({
                            "view": "front",
                            "u": anchor["u"],
                            "v": round(height * v_fraction),
                            "reason": "fresh current-frame lower facade range alternative",
                        })
                for u_offset, v_fraction in (
                    (0.0, 0.62), (-0.10, 0.68), (0.10, 0.68), (0.0, 0.74),
                ):
                    sensor_points.append({
                        "view": "front",
                        "u": int(np.clip(
                            round(anchor["u"] + width * u_offset), 0, width - 1
                        )),
                        "v": round(height * v_fraction),
                        "reason": "fresh current-frame lower approach alternative",
                    })
            sensor_points = [
                point for point in sensor_points
                if (point["u"], point["v"]) not in rt.rejected_sign_pixels
            ]
            proposals = rt.harness.normalize_proposals(sensor_points)
            raw_measurements = rt.harness.query_candidates(rt.observation, proposals)
            measurements = []
            for measurement in raw_measurements:
                measurement = self._reject_nonmoving_goal(rt, measurement)
                measurement = self._reject_recently_blocked(rt, measurement)
                if (measurement.proposal.u, measurement.proposal.v) in rt.rejected_sign_pixels:
                    measurement = replace(
                        measurement, corridor_safe=False,
                        safety_debug={**measurement.safety_debug, "reason": "sign_ocr_mismatch"},
                    )
                candidate_id = f"P{len(rt.measurements)}"
                measurement = replace(
                    measurement,
                    proposal=replace(
                        measurement.proposal,
                        candidate_id=candidate_id,
                    ),
                )
                rt.measurements[candidate_id] = measurement
                measurements.append(measurement)
            if (
                any(measurement.reachable for measurement in measurements)
                and not registry.has("SET_NAVIGATION_GOAL")
            ):
                registry.register(make_set_goal_tool())

            budget_exhausted = False
            check_budget = getattr(rt.harness, "query_budget_exhausted", None)
            if callable(check_budget):
                budget_exhausted = bool(check_budget(rt.safe.step_count))
            rt.depth_budget_exhausted = budget_exhausted
            return {
                "measurements": [item.as_dict() for item in measurements],
                "query_depth_budget_exhausted": budget_exhausted,
            }

        async def set_goal(
            candidate_id: str,
            semantic_anchor: str,
            navigation_anchor: str,
        ) -> dict[str, Any]:
            measurement = rt.measurements.get(candidate_id)
            reachable_candidate_ids = sorted(
                candidate_id
                for candidate_id, candidate in rt.measurements.items()
                if candidate.reachable
            )
            if measurement is None:
                return {
                    "accepted": False,
                    "reason": "candidate_id must come from QUERY_DEPTH in this planning cycle",
                    "available_candidate_ids": reachable_candidate_ids,
                }
            if not measurement.reachable:
                return {
                    "accepted": False,
                    "reason": "candidate is not reachable; query or select another pixel",
                    "available_candidate_ids": reachable_candidate_ids,
                }
            if not self._anchor_matches_target(rt.safe.poi_name, semantic_anchor):
                return {
                    "accepted": False,
                    "reason": (
                        "semantic_anchor names a different or unidentified POI; "
                        "inspect the current image and query fresh target pixels, "
                        "or call SCAN_360 if the named target is absent"
                    ),
                    "named_poi": rt.safe.poi_name,
                }
            if self._proposal_declares_target_absent(measurement.proposal.reason):
                return {
                    "accepted": False,
                    "reason": (
                        "candidate description says the named POI is not visible; "
                        "choose a visibly grounded target pixel or call SCAN_360"
                    ),
                }
            if self._needs_upper_sign_check(rt, measurement):
                try:
                    observed_text = await self._transcribe_upper_sign(rt, measurement)
                except Exception as exc:
                    # Optional visual cross-check must not turn a trajectory
                    # into an infrastructure failure when OCR is unavailable.
                    observed_text = ""
                    rt.tool_trace.append({
                        "tool": "UPPER_SIGN_OCR_ERROR",
                        "arguments": {"candidate_id": candidate_id},
                        "result": {"error": str(exc)[:300]},
                    })
                if self._upper_sign_text_mismatch(rt.safe.poi_name, observed_text):
                    pixel = (measurement.proposal.u, measurement.proposal.v)
                    rt.rejected_sign_pixels.add(pixel)
                    rt.local_recheck_needed = True
                    rt.measurements[candidate_id] = replace(
                        measurement, corridor_safe=False,
                        safety_debug={**measurement.safety_debug, "reason": "sign_ocr_mismatch"},
                    )
                    return {
                        "accepted": False,
                        "reason": "current-frame sign text differs from named POI; choose a fresh target pixel or scan",
                        "observed_sign_text": observed_text[:100],
                        "named_poi": rt.safe.poi_name,
                    }
            task = rt.executor.create_navigation_task(
                rt.observation, measurement, semantic_anchor, navigation_anchor
            )
            if task.status is not TaskStatus.RUNNING:
                rt.executor.finish_current_task()
                rt.measurements[candidate_id] = replace(
                    measurement,
                    corridor_safe=False,
                    safety_debug={
                        **measurement.safety_debug,
                        "reason": task.failure_reason or task.status.value,
                    },
                )
                return {
                    "accepted": False,
                    "reason": task.failure_reason or task.status.value,
                    "available_candidate_ids": [
                        key for key, item in rt.measurements.items()
                        if item.reachable
                    ],
                }
            rt.terminal = {"action": "SET_NAVIGATION_GOAL", "task": task.public_dict()}
            return rt.terminal

        def make_set_goal_tool() -> RuntimeTool:
            return RuntimeTool(
                rt,
                "SET_NAVIGATION_GOAL",
                "Create one persistent navigation task from a reachable QUERY_DEPTH candidate.",
                {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "semantic_anchor": {"type": "string"},
                        "navigation_anchor": {"type": "string"},
                    },
                    "required": ["candidate_id", "semantic_anchor", "navigation_anchor"],
                },
                set_goal,
            )

        def scan_360(reason: str) -> dict[str, Any]:
            if rt.local_recheck_needed and not rt.depth_budget_exhausted:
                rt.local_recheck_needed = False
                return {
                    "accepted": False,
                    "reason": (
                        "First recheck this current image for the named POI and "
                        "query different entrance or ground pixels if it is visible. "
                        "If the POI is absent, request SCAN_360 again."
                    ),
                }
            rt.terminal = {"action": "SCAN_360", "reason": str(reason)[:500]}
            return rt.terminal

        def search_exhausted(reason: str) -> dict[str, Any]:
            rt.terminal = {"action": "SEARCH_EXHAUSTED", "reason": str(reason)[:500]}
            return rt.terminal

        def verify_poi(
            confirmed: bool,
            reason: str,
            approach_u: int | None = None,
            approach_v: int | None = None,
            approach_reason: str = "",
        ) -> dict[str, Any]:
            if confirmed and approach_u is not None and approach_v is not None:
                sensor_point = self._vision_point_to_sensor(
                    rt,
                    {
                        "view": "front",
                        "u": approach_u,
                        "v": approach_v,
                        "reason": approach_reason
                        or "fresh visible POI approach",
                    },
                )
                points = [sensor_point]
                camera = rt.harness.camera
                if sensor_point["v"] < 0.55 * camera.height:
                    # A distant sign identifies the POI but is usually a poor
                    # walking target. Try fresh ground below the same facade.
                    for u_offset, v_fraction in (
                        (0.0, 0.62), (-0.08, 0.68), (0.08, 0.68), (0.0, 0.74),
                    ):
                        points.append({
                            "view": "front",
                            "u": int(np.clip(
                                round(sensor_point["u"] + u_offset * camera.width),
                                0, camera.width - 1,
                            )),
                            "v": round(v_fraction * camera.height),
                            "reason": "fresh current-frame ground below visible POI",
                        })
                proposals = rt.harness.normalize_proposals(points)
                measurements = rt.harness.query_candidates(rt.observation, proposals)
                reachable = [
                    measurement
                    for raw in measurements
                    if (measurement := self._reject_recently_blocked(
                        rt, self._reject_nonmoving_goal(rt, raw)
                    )).reachable
                ]
                if reachable:
                    ground = [
                        item for item in reachable
                        if item.proposal.v >= 0.6 * camera.height
                    ]
                    rt.continuation_measurement = max(
                        ground or reachable, key=self._candidate_score
                    )
            rt.terminal = (
                {
                    "action": "TERMINATE",
                    "reason": reason,
                    "target_visible": True,
                    "continuation_candidate": (
                        rt.continuation_measurement.as_dict()
                        if rt.continuation_measurement is not None
                        else None
                    ),
                }
                if confirmed
                else {
                    "action": "RETURN_TO_PLANNING",
                    "reason": reason,
                    "target_visible": False,
                }
            )
            return rt.terminal

        if rt.safe.mode is NavMode.VERIFYING:
            registry.register(
                RuntimeTool(
                    rt,
                    "VERIFY_POI",
                    "Report whether the named POI is visible. When confirmed, provide a fresh walkable ground pixel by its entrance in this current image; Python also checks nearby ground if you point at a high sign.",
                    {
                        "type": "object",
                        "properties": {
                            "confirmed": {"type": "boolean"},
                            "reason": {"type": "string"},
                            "approach_u": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": rt.vision_width - 1,
                            },
                            "approach_v": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": rt.vision_height - 1,
                            },
                            "approach_reason": {"type": "string"},
                        },
                        "required": ["confirmed", "reason"],
                    },
                    verify_poi,
                )
            )
            return

        if rt.safe.mode not in {NavMode.PLANNING, NavMode.RECOVERY}:
            raise RuntimeError(f"planner cannot act in mode {rt.safe.mode.value}")

        if rt.safe.scan_state.get("semantic_relocalization_required"):
            registry.register(
                RuntimeTool(
                    rt,
                    "SCAN_360",
                    "Start a fresh Python-controlled 360-degree semantic relocalization scan.",
                    {
                        "type": "object",
                        "properties": {"reason": {"type": "string"}},
                        "required": ["reason"],
                    },
                    scan_360,
                )
            )
            return

        registry.register(
            RuntimeTool(
                rt,
                "QUERY_DEPTH",
                "Validate semantic or range-anchor pixels with metric depth. Route safety is checked from the newest full depth map before every executor step.",
                {
                    "type": "object",
                    "properties": {
                        "points": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "view": {"type": "string", "enum": ["front"]},
                                    "u": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": rt.vision_width - 1,
                                    },
                                    "v": {
                                        "type": "integer",
                                        "minimum": 0,
                                        "maximum": rt.vision_height - 1,
                                    },
                                    "reason": {"type": "string"},
                                },
                                "required": ["u", "v", "reason"],
                            },
                            "minItems": 1,
                            "maxItems": 8,
                        }
                    },
                    "required": ["points"],
                },
                query_depth,
            )
        )
        terminal_handler = search_exhausted if rt.safe.scan_state.get("completed") else scan_360
        terminal_name = "SEARCH_EXHAUSTED" if rt.safe.scan_state.get("completed") else "SCAN_360"
        terminal_description = (
            "Report that the named POI was not found after the complete Python-controlled 360-degree scan."
            if terminal_name == "SEARCH_EXHAUSTED"
            else "Report that the named POI is not visible in the current image and request the Python-controlled 360-degree scan. Direction and angle are controlled by Python."
        )
        registry.register(
            RuntimeTool(
                rt,
                terminal_name,
                terminal_description,
                {
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                },
                terminal_handler,
            )
        )
    @staticmethod
    def _reject_nonmoving_goal(
        runtime: ToolRuntime, measurement: PixelMeasurement
    ) -> PixelMeasurement:
        tolerance = getattr(
            getattr(runtime.executor, "config", None), "local_goal_tolerance", 0.15
        )
        if measurement.reachable and float(np.linalg.norm(measurement.local_goal)) <= tolerance:
            return replace(
                measurement,
                corridor_safe=False,
                safety_debug={
                    **measurement.safety_debug,
                    "reason": "goal_already_within_local_tolerance",
                },
            )
        return measurement

    @staticmethod
    def _reject_recently_blocked(
        runtime: ToolRuntime, measurement: PixelMeasurement
    ) -> PixelMeasurement:
        check = getattr(runtime.executor, "candidate_recently_blocked", None)
        if measurement.reachable and callable(check) and check(
            runtime.observation, measurement
        ):
            return replace(
                measurement,
                corridor_safe=False,
                safety_debug={
                    **measurement.safety_debug,
                    "reason": "goal_matches_recent_blocked_region",
                },
            )
        return measurement

    @staticmethod
    def _proposal_declares_target_absent(reason: str) -> bool:
        text = reason.casefold()
        return (
            "not visible" in text
            or "目标不可见" in text
            or "未看到目标" in text
            or "看不到目标" in text
        )

    @classmethod
    def _post_scan_has_named_candidate(cls, payload: dict[str, Any], target: str) -> bool:
        candidates = payload.get("candidates", [])
        return isinstance(candidates, list) and any(
            isinstance(item, dict)
            and cls._anchor_matches_target(target, str(item.get("reason", "")))
            and not cls._proposal_declares_target_absent(
                str(item.get("reason", ""))
            )
            for item in candidates
        )

    @staticmethod
    def _anchor_matches_target(target: str, anchor: str) -> bool:
        def components(value: str) -> tuple[str, str]:
            latin = "".join(
                char.lower() for char in value
                if char.isascii() and char.isalnum()
            )
            chinese = "".join(
                char for char in value if "\u4e00" <= char <= "\u9fff"
            )
            return latin, chinese

        target_latin, target_chinese = components(target)
        anchor_latin, anchor_chinese = components(anchor)
        return (
            (len(target_latin) >= 4 and target_latin in anchor_latin)
            or (len(target_chinese) >= 2 and target_chinese in anchor_chinese)
            or (not target_latin and not target_chinese and target.strip().lower() in anchor.lower())
        )

    @staticmethod
    def _needs_upper_sign_check(
        runtime: ToolRuntime, measurement: PixelMeasurement
    ) -> bool:
        camera = getattr(runtime.harness, "camera", None)
        return bool(
            camera is not None
            and measurement.proposal.v < (
                0.40 if runtime.safe.transition_reason in {
                    "episode_start", "MIDPOINT_REACHED"
                } else 0.18
            ) * camera.height
            and measurement.reachable
            and re.fullmatch(r"[\u4e00-\u9fff]+", runtime.safe.poi_name.strip())
        )

    @staticmethod
    def _upper_sign_text_mismatch(target: str, observed: str) -> bool:
        if not re.fullmatch(r"[\u4e00-\u9fff]+", target.strip()):
            return False
        chinese = "".join(re.findall(r"[\u4e00-\u9fff]", observed))
        target = target.strip()
        # One character (including OCR's common "无" for unreadable text)
        # cannot identify a different storefront. Treat it as uncertain.
        if len(chinese) < 2:
            return False
        return not any(
            chinese[index:index + 2] in target
            for index in range(len(chinese) - 1)
        )

    async def _transcribe_upper_sign(
        self, runtime: ToolRuntime, measurement: PixelMeasurement
    ) -> str:
        image = runtime.safe.front_rgb
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        u, v = measurement.proposal.u, measurement.proposal.v
        if v < 0.18 * image.height:
            # Preserve the existing upper-sign crop for previously checked points.
            box = (
                max(0, u - 180), max(0, v - 120),
                min(image.width, u + 180), min(image.height, v + 180),
            )
        else:
            # Mid-facade anchors can lie over an adjacent storefront. Keep
            # the crop close to the chosen pixel to check that association.
            box = (
                max(0, u - 90), max(0, v - 150),
                min(image.width, u + 90), min(image.height, v + 90),
            )
        image_dir = self.workspace / "runtime_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        path = image_dir / f"sign_{runtime.safe.step_count:04d}_{u}_{v}.jpg"
        image.crop(box).save(path, quality=95)
        messages = self.context.build_messages(
            [],
            "只转写这张图片里店铺招牌上能看清的汉字。看不清的字不要猜；不要根据任何目标名称补全。只输出识别到的文字。",
            media=[str(path)],
        )
        messages[0] = {"role": "system", "content": "Read only visible sign text in the attached crop. Do not guess."}
        response = await asyncio.wait_for(
            self.provider.chat_with_retry(
                messages=messages, tools=[], model=self.model,
                max_tokens=128, temperature=0.0, tool_choice="none",
            ),
            timeout=self.request_timeout_s,
        )
        self.call_count += 1
        text = (
            str(response.content or "").strip()
            if response.finish_reason not in {"error", "length"}
            else ""
        )
        runtime.tool_trace.append({
            "tool": "UPPER_SIGN_OCR",
            "arguments": {"pixel": [u, v]},
            "result": {"observed_text": text[:100]},
        })
        return text

    @classmethod
    def _exploration_candidate_score(cls, measurement: PixelMeasurement) -> float:
        score = cls._candidate_score(measurement)
        if not np.isfinite(score):
            return score
        forward, left = np.asarray(measurement.local_goal, dtype=np.float64)[:2]
        bearing = abs(float(np.arctan2(left, forward)))
        # Without a confirmed target, tiny depth-noise differences should not
        # select an extreme side route over similarly reliable forward ground.
        return score - 0.2 * bearing / (np.pi / 2.0)

    @staticmethod
    def _candidate_score(measurement: PixelMeasurement) -> float:
        if not measurement.reachable:
            return float("-inf")
        depth = max(float(measurement.depth_m or 0.0), 1e-6)
        mad = float(measurement.depth_mad_m or 0.0)
        relative_mad = min(1.0, mad / depth)
        distance = float((measurement.local_goal ** 2).sum() ** 0.5)
        return (
            float(measurement.valid_depth_ratio)
            - 0.25 * relative_mad
            + 0.15 * min(1.0, distance / 4.0)
        )

    def _force_progress_fallback(
        self,
        runtime: ToolRuntime,
        reason: str,
    ) -> None:
        if runtime.terminal is not None:
            return
        reachable = [
            measurement
            for measurement in runtime.measurements.values()
            if measurement.reachable
        ]
        runtime.terminal = {
            "action": (
                "SEARCH_EXHAUSTED"
                if runtime.safe.scan_state.get("completed")
                else "SCAN_360"
            ),
            "reason": (
                "The high-level planner did not explicitly select a current "
                "candidate within its bounded tool rounds."
            ),
            "fallback_reason": reason,
            "reachable_candidate_ids": [
                item.proposal.candidate_id for item in reachable
            ],
        }
        runtime.tool_trace.append(
            {
                "tool": "PYTHON_PROGRESS_FALLBACK",
                "arguments": {"reason": reason},
                "result": runtime.terminal,
            }
        )

    @staticmethod
    def _vision_point_to_sensor(
        runtime: ToolRuntime,
        point: dict[str, Any],
    ) -> dict[str, Any]:
        camera = runtime.harness.camera
        if runtime.vision_width < 1 or runtime.vision_height < 1:
            raise RuntimeError("vision image dimensions are unavailable")
        mapped = dict(point)
        mapped["u"] = int(
            np.clip(
                round(float(point["u"]) * camera.width / runtime.vision_width),
                0,
                camera.width - 1,
            )
        )
        mapped["v"] = int(
            np.clip(
                round(float(point["v"]) * camera.height / runtime.vision_height),
                0,
                camera.height - 1,
            )
        )
        return mapped

    def _save_front_rgb(
        self,
        safe: AgentSafeObservation,
    ) -> tuple[Path, tuple[int, int]]:
        image_dir = self.workspace / "runtime_images"
        image_dir.mkdir(parents=True, exist_ok=True)
        path = image_dir / f"front_{safe.step_count:04d}.jpg"
        image = safe.front_rgb
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        width = max(1, round(image.width * self.image_scale))
        height = max(1, round(image.height * self.image_scale))
        if (width, height) != image.size:
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        image.save(path, quality=95)
        return path, (width, height)

    @staticmethod
    def _planner_system_prompt() -> str:
        return (
            "You are the high-level semantic planner for a robot with one "
            "forward RGB camera. Use only the attached current image, the "
            "provided leak-safe state, and tool results. Never assume target "
            "coordinates, ground-truth depth, occupancy maps, or side cameras. "
            "Pixels belong only to the current frame. Call only provided tools "
            "and finish with exactly one available terminal tool."
        )

    @staticmethod
    def _decision_prompt(safe: AgentSafeObservation) -> str:
        if safe.mode is NavMode.VERIFYING:
            return (
                "Perform exactly one visual verification of the named POI in the attached current front RGB. "
                "Inspect small and distant storefront signs carefully. A distinctive Latin or Chinese component of a mixed-language target label is sufficient visual evidence; the text need not match both scripts. "
                "Call VERIFY_POI exactly once. Set confirmed=true only when the named POI itself is visually confirmed. "
                "When confirmed, include one fresh best approach pixel using approach_u, approach_v, and approach_reason; "
                "choose the entrance, lower facade, or visible approach area and prefer useful longer range over near foreground. "
                "Python will depth-check it and continue approaching if its private distance gate says the robot is still too far. "
                "When not confirmed, set confirmed=false and omit the approach fields. Do not turn or create a navigation goal.\n\n"
                + json.dumps(
                    {
                        "poi_name": safe.poi_name,
                        "step_count": safe.step_count,
                        "mode": safe.mode.value,
                        "transition_reason": safe.transition_reason,
                        "memory": safe.memory_summary,
                    },
                    ensure_ascii=False,
                )
            )
        completed = bool(safe.scan_state.get("completed"))
        terminal_instruction = (
            "The Python-controlled 360-degree scan is complete. If the named POI is still absent, call SEARCH_EXHAUSTED."
            if completed
            else "If the named POI is absent from this image, call SCAN_360. Do not choose a turn direction or angle."
        )
        return (
            "First inspect the attached image for the named POI itself. "
            "Inspect every storefront sign, including small and distant text. A distinctive Latin or Chinese component of a mixed-language target label counts as a match; do not require both scripts to be readable. "
            "The attached image may be a magnified copy; use coordinates in the displayed image according to the tool bounds, and Python will map them back to the sensor frame. "
            "The attached image is the only current camera view. Do not infer hidden ground-truth coordinates. "
            "Pixel coordinates are valid only for this attached frame. Re-localize the named POI and select fresh coordinates from this image; never copy coordinates from memory or any earlier frame. "
            "After a blocked route, rejected neighboring sign, or failed arrival verification, inspect this current view for the named POI again. If it is still visible, query different entrance or ground pixels associated with that POI before scanning. A failed local goal is not proof that the POI is absent. "
            "If the POI is visible, identify its entrance or facade, then use QUERY_DEPTH on 2-8 distinct stable range anchors on the entrance, lower facade, or nearby floor; never query an arbitrary representative pixel. "
            "Use SET_NAVIGATION_GOAL only with a reachable returned candidate, and include the named target POI in semantic_anchor; do not name a neighboring storefront. "
            "A terminal action is exactly one of SET_NAVIGATION_GOAL, SCAN_360, or SEARCH_EXHAUSTED, depending on the tools available; stop tool use immediately after it. "
            "Verification and termination are unavailable in this mode. "
            f"{terminal_instruction}\n\n"
            + json.dumps(
                {
                    "poi_name": safe.poi_name,
                    "step_count": safe.step_count,
                    "mode": safe.mode.value,
                    "transition_reason": safe.transition_reason,
                    "scan_state": safe.scan_state,
                    "memory": safe.memory_summary,
                },
                ensure_ascii=False,
            )
        )
