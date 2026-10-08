"""Leakage-safe adapters for ABot POI observations."""

from __future__ import annotations

from typing import Any

from agentnav.abot.types import AgentSafeObservation, EpisodeMemory, NavMode


def front_rgb(observation: Any) -> Any:
    images = getattr(observation, "images", None)
    if not isinstance(images, dict) or "front" not in images:
        raise ValueError("single-front observation must contain images['front']")
    return images["front"]


def to_agent_safe_observation(
    observation: Any,
    mode: NavMode,
    memory: EpisodeMemory,
    transition_reason: str = "",
    scan_state: dict[str, Any] | None = None,
) -> AgentSafeObservation:
    """Copy only policy-approved fields; never pass the original object."""
    poi_name = str(getattr(observation, "poi_name", "")).strip()
    if not poi_name:
        raise ValueError("PoiGoalObservation.poi_name is required")
    return AgentSafeObservation(
        poi_name=poi_name,
        front_rgb=front_rgb(observation),
        step_count=int(getattr(observation, "step_count")),
        mode=mode,
        memory_summary=memory.high_level_summary(),
        transition_reason=transition_reason,
        scan_state=dict(scan_state or {}),
    )
