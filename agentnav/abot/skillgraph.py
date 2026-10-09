"""Executable skill contract for the ABot POI decision loop.

The graph describes which atomic action can finish a planning cycle and which
mode may follow it. The existing planner, harness, and executor implement the
actions; this module is the shared routing contract between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentnav.abot.types import AgentSafeObservation, NavMode


@dataclass(frozen=True)
class AtomicSkill:
    name: str
    action: str
    source_modes: frozenset[NavMode]
    next_modes: frozenset[NavMode]
    scan_completed: bool | None = None


@dataclass(frozen=True)
class MetaSkill:
    name: str
    guidance: tuple[str, ...]
    actions: tuple[str, ...]


class PoiSkillGraph:
    """Route POI meta skills to bounded, mode-aware atomic actions."""

    def __init__(self) -> None:
        planning = frozenset({NavMode.PLANNING, NavMode.RECOVERY})
        verifying = frozenset({NavMode.VERIFYING})
        self.atomic = {
            item.action: item for item in (
                AtomicSkill(
                    "navigate", "SET_NAVIGATION_GOAL", planning,
                    frozenset({NavMode.EXECUTING, NavMode.RECOVERY}),
                ),
                AtomicSkill(
                    "scan", "SCAN_360", planning,
                    frozenset({NavMode.EXECUTING, NavMode.FAILED}), False,
                ),
                AtomicSkill(
                    "explore", "SET_EXPLORATION_GOAL", planning,
                    frozenset({NavMode.EXECUTING, NavMode.RECOVERY}), True,
                ),
                AtomicSkill(
                    "exhaust", "SEARCH_EXHAUSTED", planning,
                    frozenset({NavMode.FAILED}), True,
                ),
                AtomicSkill(
                    "verify", "TERMINATE", verifying,
                    frozenset({NavMode.TERMINATED, NavMode.EXECUTING, NavMode.RECOVERY}),
                ),
                AtomicSkill(
                    "verify", "RETURN_TO_PLANNING", verifying,
                    frozenset({NavMode.RECOVERY}),
                ),
            )
        }
        self.meta = {
            "find_poi": MetaSkill(
                "find_poi", ("navigate", "locate", "explore"),
                ("SET_NAVIGATION_GOAL", "SCAN_360"),
            ),
            "assess_scan": MetaSkill(
                "assess_scan", ("locate", "explore"),
                ("SET_NAVIGATION_GOAL", "SET_EXPLORATION_GOAL", "SEARCH_EXHAUSTED"),
            ),
            "verify_arrival": MetaSkill(
                "verify_arrival", ("navigate", "locate"),
                ("TERMINATE", "RETURN_TO_PLANNING"),
            ),
        }

    def active_meta_skill(self, safe: AgentSafeObservation) -> MetaSkill:
        if safe.mode is NavMode.VERIFYING:
            return self.meta["verify_arrival"]
        if safe.mode not in {NavMode.PLANNING, NavMode.RECOVERY}:
            raise ValueError(f"no high-level skill in {safe.mode.value}")
        if safe.scan_state.get("completed"):
            return self.meta["assess_scan"]
        return self.meta["find_poi"]

    def allowed_actions(self, safe: AgentSafeObservation) -> frozenset[str]:
        meta = self.active_meta_skill(safe)
        actions = set(meta.actions)
        if (
            safe.mode in {NavMode.PLANNING, NavMode.RECOVERY}
            and safe.scan_state.get("semantic_relocalization_required")
        ):
            actions.intersection_update({"SCAN_360"})
        completed = bool(safe.scan_state.get("completed"))
        return frozenset(
            action for action in actions
            if safe.mode in self.atomic[action].source_modes
            and (
                self.atomic[action].scan_completed is None
                or self.atomic[action].scan_completed is completed
            )
        )

    def validate_terminal(
        self, safe: AgentSafeObservation, terminal: dict[str, Any]
    ) -> AtomicSkill:
        action = terminal.get("action")
        if action not in self.allowed_actions(safe):
            raise ValueError(
                f"action {action!r} is invalid for {safe.mode.value} "
                f"with scan_completed={bool(safe.scan_state.get('completed'))}"
            )
        return self.atomic[action]

    def validate_transition(
        self, safe: AgentSafeObservation, action: str, mode: NavMode
    ) -> None:
        skill = self.validate_terminal(safe, {"action": action})
        if mode not in skill.next_modes:
            raise ValueError(
                f"{action} cannot transition from {safe.mode.value} to {mode.value}"
            )


POI_SKILL_GRAPH = PoiSkillGraph()
