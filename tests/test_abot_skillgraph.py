"""The skill graph must reject actions outside the current POI phase."""

import pytest

from agentnav.abot.skillgraph import POI_SKILL_GRAPH
from agentnav.abot.types import AgentSafeObservation, NavMode


def safe(mode: NavMode, **scan_state: bool) -> AgentSafeObservation:
    return AgentSafeObservation(
        poi_name="library", front_rgb=None, step_count=0, mode=mode,
        memory_summary={}, scan_state=scan_state,
    )


def test_graph_routes_normal_scan_and_verification_phases() -> None:
    graph = POI_SKILL_GRAPH
    planning = safe(NavMode.PLANNING)
    scanned = safe(NavMode.RECOVERY, completed=True)
    verifying = safe(NavMode.VERIFYING)

    assert graph.active_meta_skill(planning).guidance == (
        "navigate", "locate", "explore"
    )
    assert graph.allowed_actions(planning) == {
        "SET_NAVIGATION_GOAL", "SCAN_360"
    }
    assert graph.active_meta_skill(scanned).guidance == ("locate", "explore")
    assert graph.allowed_actions(scanned) == {
        "SET_NAVIGATION_GOAL", "SET_EXPLORATION_GOAL", "SEARCH_EXHAUSTED"
    }
    assert graph.allowed_actions(verifying) == {
        "TERMINATE", "RETURN_TO_PLANNING"
    }


def test_graph_rejects_wrong_phase_or_successor() -> None:
    graph = POI_SKILL_GRAPH
    planning = safe(NavMode.PLANNING)
    scanned = safe(NavMode.PLANNING, completed=True)

    with pytest.raises(ValueError, match="invalid"):
        graph.validate_terminal(planning, {"action": "TERMINATE"})
    with pytest.raises(ValueError, match="invalid"):
        graph.validate_terminal(scanned, {"action": "SCAN_360"})
    with pytest.raises(ValueError, match="cannot transition"):
        graph.validate_transition(planning, "SET_NAVIGATION_GOAL", NavMode.TERMINATED)


def test_semantic_relocalization_only_allows_scan() -> None:
    graph = POI_SKILL_GRAPH
    recovery = safe(NavMode.RECOVERY, semantic_relocalization_required=True)
    assert graph.allowed_actions(recovery) == {"SCAN_360"}
