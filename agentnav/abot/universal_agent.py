"""Common ABot agent entry; navigation policies are supplied per goal type.

Point/Object/VLN policies can be added later through ``backend_module``. The
existing POI agent is the only bundled policy at this entry point today.
"""

from __future__ import annotations

from typing import Any

from abotn_evaluator.interface.poi_goal import BasePoiGoalAgent
from abotn_evaluator.interface.point_goal import BasePointGoalAgent, WaypointPrediction

from agentnav.abot.goal import GoalKind, GoalSpec, parse_goal_kind


class _PoiObservationView:
    """Pass an explicit POI goal to the existing POI policy."""

    def __init__(self, observation: Any, poi_name: str) -> None:
        self._observation = observation
        self.poi_name = poi_name

    def __getattr__(self, name: str) -> Any:
        return getattr(self._observation, name)


class AgentNavUniversalGoalAgent(BasePointGoalAgent, BasePoiGoalAgent):
    """Normalize a goal, then call its backend's ``predict_goal`` method.

    A backend class accepts YAML keyword arguments, implements ``reset()``,
    and implements ``predict_goal(observation, goal: GoalSpec)``. Its return
    value is ABot ``WaypointPrediction``. Only POI has a bundled backend;
    that legacy backend uses ``predict(observation)`` and is adapted here.
    """

    def __init__(
        self,
        goal_type: GoalKind | str,
        backend_module: str | None = None,
        **config: Any,
    ) -> None:
        self.goal_type = parse_goal_kind(goal_type)
        self._legacy_poi = backend_module is None and self.goal_type is GoalKind.POI
        if self._legacy_poi:
            from agentnav.abot.poi_agent import AgentNavPoiGoalAgent

            backend_class = AgentNavPoiGoalAgent
        elif backend_module is not None:
            from abotn_evaluator.agent_loader import load_class

            backend_class = load_class(backend_module)
        else:
            raise ValueError(
                f"{self.goal_type.value} requires backend_module; only the POI "
                "navigation policy is bundled"
            )
        self.backend = backend_class(**config)
        if not self._legacy_poi and not callable(getattr(self.backend, "predict_goal", None)):
            raise TypeError("backend must implement predict_goal(observation, goal)")

    def reset(self) -> None:
        self.backend.reset()

    def predict(self, observation: Any) -> WaypointPrediction:
        goal = GoalSpec.from_observation(self.goal_type, observation)
        return self.predict_goal(observation, goal)

    def predict_goal(self, observation: Any, goal: GoalSpec) -> WaypointPrediction:
        if goal.kind is not self.goal_type:
            raise ValueError(f"configured for {self.goal_type.value}, got {goal.kind.value}")
        if self._legacy_poi:
            return self.backend.predict(_PoiObservationView(observation, goal.text))
        return self.backend.predict_goal(observation, goal)

    def __getattr__(self, name: str) -> Any:
        # Preserve the existing POI evaluator's access to runtime/log fields.
        return getattr(self.backend, name)
