"""Task-neutral goal input at the ABot agent boundary.

Coordinates use the evaluator observation convention: [forward, left] metres.
Only point goals expose coordinates to a policy. Semantic goals carry text.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np


class GoalKind(str, Enum):
    POINT = "point"
    POI = "poi"
    OBJECT = "object"
    VLN = "vln"


def parse_goal_kind(kind: GoalKind | str) -> GoalKind:
    if isinstance(kind, GoalKind):
        return kind
    name = str(kind).strip().lower().replace("-", "_")
    if name.endswith("_goal"):
        name = name[:-5]
    elif name.endswith("goal"):
        name = name[:-4]
    return GoalKind(name)


_TEXT_FIELDS = {
    GoalKind.POI: ("poi_name",),
    GoalKind.OBJECT: ("object_name", "object_category"),
    GoalKind.VLN: ("instruction", "route_instruction"),
}


@dataclass(frozen=True)
class GoalSpec:
    kind: GoalKind
    target_position: tuple[float, float] | None = None
    text: str | None = None

    @classmethod
    def from_input(cls, kind: GoalKind | str, value: Any) -> GoalSpec:
        kind = parse_goal_kind(kind)
        if kind is GoalKind.POINT:
            position = np.asarray(value, dtype=np.float64)
            if position.shape != (2,) or not np.all(np.isfinite(position)):
                raise ValueError("point goal must be a finite [forward, left] pair")
            return cls(kind=kind, target_position=(float(position[0]), float(position[1])))
        text = str(value).strip() if value is not None else ""
        if not text:
            raise ValueError(f"{kind.value} goal requires nonempty text")
        return cls(kind=kind, text=text)

    @classmethod
    def from_observation(cls, kind: GoalKind | str, observation: Any) -> GoalSpec:
        kind = parse_goal_kind(kind)
        if kind is GoalKind.POINT:
            return cls.from_input(kind, _field(observation, "target_position"))
        for name in _TEXT_FIELDS[kind]:
            value = _field(observation, name)
            if value is not None and str(value).strip():
                return cls.from_input(kind, value)
        raise ValueError(
            f"{kind.value} observation requires one of: {', '.join(_TEXT_FIELDS[kind])}"
        )


def _field(observation: Any, name: str) -> Any:
    if isinstance(observation, Mapping):
        return observation.get(name)
    return getattr(observation, name, None)
