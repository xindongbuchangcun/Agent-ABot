"""Small deterministic agent used only for evaluator smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from abotn_evaluator.interface.poi_goal import BasePoiGoalAgent
from abotn_evaluator.interface.point_goal import WaypointPrediction


class DummyPoiGoalAgent(BasePoiGoalAgent):
    def __init__(self, trace_path: str = "/tmp/abot_dummy_poi_trace.jsonl", **_: Any) -> None:
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self) -> None:
        self.calls = 0

    def predict(self, observation: Any) -> WaypointPrediction:
        image = observation.images["front"]
        array = np.asarray(image)
        record = {
            "step": int(observation.step_count),
            "poi_name": str(observation.poi_name),
            "image_keys": list(observation.images),
            "front_shape": list(array.shape),
            "front_mean": float(array.mean()),
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self.calls += 1
        return WaypointPrediction(
            waypoint=np.array([[0.1, 0.0]], dtype=np.float32),
            arrive=False,
            confidence=1.0,
        )
