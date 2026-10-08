from types import SimpleNamespace

from PIL import Image

from agentnav.abot.evaluator import AgentNavPoiGoalEvaluator


class Memory:
    def __init__(self, image):
        self.image = image

    def get_current_images(self):
        return [self.image]


def test_single_front_adapter_uses_renderer_image_not_placeholder(monkeypatch):
    real = Image.new("RGB", (8, 8), "red")
    placeholder = Image.new("RGB", (8, 8), "black")
    observation = SimpleNamespace(
        images={"left": real, "front": placeholder, "right": placeholder},
        history_images=[{}],
        history_poses=[object()],
        occ_map=object(),
        height_map=object(),
        meta_data={"privileged": True},
    )
    monkeypatch.setattr(
        "abotn_evaluator.poi_goal.evaluator.PoiGoalEvaluator._build_poi_observation",
        lambda self, *args, **kwargs: observation,
    )
    evaluator = object.__new__(AgentNavPoiGoalEvaluator)
    result = evaluator._build_poi_observation(short_memory=Memory(real))
    assert list(result.images) == ["front"]
    assert result.images["front"] is real
    assert result.images["front"] is not placeholder
    assert result.occ_map is None
    assert result.meta_data is None




def test_single_front_evaluation_configures_short_memory_before_frames(monkeypatch, tmp_path):
    seen = {}

    def fake_evaluate(self, agent, episode, task, short_memory, episode_dir):
        seen["num_current_views"] = short_memory.num_current_views
        seen["reorder_views"] = short_memory.reorder_views
        return {
            "status": "stop", "success": False, "oracle_success": False,
            "spl": 0.0, "steps": 0, "travel_length": 0.0,
            "distance_to_goal": 10.0, "metrics": {},
        }

    monkeypatch.setattr(
        "abotn_evaluator.poi_goal.evaluator.PoiGoalEvaluator._evaluate_task",
        fake_evaluate,
    )
    evaluator = object.__new__(AgentNavPoiGoalEvaluator)
    monkeypatch.setattr(evaluator, "_save_task_result", lambda result, path: None)
    memory = SimpleNamespace(num_current_views=3, reorder_views=True)
    agent = SimpleNamespace(runtime=SimpleNamespace(stop_reason=""))

    evaluator._evaluate_task(
        agent, object(), SimpleNamespace(task_id="task1"), memory, str(tmp_path)
    )

    assert seen == {"num_current_views": 1, "reorder_views": False}


def test_stalled_stop_is_failure_even_if_parent_reports_arrival(monkeypatch, tmp_path):
    from agentnav.abot.types import RuntimeState

    monkeypatch.setattr(
        "abotn_evaluator.poi_goal.evaluator.PoiGoalEvaluator._evaluate_task",
        lambda *args, **kwargs: {
            "status": "stop", "success": True, "oracle_success": True,
            "spl": 1.0, "steps": 24, "travel_length": 3.0,
            "distance_to_goal": 1.0, "metrics": {},
        },
    )
    evaluator = object.__new__(AgentNavPoiGoalEvaluator)
    saved = []
    monkeypatch.setattr(evaluator, "_save_task_result", lambda result, path: saved.append(result.copy()))
    agent = SimpleNamespace(runtime=RuntimeState(stop_reason="no_new_position_region"))
    result = evaluator._evaluate_task(
        agent, object(), SimpleNamespace(task_id="task1"), SimpleNamespace(num_current_views=3, reorder_views=True), str(tmp_path)
    )
    assert result["status"] == "stalled"
    assert result["success"] is False
    assert result["oracle_success"] is False
    assert result["spl"] == 0.0
    assert result["steps"] == 24
    assert result["travel_length"] == 3.0
    assert saved[-1]["status"] == "stalled"


def test_completed_scan_without_target_has_search_exhausted_status(monkeypatch, tmp_path):
    from agentnav.abot.types import RuntimeState

    monkeypatch.setattr(
        "abotn_evaluator.poi_goal.evaluator.PoiGoalEvaluator._evaluate_task",
        lambda *args, **kwargs: {
            "status": "stop", "success": True, "oracle_success": True,
            "spl": 1.0, "steps": 16, "travel_length": 0.0,
            "distance_to_goal": 8.0, "metrics": {},
        },
    )
    evaluator = object.__new__(AgentNavPoiGoalEvaluator)
    monkeypatch.setattr(evaluator, "_save_task_result", lambda result, path: None)
    agent = SimpleNamespace(
        runtime=RuntimeState(stop_reason="scan_360_target_not_found")
    )
    result = evaluator._evaluate_task(
        agent, object(), SimpleNamespace(task_id="task1"), SimpleNamespace(num_current_views=3, reorder_views=True), str(tmp_path)
    )
    assert result["status"] == "search_exhausted"
    assert result["success"] is False
    assert result["oracle_success"] is False
    assert result["spl"] == 0.0
