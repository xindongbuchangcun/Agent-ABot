#!/usr/bin/env python3
"""Render a legacy successful ABot run from its saved RGB and action trace.

This is offline visualization only. It never calls the agent or evaluator, and
it refuses to draw a reconstructed path unless it matches official metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentnav.abot.visualize import render_task_video


def backfill_trace(task_dir: Path, trace_path: Path, annotation_path: Path) -> tuple[Path, np.ndarray]:
    result = json.loads((task_dir / "result.json").read_text(encoding="utf-8"))
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not result.get("success") or annotation_path.stem != result.get("task_id"):
        raise ValueError("annotation/task mismatch or task was not successful")
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
    expected_steps = int(result["steps"])
    if [record["step"] for record in records] != list(range(expected_steps)):
        raise ValueError("action trace does not contain every environment step")
    frames = {int(path.name.split("_", 1)[0]) for path in (task_dir / "render_images").glob("*_front.jpg")}
    if frames != set(range(expected_steps + 1)):
        raise ValueError("front RGB frames are incomplete")

    start, goal = annotation["trajectory"][0], annotation["trajectory"][-1]
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = Rotation.from_euler(
        "xyz", [start["roll"], start["pitch"], start["yaw"]]
    ).as_matrix()
    pose[:3, 3] = [start["x"], start["y"], start["z"]]
    travelled = 0.0
    for record in records:
        record["visual_pose_world_xy_m"] = pose[:2, 3].tolist()
        record["visual_heading_deg"] = math.degrees(math.atan2(pose[1, 0], pose[0, 0]))
        plan = record.get("executor_plan") or {}
        local_goal = plan.get("goal_local_front_left_m")
        if local_goal is not None:
            world_goal = pose[:3, 3] + pose[:3, :3] @ np.array([*local_goal, 0.0])
            record["visual_active_goal_world_xy_m"] = world_goal[:2].tolist()

        waypoint = plan.get("selected_waypoint_front_left_m")
        command_deg = plan.get("command_deg")
        if waypoint is not None:
            # Match the evaluator's float32 API waypoint and direction transform.
            forward, left = np.asarray(waypoint, dtype=np.float32).astype(float)
            local_motion = np.array([forward, left, 0.0])
            next_position = pose[:3, 3] + pose[:3, :3] @ local_motion
            world_direction = pose[:3, :3] @ local_motion
        elif command_deg is not None:
            next_position = pose[:3, 3].copy()
            angle = math.radians(float(command_deg))
            world_direction = pose[:3, :3] @ np.array([math.cos(angle), math.sin(angle), 0.0])
        else:
            next_position = pose[:3, 3].copy()
            world_direction = pose[:3, 0]
        travelled += float(np.linalg.norm(next_position[:2] - pose[:2, 3]))
        yaw = math.atan2(world_direction[1], world_direction[0])
        pose[:3, 3] = next_position
        pose[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()

    for index, record in enumerate(records[:-1]):
        if (record.get("vlm") or {}).get("terminal", {}).get("action") == "SET_NAVIGATION_GOAL":
            goal_xy = records[index + 1].get("visual_active_goal_world_xy_m")
            if goal_xy is not None:
                record["visual_active_goal_world_xy_m"] = goal_xy

    final_distance = float(np.linalg.norm(pose[:2, 3] - [goal["x"], goal["y"]]))
    official_distance = float(result["metrics"]["final_distance_to_goal"])
    if abs(travelled - float(result["travel_length"])) > 1e-4 or abs(final_distance - official_distance) > 1e-4:
        raise ValueError(
            f"reconstructed path disagrees with official result: "
            f"travel={travelled:.6f}/{result['travel_length']:.6f}, "
            f"final_distance={final_distance:.6f}/{official_distance:.6f}"
        )
    enriched = task_dir / "agentnav_motion_trace.jsonl"
    enriched.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    print(f"validated {result['task_id']}: {expected_steps} steps, travel={travelled:.4f} m, final distance={final_distance:.4f} m")
    return enriched, pose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("annotation", type=Path)
    args = parser.parse_args()
    enriched, final_pose = backfill_trace(args.task_dir, args.trace, args.annotation)
    video = render_task_video(args.task_dir, enriched, final_pose=final_pose)
    print(video)


if __name__ == "__main__":
    main()
