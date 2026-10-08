"""Render an ABot task's front-view motion and decision timeline as MP4."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


VIDEO_SIZE = (1600, 900)
SCENE_BOX = (0, 48, 900, 848)
FONT_PATHS = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
)


def _ffmpeg_binary() -> str:
    """Find an FFmpeg executable for browser-compatible H.264 output."""
    configured = os.environ.get("AGENTNAV_FFMPEG")
    if configured:
        if Path(configured).is_file():
            return configured
        raise FileNotFoundError(f"AGENTNAV_FFMPEG does not exist: {configured}")
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        pass
    # Evaluation Python may be in a minimal Conda env while FFmpeg is
    # installed in a sibling env on the same server.
    python_env = Path(sys.executable).resolve().parent.parent
    envs_dir = python_env.parent if python_env.parent.name == "envs" else python_env / "envs"
    for candidate in sorted(envs_dir.glob("*/bin/ffmpeg")):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise FileNotFoundError("FFmpeg is required for VS Code-compatible H.264 video; set AGENTNAV_FFMPEG")


def _encode_h264(source: Path, output: Path, ffmpeg: str) -> None:
    encoded = output.with_name(output.stem + ".h264.tmp.mp4")
    try:
        completed = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
             "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(encoded)],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"FFmpeg H.264 encoding failed: {completed.stderr.strip()}")
        encoded.replace(output)
    finally:
        encoded.unlink(missing_ok=True)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _read_trace(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[int(record["step"])] = record
    return records


def _frame_files(task_dir: Path) -> list[tuple[int, Path]]:
    files = []
    for path in (task_dir / "render_images").glob("*_front.jpg"):
        try:
            files.append((int(path.name.split("_", 1)[0]), path))
        except ValueError:
            continue
    return sorted(files)


def _short(value: Any, limit: int = 68) -> str:
    value = " ".join(str(value or "").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _step_event(step: int, record: dict[str, Any]) -> dict[str, Any]:
    """Describe the action issued at this observation, including Python-only steps."""
    plan = record.get("executor_plan") or {}
    terminal = (record.get("vlm") or {}).get("terminal") or {}
    previous_status = record.get("task_status")
    next_status = (
        record.get("new_task_status")
        or (terminal.get("task") or {}).get("task_status")
    )
    status = (
        f"{previous_status} → {next_status}"
        if previous_status and next_status and previous_status != next_status
        else next_status or previous_status or record.get("mode_before") or "未知"
    )
    waypoint = plan.get("selected_waypoint_front_left_m")
    command_deg = plan.get("command_deg")
    if waypoint is not None:
        forward, left = float(waypoint[0]), float(waypoint[1])
        distance = math.hypot(forward, left)
        bearing = math.degrees(math.atan2(left, forward))
        direction = "正前方" if abs(bearing) < 1.0 else f"{'左' if bearing > 0 else '右'}{abs(bearing):.0f}°"
        action = f"移动 {distance:.2f} m"
        detail = f"状态 {status} · 运动方向 {direction}"
    elif command_deg is not None:
        angle = float(command_deg)
        action = f"原地{'左' if angle >= 0 else '右'}转 {abs(angle):.1f}°"
        detail = f"状态 {status} · {plan.get('type', 'turn')}"
    elif plan.get("type") in {"navigation", "depth_retreat"}:
        action = "停止移动 · 无安全短步"
        detail = f"状态 BLOCKED · {record.get('transition_reason', '')}"
    elif record.get("stop_reason"):
        action = "结束任务 · 本步未移动"
        detail = f"状态 {status} · {record['stop_reason']}"
    elif terminal.get("action") in {"SET_NAVIGATION_GOAL", "SET_EXPLORATION_GOAL"}:
        action = "设定导航目标 · 本步未移动"
        detail = f"状态 {status} · {terminal['action']}"
    elif terminal.get("action"):
        action = "观察/决策 · 本步未移动"
        detail = f"状态 {status} · {terminal['action']}"
    else:
        action = "原地等待 · 本步未移动"
        detail = f"状态 {status}"
    return {"step": step, "title": action, "detail": detail, "kind": "step"}


def _events(records: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for step, record in sorted(records.items()):
        vlm = record.get("vlm") or {}
        if not record.get("vlm_called"):
            events.append(_step_event(step, record))
            continue
        if (vlm.get("safe_input") or {}).get("scan_state", {}).get("completed"):
            events.append({"step": step, "title": "扫描后路线评估", "detail": "VLM 返回当前帧路线 JSON"})
        for item in vlm.get("tool_trace") or []:
            name = str(item.get("tool", ""))
            args = item.get("arguments") or {}
            result = item.get("result") or {}
            pixel = None
            if name == "QUERY_DEPTH":
                measurements = result.get("measurements") or []
                reachable = [m.get("candidate_id") for m in measurements if m.get("reachable")]
                detail = f"{len(measurements)} 点；可用 {','.join(map(str, reachable)) or '无'}"
            elif name in {"SET_NAVIGATION_GOAL", "SET_EXPLORATION_GOAL"}:
                task = result.get("task") or {}
                candidate = args.get("candidate_id") or result.get("candidate_id") or "?"
                accepted = bool(task) and result.get("accepted") is not False
                detail = f"{candidate} {'已创建任务' if accepted else '被拒绝'}"
                if accepted and task.get("selected_pixel") is not None:
                    pixel = tuple(int(v) for v in task["selected_pixel"])
                    detail += f" · 像素 {pixel}"
            elif name == "SCAN_360":
                detail = _short(args.get("reason"), 58)
            elif name == "VERIFY_POI":
                detail = f"confirmed={args.get('confirmed')} · {_short(args.get('reason'), 42)}"
            elif name == "UPPER_SIGN_OCR":
                detail = f"读取店牌：{_short(result.get('observed_text'), 45)}"
            elif name in {"SEARCH_EXHAUSTED", "POST_SCAN_UNGROUNDED_TARGET"}:
                detail = _short(args.get("reason") or result.get("reason"), 58)
            elif name.startswith("PYTHON_"):
                detail = _short(result.get("reason") or result.get("fallback_reason"), 58)
            else:
                continue
            events.append({"step": step, "title": name, "detail": detail, "pixel": pixel})
        if not vlm.get("tool_trace"):
            terminal = vlm.get("terminal") or {}
            events.append({
                "step": step,
                "title": str(terminal.get("action", "VLM")),
                "detail": _short(terminal.get("reason"), 58),
            })
        events.append(_step_event(step, record))
    return events


def _draw_cross(draw: ImageDraw.ImageDraw, xy: tuple[float, float], color: str, size: int = 11) -> None:
    x, y = xy
    draw.ellipse((x - size, y - size, x + size, y + size), outline=color, width=3)
    draw.line((x - size - 6, y, x + size + 6, y), fill=color, width=3)
    draw.line((x, y - size - 6, x, y + size + 6), fill=color, width=3)


def _draw_future_path(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, np.ndarray, float]],
    current_step: int,
    scene_box: tuple[int, int, int, int],
    source_size: tuple[int, int],
    camera: Any,
) -> None:
    """Project a short, already-recorded future path onto the current RGB."""
    current = next((item for item in points if item[0] == current_step), None)
    if current is None:
        return
    origin = current[1]
    heading = math.radians(current[2])
    cos_heading, sin_heading = math.cos(heading), math.sin(heading)
    width, height = source_size
    intrinsic_width = float(getattr(camera, "width", width))
    intrinsic_height = float(getattr(camera, "height", height))
    fx = float(getattr(camera, "fx", 252.075)) * width / intrinsic_width
    fy = float(getattr(camera, "fy", 252.075)) * height / intrinsic_height
    cx = float(getattr(camera, "cx", intrinsic_width / 2)) * width / intrinsic_width
    cy = float(getattr(camera, "cy", intrinsic_height / 2)) * height / intrinsic_height
    camera_height = float(getattr(camera, "extrinsic_height", 0.65))
    x0, y0, x1, y1 = scene_box

    def project(pos: np.ndarray) -> tuple[float, float] | None:
        dx, dy = np.asarray(pos, dtype=float) - origin
        forward = dx * cos_heading + dy * sin_heading
        left = -dx * sin_heading + dy * cos_heading
        if forward < 0.35:
            return None
        u = cx - fx * left / forward
        v = cy + fy * camera_height / forward
        x = x0 + u * (x1 - x0) / width
        y = y0 + v * (y1 - y0) / height
        if x0 <= x < x1 and y0 <= y < y1:
            return x, y
        return None

    future = [item for item in points if current_step <= item[0] <= current_step + 20]
    previous = origin
    travelled_m = 0.0
    visible_segments: list[list[tuple[float, float]]] = []
    segment: list[tuple[float, float]] = []
    for _, position, _ in future[1:]:
        length = float(np.linalg.norm(position - previous))
        if length <= 1e-5:
            continue
        available = 4.0 - travelled_m
        if available <= 0:
            break
        end = previous + (position - previous) * min(1.0, available / length)
        samples = max(2, math.ceil(float(np.linalg.norm(end - previous)) / 0.05))
        for index in range(samples + 1):
            point = project(previous + (end - previous) * (index / samples))
            if point is None:
                if len(segment) >= 2:
                    visible_segments.append(segment)
                segment = []
            elif not segment or math.dist(point, segment[-1]) >= 1.0:
                segment.append(point)
        travelled_m += float(np.linalg.norm(end - previous))
        previous = position
        if travelled_m >= 4.0 - 1e-5:
            break
    if len(segment) >= 2:
        visible_segments.append(segment)
    if not visible_segments:
        return
    for segment in visible_segments:
        draw.line(segment, fill="#163827", width=10, joint="curve")
        draw.line(segment, fill="#6af49a", width=5, joint="curve")
    end_x, end_y = visible_segments[-1][-1]
    draw.ellipse((end_x - 6, end_y - 6, end_x + 6, end_y + 6), fill="#6af49a", outline="#163827", width=2)
    draw.text((x0 + 12, y1 - 36), "后续实际路径 · 回放投影", font=_font(18), fill="#6af49a", stroke_width=2, stroke_fill="black")


def _draw_source_pixel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    selection: dict[str, Any] | None,
    frames: dict[int, Path],
) -> None:
    box = (918, 88, 1582, 365)
    draw.rounded_rectangle(box, radius=10, fill="#172330")
    draw.text((1250, 147), "绿色：后续实际运动", font=_font(18), fill="#6af49a")
    draw.text((1250, 180), "仅在视频回放中投影", font=_font(16), fill="#aab9c9")
    draw.text((1250, 211), "不会提供给机器人规划", font=_font(16), fill="#aab9c9")
    if selection is None or selection["step"] not in frames:
        draw.text((934, 190), "尚未选定目标像素", font=_font(19), fill="#aab9c9")
        return
    step = selection["step"]
    pixel = selection["pixel"]
    draw.text((934, 96), f"目标像素 · 源帧 {step}", font=_font(17), fill="#eaf1f7")
    with Image.open(frames[step]) as source:
        source = source.convert("RGB")
        thumb = ImageOps.contain(source, (300, 224))
        x0, y0 = 934, 131
        canvas.paste(thumb, (x0, y0))
        sx, sy = thumb.width / source.width, thumb.height / source.height
        _draw_cross(draw, (x0 + pixel[0] * sx, y0 + pixel[1] * sy), "#ffb648", 7)
    draw.text((1250, 255), f"(u, v) = {pixel}", font=_font(18), fill="#ffcf90")


def _draw_timeline(draw: ImageDraw.ImageDraw, events: list[dict[str, Any]], step: int) -> None:
    draw.rounded_rectangle((918, 376, 1582, 844), radius=10, fill="#172330")
    shown = [event for event in events if event["step"] <= step]
    draw.text((936, 387), f"VLM / 工具 / 环境步 · {len(shown)}/{len(events)}", font=_font(21), fill="white")
    current_step = next(
        (event for event in reversed(shown) if event["step"] == step and event.get("kind") == "step"),
        None,
    )
    if current_step is not None:
        draw.text((940, 417), f"本步 {step:02d}  {_short(current_step['title'], 34)}", font=_font(17), fill="#ffcf90")
        draw.text((960, 439), _short(current_step["detail"], 70), font=_font(14), fill="#aab9c9")
    timeline = [
        (index, event)
        for index, event in enumerate(shown, 1)
        if event is not current_step
    ]
    for row, (index, event) in enumerate(timeline[-6:]):
        y = 477 + row * 60
        current = event["step"] == step
        if current:
            draw.rounded_rectangle((930, y - 3, 1569, y + 51), radius=6, fill="#284154")
        color = "#ffcf90" if current else ("#87f1e9" if event.get("kind") == "step" else "#e0edf6")
        draw.text((940, y), _short(f"{index:02d}  步 {event['step']:02d}  {event['title']}", 42), font=_font(17), fill=color)
        draw.text((975, y + 25), _short(event["detail"], 68), font=_font(14), fill="#aab9c9")


def render_task_video(
    task_dir: str | Path,
    trace_path: str | Path,
    result: dict[str, Any] | None = None,
    *,
    final_pose: Any = None,
    camera: Any = None,
    fps: int = 2,
) -> Path:
    """Create a complete video without treating stale pixels as fresh detections."""
    task_dir = Path(task_dir)
    records = _read_trace(Path(trace_path))
    frame_list = _frame_files(task_dir)
    if not frame_list:
        raise FileNotFoundError(f"no front RGB frames in {task_dir}")
    frames = dict(frame_list)
    if result is None:
        result_file = task_dir / "result.json"
        result = json.loads(result_file.read_text(encoding="utf-8")) if result_file.is_file() else {}
    events = _events(records)
    if frame_list[-1][0] not in records:
        events.append({
            "step": frame_list[-1][0],
            "title": "最终观测 · 无新动作",
            "detail": f"状态 {'成功' if result.get('success') else result.get('status', '结束')}",
            "kind": "step",
        })
    positions: list[tuple[int, np.ndarray, float]] = []
    for step, record in sorted(records.items()):
        xy = record.get("visual_pose_world_xy_m")
        if isinstance(xy, list) and len(xy) == 2:
            positions.append((step, np.asarray(xy, dtype=float), float(record.get("visual_heading_deg", 0.0))))
    if final_pose is not None:
        pose = np.asarray(final_pose, dtype=float)
        if pose.shape == (4, 4) and np.all(np.isfinite(pose)):
            positions.append((frame_list[-1][0], pose[:2, 3], math.degrees(math.atan2(pose[1, 0], pose[0, 0]))))
    elif positions and positions[-1][0] == frame_list[-1][0] - 1:
        # Older saved traces have an action for every step but no final pose.
        # Recover only the last display point from the recorded local action.
        last_step, last_xy, heading_deg = positions[-1]
        waypoint = (records.get(last_step, {}).get("executor_plan") or {}).get(
            "selected_waypoint_front_left_m"
        )
        final_xy = last_xy.copy()
        if waypoint is not None:
            forward, left = map(float, waypoint)
            heading = math.radians(heading_deg)
            final_xy += np.array([
                forward * math.cos(heading) - left * math.sin(heading),
                forward * math.sin(heading) + left * math.cos(heading),
            ])
        positions.append((frame_list[-1][0], final_xy, heading_deg))
    positions.sort(key=lambda item: item[0])
    output = task_dir / "agentnav_motion.mp4"
    ffmpeg = _ffmpeg_binary()
    intermediate = task_dir / "agentnav_motion.mp4v.tmp.mp4"
    writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, VIDEO_SIZE)
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 video writer")
    selection = None
    try:
        for step, image_path in frame_list:
            for event in events:
                if event["step"] == step and event.get("pixel") is not None:
                    selection = event
            canvas = Image.new("RGB", VIDEO_SIZE, "#0c141d")
            draw = ImageDraw.Draw(canvas)
            with Image.open(image_path) as source:
                source = source.convert("RGB")
                scene = ImageOps.contain(source, (SCENE_BOX[2], SCENE_BOX[3] - SCENE_BOX[1]))
                scene_x = SCENE_BOX[0] + (SCENE_BOX[2] - scene.width) // 2
                scene_y = SCENE_BOX[1] + (SCENE_BOX[3] - SCENE_BOX[1] - scene.height) // 2
                canvas.paste(scene, (scene_x, scene_y))
                _draw_future_path(
                    draw, positions, step,
                    (scene_x, scene_y, scene_x + scene.width, scene_y + scene.height),
                    source.size, camera,
                )
                sx, sy = scene.width / source.width, scene.height / source.height
                if selection is not None and selection["step"] == step:
                    u, v = selection["pixel"]
                    _draw_cross(draw, (scene_x + u * sx, scene_y + v * sy), "#ffb648")
                    draw.text((scene_x + 12, scene_y + 12), "本帧 VLM 目标像素", font=_font(20), fill="#ffcf90", stroke_width=2, stroke_fill="black")
                else:
                    record = records.get(step) or {}
                    goal = record.get("visual_active_goal_world_xy_m")
                    pose_xy = record.get("visual_pose_world_xy_m")
                    if goal is not None and pose_xy is not None:
                        delta = np.asarray(goal, dtype=float) - np.asarray(pose_xy, dtype=float)
                        angle = math.atan2(delta[1], delta[0]) - math.radians(float(record.get("visual_heading_deg", 0.0)))
                        angle = (angle + math.pi) % (2 * math.pi) - math.pi
                        if abs(angle) < math.pi / 2:
                            width = float(getattr(camera, "width", source.width))
                            cx = float(getattr(camera, "cx", width / 2))
                            fx = float(getattr(camera, "fx", 252.075))
                            u = cx - fx * math.tan(angle)
                            if 0 <= u < width:
                                x = scene_x + u * scene.width / width
                                for y in range(scene_y + 65, scene_y + scene.height - 20, 22):
                                    draw.line((x, y, x, min(y + 11, scene_y + scene.height - 20)), fill="#55e4df", width=2)
                                draw.text((scene_x + 12, scene_y + 12), "目标方向投影 · 非新像素", font=_font(20), fill="#87f1e9", stroke_width=2, stroke_fill="black")
            if step == frame_list[-1][0]:
                status = "成功" if result.get("success") else str(result.get("status", "结束"))
            else:
                status = str((records.get(step) or {}).get("mode_before", "EXECUTING"))
            draw.text((20, 8), f"{result.get('task_id', task_dir.name)}  |  {result.get('target_label', '')}", font=_font(24), fill="#f4f8fc")
            draw.text((1010, 9), f"环境步 {step}/{frame_list[-1][0]}  |  {status}", font=_font(22), fill="#ffcf90")
            _draw_source_pixel(canvas, draw, selection, frames)
            _draw_timeline(draw, events, step)
            draw.text((22, 856), "橙色：当前帧选点 / 源帧像素    青色：活动目标方向    绿色：后续实际路径（回放投影）", font=_font(18), fill="#d3e1ee")
            draw.rectangle((22, 886, 1578, 892), fill="#314656")
            progress = (step - frame_list[0][0]) / max(1, frame_list[-1][0] - frame_list[0][0])
            draw.rectangle((22, 886, 22 + 1556 * progress, 892), fill="#55d5be")
            video_frame = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
            repeats = max(fps * 2, 1) if step == frame_list[-1][0] else (fps + 1 if any(event["step"] == step for event in events) else 1)
            for _ in range(repeats):
                writer.write(video_frame)
    finally:
        writer.release()
    try:
        _encode_h264(intermediate, output, ffmpeg)
    finally:
        intermediate.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an ABot POI task video")
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    video = render_task_video(args.task_dir, args.trace)
    print(video)


if __name__ == "__main__":
    main()
