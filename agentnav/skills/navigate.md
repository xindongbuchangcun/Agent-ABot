# Skill: navigate — move the robot to a target

Full workflow from natural-language instruction to physical arrival.

---

## Standard workflow (target visible)

```
robot_capture()
→ [agent perceives scene, estimates pixel (u, v) of target]

pixel_to_pose(u, v)
→ {x: 1.8, y: 0.3, theta: 0.1}    ← base_link frame, metres/radians

s1_move({x: 1.8, y: 0.3, theta: 0.1})
→ {task_id: "a1b2c3d4", status: "started"}

task_status("a1b2c3d4")   ← poll every 3–5 s
→ {status: "running", phase: "moving", distance_to_goal_m: 1.2, elapsed_s: 4.1}

task_status("a1b2c3d4")
→ {status: "completed", phase: "arrived", distance_to_goal_m: 0.0, elapsed_s: 9.3}

robot_capture()
→ [confirm arrival — describe what you see]
```

---

## When target is not visible

Use `robot_scan()` first to find the target direction, then navigate:

```
robot_scan()
→ [frame at 0°, 90°, 180°, 270° — agent picks best direction]

pixel_to_pose(u, v)   ← from the chosen scan frame
→ pose

s1_move(pose) → task_id → [poll → arrived]
robot_capture() → [confirm]
```

See `skills/explore.md` for the full multi-step exploration loop.

---

## Failure recovery

```
task_status(task_id)
→ {status: "failed", error: "Navigation failed (Nav2 status=4)"}

robot_capture()
→ [agent re-evaluates scene — target may be partially occluded]

pixel_to_pose(u2, v2)   ← better pixel estimate
→ new_pose

s1_move(new_pose) → new_task_id → [poll → arrived]
```

Common failure reasons:
- `Nav2 rejected the goal` — pose is outside the map or in an obstacle; try a nearer pixel
- `depth out of valid range` — pixel is on glass/mirror or too close; pick a different pixel
- `Nav2 not connected` — Nav2 stack is not running; call `ros_list_nodes()` to diagnose

---

## Emergency stop

```
robot_stop()    ← cancels all tasks, publishes zero velocity (<50 ms)
```

To abort a specific task without stopping all motion:

```
task_cancel(task_id)
```

---

## Distant targets (x > 4 m)

If `pixel_to_pose` returns `x > 4.0 m`, do **not** navigate directly — depth accuracy
degrades at long range and Nav2 may reject goals that project outside the current map.
Use a midpoint approach instead:

```
pixel_to_pose(u, v) → {x: 7.2, y: 0.4, theta: 0.1}   ← too far

# Navigate to half the distance
s1_move({x: 3.5, y: 0.2, theta: 0.1}) → task_id → [arrived]

# Re-assess from the new position
robot_capture()
pixel_to_pose(u2, v2) → {x: 3.1, y: 0.1, theta: 0.05}   ← now within range
s1_move({x: 3.1, ...}) → task_id → [arrived]
robot_capture() → [confirm]
```

---

## Key rules

1. Always call `robot_capture()` **before** to understand the scene.
2. Always call `robot_capture()` **after** to confirm arrival.
3. Poll `task_status` every **3–5 seconds** — not faster (Nav2 needs time to plan).
4. Stop polling when `status` is `completed`, `failed`, or `cancelled`.
5. If `pixel_to_pose` returns an error, pick a different pixel — do not retry the same one.
6. If navigation fails twice, use `robot_scan()` to reorient and try from a fresh perspective.
7. If `pixel_to_pose` returns `x > 4.0 m`, use the midpoint strategy above.
