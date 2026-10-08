# explore skill

When the target is not in the current field of view, use the following steps to actively search:

1. `robot_scan()` — rotate 360°, get scene descriptions in each direction
2. Based on scan results, determine the most likely direction (e.g. "there's a door at 180°, kitchen is likely behind it")
3. `s2_locate("move in that direction")` → pose
4. `s1_move(pose)` → task_id, poll with `task_status` until `phase == "arrived"`
5. `robot_look(focus=target)` — check if target is visible
6. If visible: execute standard navigation (robot_look → s2_locate → s1_move → poll → verify)
7. If not visible and exploration count < 3: go back to step 1
8. If count > 3: inform user target not found, request more information or suggest alternatives

## Notes

- Before each scan, use `robot_look()` to check the current direction first, to avoid redundant rotation
- Prefer extending search in directions with "doors", "corridors", or "openings"
- If `task_status` returns `status: "failed"`, analyze `error` / `s2_interpretation` before deciding whether to retry
- `scan` defaults to 0°, 90°, 180°, 270°; if one direction looks more promising, pass `angles=[angle]` to scan only that direction
