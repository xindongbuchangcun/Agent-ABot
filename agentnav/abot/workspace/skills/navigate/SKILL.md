---
name: navigate
description: Choose semantic POI goals and verify local task outcomes in the ABot navigation state machine.
---
# Navigate

Skill graph role: `find_poi` uses this skill in `PLANNING` and `RECOVERY`; `verify_arrival` uses it in `VERIFYING`. A reachable current-frame candidate leads to `SET_NAVIGATION_GOAL` and a persistent `EXECUTING` task. A reached local goal leads to `VERIFYING`; a failed task leads to `RECOVERY`.

Use the named POI, the attached **current front RGB**, leak-safe state, and tool results. You choose a semantic goal; Python owns depth estimation, route scoring, short-step motion, collision checks, task status, and turn angles. Never infer target coordinates or success from hidden simulator state.

- In `PLANNING` or `RECOVERY`, first decide whether the named POI is visually grounded in this image. If visible, follow `locate`: query fresh approach pixels and select a `reachable=true` candidate with `SET_NAVIGATION_GOAL`. If absent, follow `explore`.
- `EXECUTING` is a persistent Python task. Do not choose its individual movement steps or poll task status with VLM tools. A midpoint is staged progress, not arrival; inspect its new image and choose a fresh continuation.
- A navigation `GOAL_REACHED` means the selected local point was reached. In `VERIFYING`, call `VERIFY_POI` exactly once. Set `confirmed=true` only with current-image evidence of the named POI, and provide a fresh approach pixel when visible. Python alone checks the private arrival distance and may return to planning.
- After `BLOCKED`, `STUCK`, `TIMEOUT`, `INVALID_GOAL`, or failed verification, inspect the new image before replanning. A failed route does not prove the POI is absent. Change the approach or viewpoint; do not resubmit an equivalent blocked route. Treat `SYSTEM_ERROR` as infrastructure failure.

End a tool-based planning cycle with exactly one terminal tool that is actually offered. After a terminal tool succeeds, stop calling tools. A completed scan uses the separate JSON route-assessment step described in `explore`; do not invent tool calls for that step.
