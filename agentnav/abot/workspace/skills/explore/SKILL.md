---
name: explore
description: Search and choose fresh exploratory viewpoints when the named POI is absent from the current RGB.
---
# Explore

Skill graph role: `find_poi` may request `SCAN_360` while the target is absent; after the scan, `assess_scan` may create `SET_EXPLORATION_GOAL` or report `SEARCH_EXHAUSTED`. Each exploratory move returns to a fresh image and a new POI search.

If the named POI is absent before a completed scan, call `SCAN_360`. Python rotates in one fixed direction and presents each new front image. Do not specify a physical turn direction or angle. If the POI is still visible after a blocked route, first query different current-image approach pixels; scan when no viable current-view approach remains. An old remembered target bearing may help Python reacquire a view, but it is not current visual proof.

After a full scan, the planner's route-assessment call expects **one JSON object**, not tool calls: `{"target_visible": boolean, "candidates": [{"u": integer, "v": integer, "reason": string}, ...]}`. Recheck the current image. If the named POI is visible, give 2–4 fresh approach pixels and explain their visible link to that POI. Otherwise give 2–4 fresh ground-route pixels toward open passages, storefront clusters, or places likely to reveal new views. Return an empty list only when no walkable route is visible. Do not assert that an unseen POI is behind a particular direction.

Python depth-checks the proposals, previews nearby route safety, and limits each exploratory movement before re-observation. Those checks do not guarantee the whole route is obstacle-free. `SEARCH_EXHAUSTED` is Python's bounded outcome when no reachable exploratory route remains; do not declare the POI absent from the environment merely because one scan did not show it. On the next image, look for the named POI again and choose new pixels.
