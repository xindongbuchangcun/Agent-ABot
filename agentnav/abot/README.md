# AgentNav on ABot POI Goal

This backend keeps ABot's official simulator, motion conversion, termination,
collision handling, and SR/SPL/NE evaluation. AgentNav/nanobot provides the
high-level semantic tool loop. A Metric3D-based Harness validates front-image
pixels and a persistent S1-style executor performs short local steps.
`skillgraph.py` routes the `find_poi`, `assess_scan`, and `verify_arrival` meta
skills to phase-specific Markdown guidance and validates terminal actions and
mode transitions before the agent accepts them.

Policy-visible input is restricted to POI name, one current front RGB, step,
mode, and coordinate-free episode memory. Ground-truth target coordinates,
distance, occupancy maps, and renderer depth remain evaluator-only and are not
serialized into VLM messages.

ABot `max_steps`, `TERMINATE`, collision termination, a cross-task stall guard,
or a system error own the episode lifecycle. The stall guard stops after
`max_no_progress_steps` (default 24) environment steps without reaching a new
XY region: the current position must be at least `progress_radius_m` (default
0.5 m) from every saved region representative to reset the counter. Turns,
new task IDs, and revisiting old regions do not reset it. Increase this budget
for tasks requiring prolonged backtracking. A pending final visual verification
is allowed before stopping. The AgentNav evaluator records guard stops as
`status=stalled`, `success=false`, and `spl=0`, even though ABot's interface uses
`arrive=true` as its only stop signal. Use the AgentNav evaluator adapter when
running this agent so these stops cannot be counted as successful arrivals.

The VLM has no direct turn action. When the named POI is absent from the current
RGB it requests `SCAN_360`; Python then turns in `scan_direction` (default
`left`) by at most `scan_increment_deg` (default 45 degrees), presents every
new front RGB to the VLM, and accumulates the actual completed heading change.
The scan stops immediately when a reachable target approach is selected. After
repeated rotation failures, `max_scan_rotation_failures` stops the scan instead
of retrying forever. After one full circle without identifying the POI, the planner must depth-check
fresh route anchors toward visible storefronts or open commercial areas and use
`SET_EXPLORATION_GOAL` to move closer. `SEARCH_EXHAUSTED` becomes available only
when bounded depth checks find no reachable exploration route. Fixed-direction
scanning still prevents left-right oscillation and uses actual heading change
rather than command counts.

An approach that passes visual verification but fails
the private distance gate is temporarily excluded using the same goal-region
TTL as other unconfirmed approaches; its memory retains `confirmed=true` and
does not expose the private distance.

Each nanobot tool loop is bounded by `max_tool_iterations` (default 40).
Each asynchronous VLM request, including provider retries, is bounded by
`vlm_request_timeout_s` (default 120 seconds). This timeout does not interrupt
synchronous Metric3D/GPU inference or renderer calls.

Start vLLM with `scripts/start_abot_vllm.sh`, start the existing ABot renderer,
then run `scripts/run_abot_poi.sh`. Install the adapter-specific packages from
`agentnav/abot/requirements.txt`. JSONL architecture traces are isolated under
`outputs/abot_agentnav_logs/run_<UTC timestamp>`; official task results remain
in the evaluator output directory.

The default three-V100 placement is renderer on physical GPU 0, vLLM on GPU 1,
and the ABot agent/Metric3D process on GPU 2. `run_abot_poi.sh` exposes physical
GPU 2 as process-local `cuda:0`, matching Metric3D's internal tensor creation.
