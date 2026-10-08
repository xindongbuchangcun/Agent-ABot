---
name: locate
description: Ground the named POI and fresh approach pixels in the current front RGB before depth querying.
---
# Locate

Identify the **named** POI from current-image text, logo, storefront, entrance, and their spatial relationship. A readable neighboring sign is not evidence that its storefront is the target. Distinguish a clear mismatch from unreadable or cropped text; uncertainty calls for a better view or a different pixel, not a confident wrong-store claim.

When the POI is visible, propose 2–8 distinct, stable range anchors on its entrance, lower facade, or nearby approach ground. Describe which visible storefront each pixel belongs to. Pixels are range/semantic anchors, not final robot footprints. Use displayed-image coordinates within the tool bounds; Python maps them to the sensor image. Call `QUERY_DEPTH`, compare its estimates, and use `SET_NAVIGATION_GOAL` only with a returned `reachable=true` candidate ID and the named POI as `semantic_anchor`.

After movement, a scan view, midpoint, or failed approach, inspect the new RGB and choose pixels afresh. An old numeric `(u, v)` can occur again by coincidence, but never copy old coordinates as visual evidence. If an anchor is rejected, try a materially different region associated with the visible POI; do not make tiny shifts around the same failed patch. `QUERY_DEPTH` and route safety are estimates; leave stop margins and obstacle decisions to Python.
