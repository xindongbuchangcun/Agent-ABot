"""Depth, pixel geometry, safety, and query validation for ABot POI navigation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from agentnav.abot.depth import ObservationDepthCache
from agentnav.abot.geometry import pixel_to_local_goal
from agentnav.abot.observation import front_rgb
from agentnav.abot.types import CameraIntrinsics, PixelMeasurement, PixelProposal


class PixelHarness:
    """Translate visual anchors into safe local goals without policy decisions."""

    def __init__(
        self,
        depth_cache: ObservationDepthCache,
        camera: CameraIntrinsics,
        stop_margin_m: float = 0.35,
        robot_radius_m: float = 0.25,
        robot_height_m: float = 1.2,
        min_obstacle_height_m: float = 0.08,
        depth_safety_margin_m: float = 0.10,
        depth_corridor_lookahead_m: float = 0.70,
        depth_stride_pixels: int = 4,
        depth_min_blocking_points: int = 4,
        depth_min_blocking_columns: int = 2,
        depth_min_corridor_points: int = 8,
        depth_patch_size: int = 5,
        depth_valid_ratio_threshold: float = 0.6,
        max_reliable_depth_m: float = 30.0,
        query_budget: int = 8,
        duplicate_query_budget: int = 1,
        failed_pixel_radius: int = 24,
        max_validation_distance_m: float = 1.0,
    ) -> None:
        self.depth_cache = depth_cache
        self.camera = camera
        self.stop_margin_m = float(stop_margin_m)
        self.robot_radius_m = float(robot_radius_m)
        self.robot_height_m = float(robot_height_m)
        self.min_obstacle_height_m = float(min_obstacle_height_m)
        self.depth_safety_margin_m = float(depth_safety_margin_m)
        self.depth_corridor_lookahead_m = max(
            0.0, float(depth_corridor_lookahead_m)
        )
        self.depth_stride_pixels = max(1, int(depth_stride_pixels))
        self.depth_min_blocking_points = max(1, int(depth_min_blocking_points))
        self.depth_min_blocking_columns = max(1, int(depth_min_blocking_columns))
        self.depth_min_corridor_points = max(1, int(depth_min_corridor_points))
        self.depth_patch_radius = max(0, int(depth_patch_size) // 2)
        self.depth_valid_ratio_threshold = float(depth_valid_ratio_threshold)
        self.max_reliable_depth_m = float(max_reliable_depth_m)
        self.query_budget = max(1, int(query_budget))
        self.duplicate_query_budget = max(0, int(duplicate_query_budget))
        self.failed_pixel_radius = max(0, int(failed_pixel_radius))
        self.max_validation_distance_m = float(max_validation_distance_m)
        self.reset()

    def reset(self) -> None:
        self.depth_cache.reset()
        self._query_count: dict[int, int] = defaultdict(int)
        self._pixel_counts: dict[tuple[int, int, int], int] = defaultdict(int)
        self._measurements: dict[tuple[int, int, int], PixelMeasurement] = {}
        self._failed_pixels: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self._ground_plane_cache_depth: np.ndarray | None = None
        self._ground_plane_cache_model: dict[str, float] | None = None
        self._ground_plane_cache_debug: dict[str, Any] | None = None
        self._last_reliable_ground_scale: float | None = None

    @property
    def total_query_count(self) -> int:
        return sum(self._query_count.values())

    def remaining_query_budget(self, frame_id: int) -> int:
        return max(0, self.query_budget - self._query_count[int(frame_id)])

    def query_budget_exhausted(self, frame_id: int) -> bool:
        return self.remaining_query_budget(frame_id) == 0

    def invalidate_frame_queries(self, frame_id: int) -> None:
        """Old image pixels are never carried into a new navigation task."""
        for key in [key for key in self._measurements if key[0] == int(frame_id)]:
            self._measurements.pop(key, None)
        self._failed_pixels.pop(int(frame_id), None)

    def normalize_proposals(self, raw_points: Iterable[dict[str, Any]]) -> list[PixelProposal]:
        proposals = []
        seen = set()
        for raw in list(raw_points)[: self.query_budget]:
            try:
                view = str(raw.get("view", "front")).lower()
                u = int(round(float(raw["u"])))
                v = int(round(float(raw["v"])))
            except (KeyError, TypeError, ValueError):
                continue
            if view != "front" or not (0 <= u < self.camera.width and 0 <= v < self.camera.height):
                continue
            key = (u, v)
            if key in seen:
                continue
            seen.add(key)
            proposals.append(
                PixelProposal(
                    candidate_id=f"P{len(proposals)}",
                    u=u,
                    v=v,
                    reason=str(raw.get("reason", ""))[:300],
                )
            )
        return proposals

    def query_candidates(
        self,
        observation: Any,
        proposals: Iterable[PixelProposal],
    ) -> list[PixelMeasurement]:
        frame_id = int(observation.step_count)
        proposals = list(proposals)
        remaining = max(0, self.query_budget - self._query_count[frame_id])
        proposals = proposals[:remaining]
        if not proposals:
            return []
        prediction = None
        target_depth = None
        target_ground_debug: dict[str, Any] = {}
        failed_regions = tuple(self._failed_pixels[frame_id])
        results = []
        for proposal in proposals:
            key = (frame_id, proposal.u, proposal.v)
            if prediction is None:
                prediction = self.depth_cache.get(frame_id, front_rgb(observation))
                ground_scale, target_ground_debug = self.estimate_ground_scale(
                    prediction.depth
                )
                target_depth = (
                    prediction.depth * ground_scale
                    if ground_scale is not None
                    else prediction.depth
                )
            count = self._pixel_counts[key]
            if count > self.duplicate_query_budget:
                continue
            self._pixel_counts[key] += 1
            self._query_count[frame_id] += 1
            # Image-corner depth patches lack surrounding context and can
            # turn a mislocated storefront sign into a spurious near goal.
            if (
                (proposal.u <= 1 or proposal.u >= self.camera.width - 2)
                and (proposal.v <= 1 or proposal.v >= self.camera.height - 2)
            ):
                measurement = PixelMeasurement(
                    proposal=proposal,
                    depth_m=None,
                    depth_mad_m=None,
                    valid_depth_ratio=0.0,
                    local_goal=np.zeros(2, dtype=np.float64),
                    depth_reliable=False,
                    corridor_safe=False,
                    safety_debug={"reason": "image_corner_anchor_unreliable"},
                )
                self._measurements[key] = measurement
                self._failed_pixels[frame_id].append((proposal.u, proposal.v))
                results.append(measurement)
                continue
            if any(
                (proposal.u - failed_u) ** 2 + (proposal.v - failed_v) ** 2
                <= self.failed_pixel_radius**2
                for failed_u, failed_v in failed_regions
            ):
                measurement = PixelMeasurement(
                    proposal=proposal,
                    depth_m=None,
                    depth_mad_m=None,
                    valid_depth_ratio=0.0,
                    local_goal=np.zeros(2, dtype=np.float64),
                    depth_reliable=False,
                    corridor_safe=False,
                    safety_debug={"reason": "already_failed_pixel_region"},
                )
                self._measurements[key] = measurement
                results.append(measurement)
                continue
            if key in self._measurements:
                results.append(self._measurements[key])
                continue
            depth, mad, valid_ratio = self.probe_depth(
                target_depth, proposal.u, proposal.v, self.depth_patch_radius
            )
            stable_patch = bool(
                depth is not None
                and valid_ratio >= self.depth_valid_ratio_threshold
                and mad is not None
                and mad <= max(0.35, depth * 0.2)
            )
            far_bearing_fallback = bool(
                stable_patch and depth is not None and depth >= self.max_reliable_depth_m
            )
            reliable = bool(
                stable_patch
                and depth is not None
                and (0.25 < depth < self.max_reliable_depth_m or far_bearing_fallback)
            )
            projection_depth = (
                min(float(depth), 12.0)
                if far_bearing_fallback and depth is not None
                else depth
            )
            local_goal = (
                pixel_to_local_goal(
                    proposal.u,
                    proposal.v,
                    projection_depth,
                    self.camera,
                    self.stop_margin_m,
                )
                if reliable and projection_depth is not None
                else np.zeros(2, dtype=np.float64)
            )
            if depth is None:
                safe, debug = False, {"reason": "invalid_depth"}
            elif depth <= 0.25:
                safe, debug = False, {"reason": "depth_below_reliable_range"}
            elif far_bearing_fallback:
                safe, debug = None, {
                    "reason": "far_depth_bearing_fallback",
                    "method": "bounded_visual_bearing_goal",
                    "projection_depth_m": float(projection_depth),
                }
            elif valid_ratio < self.depth_valid_ratio_threshold:
                safe, debug = False, {"reason": "insufficient_valid_depth"}
            elif mad is None or mad > max(0.35, depth * 0.2):
                safe, debug = False, {"reason": "unstable_depth"}
            else:
                # A target pixel establishes a semantic/range goal only. Route
                # safety is evaluated from the newest full depth map before
                # every short executor step.
                safe, debug = None, {
                    "reason": "route_check_deferred_to_executor",
                    "method": "target_pixel_depth_only",
                }
            debug["ground_calibration"] = target_ground_debug
            measurement = PixelMeasurement(
                proposal=proposal,
                depth_m=depth,
                depth_mad_m=mad,
                valid_depth_ratio=valid_ratio,
                local_goal=local_goal,
                depth_reliable=reliable,
                corridor_safe=safe,
                safety_debug=debug,
            )
            self._measurements[key] = measurement
            if not measurement.reachable:
                self._failed_pixels[frame_id].append((proposal.u, proposal.v))
            results.append(measurement)
        return results

    def measurement_for_pixel(
        self,
        observation: Any,
        u: int,
        v: int,
        reason: str = "navigation anchor",
    ) -> PixelMeasurement | None:
        proposal = PixelProposal("selected", int(u), int(v), reason)
        results = self.query_candidates(observation, [proposal])
        return results[0] if results else None

    def current_depth(self, observation: Any) -> np.ndarray:
        return self.depth_cache.get(
            int(observation.step_count), front_rgb(observation)
        ).depth

    def _estimate_ground_plane(
        self,
        depth: np.ndarray | None,
    ) -> tuple[dict[str, float] | None, dict[str, Any]]:
        """Fit a scale-aware 3D floor plane from lower-image depth points."""
        if (
            depth is not None
            and self._ground_plane_cache_depth is depth
            and self._ground_plane_cache_debug is not None
        ):
            model = self._ground_plane_cache_model
            return (None if model is None else model.copy()), dict(
                self._ground_plane_cache_debug
            )

        debug: dict[str, Any] = {
            "ground_model": "robust_3d_plane",
            "ground_plane_found": False,
            "ground_inlier_points": 0,
            "ground_support_ratio": 0.0,
            "ground_column_coverage_ratio": 0.0,
            "ground_row_span_ratio": 0.0,
            "ground_bottom_inlier_points": 0,
            "ground_tilt_deg": None,
            "ground_plane_coefficients_raw": None,
            "depth_scale_correction": None,
        }
        model: dict[str, float] | None = None
        if depth is None or depth.ndim != 2:
            return None, debug

        sampled = np.asarray(
            depth[:: self.depth_stride_pixels, :: self.depth_stride_pixels],
            dtype=np.float64,
        )
        rows, cols = np.indices(sampled.shape, dtype=np.float64)
        u = cols * self.depth_stride_pixels
        v = rows * self.depth_stride_pixels
        valid = np.isfinite(sampled) & (sampled > 0.05) & (sampled < 100.0)
        lower = valid & (
            v >= self.camera.cy + max(24.0, 0.06 * self.camera.height)
        )
        if not np.any(lower):
            self._cache_ground_plane(depth, None, debug)
            return None, debug

        z = sampled[lower]
        x = (u[lower] - self.camera.cx) * z / self.camera.fx
        y = (v[lower] - self.camera.cy) * z / self.camera.fy
        column_indices = cols[lower].astype(np.int64)
        pixel_rows = v[lower]
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (y > 0.02)
        x, y, z, column_indices, pixel_rows = (
            x[finite],
            y[finite],
            z[finite],
            column_indices[finite],
            pixel_rows[finite],
        )
        min_ground_points = max(40, self.depth_min_corridor_points * 2)
        if x.size < min_ground_points:
            self._cache_ground_plane(depth, None, debug)
            return None, debug

        design = np.column_stack((x, z, np.ones_like(x)))
        rng = np.random.default_rng(0)
        best_inliers: np.ndarray | None = None
        best_score: tuple[int, int, int] = (-1, -1, -1)
        iterations = min(160, max(64, int(x.size // 60)))
        for _ in range(iterations):
            indices = rng.choice(x.size, size=3, replace=False)
            sample_design = design[indices]
            if abs(float(np.linalg.det(sample_design))) < 1e-7:
                continue
            try:
                a, b, c = np.linalg.solve(sample_design, y[indices])
            except np.linalg.LinAlgError:
                continue
            norm = float(np.sqrt(a * a + b * b + 1.0))
            vertical_alignment = 1.0 / norm
            plane_distance = float(c / norm)
            if vertical_alignment < 0.55 or plane_distance <= 0.0:
                continue
            scale = self.camera.extrinsic_height / plane_distance
            if not 0.4 <= scale <= 1.85:
                continue
            tolerance_raw = max(0.025 / scale, 0.04 * plane_distance)
            residual = np.abs(design @ np.array([a, b, c]) - y) / norm
            inliers = residual <= tolerance_raw
            count = int(np.count_nonzero(inliers))
            coverage = int(np.unique(column_indices[inliers]).size)
            bottom_count = int(
                np.count_nonzero(inliers & (pixel_rows >= 0.8 * self.camera.height))
            )
            score = (count + 2 * bottom_count, count, coverage)
            if score > best_score:
                best_score = score
                best_inliers = inliers

        if best_inliers is None or np.count_nonzero(best_inliers) < min_ground_points:
            self._cache_ground_plane(depth, None, debug)
            return None, debug

        inliers = best_inliers
        for _ in range(2):
            try:
                a, b, c = np.linalg.lstsq(design[inliers], y[inliers], rcond=None)[0]
            except np.linalg.LinAlgError:
                self._cache_ground_plane(depth, None, debug)
                return None, debug
            norm = float(np.sqrt(a * a + b * b + 1.0))
            plane_distance = float(c / norm)
            if plane_distance <= 0.0:
                self._cache_ground_plane(depth, None, debug)
                return None, debug
            scale = self.camera.extrinsic_height / plane_distance
            if not 0.4 <= scale <= 1.85:
                self._cache_ground_plane(depth, None, debug)
                return None, debug
            tolerance_raw = max(0.025 / scale, 0.04 * plane_distance)
            residual = np.abs(design @ np.array([a, b, c]) - y) / norm
            inliers = residual <= tolerance_raw

        ground_inliers = int(np.count_nonzero(inliers))
        coverage = int(np.unique(column_indices[inliers]).size)
        total_columns = max(1, int(np.unique(column_indices).size))
        support_ratio = ground_inliers / float(x.size)
        coverage_ratio = coverage / float(total_columns)
        inlier_rows = pixel_rows[inliers]
        lower_start = self.camera.cy + max(24.0, 0.06 * self.camera.height)
        row_span_ratio = (
            float(np.ptp(inlier_rows))
            / max(1.0, self.camera.height - lower_start)
            if inlier_rows.size
            else 0.0
        )
        bottom_inliers = int(
            np.count_nonzero(inlier_rows >= 0.8 * self.camera.height)
        )
        vertical_alignment = 1.0 / norm
        debug.update(
            {
                "ground_inlier_points": ground_inliers,
                "ground_support_ratio": round(support_ratio, 4),
                "ground_column_coverage_ratio": round(coverage_ratio, 4),
                "ground_row_span_ratio": round(row_span_ratio, 4),
                "ground_bottom_inlier_points": bottom_inliers,
                "ground_tilt_deg": round(
                    float(np.degrees(np.arccos(np.clip(vertical_alignment, 0.0, 1.0)))),
                    3,
                ),
            }
        )
        if (
            ground_inliers < min_ground_points
            or support_ratio < 0.08
            or coverage_ratio < 0.20
            or row_span_ratio < 0.35
            or bottom_inliers < max(20, min_ground_points // 2)
        ):
            self._cache_ground_plane(depth, None, debug)
            return None, debug

        model = {
            "a": float(a),
            "b": float(b),
            "c": float(c),
            "normalizer": norm,
            "scale": float(scale),
        }
        debug.update(
            {
                "ground_plane_found": True,
                "ground_plane_coefficients_raw": [
                    round(float(a / norm), 6),
                    round(float(-1.0 / norm), 6),
                    round(float(b / norm), 6),
                    round(float(c / norm), 6),
                ],
                "depth_scale_correction": round(float(scale), 4),
            }
        )
        self._cache_ground_plane(depth, model, debug)
        return model.copy(), dict(debug)

    def _cache_ground_plane(
        self,
        depth: np.ndarray,
        model: dict[str, float] | None,
        debug: dict[str, Any],
    ) -> None:
        self._ground_plane_cache_depth = depth
        self._ground_plane_cache_model = None if model is None else model.copy()
        self._ground_plane_cache_debug = dict(debug)

    def estimate_ground_scale(
        self,
        depth: np.ndarray | None,
    ) -> tuple[float | None, dict[str, Any]]:
        model, debug = self._estimate_ground_plane(depth)
        if model is not None:
            self._last_reliable_ground_scale = float(model["scale"])
        return (None if model is None else model["scale"]), debug

    def depth_corridor_is_safe(
        self,
        depth: np.ndarray | None,
        waypoint: np.ndarray,
        max_lookahead_m: float | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        travel_m = float(np.linalg.norm(waypoint))
        debug: dict[str, Any] = {
            "method": "estimated_depth_backprojection_corridor",
            "view": "front",
            "travel_m": round(travel_m, 3),
            "occupancy_map_used": False,
            "gt_depth_used": False,
        }
        if travel_m <= 0.0:
            debug["reason"] = "zero_travel"
            return False, debug
        if depth is None or depth.ndim != 2:
            debug["reason"] = "missing_depth"
            return False, debug

        sampled = np.asarray(
            depth[:: self.depth_stride_pixels, :: self.depth_stride_pixels],
            dtype=np.float64,
        )
        rows, cols = np.indices(sampled.shape, dtype=np.float64)
        u = cols * self.depth_stride_pixels
        v = rows * self.depth_stride_pixels
        valid = np.isfinite(sampled) & (sampled > 0.05) & (sampled < 100.0)
        if not np.any(valid):
            debug["reason"] = "no_valid_depth"
            return False, debug

        z_camera_raw = sampled[valid]
        x_camera_raw = (
            (u[valid] - self.camera.cx) * z_camera_raw / self.camera.fx
        )
        y_camera_raw = (
            (v[valid] - self.camera.cy) * z_camera_raw / self.camera.fy
        )

        # Fit the floor in 3D, then use its camera distance to correct the
        # monocular scale. Obstacle height is measured normal to this plane.
        ground_model, ground_debug = self._estimate_ground_plane(depth)
        debug.update(ground_debug)
        if ground_model is None:
            # Floor visibility can disappear briefly between adjacent frames.
            # Reuse only the latest episode-local scale and assume a level
            # floor; every short executor step obtains a new depth map.
            if self._last_reliable_ground_scale is None:
                debug["reason"] = "insufficient_ground_plane_support"
                return False, debug
            ground_scale = self._last_reliable_ground_scale
            ground_model = {
                "a": 0.0,
                "b": 0.0,
                "c": self.camera.extrinsic_height / ground_scale,
                "normalizer": 1.0,
                "scale": ground_scale,
            }
            debug.update(
                {
                    "ground_model": "cached_scale_horizontal_plane",
                    "ground_scale_reused": True,
                    "depth_scale_correction": round(ground_scale, 4),
                }
            )
        else:
            self._last_reliable_ground_scale = float(ground_model["scale"])

        ground_scale = ground_model["scale"]
        z_camera = z_camera_raw * ground_scale
        x_camera = x_camera_raw * ground_scale
        # Front camera: camera +z is robot forward; camera +x is robot right.
        forward = z_camera
        left = -x_camera
        direction = np.asarray(waypoint, dtype=np.float64) / travel_m
        configured_lookahead_m = self.depth_corridor_lookahead_m
        if max_lookahead_m is not None:
            configured_lookahead_m = min(
                configured_lookahead_m,
                max(travel_m, float(max_lookahead_m)),
            )
        corridor_lookahead_m = max(travel_m, configured_lookahead_m)
        longitudinal = forward * direction[0] + left * direction[1]
        lateral = np.abs(direction[0] * left - direction[1] * forward)
        corridor_radius = self.robot_radius_m + self.depth_safety_margin_m
        corridor = (
            (longitudinal > 0.05)
            & (
                longitudinal
                <= corridor_lookahead_m + self.depth_safety_margin_m
            )
            & (lateral <= corridor_radius)
        )
        corridor_points = int(np.count_nonzero(corridor))
        # Near a local goal, obstacle checking stops at the endpoint, but
        # floor/depth evidence must still reach the camera's normal lookahead.
        # Otherwise a clear final few centimetres are declared unseen.
        evidence_lookahead_m = max(
            corridor_lookahead_m, self.depth_corridor_lookahead_m
        )
        evidence_corridor = (
            (longitudinal > 0.05)
            & (longitudinal <= evidence_lookahead_m + self.depth_safety_margin_m)
            & (lateral <= corridor_radius)
        )
        evidence_points = int(np.count_nonzero(evidence_corridor))
        point_height = (
            ground_model["a"] * x_camera_raw
            - y_camera_raw
            + ground_model["b"] * z_camera_raw
            + ground_model["c"]
        ) / ground_model["normalizer"] * ground_scale
        obstacle_height = (
            (point_height >= self.min_obstacle_height_m)
            & (point_height <= self.robot_height_m)
        )
        # A dominant nearby facade can occasionally be absorbed into the
        # fitted floor plane, making its plane-relative height nearly zero.
        # Independently protect the robot's near-field body band. At this
        # range, true floor pixels project below the band and remain excluded.
        valid_rows = v[valid]
        near_body_band = (
            (valid_rows >= self.camera.cy - 0.35 * self.camera.height)
            & (valid_rows <= self.camera.cy + 0.30 * self.camera.height)
        )
        # A supported 3D floor fit already classifies these pixels by height.
        # Applying the image-row fallback as well counts nearby floor as an
        # obstacle; reserve it for frames using only a cached scale.
        near_body_blocking = (
            corridor & near_body_band
            & (debug.get("ground_model") == "cached_scale_horizontal_plane")
        )
        blocking = (corridor & obstacle_height) | near_body_blocking
        blocking_points = int(np.count_nonzero(blocking))
        blocking_columns = int(
            np.unique(cols[valid][blocking].astype(np.int64)).size
        )
        blocking_ratio = (
            blocking_points / float(corridor_points) if corridor_points else 0.0
        )
        nearest_blocking_m = (
            float(np.min(longitudinal[blocking])) if blocking_points else None
        )
        obstacle_risk = min(
            1.0,
            max(
                blocking_points / float(self.depth_min_blocking_points),
                blocking_columns / float(self.depth_min_blocking_columns),
            ),
        )
        debug.update(
            {
                "corridor_points": corridor_points,
                "evidence_points": evidence_points,
                "evidence_lookahead_m": round(evidence_lookahead_m, 3),
                "blocking_points": blocking_points,
                "blocking_columns": blocking_columns,
                "near_body_blocking_points": int(np.count_nonzero(near_body_blocking)),
                "blocking_ratio": round(blocking_ratio, 4),
                "obstacle_risk": round(obstacle_risk, 4),
                "corridor_radius_m": round(corridor_radius, 3),
                "corridor_lookahead_m": round(corridor_lookahead_m, 3),
                "max_lookahead_m": (
                    None
                    if max_lookahead_m is None
                    else round(float(max_lookahead_m), 3)
                ),
                "min_required_corridor_points": self.depth_min_corridor_points,
                "blocking_threshold_points": self.depth_min_blocking_points,
                "blocking_threshold_columns": self.depth_min_blocking_columns,
            }
        )
        debug["corridor_evidence_sparse"] = (
            evidence_points < self.depth_min_corridor_points
        )
        if nearest_blocking_m is not None:
            debug["nearest_obstacle_m"] = round(nearest_blocking_m, 3)
        if (
            blocking_points >= self.depth_min_blocking_points
            and blocking_columns >= self.depth_min_blocking_columns
        ):
            debug["reason"] = "depth_obstacle"
            return False, debug
        if debug["corridor_evidence_sparse"]:
            # A fitted floor establishes height and scale, but cannot prove that
            # an unseen swept volume is obstacle-free.
            debug["reason"] = "insufficient_corridor_evidence"
            debug["obstacle_risk"] = 1.0
            return False, debug
        debug["reason"] = "clear_depth_corridor"
        return True, debug

    @staticmethod
    def probe_depth(
        depth: np.ndarray | None,
        u: int,
        v: int,
        radius: int,
    ) -> tuple[float | None, float | None, float]:
        if depth is None or depth.ndim != 2:
            return None, None, 0.0
        height, width = depth.shape
        patch = np.asarray(
            depth[
                max(0, v - radius) : min(height, v + radius + 1),
                max(0, u - radius) : min(width, u + radius + 1),
            ],
            dtype=np.float64,
        )
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        ratio = float(valid.size / max(1, patch.size))
        if valid.size == 0:
            return None, None, ratio
        median = float(np.median(valid))
        mad = float(np.median(np.abs(valid - median)))
        return median, mad, ratio

    @staticmethod
    def clip_waypoint(waypoint: np.ndarray, maximum: float) -> np.ndarray:
        waypoint = np.asarray(waypoint, dtype=np.float64)
        distance = float(np.linalg.norm(waypoint))
        if distance <= maximum or distance <= 1e-9:
            return waypoint.copy()
        return waypoint * (float(maximum) / distance)
