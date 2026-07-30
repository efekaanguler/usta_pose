#!/usr/bin/env python3
"""Live multi-camera depth/PCL alignment validation.

The checker deliberately does not run ICP or modify calibration. It projects
measured depth from one camera into another camera through the exact transform
chain used by the point-cloud pipeline and measures the remaining depth
disagreement.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_CAMERA_PAIRS = (
    (1, 2),
    (1, 3),
    (2, 4),
    (2, 3),
    (1, 4),
)


@dataclass(frozen=True)
class DirectionMetrics:
    source_points: int
    projected_points: int
    compared_points: int
    overlap_ratio: float
    median_mm: float
    p75_mm: float
    p90_mm: float
    inlier_ratio: float


@dataclass(frozen=True)
class PairMetrics:
    camera_a: int
    camera_b: int
    forward: DirectionMetrics
    reverse: DirectionMetrics
    status: str
    reason: str


@dataclass(frozen=True)
class AlignmentCheckResult:
    status: str
    ok: bool
    pairs: tuple
    failed_pairs: tuple
    warning_pairs: tuple
    disconnected_cameras: tuple


def _empty_direction_metrics():
    return DirectionMetrics(
        source_points=0,
        projected_points=0,
        compared_points=0,
        overlap_ratio=0.0,
        median_mm=float("inf"),
        p75_mm=float("inf"),
        p90_mm=float("inf"),
        inlier_ratio=0.0,
    )


def _extrinsics_matrix(extrinsics):
    if not extrinsics:
        raise KeyError("Missing RealSense stream extrinsics")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        extrinsics["rotation"], dtype=np.float64
    ).reshape(3, 3, order="F")
    transform[:3, 3] = np.asarray(
        extrinsics["translation"], dtype=np.float64
    ).reshape(3)
    return transform


def _apply_transform(transform, points):
    return (
        np.asarray(transform[:3, :3], dtype=np.float64) @ points.T
    ).T + np.asarray(transform[:3, 3], dtype=np.float64)


class PCLAlignmentChecker:
    """Check whether live depth clouds still agree under morning calibration."""

    def __init__(
        self,
        calibration_npz,
        camera_pairs=DEFAULT_CAMERA_PAIRS,
        sample_step=6,
        min_depth_m=0.30,
        max_depth_m=3.50,
        depth_edge_threshold_m=0.080,
        temporal_mad_threshold_m=0.030,
        occlusion_margin_m=0.080,
        min_compared_points=350,
        min_overlap_ratio=0.010,
        pass_median_mm=35.0,
        pass_p75_mm=65.0,
        pass_inlier_ratio=0.35,
        warn_median_mm=50.0,
        warn_p75_mm=100.0,
        warn_inlier_ratio=0.20,
    ):
        self.calibration_path = Path(calibration_npz)
        if not self.calibration_path.exists():
            raise FileNotFoundError(
                f"Missing multicamera calibration: {self.calibration_path}"
            )

        with np.load(self.calibration_path) as calibration:
            self.calibration = {
                key: np.asarray(calibration[key]) for key in calibration.files
            }

        self.camera_pairs = tuple(
            (int(camera_a), int(camera_b))
            for camera_a, camera_b in camera_pairs
        )
        self.sample_step = max(1, int(sample_step))
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.depth_edge_threshold_m = float(depth_edge_threshold_m)
        self.temporal_mad_threshold_m = float(temporal_mad_threshold_m)
        self.occlusion_margin_m = float(occlusion_margin_m)
        self.min_compared_points = int(min_compared_points)
        self.min_overlap_ratio = float(min_overlap_ratio)
        self.pass_median_mm = float(pass_median_mm)
        self.pass_p75_mm = float(pass_p75_mm)
        self.pass_inlier_ratio = float(pass_inlier_ratio)
        self.warn_median_mm = float(warn_median_mm)
        self.warn_p75_mm = float(warn_p75_mm)
        self.warn_inlier_ratio = float(warn_inlier_ratio)

        self.camera_to_reference = {}
        camera_ids = sorted(
            {
                camera_id
                for pair in self.camera_pairs
                for camera_id in pair
            }
        )
        for camera_id in camera_ids:
            self.camera_to_reference[camera_id] = (
                self._load_camera_to_reference(camera_id)
            )

    def _load_camera_to_reference(self, camera_id):
        explicit_key = f"T_cam{camera_id}_to_ref"
        if explicit_key in self.calibration:
            transform = np.asarray(
                self.calibration[explicit_key], dtype=np.float64
            )
            if transform.shape != (4, 4):
                raise ValueError(
                    f"{explicit_key} must be 4x4, got {transform.shape}"
                )
            return transform

        rotation_key = f"R_{camera_id}_to_ref"
        translation_key = f"t_{camera_id}_to_ref"
        if (
            rotation_key not in self.calibration
            or translation_key not in self.calibration
        ):
            raise KeyError(
                f"Calibration lacks transform for camera {camera_id}"
            )

        reference_to_camera = np.eye(4, dtype=np.float64)
        reference_to_camera[:3, :3] = np.asarray(
            self.calibration[rotation_key], dtype=np.float64
        ).reshape(3, 3)
        reference_to_camera[:3, 3] = np.asarray(
            self.calibration[translation_key], dtype=np.float64
        ).reshape(3)
        return np.linalg.inv(reference_to_camera)

    @staticmethod
    def _intrinsics(model, aligned_to_color):
        key = "color_intrinsics" if aligned_to_color else "depth_intrinsics"
        intrinsics = model.get(key)
        if not intrinsics:
            raise KeyError(f"Camera model lacks {key}")
        return intrinsics

    @staticmethod
    def _deproject(pixel_x, pixel_y, depth_m, intrinsics):
        x = (
            (pixel_x - float(intrinsics["ppx"]))
            * depth_m
            / float(intrinsics["fx"])
        )
        y = (
            (pixel_y - float(intrinsics["ppy"]))
            * depth_m
            / float(intrinsics["fy"])
        )
        return np.stack((x, y, depth_m), axis=1)

    @staticmethod
    def _project(points, intrinsics):
        z = points[:, 2]
        pixel_x = (
            float(intrinsics["fx"]) * points[:, 0] / z
            + float(intrinsics["ppx"])
        )
        pixel_y = (
            float(intrinsics["fy"]) * points[:, 1] / z
            + float(intrinsics["ppy"])
        )
        return pixel_x, pixel_y

    @staticmethod
    def _z_buffer(projected_u, projected_v, projected_z, width, height):
        flat_pixels = projected_v * width + projected_u
        depth_buffer = np.full(width * height, np.inf, dtype=np.float64)
        np.minimum.at(depth_buffer, flat_pixels, projected_z)

        occupied = np.flatnonzero(np.isfinite(depth_buffer))
        return (
            occupied % width,
            occupied // width,
            depth_buffer[occupied],
        )

    def _temporal_depth(self, raw_frames, model):
        if not raw_frames:
            raise ValueError("No depth frames supplied")

        shapes = {np.asarray(frame).shape for frame in raw_frames}
        if len(shapes) != 1:
            raise ValueError(f"Depth frame shapes differ: {sorted(shapes)}")

        depth_scale = float(model.get("depth_scale_meters_per_unit") or 0.0)
        if depth_scale <= 0.0:
            raise ValueError("Invalid RealSense depth scale")

        stack = np.stack(
            [np.asarray(frame, dtype=np.float32) for frame in raw_frames],
            axis=0,
        )
        valid_samples = stack > 0.0
        stack *= depth_scale
        stack[~valid_samples] = np.nan

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="All-NaN slice encountered",
                category=RuntimeWarning,
            )
            median_depth = np.nanmedian(stack, axis=0)
            temporal_mad = np.nanmedian(
                np.abs(stack - median_depth[None, :, :]),
                axis=0,
            )

        required_samples = max(2, int(np.ceil(len(raw_frames) * 0.60)))
        temporal_valid = (
            (np.sum(valid_samples, axis=0) >= required_samples)
            & np.isfinite(median_depth)
            & np.isfinite(temporal_mad)
            & (temporal_mad <= self.temporal_mad_threshold_m)
        )
        median_depth[~np.isfinite(median_depth)] = 0.0
        return median_depth.astype(np.float32), temporal_valid

    def _continuous_surface_mask(self, depth_m, temporal_valid):
        valid = (
            temporal_valid
            & (depth_m >= self.min_depth_m)
            & (depth_m <= self.max_depth_m)
        )
        stable = valid.copy()

        for row_shift, column_shift in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            shifted_depth = np.roll(
                depth_m, shift=(row_shift, column_shift), axis=(0, 1)
            )
            shifted_valid = np.roll(
                valid, shift=(row_shift, column_shift), axis=(0, 1)
            )
            stable &= shifted_valid
            stable &= (
                np.abs(depth_m - shifted_depth)
                <= self.depth_edge_threshold_m
            )

        stable[[0, -1], :] = False
        stable[:, [0, -1]] = False
        return stable

    def _points_in_color_camera(
        self,
        camera_id,
        depth_m,
        surface_mask,
        model,
    ):
        rows = np.arange(0, depth_m.shape[0], self.sample_step)
        columns = np.arange(0, depth_m.shape[1], self.sample_step)
        pixel_x, pixel_y = np.meshgrid(columns, rows)
        sampled_valid = surface_mask[pixel_y, pixel_x]
        sampled_depth = depth_m[pixel_y, pixel_x]

        pixel_x = pixel_x[sampled_valid].astype(np.float64)
        pixel_y = pixel_y[sampled_valid].astype(np.float64)
        sampled_depth = sampled_depth[sampled_valid].astype(np.float64)
        intrinsics = self._intrinsics(
            model, bool(model.get("aligned_to_color", False))
        )
        points = self._deproject(
            pixel_x, pixel_y, sampled_depth, intrinsics
        )

        if not model.get("aligned_to_color", False):
            points = _apply_transform(
                _extrinsics_matrix(model.get("depth_to_color_extrinsics")),
                points,
            )

        return points

    def _color_points_to_target_depth(
        self,
        points_color,
        target_model,
    ):
        if target_model.get("aligned_to_color", False):
            return points_color
        return _apply_transform(
            _extrinsics_matrix(target_model.get("color_to_depth_extrinsics")),
            points_color,
        )

    def _direction_metrics(
        self,
        source_camera,
        target_camera,
        depths,
        surface_masks,
        camera_models,
    ):
        source_model = camera_models[source_camera]
        target_model = camera_models[target_camera]
        source_points = self._points_in_color_camera(
            source_camera,
            depths[source_camera],
            surface_masks[source_camera],
            source_model,
        )
        source_count = int(len(source_points))
        if source_count == 0:
            return _empty_direction_metrics()

        points_reference = _apply_transform(
            self.camera_to_reference[source_camera],
            source_points,
        )
        reference_to_target = np.linalg.inv(
            self.camera_to_reference[target_camera]
        )
        points_target_color = _apply_transform(
            reference_to_target,
            points_reference,
        )
        points_target_depth = self._color_points_to_target_depth(
            points_target_color,
            target_model,
        )

        positive = points_target_depth[:, 2] > self.min_depth_m
        points_target_depth = points_target_depth[positive]
        if len(points_target_depth) == 0:
            return _empty_direction_metrics()

        target_intrinsics = self._intrinsics(
            target_model,
            bool(target_model.get("aligned_to_color", False)),
        )
        projected_x, projected_y = self._project(
            points_target_depth, target_intrinsics
        )
        projected_u = np.rint(projected_x).astype(np.int32)
        projected_v = np.rint(projected_y).astype(np.int32)

        target_depth = depths[target_camera]
        height, width = target_depth.shape
        inside = (
            (projected_u >= 0)
            & (projected_u < width)
            & (projected_v >= 0)
            & (projected_v < height)
        )
        if not np.any(inside):
            return DirectionMetrics(
                source_points=source_count,
                projected_points=0,
                compared_points=0,
                overlap_ratio=0.0,
                median_mm=float("inf"),
                p75_mm=float("inf"),
                p90_mm=float("inf"),
                inlier_ratio=0.0,
            )

        target_u, target_v, target_z_predicted = self._z_buffer(
            projected_u[inside],
            projected_v[inside],
            points_target_depth[inside, 2],
            width,
            height,
        )
        projected_count = int(len(target_z_predicted))
        observed_z = target_depth[target_v, target_u]
        stable_target = surface_masks[target_camera][target_v, target_u]

        # A closer target-camera surface means that the projected source point
        # is genuinely occluded. Such points are not calibration evidence.
        visible = (
            stable_target
            & (
                target_z_predicted - observed_z
                <= self.occlusion_margin_m
            )
        )
        residuals = np.abs(
            target_z_predicted[visible] - observed_z[visible]
        )
        compared_count = int(len(residuals))
        if compared_count == 0:
            return DirectionMetrics(
                source_points=source_count,
                projected_points=projected_count,
                compared_points=0,
                overlap_ratio=0.0,
                median_mm=float("inf"),
                p75_mm=float("inf"),
                p90_mm=float("inf"),
                inlier_ratio=0.0,
            )

        observed_for_inliers = observed_z[visible]
        inlier_threshold = np.maximum(
            self.pass_median_mm / 1000.0,
            0.02 * observed_for_inliers,
        )
        return DirectionMetrics(
            source_points=source_count,
            projected_points=projected_count,
            compared_points=compared_count,
            overlap_ratio=compared_count / max(source_count, 1),
            median_mm=float(np.median(residuals) * 1000.0),
            p75_mm=float(np.percentile(residuals, 75) * 1000.0),
            p90_mm=float(np.percentile(residuals, 90) * 1000.0),
            inlier_ratio=float(np.mean(residuals <= inlier_threshold)),
        )

    def _grade_pair(self, forward, reverse):
        directions = (forward, reverse)
        if any(
            metrics.compared_points < self.min_compared_points
            or metrics.overlap_ratio < self.min_overlap_ratio
            for metrics in directions
        ):
            return "FAIL", "insufficient common depth support"

        median_mm = max(metrics.median_mm for metrics in directions)
        p75_mm = max(metrics.p75_mm for metrics in directions)
        inlier_ratio = min(metrics.inlier_ratio for metrics in directions)

        if (
            median_mm <= self.pass_median_mm
            and p75_mm <= self.pass_p75_mm
            and inlier_ratio >= self.pass_inlier_ratio
        ):
            return "PASS", "depth surfaces agree"

        if (
            median_mm <= self.warn_median_mm
            and p75_mm <= self.warn_p75_mm
            and inlier_ratio >= self.warn_inlier_ratio
        ):
            return "WARN", "usable but close to the tolerance boundary"

        return (
            "FAIL",
            (
                f"depth mismatch: median={median_mm:.1f}mm, "
                f"p75={p75_mm:.1f}mm, inlier={inlier_ratio:.0%}"
            ),
        )

    @staticmethod
    def _disconnected_cameras(camera_ids, accepted_pairs):
        camera_ids = set(camera_ids)
        if not camera_ids:
            return tuple()

        visited = {min(camera_ids)}
        changed = True
        while changed:
            changed = False
            for camera_a, camera_b in accepted_pairs:
                if camera_a in visited and camera_b not in visited:
                    visited.add(camera_b)
                    changed = True
                elif camera_b in visited and camera_a not in visited:
                    visited.add(camera_a)
                    changed = True
        return tuple(sorted(camera_ids - visited))

    def check(self, depth_frames_by_camera, camera_models):
        required_camera_ids = sorted(
            {
                camera_id
                for pair in self.camera_pairs
                for camera_id in pair
            }
        )
        missing_frames = [
            camera_id
            for camera_id in required_camera_ids
            if not depth_frames_by_camera.get(camera_id)
        ]
        missing_models = [
            camera_id
            for camera_id in required_camera_ids
            if camera_id not in camera_models
        ]
        if missing_frames or missing_models:
            missing = sorted(set(missing_frames + missing_models))
            raise RuntimeError(
                f"Missing live depth data/model for cameras: {missing}"
            )

        depths = {}
        surface_masks = {}
        for camera_id in required_camera_ids:
            depth_m, temporal_valid = self._temporal_depth(
                depth_frames_by_camera[camera_id],
                camera_models[camera_id],
            )
            depths[camera_id] = depth_m
            surface_masks[camera_id] = self._continuous_surface_mask(
                depth_m, temporal_valid
            )

        pair_results = []
        for camera_a, camera_b in self.camera_pairs:
            forward = self._direction_metrics(
                camera_a,
                camera_b,
                depths,
                surface_masks,
                camera_models,
            )
            reverse = self._direction_metrics(
                camera_b,
                camera_a,
                depths,
                surface_masks,
                camera_models,
            )
            status, reason = self._grade_pair(forward, reverse)
            pair_results.append(
                PairMetrics(
                    camera_a=camera_a,
                    camera_b=camera_b,
                    forward=forward,
                    reverse=reverse,
                    status=status,
                    reason=reason,
                )
            )

        accepted_pairs = [
            (pair.camera_a, pair.camera_b)
            for pair in pair_results
            if pair.status != "FAIL"
        ]
        disconnected = self._disconnected_cameras(
            required_camera_ids, accepted_pairs
        )
        failed_pairs = tuple(
            (pair.camera_a, pair.camera_b)
            for pair in pair_results
            if pair.status == "FAIL"
        )
        warning_pairs = tuple(
            (pair.camera_a, pair.camera_b)
            for pair in pair_results
            if pair.status == "WARN"
        )

        max_tolerated_failed_edges = 1
        if disconnected or len(failed_pairs) > max_tolerated_failed_edges:
            status = "FAIL"
        elif failed_pairs or warning_pairs:
            status = "WARN"
        else:
            status = "PASS"

        return AlignmentCheckResult(
            status=status,
            ok=status != "FAIL",
            pairs=tuple(pair_results),
            failed_pairs=failed_pairs,
            warning_pairs=warning_pairs,
            disconnected_cameras=disconnected,
        )

    @staticmethod
    def print_report(result):
        colors = {
            "PASS": "\033[1;32m",
            "WARN": "\033[1;33m",
            "FAIL": "\033[1;31m",
        }
        reset = "\033[0m"
        print("\nDepth/PCL camera-pair consistency:")
        for pair in result.pairs:
            forward = pair.forward
            reverse = pair.reverse
            color = colors[pair.status]
            print(
                f"  {color}{pair.status:4s}{reset} "
                f"cam{pair.camera_a}<->cam{pair.camera_b} | "
                f"median {forward.median_mm:.1f}/{reverse.median_mm:.1f}mm | "
                f"p75 {forward.p75_mm:.1f}/{reverse.p75_mm:.1f}mm | "
                f"inlier {forward.inlier_ratio:.0%}/{reverse.inlier_ratio:.0%} | "
                f"support {forward.compared_points}/{reverse.compared_points}"
            )

        color = colors[result.status]
        print(f"\n{color}{'=' * 72}")
        if result.status == "PASS":
            print("PCL ALIGNMENT OK: camera depth surfaces agree.")
        elif result.status == "WARN":
            print(
                "PCL ALIGNMENT WARNING: usable, but recheck camera stability "
                "and scene motion."
            )
        else:
            print(
                "PCL ALIGNMENT FAILED: calibration/camera placement is not "
                "safe for recording."
            )
        print(f"{'=' * 72}{reset}")
        if result.failed_pairs:
            print(f"Failed pairs: {list(result.failed_pairs)}")
        if result.warning_pairs:
            print(f"Warning pairs: {list(result.warning_pairs)}")
        if result.disconnected_cameras:
            print(
                "Cameras disconnected from the accepted calibration graph: "
                f"{list(result.disconnected_cameras)}"
            )
