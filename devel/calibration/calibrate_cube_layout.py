#!/usr/bin/env python3
"""Estimate the physical AprilTag rig geometry of the calibration cube."""

from __future__ import annotations

import argparse
import heapq
import json
import math
from copy import deepcopy
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from cube_calibration import (
    create_apriltag_detector,
    detect_known_cube_tags,
    estimate_cube_pose,
    invert_transform,
    make_transform,
    robust_average_transforms,
    transform_to_vector,
    vector_to_transform,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEVEL_DIR = SCRIPT_DIR.parent
RECORD_DIR = DEVEL_DIR / "record"
RECORDINGS_DIR = RECORD_DIR / "recordings"
DEFAULT_INTRINSICS = RECORDINGS_DIR / "calib_data" / "master_intrinsics.npz"
DEFAULT_CAPTURES = (
    RECORDINGS_DIR / "calib_data" / "extrinsic" / "current"
)
DEFAULT_INPUT_LAYOUT = RECORD_DIR / "apriltag_cube_layout.json"
DEFAULT_OUTPUT_LAYOUT = RECORD_DIR / "apriltag_cube_layout_calibrated.json"


def load_intrinsics(path: Path, num_cameras: int):
    intrinsics = {}
    with np.load(path) as data:
        for camera_id in range(1, num_cameras + 1):
            intrinsics[camera_id] = (
                np.asarray(data[f"K{camera_id}"], dtype=np.float64),
                np.asarray(data[f"dist{camera_id}"], dtype=np.float64),
            )
    return intrinsics


def rigid_alignment(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (
        (source - source_center).T @ (target - target_center)
    )
    left, _singular_values, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    translation = target_center - rotation @ source_center
    return make_transform(rotation, translation)


def load_tag_models(layout_path: Path):
    with open(layout_path, "r", encoding="utf-8") as layout_file:
        raw_layout = json.load(layout_file)

    unit = str(raw_layout.get("unit", "m")).lower()
    scale = {"m": 1.0, "cm": 0.01, "mm": 0.001}.get(unit)
    if scale is None:
        raise ValueError(f"Unsupported layout unit: {unit}")

    local_points = {}
    ideal_poses = {}
    tags_by_id = {}
    for tag in raw_layout["tags"]:
        tag_id = int(tag["id"])
        corners = np.asarray(tag["corners"], dtype=np.float64) * scale
        edge_lengths = [
            np.linalg.norm(corners[(index + 1) % 4] - corners[index])
            for index in range(4)
        ]
        half_size = float(np.median(edge_lengths)) / 2.0
        canonical = np.asarray([
            [-half_size, -half_size, 0.0],
            [half_size, -half_size, 0.0],
            [half_size, half_size, 0.0],
            [-half_size, half_size, 0.0],
        ], dtype=np.float64)
        local_points[tag_id] = canonical
        ideal_poses[tag_id] = rigid_alignment(canonical, corners)
        tags_by_id[tag_id] = tag
    return raw_layout, tags_by_id, local_points, ideal_poses


def collect_tag_pair_measurements(
    captures_dir: Path,
    intrinsics,
    detector,
    local_points,
    args,
):
    measurements: Dict[Tuple[int, int], List[np.ndarray]] = {}
    validation_frames = []
    image_paths = sorted(captures_dir.rglob("camera_*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No camera_*.png images under {captures_dir}")

    for image_path in image_paths:
        camera_id = int(image_path.stem.split("_")[-1])
        if camera_id not in intrinsics:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        detections = detect_known_cube_tags(
            image,
            detector,
            local_points.keys(),
            min_decision_margin=args.min_decision_margin,
            max_hamming=args.max_hamming,
            min_tag_edge_px=args.min_tag_edge_px,
        )
        camera_matrix, distortion = intrinsics[camera_id]
        poses = {}
        accepted_detections = []
        for detection in detections:
            pose, error = estimate_cube_pose(
                local_points[detection.tag_id],
                detection.corners,
                camera_matrix,
                distortion,
            )
            if pose is None or error > args.max_single_tag_reproj_px:
                continue
            poses[detection.tag_id] = pose
            accepted_detections.append(detection)

        if len(poses) < 2:
            continue
        for first_id, second_id in combinations(sorted(poses), 2):
            first_from_second = (
                invert_transform(poses[first_id]) @ poses[second_id]
            )
            measurements.setdefault((first_id, second_id), []).append(
                first_from_second
            )
        validation_frames.append(
            (camera_id, accepted_detections, camera_matrix, distortion)
        )
    return measurements, validation_frames


def build_tag_edges(measurements, min_pair_views: int):
    edges = {}
    print("Physical tag-pair measurements:")
    for pair, transforms in sorted(measurements.items()):
        if len(transforms) < min_pair_views:
            print(f"  id{pair[0]}<->id{pair[1]}: {len(transforms)} views (skipped)")
            continue
        average, inliers, rotation_spread, translation_spread = (
            robust_average_transforms(
                transforms,
                max_rotation_deviation_deg=10.0,
                max_translation_deviation_m=0.04,
            )
        )
        inlier_count = int(np.sum(inliers))
        if inlier_count < min_pair_views:
            continue
        weight = (
            rotation_spread
            + translation_spread * 100.0
            + 1.0 / inlier_count
        )
        edges[pair] = (average, weight, inlier_count)
        print(
            f"  id{pair[0]}<->id{pair[1]}: inliers={inlier_count}, "
            f"rot_median={rotation_spread:.3f}deg, "
            f"trans_median={translation_spread * 1000.0:.2f}mm"
        )
    if not edges:
        raise RuntimeError("No sufficiently supported co-visible tag pairs.")
    return edges


def initialize_tag_poses(edges, ideal_poses, anchor_id: int):
    adjacency = {tag_id: [] for tag_id in ideal_poses}
    for (first_id, second_id), (first_from_second, weight, _count) in edges.items():
        adjacency[first_id].append((second_id, first_from_second, weight))
        adjacency[second_id].append(
            (first_id, invert_transform(first_from_second), weight)
        )

    poses = {anchor_id: ideal_poses[anchor_id]}
    distances = {tag_id: float("inf") for tag_id in ideal_poses}
    distances[anchor_id] = 0.0
    queue = [(0.0, anchor_id)]
    while queue:
        distance, source_id = heapq.heappop(queue)
        if distance > distances[source_id]:
            continue
        for target_id, source_from_target, weight in adjacency[source_id]:
            candidate_distance = distance + weight
            if candidate_distance >= distances[target_id]:
                continue
            distances[target_id] = candidate_distance
            poses[target_id] = poses[source_id] @ source_from_target
            heapq.heappush(queue, (candidate_distance, target_id))

    missing = sorted(set(ideal_poses) - set(poses))
    if missing:
        raise RuntimeError(
            f"Physical tag graph is disconnected from anchor id{anchor_id}: {missing}"
        )
    return poses


def optimize_tag_pose_graph(edges, initial_poses, anchor_id: int):
    variable_ids = sorted(tag_id for tag_id in initial_poses if tag_id != anchor_id)
    initial_parameters = np.concatenate([
        transform_to_vector(initial_poses[tag_id])
        for tag_id in variable_ids
    ])

    def unpack(parameters: np.ndarray):
        poses = {anchor_id: initial_poses[anchor_id]}
        for index, tag_id in enumerate(variable_ids):
            poses[tag_id] = vector_to_transform(
                parameters[index * 6:(index + 1) * 6]
            )
        return poses

    def residuals(parameters: np.ndarray):
        poses = unpack(parameters)
        blocks = []
        for (first_id, second_id), (measured, weight, count) in edges.items():
            predicted = invert_transform(poses[first_id]) @ poses[second_id]
            delta = invert_transform(measured) @ predicted
            confidence = math.sqrt(count) / math.sqrt(max(weight, 0.05))
            blocks.extend(
                confidence
                * Rotation.from_matrix(delta[:3, :3]).as_rotvec()
                / math.radians(0.5)
            )
            blocks.extend(confidence * delta[:3, 3] / 0.002)
        return np.asarray(blocks, dtype=np.float64)

    result = least_squares(
        residuals,
        initial_parameters,
        method="trf",
        loss="huber",
        f_scale=1.0,
        max_nfev=200,
        x_scale="jac",
    )
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError("Physical tag pose graph produced non-finite parameters.")
    return unpack(result.x), result


def validate_layout(validation_frames, tag_poses, local_points):
    errors_by_camera: Dict[int, List[float]] = {}
    for camera_id, detections, camera_matrix, distortion in validation_frames:
        usable = [
            detection
            for detection in detections
            if detection.tag_id in tag_poses
        ]
        if len(usable) < 2:
            continue
        object_points = np.concatenate([
            (
                tag_poses[detection.tag_id][:3, :3]
                @ local_points[detection.tag_id].T
                + tag_poses[detection.tag_id][:3, 3].reshape(3, 1)
            ).T
            for detection in usable
        ])
        image_points = np.concatenate([
            detection.corners for detection in usable
        ])
        _pose, error = estimate_cube_pose(
            object_points,
            image_points,
            camera_matrix,
            distortion,
        )
        if np.isfinite(error):
            errors_by_camera.setdefault(camera_id, []).append(error)

    print("Refined multi-tag layout diagnostics:")
    for camera_id, errors in sorted(errors_by_camera.items()):
        values = np.asarray(errors, dtype=np.float64)
        print(
            f"  cam{camera_id}: multi-tag frames={len(values)}, "
            f"median_mean={np.median(values):.3f}px, "
            f"P95_mean={np.percentile(values, 95):.3f}px"
        )


def write_refined_layout(
    output_path: Path,
    raw_layout,
    tags_by_id,
    local_points,
    tag_poses,
    anchor_id: int,
    captures_dir: Path,
):
    refined = deepcopy(raw_layout)
    refined["unit"] = "m"
    refined["description"] = (
        "Measured physical AprilTag rig geometry. Generated from co-visible "
        "tag pairs with fixed master intrinsics; original ideal layout is preserved."
    )
    refined["layout_refinement"] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "co-visible_single-camera_tag_pose_graph",
        "anchor_tag_id": anchor_id,
        "captures_dir": str(captures_dir),
    }
    refined_tags = []
    for tag_id in sorted(tag_poses):
        pose = tag_poses[tag_id]
        corners = (
            pose[:3, :3] @ local_points[tag_id].T
            + pose[:3, 3].reshape(3, 1)
        ).T
        tag = deepcopy(tags_by_id[tag_id])
        tag["center"] = pose[:3, 3].round(9).tolist()
        tag["normal"] = pose[:3, 2].round(9).tolist()
        tag["corners"] = corners.round(9).tolist()
        refined_tags.append(tag)
    refined["tags"] = refined_tags

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(refined, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the physical five-face AprilTag cube layout from "
            "co-visible tags before solving daily camera extrinsics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--master-intrinsics", type=Path, default=DEFAULT_INTRINSICS)
    parser.add_argument("--captures-dir", type=Path, default=DEFAULT_CAPTURES)
    parser.add_argument("--input-layout", type=Path, default=DEFAULT_INPUT_LAYOUT)
    parser.add_argument("--output-layout", type=Path, default=DEFAULT_OUTPUT_LAYOUT)
    parser.add_argument("--num-cameras", type=int, default=4)
    parser.add_argument("--anchor-tag-id", type=int, default=None)
    parser.add_argument("--apriltag-family", type=str, default="tag36h11")
    parser.add_argument("--min-pair-views", type=int, default=5)
    parser.add_argument("--min-decision-margin", type=float, default=20.0)
    parser.add_argument("--max-hamming", type=int, default=0)
    parser.add_argument("--min-tag-edge-px", type=float, default=45.0)
    parser.add_argument("--max-single-tag-reproj-px", type=float, default=2.5)
    return parser.parse_args()


def main():
    args = parse_args()
    intrinsics = load_intrinsics(args.master_intrinsics, args.num_cameras)
    raw_layout, tags_by_id, local_points, ideal_poses = load_tag_models(
        args.input_layout
    )
    front_id = raw_layout.get("faces", {}).get("front")
    anchor_id = int(
        args.anchor_tag_id
        if args.anchor_tag_id is not None
        else front_id if front_id is not None
        else sorted(local_points)[0]
    )
    if anchor_id not in local_points:
        raise ValueError(f"Anchor tag id{anchor_id} is not in the input layout.")

    detector = create_apriltag_detector(args.apriltag_family)
    measurements, validation_frames = collect_tag_pair_measurements(
        args.captures_dir,
        intrinsics,
        detector,
        local_points,
        args,
    )
    edges = build_tag_edges(measurements, args.min_pair_views)
    initial_poses = initialize_tag_poses(edges, ideal_poses, anchor_id)
    refined_poses, result = optimize_tag_pose_graph(
        edges,
        initial_poses,
        anchor_id,
    )
    validate_layout(validation_frames, refined_poses, local_points)
    write_refined_layout(
        args.output_layout,
        raw_layout,
        tags_by_id,
        local_points,
        refined_poses,
        anchor_id,
        args.captures_dir,
    )
    print(f"Optimizer: {result.message}")
    print(f"Physical cube layout written: {args.output_layout}")
    print("Run a fresh daily capture to validate this layout independently.")


if __name__ == "__main__":
    main()
