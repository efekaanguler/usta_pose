#!/usr/bin/env python3
"""AprilTag-cube based extrinsic calibration for a fixed multi-camera rig."""

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
except ImportError as exc:
    raise ImportError(
        "cube_calibration.py requires scipy. Install it with: "
        "python3 -m pip install scipy"
    ) from exc


def create_apriltag_detector(families: str):
    try:
        from pupil_apriltags import Detector

        return Detector(
            families=families,
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
        )
    except ImportError:
        pass

    try:
        from dt_apriltags import Detector

        return Detector(
            families=families,
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
        )
    except ImportError as exc:
        raise ImportError(
            "Install pupil-apriltags or dt-apriltags for cube calibration."
        ) from exc


def unit_scale(unit: str) -> float:
    normalized = unit.strip().lower()
    if normalized in ("m", "meter", "meters"):
        return 1.0
    if normalized in ("cm", "centimeter", "centimeters"):
        return 0.01
    if normalized in ("mm", "millimeter", "millimeters"):
        return 0.001
    raise ValueError(f"Unsupported cube layout unit: {unit}")


def load_cube_layout(layout_path: Path) -> Dict[int, np.ndarray]:
    with open(layout_path, "r", encoding="utf-8") as layout_file:
        raw = json.load(layout_file)

    if "tags" not in raw:
        raise KeyError(f"Cube layout must contain a 'tags' list: {layout_path}")

    default_unit = raw.get("unit", "m")
    tag_corners = {}
    for tag in raw["tags"]:
        tag_id = int(tag["id"])
        scale = unit_scale(tag.get("unit", default_unit))
        corners = np.asarray(tag["corners"], dtype=np.float64)
        if corners.shape != (4, 3):
            raise ValueError(f"Tag {tag_id} corners must have shape 4x3.")
        corners = corners * scale
        if "normal" in tag:
            expected_normal = np.asarray(tag["normal"], dtype=np.float64)
            expected_normal /= np.linalg.norm(expected_normal)
            winding_normal = np.cross(
                corners[1] - corners[0],
                corners[2] - corners[0],
            )
            winding_normal /= np.linalg.norm(winding_normal)
            if float(np.dot(expected_normal, winding_normal)) < 0.95:
                raise ValueError(
                    f"Tag {tag_id} corner winding disagrees with its face normal."
                )
        tag_corners[tag_id] = corners

    if not tag_corners:
        raise ValueError(f"No AprilTags found in cube layout: {layout_path}")
    return tag_corners


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverted = np.eye(4, dtype=np.float64)
    inverted[:3, :3] = rotation.T
    inverted[:3, 3] = -rotation.T @ translation
    return inverted


def transform_to_vector(transform: np.ndarray) -> np.ndarray:
    rotation_vector = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    return np.concatenate([rotation_vector, transform[:3, 3]])


def vector_to_transform(vector: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(np.asarray(vector[:3], dtype=np.float64)).as_matrix()
    return make_transform(rotation, vector[3:6])


def rotation_distance_degrees(first: np.ndarray, second: np.ndarray) -> float:
    delta = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def tag_edge_length_px(corners: np.ndarray) -> float:
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    lengths = [
        np.linalg.norm(points[(index + 1) % 4] - points[index])
        for index in range(4)
    ]
    return float(np.median(lengths))


@dataclass
class DetectedCubeTag:
    tag_id: int
    corners: np.ndarray
    decision_margin: float
    hamming: int
    edge_length_px: float


@dataclass
class CubeObservation:
    camera_id: int
    capture_id: str
    image_path: Path
    object_points: np.ndarray
    image_points: np.ndarray
    tag_ids: List[int]
    decision_margin_min: float
    tag_edge_min_px: float
    initial_transform: Optional[np.ndarray] = None
    initial_reprojection_error_px: float = float("inf")
    inlier_mask: np.ndarray = field(default_factory=lambda: np.ones(0, dtype=bool))


@dataclass
class CameraQuality:
    camera_id: int
    observations: int
    corners: int
    median_reprojection_error_px: float
    p95_reprojection_error_px: float
    max_reprojection_error_px: float
    covariance_6x6: np.ndarray


@dataclass
class CubeCalibrationResult:
    reference_camera: int
    camera_transforms: Dict[int, np.ndarray]
    cube_transforms: Dict[str, np.ndarray]
    camera_quality: Dict[int, CameraQuality]
    optimizer_success: bool
    optimizer_message: str
    optimizer_cost: float
    residual_count: int
    rejected_corners: int
    image_size: Tuple[int, int]


def detect_known_cube_tags(
    image_bgr: np.ndarray,
    detector,
    known_tag_ids: Iterable[int],
    *,
    min_decision_margin: float,
    max_hamming: int,
    min_tag_edge_px: float,
) -> List[DetectedCubeTag]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    known = set(int(tag_id) for tag_id in known_tag_ids)
    accepted = []

    for detection in detector.detect(gray):
        tag_id = int(detection.tag_id)
        if tag_id not in known:
            continue

        hamming = int(getattr(detection, "hamming", 0))
        decision_margin = float(getattr(detection, "decision_margin", float("inf")))
        corners = np.asarray(detection.corners, dtype=np.float64).reshape(4, 2)
        edge_length = tag_edge_length_px(corners)

        if hamming > max_hamming:
            continue
        if decision_margin < min_decision_margin:
            continue
        if edge_length < min_tag_edge_px:
            continue

        accepted.append(DetectedCubeTag(
            tag_id=tag_id,
            corners=corners,
            decision_margin=decision_margin,
            hamming=hamming,
            edge_length_px=edge_length,
        ))

    return accepted


def project_points(
    object_points: np.ndarray,
    transform_camera_from_object: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    rotation_vector = Rotation.from_matrix(
        transform_camera_from_object[:3, :3]
    ).as_rotvec().reshape(3, 1)
    translation = transform_camera_from_object[:3, 3].reshape(3, 1)
    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float64),
        rotation_vector,
        translation,
        camera_matrix,
        distortion,
    )
    return projected.reshape(-1, 2)


def pose_reprojection_errors(
    object_points: np.ndarray,
    image_points: np.ndarray,
    transform_camera_from_object: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    projected = project_points(
        object_points,
        transform_camera_from_object,
        camera_matrix,
        distortion,
    )
    return np.linalg.norm(projected - image_points, axis=1)


def estimate_cube_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> Tuple[Optional[np.ndarray], float]:
    object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    centered = object_points - object_points.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    planar = singular_values[-1] <= max(singular_values[0] * 1e-5, 1e-9)
    plane_normal = None
    if planar:
        plane_normal = np.cross(
            object_points[1] - object_points[0],
            object_points[2] - object_points[0],
        )
        plane_normal /= np.linalg.norm(plane_normal)
    candidates: List[np.ndarray] = []

    if planar:
        try:
            solution = cv2.solvePnPGeneric(
                object_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if bool(solution[0]):
                for rotation_vector, translation in zip(solution[1], solution[2]):
                    rotation, _ = cv2.Rodrigues(rotation_vector)
                    candidates.append(make_transform(rotation, translation))
        except cv2.error:
            candidates = []

    if not candidates:
        flag = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_ITERATIVE)
        success, rotation_vector, translation = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=flag,
        )
        if success:
            rotation, _ = cv2.Rodrigues(rotation_vector)
            candidates.append(make_transform(rotation, translation))

    best_transform = None
    best_error = float("inf")
    for candidate in candidates:
        camera_points = (
            candidate[:3, :3] @ object_points.T
            + candidate[:3, 3].reshape(3, 1)
        ).T
        if np.any(camera_points[:, 2] <= 0.0):
            continue
        if plane_normal is not None:
            normal_camera = candidate[:3, :3] @ plane_normal
            if float(np.dot(normal_camera, camera_points.mean(axis=0))) >= 0.0:
                continue

        errors = pose_reprojection_errors(
            object_points,
            image_points,
            candidate,
            camera_matrix,
            distortion,
        )
        mean_error = float(np.mean(errors))
        if mean_error < best_error:
            best_transform = candidate
            best_error = mean_error

    if best_transform is None:
        return None, float("inf")

    try:
        initial_rotation = Rotation.from_matrix(best_transform[:3, :3]).as_rotvec().reshape(3, 1)
        initial_translation = best_transform[:3, 3].reshape(3, 1)
        refined_rotation, refined_translation = cv2.solvePnPRefineLM(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            initial_rotation,
            initial_translation,
        )
        refined_matrix, _ = cv2.Rodrigues(refined_rotation)
        refined_transform = make_transform(refined_matrix, refined_translation)
        refined_error = float(np.mean(pose_reprojection_errors(
            object_points,
            image_points,
            refined_transform,
            camera_matrix,
            distortion,
        )))
        if refined_error <= best_error:
            best_transform = refined_transform
            best_error = refined_error
    except (AttributeError, cv2.error):
        pass

    return best_transform, best_error


def robust_average_transforms(
    transforms: Sequence[np.ndarray],
    *,
    max_rotation_deviation_deg: float = 8.0,
    max_translation_deviation_m: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    if not transforms:
        raise ValueError("Cannot average an empty transform sequence.")

    rotations = np.asarray([transform[:3, :3] for transform in transforms])
    translations = np.asarray([transform[:3, 3] for transform in transforms])
    inliers = np.ones(len(transforms), dtype=bool)

    for _iteration in range(4):
        selected_rotations = Rotation.from_matrix(rotations[inliers])
        mean_rotation = selected_rotations.mean().as_matrix()
        median_translation = np.median(translations[inliers], axis=0)
        center_transform = make_transform(mean_rotation, median_translation)

        rotation_errors = np.asarray([
            rotation_distance_degrees(center_transform, transform)
            for transform in transforms
        ])
        translation_errors = np.linalg.norm(
            translations - median_translation.reshape(1, 3), axis=1
        )

        rotation_median = float(np.median(rotation_errors[inliers]))
        translation_median = float(np.median(translation_errors[inliers]))
        rotation_mad = float(np.median(np.abs(rotation_errors[inliers] - rotation_median)))
        translation_mad = float(np.median(np.abs(
            translation_errors[inliers] - translation_median
        )))

        rotation_limit = min(
            max_rotation_deviation_deg,
            max(1.0, rotation_median + 3.5 * 1.4826 * rotation_mad),
        )
        translation_limit = min(
            max_translation_deviation_m,
            max(0.005, translation_median + 3.5 * 1.4826 * translation_mad),
        )
        updated = (
            (rotation_errors <= rotation_limit)
            & (translation_errors <= translation_limit)
        )
        if not np.any(updated):
            updated[np.argmin(rotation_errors + translation_errors * 50.0)] = True
        if np.array_equal(updated, inliers):
            break
        inliers = updated

    final_rotation = Rotation.from_matrix(rotations[inliers]).mean().as_matrix()
    final_translation = np.median(translations[inliers], axis=0)
    final_transform = make_transform(final_rotation, final_translation)
    final_rotation_errors = np.asarray([
        rotation_distance_degrees(final_transform, transform)
        for transform in np.asarray(transforms, dtype=np.float64)[inliers]
    ])
    final_translation_errors = np.linalg.norm(
        translations[inliers] - final_translation.reshape(1, 3), axis=1
    )
    return (
        final_transform,
        inliers,
        float(np.median(final_rotation_errors)),
        float(np.median(final_translation_errors)),
    )


class CubeMulticamCalibrator:
    def __init__(
        self,
        intrinsics: Dict[int, Tuple[np.ndarray, np.ndarray, float]],
        cube_layout: Path,
        *,
        reference_camera: int = 1,
        families: str = "tag36h11",
        min_decision_margin: float = 20.0,
        max_hamming: int = 0,
        min_tag_edge_px: float = 45.0,
        max_initial_reprojection_error_px: float = 3.0,
        min_views_per_camera: int = 10,
        robust_loss: str = "huber",
        robust_scale_px: float = 1.0,
        max_final_p95_px: float = 2.0,
    ):
        self.intrinsics = intrinsics
        self.reference_camera = int(reference_camera)
        self.tag_corners = load_cube_layout(Path(cube_layout))
        self.detector = create_apriltag_detector(families)
        self.min_decision_margin = float(min_decision_margin)
        self.max_hamming = int(max_hamming)
        self.min_tag_edge_px = float(min_tag_edge_px)
        self.max_initial_reprojection_error_px = float(
            max_initial_reprojection_error_px
        )
        self.min_views_per_camera = int(min_views_per_camera)
        self.robust_loss = robust_loss
        self.robust_scale_px = float(robust_scale_px)
        self.max_final_p95_px = float(max_final_p95_px)
        self.observations: List[CubeObservation] = []
        self.image_size: Optional[Tuple[int, int]] = None

    def load_capture_directory(self, captures_dir: Path) -> List[CubeObservation]:
        captures_dir = Path(captures_dir)
        capture_dirs = sorted(
            path for path in captures_dir.glob("capture_*") if path.is_dir()
        )
        if not capture_dirs:
            raise FileNotFoundError(
                f"No capture_* directories found under {captures_dir}."
            )

        observations = []
        for capture_dir in capture_dirs:
            for camera_id in sorted(self.intrinsics):
                image_path = capture_dir / f"camera_{camera_id}.png"
                if not image_path.exists():
                    continue
                image = cv2.imread(str(image_path))
                if image is None:
                    continue

                current_size = (image.shape[1], image.shape[0])
                if self.image_size is None:
                    self.image_size = current_size
                elif self.image_size != current_size:
                    raise ValueError(
                        f"Mixed image sizes: {self.image_size} and {current_size}."
                    )

                detections = detect_known_cube_tags(
                    image,
                    self.detector,
                    self.tag_corners.keys(),
                    min_decision_margin=self.min_decision_margin,
                    max_hamming=self.max_hamming,
                    min_tag_edge_px=self.min_tag_edge_px,
                )
                if not detections:
                    continue

                object_points = np.concatenate([
                    self.tag_corners[detection.tag_id] for detection in detections
                ], axis=0)
                image_points = np.concatenate([
                    detection.corners for detection in detections
                ], axis=0)
                camera_matrix, distortion, _error = self.intrinsics[camera_id]
                initial_transform, initial_error = estimate_cube_pose(
                    object_points,
                    image_points,
                    camera_matrix,
                    distortion,
                )
                if initial_transform is None:
                    continue
                if initial_error > self.max_initial_reprojection_error_px:
                    continue

                observation = CubeObservation(
                    camera_id=camera_id,
                    capture_id=capture_dir.name,
                    image_path=image_path,
                    object_points=object_points,
                    image_points=image_points,
                    tag_ids=sorted({detection.tag_id for detection in detections}),
                    decision_margin_min=min(
                        detection.decision_margin for detection in detections
                    ),
                    tag_edge_min_px=min(
                        detection.edge_length_px for detection in detections
                    ),
                    initial_transform=initial_transform,
                    initial_reprojection_error_px=initial_error,
                    inlier_mask=np.ones(len(object_points), dtype=bool),
                )
                observations.append(observation)

        self.observations = observations
        self._validate_observation_graph()
        return observations

    def _validate_observation_graph(self) -> None:
        if not self.observations:
            raise RuntimeError("No valid cube observations were loaded.")

        view_counts = {camera_id: 0 for camera_id in self.intrinsics}
        cameras_by_capture: Dict[str, set] = {}
        for observation in self.observations:
            view_counts[observation.camera_id] += 1
            cameras_by_capture.setdefault(observation.capture_id, set()).add(
                observation.camera_id
            )

        insufficient = {
            camera_id: count
            for camera_id, count in view_counts.items()
            if count < self.min_views_per_camera
        }
        if insufficient:
            raise RuntimeError(
                "Insufficient accepted cube views per camera: "
                f"{insufficient}; minimum={self.min_views_per_camera}."
            )

        adjacency = {camera_id: set() for camera_id in self.intrinsics}
        for camera_ids in cameras_by_capture.values():
            for source in camera_ids:
                adjacency[source].update(camera_ids - {source})

        reachable = {self.reference_camera}
        frontier = [self.reference_camera]
        while frontier:
            source = frontier.pop()
            for target in adjacency[source]:
                if target not in reachable:
                    reachable.add(target)
                    frontier.append(target)

        missing = sorted(set(self.intrinsics) - reachable)
        if missing:
            raise RuntimeError(
                "Cube observation graph is disconnected from reference camera "
                f"{self.reference_camera}; unreachable cameras: {missing}."
            )

    def _build_pairwise_initialization(self):
        observations_by_capture: Dict[str, Dict[int, CubeObservation]] = {}
        for observation in self.observations:
            observations_by_capture.setdefault(observation.capture_id, {})[
                observation.camera_id
            ] = observation

        candidates: Dict[Tuple[int, int], List[np.ndarray]] = {}
        for camera_observations in observations_by_capture.values():
            camera_ids = sorted(camera_observations)
            for source_index, source_camera in enumerate(camera_ids):
                source_pose = camera_observations[source_camera].initial_transform
                for target_camera in camera_ids[source_index + 1:]:
                    target_pose = camera_observations[target_camera].initial_transform
                    transform_target_from_source = (
                        target_pose @ invert_transform(source_pose)
                    )
                    candidates.setdefault(
                        (source_camera, target_camera), []
                    ).append(transform_target_from_source)

        edges = {}
        for pair, transforms in candidates.items():
            average, inliers, rotation_spread, translation_spread = (
                robust_average_transforms(transforms)
            )
            inlier_count = int(np.sum(inliers))
            if inlier_count < 2:
                continue
            weight = (
                rotation_spread
                + translation_spread * 50.0
                + 1.0 / max(inlier_count, 1)
            )
            edges[pair] = (average, weight, inlier_count)
        return edges

    def _initial_camera_transforms(self) -> Dict[int, np.ndarray]:
        edges = self._build_pairwise_initialization()
        adjacency: Dict[int, List[Tuple[int, np.ndarray, float]]] = {
            camera_id: [] for camera_id in self.intrinsics
        }
        for (source, target), (transform, weight, _count) in edges.items():
            adjacency[source].append((target, transform, weight))
            adjacency[target].append((source, invert_transform(transform), weight))

        distances = {camera_id: float("inf") for camera_id in self.intrinsics}
        transforms = {self.reference_camera: np.eye(4, dtype=np.float64)}
        distances[self.reference_camera] = 0.0
        queue = [(0.0, self.reference_camera)]

        while queue:
            distance, source = heapq.heappop(queue)
            if distance > distances[source]:
                continue
            for target, transform_target_from_source, weight in adjacency[source]:
                candidate_distance = distance + weight
                if candidate_distance >= distances[target]:
                    continue
                distances[target] = candidate_distance
                transforms[target] = (
                    transform_target_from_source @ transforms[source]
                )
                heapq.heappush(queue, (candidate_distance, target))

        missing = sorted(set(self.intrinsics) - set(transforms))
        if missing:
            raise RuntimeError(
                f"Could not initialize camera transforms for cameras: {missing}."
            )
        return transforms

    def _initial_cube_transforms(
        self, camera_transforms: Dict[int, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        candidates: Dict[str, List[np.ndarray]] = {}
        for observation in self.observations:
            transform_ref_from_camera = invert_transform(
                camera_transforms[observation.camera_id]
            )
            candidates.setdefault(observation.capture_id, []).append(
                transform_ref_from_camera @ observation.initial_transform
            )

        cube_transforms = {}
        for capture_id, transforms in candidates.items():
            average, _inliers, _rotation_spread, _translation_spread = (
                robust_average_transforms(transforms)
            )
            cube_transforms[capture_id] = average
        return cube_transforms

    def _pack_parameters(
        self,
        camera_transforms: Dict[int, np.ndarray],
        cube_transforms: Dict[str, np.ndarray],
        camera_ids: Sequence[int],
        capture_ids: Sequence[str],
    ) -> np.ndarray:
        vectors = [
            transform_to_vector(camera_transforms[camera_id])
            for camera_id in camera_ids
        ]
        vectors.extend(
            transform_to_vector(cube_transforms[capture_id])
            for capture_id in capture_ids
        )
        return np.concatenate(vectors)

    def _unpack_parameters(
        self,
        parameters: np.ndarray,
        camera_ids: Sequence[int],
        capture_ids: Sequence[str],
    ) -> Tuple[Dict[int, np.ndarray], Dict[str, np.ndarray]]:
        camera_transforms = {
            self.reference_camera: np.eye(4, dtype=np.float64)
        }
        offset = 0
        for camera_id in camera_ids:
            camera_transforms[camera_id] = vector_to_transform(
                parameters[offset:offset + 6]
            )
            offset += 6

        cube_transforms = {}
        for capture_id in capture_ids:
            cube_transforms[capture_id] = vector_to_transform(
                parameters[offset:offset + 6]
            )
            offset += 6
        return camera_transforms, cube_transforms

    def _residuals(
        self,
        parameters: np.ndarray,
        camera_ids: Sequence[int],
        capture_ids: Sequence[str],
    ) -> np.ndarray:
        camera_transforms, cube_transforms = self._unpack_parameters(
            parameters, camera_ids, capture_ids
        )
        residuals = []
        for observation in self.observations:
            mask = observation.inlier_mask
            if int(np.sum(mask)) < 4:
                continue
            transform_camera_from_cube = (
                camera_transforms[observation.camera_id]
                @ cube_transforms[observation.capture_id]
            )
            camera_matrix, distortion, _error = self.intrinsics[
                observation.camera_id
            ]
            projected = project_points(
                observation.object_points[mask],
                transform_camera_from_cube,
                camera_matrix,
                distortion,
            )
            residuals.append(
                (projected - observation.image_points[mask]).reshape(-1)
            )
        if not residuals:
            raise RuntimeError("No residuals available for cube bundle adjustment.")
        return np.concatenate(residuals)

    def _corner_errors(
        self,
        camera_transforms: Dict[int, np.ndarray],
        cube_transforms: Dict[str, np.ndarray],
    ) -> Dict[int, List[float]]:
        errors_by_camera: Dict[int, List[float]] = {
            camera_id: [] for camera_id in self.intrinsics
        }
        for observation in self.observations:
            transform_camera_from_cube = (
                camera_transforms[observation.camera_id]
                @ cube_transforms[observation.capture_id]
            )
            camera_matrix, distortion, _error = self.intrinsics[
                observation.camera_id
            ]
            errors = pose_reprojection_errors(
                observation.object_points,
                observation.image_points,
                transform_camera_from_cube,
                camera_matrix,
                distortion,
            )
            errors_by_camera[observation.camera_id].extend(
                errors[observation.inlier_mask].tolist()
            )
        return errors_by_camera

    def _reject_corner_outliers(
        self,
        camera_transforms: Dict[int, np.ndarray],
        cube_transforms: Dict[str, np.ndarray],
    ) -> int:
        all_errors = []
        errors_by_observation = []
        for observation in self.observations:
            transform_camera_from_cube = (
                camera_transforms[observation.camera_id]
                @ cube_transforms[observation.capture_id]
            )
            camera_matrix, distortion, _error = self.intrinsics[
                observation.camera_id
            ]
            errors = pose_reprojection_errors(
                observation.object_points,
                observation.image_points,
                transform_camera_from_cube,
                camera_matrix,
                distortion,
            )
            errors_by_observation.append(errors)
            all_errors.extend(errors.tolist())

        all_errors_array = np.asarray(all_errors, dtype=np.float64)
        median_error = float(np.median(all_errors_array))
        mad = float(np.median(np.abs(all_errors_array - median_error)))
        threshold = min(4.0, max(1.5, median_error + 4.0 * 1.4826 * mad))

        rejected = 0
        for observation, errors in zip(self.observations, errors_by_observation):
            mask = errors <= threshold
            if int(np.sum(mask)) < 4:
                best_indices = np.argsort(errors)[:4]
                mask = np.zeros(len(errors), dtype=bool)
                mask[best_indices] = True
            rejected += int(np.sum(~mask))
            observation.inlier_mask = mask
        return rejected

    @staticmethod
    def _camera_covariances(
        optimizer_result,
        camera_ids: Sequence[int],
        reference_camera: int,
    ) -> Dict[int, np.ndarray]:
        covariances = {
            reference_camera: np.zeros((6, 6), dtype=np.float64)
        }
        jacobian = np.asarray(optimizer_result.jac, dtype=np.float64)
        degrees_of_freedom = max(jacobian.shape[0] - jacobian.shape[1], 1)
        residual_variance = float(
            2.0 * optimizer_result.cost / degrees_of_freedom
        )
        covariance = np.linalg.pinv(jacobian.T @ jacobian) * residual_variance
        for camera_index, camera_id in enumerate(camera_ids):
            start = camera_index * 6
            covariances[camera_id] = covariance[start:start + 6, start:start + 6]
        return covariances

    def calibrate(self) -> CubeCalibrationResult:
        self._validate_observation_graph()
        initial_cameras = self._initial_camera_transforms()
        initial_cubes = self._initial_cube_transforms(initial_cameras)
        camera_ids = sorted(
            camera_id for camera_id in self.intrinsics
            if camera_id != self.reference_camera
        )
        capture_ids = sorted(initial_cubes)
        initial_parameters = self._pack_parameters(
            initial_cameras,
            initial_cubes,
            camera_ids,
            capture_ids,
        )

        first_result = least_squares(
            self._residuals,
            initial_parameters,
            args=(camera_ids, capture_ids),
            method="trf",
            loss=self.robust_loss,
            f_scale=self.robust_scale_px,
            max_nfev=300,
            x_scale="jac",
            verbose=0,
        )
        first_cameras, first_cubes = self._unpack_parameters(
            first_result.x, camera_ids, capture_ids
        )
        rejected_corners = self._reject_corner_outliers(
            first_cameras, first_cubes
        )

        final_result = least_squares(
            self._residuals,
            first_result.x,
            args=(camera_ids, capture_ids),
            method="trf",
            loss=self.robust_loss,
            f_scale=self.robust_scale_px,
            max_nfev=300,
            x_scale="jac",
            verbose=0,
        )
        camera_transforms, cube_transforms = self._unpack_parameters(
            final_result.x, camera_ids, capture_ids
        )
        if not final_result.success:
            raise RuntimeError(
                "Cube bundle adjustment did not converge: "
                f"{final_result.message}"
            )
        errors_by_camera = self._corner_errors(
            camera_transforms, cube_transforms
        )
        covariances = self._camera_covariances(
            final_result, camera_ids, self.reference_camera
        )

        camera_quality = {}
        for camera_id, errors in errors_by_camera.items():
            error_array = np.asarray(errors, dtype=np.float64)
            observation_count = sum(
                observation.camera_id == camera_id
                for observation in self.observations
            )
            camera_quality[camera_id] = CameraQuality(
                camera_id=camera_id,
                observations=observation_count,
                corners=len(error_array),
                median_reprojection_error_px=float(np.median(error_array)),
                p95_reprojection_error_px=float(np.percentile(error_array, 95)),
                max_reprojection_error_px=float(np.max(error_array)),
                covariance_6x6=covariances[camera_id],
            )

        poor_cameras = [
            camera_id
            for camera_id, quality in camera_quality.items()
            if quality.p95_reprojection_error_px > self.max_final_p95_px
        ]
        if poor_cameras:
            raise RuntimeError(
                "Final cube calibration failed the reprojection quality gate; "
                f"P95 > {self.max_final_p95_px:.2f}px for cameras {poor_cameras}."
            )

        return CubeCalibrationResult(
            reference_camera=self.reference_camera,
            camera_transforms=camera_transforms,
            cube_transforms=cube_transforms,
            camera_quality=camera_quality,
            optimizer_success=bool(final_result.success),
            optimizer_message=str(final_result.message),
            optimizer_cost=float(final_result.cost),
            residual_count=len(final_result.fun),
            rejected_corners=rejected_corners,
            image_size=self.image_size or (0, 0),
        )


def save_cube_calibration(
    output_path: Path,
    intrinsics: Dict[int, Tuple[np.ndarray, np.ndarray, float]],
    result: CubeCalibrationResult,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "ref_camera": np.asarray(result.reference_camera, dtype=np.int32),
        "num_cameras": np.asarray(len(intrinsics), dtype=np.int32),
        "image_size": np.asarray(result.image_size, dtype=np.int32),
        "calibration_method": np.asarray("apriltag_cube_bundle_adjustment"),
        "transform_convention": np.asarray(
            "T_ref_to_camN maps reference-camera coordinates into camera N"
        ),
        "optimizer_success": np.asarray(result.optimizer_success),
        "optimizer_cost": np.asarray(result.optimizer_cost, dtype=np.float64),
        "optimizer_residual_count": np.asarray(result.residual_count, dtype=np.int32),
        "rejected_corners": np.asarray(result.rejected_corners, dtype=np.int32),
        "num_cube_poses": np.asarray(len(result.cube_transforms), dtype=np.int32),
    }

    for camera_id in sorted(intrinsics):
        camera_matrix, distortion, intrinsic_error = intrinsics[camera_id]
        transform_camera_from_ref = result.camera_transforms[camera_id]
        transform_ref_from_camera = invert_transform(transform_camera_from_ref)
        quality = result.camera_quality[camera_id]

        data[f"K{camera_id}"] = camera_matrix
        data[f"dist{camera_id}"] = distortion
        data[f"intrinsic_error{camera_id}"] = np.asarray(
            intrinsic_error, dtype=np.float64
        )

        # Legacy compatibility: this project historically stores ref -> camera
        # under keys named R_N_to_ref / t_N_to_ref.
        data[f"R_{camera_id}_to_ref"] = transform_camera_from_ref[:3, :3]
        data[f"t_{camera_id}_to_ref"] = transform_camera_from_ref[:3, 3]
        data[f"T_ref_to_cam{camera_id}"] = transform_camera_from_ref
        data[f"T_cam{camera_id}_to_ref"] = transform_ref_from_camera
        data[f"reproj_median_px_cam{camera_id}"] = np.asarray(
            quality.median_reprojection_error_px, dtype=np.float64
        )
        data[f"reproj_p95_px_cam{camera_id}"] = np.asarray(
            quality.p95_reprojection_error_px, dtype=np.float64
        )
        data[f"reproj_max_px_cam{camera_id}"] = np.asarray(
            quality.max_reprojection_error_px, dtype=np.float64
        )
        data[f"extrinsic_covariance_cam{camera_id}"] = quality.covariance_6x6

    if 1 in result.camera_transforms and 2 in result.camera_transforms:
        transform_cam2_from_cam1 = (
            result.camera_transforms[2]
            @ invert_transform(result.camera_transforms[1])
        )
        data["R_1_to_2"] = transform_cam2_from_cam1[:3, :3]
        data["t_1_to_2"] = transform_cam2_from_cam1[:3, 3]

    temporary_path = output_path.with_name(f".{output_path.name}.tmp.npz")
    try:
        np.savez(temporary_path, **data)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
