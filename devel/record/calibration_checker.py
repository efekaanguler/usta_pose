#!/usr/bin/env python3
"""
AprilTag cube based pre-recording calibration sanity check.

The checker compares the fixed morning multi-camera calibration against a
fresh cube observation captured immediately before a recording session.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np


def _load_apriltag_detector(families: str):
    try:
        from pupil_apriltags import Detector

        return Detector(families=families)
    except ImportError:
        pass

    try:
        from dt_apriltags import Detector

        return Detector(families=families)
    except ImportError as exc:
        raise ImportError(
            "Install pupil-apriltags or dt-apriltags to use CalibrationChecker."
        ) from exc


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = rotation.T
    inv[:3, 3] = -rotation.T @ translation
    return inv


def rotation_error_degrees(reference_rotation: np.ndarray, measured_rotation: np.ndarray) -> float:
    delta = reference_rotation.T @ measured_rotation
    cos_angle = (np.trace(delta) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


@dataclass
class CameraPose:
    cam_id: int
    tag_ids: List[int]
    num_points: int
    reprojection_error_px: float
    normal_facing_min_cos: float
    normal_checks: Dict[int, float]
    T_cube_to_cam: np.ndarray


@dataclass
class PairCheck:
    cam_a: int
    cam_b: int
    rotation_error_deg: float
    translation_error_mm: float
    ok: bool


@dataclass
class CalibrationCheckResult:
    ok: bool
    camera_poses: Dict[int, CameraPose]
    pair_checks: List[PairCheck]
    failed_cameras: List[int]
    message: str


class CalibrationChecker:
    """
    Validate morning extrinsics using a cube with known AprilTag 3D corners.

    solvePnP convention:
        T_cube_to_cam maps cube/CAD coordinates into a camera frame.

    Calibration NPZ convention used by this project:
        R_N_to_ref / t_N_to_ref keys store ref -> cam_N transforms.
        That means P_cam_N = R @ P_ref + t.
    """

    def __init__(
        self,
        calibration_npz: Union[str, Path],
        cube_layout_json: Union[str, Path],
        *,
        families: str = "tag36h11",
        reference_camera: int = 1,
        min_tags_per_camera: int = 1,
        min_points_per_camera: int = 4,
        max_reprojection_error_px: float = 3.0,
        min_normal_facing_cos: float = 0.05,
        max_rotation_error_deg: float = 2.0,
        max_translation_error_mm: float = 20.0,
        compare_to_reference_only: bool = True,
    ):
        self.calibration_npz = Path(calibration_npz)
        self.cube_layout_json = Path(cube_layout_json)
        self.families = families
        self.reference_camera = int(reference_camera)
        self.min_tags_per_camera = int(min_tags_per_camera)
        self.min_points_per_camera = int(min_points_per_camera)
        self.max_reprojection_error_px = float(max_reprojection_error_px)
        self.min_normal_facing_cos = float(min_normal_facing_cos)
        self.max_rotation_error_deg = float(max_rotation_error_deg)
        self.max_translation_error_mm = float(max_translation_error_mm)
        self.compare_to_reference_only = bool(compare_to_reference_only)

        self.detector = _load_apriltag_detector(families)
        self.tag_corners, self.tag_centers, self.tag_normals = self._load_cube_layout(self.cube_layout_json)
        self.K, self.dist, self.T_ref_to_cam = self._load_calibration(self.calibration_npz)

    @staticmethod
    def _unit_scale(unit: str) -> float:
        unit = unit.lower()
        if unit in ("m", "meter", "meters"):
            return 1.0
        if unit in ("mm", "millimeter", "millimeters"):
            return 0.001
        if unit in ("cm", "centimeter", "centimeters"):
            return 0.01
        raise ValueError(f"Unsupported cube layout unit: {unit}")

    @staticmethod
    def _normal_from_corners(corners: np.ndarray) -> np.ndarray:
        candidate = np.cross(corners[1] - corners[0], corners[2] - corners[0])
        norm = float(np.linalg.norm(candidate))
        if norm <= 1e-12:
            center = corners.mean(axis=0)
            center_norm = float(np.linalg.norm(center))
            if center_norm <= 1e-12:
                return np.array([0.0, 0.0, 1.0], dtype=np.float64)
            return center / center_norm

        normal = candidate / norm
        center = corners.mean(axis=0)
        if float(np.dot(normal, center)) < 0.0:
            normal = -normal
        return normal

    @classmethod
    def _load_cube_layout(
        cls, layout_path: Path
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        Load tag corner coordinates in cube/CAD coordinates.

        Preferred JSON schema:
            {
              "unit": "m",
              "tags": [
                {"id": 0, "corners": [[x,y,z], [x,y,z], [x,y,z], [x,y,z]]}
              ]
            }

        The corner order must match the AprilTag detector corner order.
        """
        with open(layout_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        tags = {}
        centers = {}
        normals = {}
        default_scale = cls._unit_scale(raw.get("unit", "m"))

        if "tags" in raw:
            iterable = raw["tags"]
        elif "tagCoordinates" in raw:
            iterable = raw["tagCoordinates"][0]
        else:
            raise KeyError("Cube layout JSON must contain 'tags' or 'tagCoordinates'.")

        for tag in iterable:
            tag_id = int(tag["id"])
            scale = cls._unit_scale(tag.get("unit", raw.get("unit", "m")))
            corners = np.asarray(tag["corners"], dtype=np.float64)
            if corners.shape == (4, 2):
                corners = np.column_stack([corners, np.zeros(4, dtype=np.float64)])
            if corners.shape != (4, 3):
                raise ValueError(f"Tag {tag_id} corners must be 4x3 or 4x2.")
            corners_m = (corners * scale / default_scale) * default_scale
            tags[tag_id] = corners_m

            center = np.asarray(tag.get("center", corners.mean(axis=0)), dtype=np.float64).reshape(3)
            centers[tag_id] = (center * scale / default_scale) * default_scale

            if "normal" in tag:
                normal = np.asarray(tag["normal"], dtype=np.float64).reshape(3)
                normal_norm = float(np.linalg.norm(normal))
                if normal_norm <= 1e-12:
                    normal = cls._normal_from_corners(corners_m)
                else:
                    normal = normal / normal_norm
            else:
                normal = cls._normal_from_corners(corners_m)
            normals[tag_id] = normal

        if not tags:
            raise ValueError(f"No tags loaded from {layout_path}")
        return tags, centers, normals

    @staticmethod
    def _tag_edge_length(corners: np.ndarray) -> float:
        edges = [
            np.linalg.norm(corners[(idx + 1) % 4] - corners[idx])
            for idx in range(4)
        ]
        return float(np.median(edges))

    @staticmethod
    def _distortion_from_npz(data, cam_id: int) -> np.ndarray:
        key = f"dist{cam_id}"
        if key not in data:
            return np.zeros((5, 1), dtype=np.float64)
        return np.asarray(data[key], dtype=np.float64).reshape(-1, 1)

    @classmethod
    def _load_calibration(
        cls, calibration_path: Path
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        if not calibration_path.exists():
            raise FileNotFoundError(f"Calibration NPZ not found: {calibration_path}")

        K = {}
        dist = {}
        T_ref_to_cam = {}

        with np.load(calibration_path) as data:
            cam_ids = sorted(
                int(key[1:]) for key in data.files
                if key.startswith("K") and key[1:].isdigit()
            )
            if not cam_ids:
                raise KeyError(f"No K1..KN camera matrices found in {calibration_path}")

            for cam_id in cam_ids:
                K[cam_id] = np.asarray(data[f"K{cam_id}"], dtype=np.float64)
                dist[cam_id] = cls._distortion_from_npz(data, cam_id)

                r_key = f"R_{cam_id}_to_ref"
                t_key = f"t_{cam_id}_to_ref"
                if r_key in data and t_key in data:
                    T_ref_to_cam[cam_id] = make_transform(data[r_key], data[t_key])
                else:
                    T_ref_to_cam[cam_id] = np.eye(4, dtype=np.float64)

        return K, dist, T_ref_to_cam

    @staticmethod
    def _detect_corners(detector, image_bgr: np.ndarray):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return detector.detect(gray)

    def estimate_camera_pose(self, cam_id: int, image_bgr: np.ndarray) -> Optional[CameraPose]:
        detections = self._detect_corners(self.detector, image_bgr)
        object_points = []
        image_points = []
        tag_ids = []

        for detection in detections:
            tag_id = int(detection.tag_id)
            if tag_id not in self.tag_corners:
                continue
            object_points.extend(self.tag_corners[tag_id])
            image_points.extend(np.asarray(detection.corners, dtype=np.float64))
            tag_ids.append(tag_id)

        if len(tag_ids) < self.min_tags_per_camera:
            return None
        if len(object_points) < self.min_points_per_camera:
            return None

        obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
        img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)

        flag = getattr(cv2, "SOLVEPNP_SQPNP", cv2.SOLVEPNP_ITERATIVE)
        success, rvec, tvec = cv2.solvePnP(
            obj,
            img,
            self.K[cam_id],
            self.dist.get(cam_id, np.zeros((5, 1), dtype=np.float64)),
            flags=flag,
        )
        if not success:
            return None

        rotation, _ = cv2.Rodrigues(rvec)
        projected, _ = cv2.projectPoints(
            obj,
            rvec,
            tvec,
            self.K[cam_id],
            self.dist.get(cam_id, np.zeros((5, 1), dtype=np.float64)),
        )
        projected = projected.reshape(-1, 2)
        reproj_error = float(np.mean(np.linalg.norm(projected - img, axis=1)))
        normal_checks = self._normal_facing_scores(rotation, tvec.reshape(3), sorted(set(tag_ids)))
        normal_min = min(normal_checks.values()) if normal_checks else 1.0

        return CameraPose(
            cam_id=cam_id,
            tag_ids=sorted(set(tag_ids)),
            num_points=len(obj),
            reprojection_error_px=reproj_error,
            normal_facing_min_cos=float(normal_min),
            normal_checks=normal_checks,
            T_cube_to_cam=make_transform(rotation, tvec),
        )

    def _normal_facing_scores(
        self,
        rotation: np.ndarray,
        translation: np.ndarray,
        tag_ids: Iterable[int],
    ) -> Dict[int, float]:
        """
        For a visible cube face, the outward face normal should point roughly
        toward the camera. In OpenCV camera coordinates the tag center vector
        points from camera to tag, so a valid visible normal has negative dot
        product with that view vector. We report -dot(normal_cam, view_dir):
        +1 means facing camera, 0 means grazing, negative means impossible/away.
        """
        scores = {}
        t = np.asarray(translation, dtype=np.float64).reshape(3)
        for tag_id in tag_ids:
            if tag_id not in self.tag_normals or tag_id not in self.tag_centers:
                continue
            normal_cam = rotation @ self.tag_normals[tag_id]
            center_cam = rotation @ self.tag_centers[tag_id] + t
            center_norm = float(np.linalg.norm(center_cam))
            normal_norm = float(np.linalg.norm(normal_cam))
            if center_norm <= 1e-12 or normal_norm <= 1e-12:
                continue
            view_dir = center_cam / center_norm
            normal_dir = normal_cam / normal_norm
            scores[int(tag_id)] = float(-np.dot(normal_dir, view_dir))
        return scores

    def draw_pose_overlay(
        self,
        cam_id: int,
        image_bgr: np.ndarray,
        pose: Optional[CameraPose],
    ) -> np.ndarray:
        """
        Draw detected known tags and a projected virtual outward-normal arrow.

        The magenta arrow is the CAD face normal transformed by solvePnP and
        projected into the image. If the face is almost fronto-parallel, the
        projected arrow can be short; the numeric facing score is printed too.
        """
        overlay = image_bgr.copy()
        detections = self._detect_corners(self.detector, image_bgr)
        known_detections = [
            detection for detection in detections
            if int(detection.tag_id) in self.tag_corners
        ]

        for detection in known_detections:
            tag_id = int(detection.tag_id)
            pts = np.asarray(detection.corners, dtype=np.int32).reshape(4, 2)
            cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
            center_2d = tuple(np.asarray(detection.center, dtype=np.int32).reshape(2))
            cv2.putText(
                overlay,
                f"ID {tag_id}",
                center_2d,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if pose is not None:
            rotation = pose.T_cube_to_cam[:3, :3]
            tvec = pose.T_cube_to_cam[:3, 3].reshape(3, 1)
            rvec, _ = cv2.Rodrigues(rotation)

            for tag_id in pose.tag_ids:
                if tag_id not in self.tag_centers or tag_id not in self.tag_normals:
                    continue
                length = self._tag_edge_length(self.tag_corners[tag_id]) * 0.75
                center = self.tag_centers[tag_id]
                endpoint = center + self.tag_normals[tag_id] * length
                points_3d = np.asarray([center, endpoint], dtype=np.float64).reshape(-1, 3)
                projected, _ = cv2.projectPoints(
                    points_3d,
                    rvec,
                    tvec,
                    self.K[cam_id],
                    self.dist.get(cam_id, np.zeros((5, 1), dtype=np.float64)),
                )
                projected = projected.reshape(-1, 2)
                start = tuple(np.round(projected[0]).astype(int))
                end = tuple(np.round(projected[1]).astype(int))

                if np.linalg.norm(projected[1] - projected[0]) < 12.0:
                    normal_cam = rotation @ self.tag_normals[tag_id]
                    fallback_end = projected[0] + normal_cam[:2] * 60.0
                    end = tuple(np.round(fallback_end).astype(int))
                    cv2.circle(overlay, start, 9, (255, 0, 255), 2)

                score = pose.normal_checks.get(tag_id, float("nan"))
                color = (255, 0, 255) if score >= self.min_normal_facing_cos else (0, 0, 255)
                cv2.arrowedLine(overlay, start, end, color, 3, cv2.LINE_AA, tipLength=0.25)
                cv2.putText(
                    overlay,
                    f"normal {tag_id}: {score:.2f}",
                    (start[0] + 8, start[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                overlay,
                f"cam{cam_id} reproj={pose.reprojection_error_px:.2f}px normal_min={pose.normal_facing_min_cos:.2f}",
                (24, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                overlay,
                f"cam{cam_id}: no valid cube pose",
                (24, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return overlay

    def _calibrated_cam_to_cam(self, cam_a: int, cam_b: int) -> np.ndarray:
        return self.T_ref_to_cam[cam_b] @ invert_transform(self.T_ref_to_cam[cam_a])

    @staticmethod
    def _measured_cam_to_cam(pose_a: CameraPose, pose_b: CameraPose) -> np.ndarray:
        return pose_b.T_cube_to_cam @ invert_transform(pose_a.T_cube_to_cam)

    def _pairs_to_compare(self, cam_ids: Iterable[int]) -> List[Tuple[int, int]]:
        cam_ids = sorted(cam_ids)
        if self.compare_to_reference_only and self.reference_camera in cam_ids:
            return [(self.reference_camera, cam_id) for cam_id in cam_ids if cam_id != self.reference_camera]

        pairs = []
        for i, cam_a in enumerate(cam_ids):
            for cam_b in cam_ids[i + 1:]:
                pairs.append((cam_a, cam_b))
        return pairs

    def check(self, frames_by_camera: Dict[int, np.ndarray]) -> CalibrationCheckResult:
        poses = {}
        for cam_id, image in sorted(frames_by_camera.items()):
            if cam_id not in self.K:
                continue
            pose = self.estimate_camera_pose(cam_id, image)
            if pose is not None:
                poses[cam_id] = pose

        if len(poses) < 2:
            return CalibrationCheckResult(
                ok=False,
                camera_poses=poses,
                pair_checks=[],
                failed_cameras=sorted(set(frames_by_camera.keys()) - set(poses.keys())),
                message="Need at least two cameras with valid AprilTag cube pose.",
            )

        pair_checks = []
        failed_cameras = set()

        for cam_a, cam_b in self._pairs_to_compare(poses.keys()):
            T_calib = self._calibrated_cam_to_cam(cam_a, cam_b)
            T_meas = self._measured_cam_to_cam(poses[cam_a], poses[cam_b])

            rot_err = rotation_error_degrees(T_calib[:3, :3], T_meas[:3, :3])
            trans_err = float(np.linalg.norm(T_calib[:3, 3] - T_meas[:3, 3]) * 1000.0)

            reproj_ok = (
                poses[cam_a].reprojection_error_px <= self.max_reprojection_error_px and
                poses[cam_b].reprojection_error_px <= self.max_reprojection_error_px
            )
            normal_ok = (
                poses[cam_a].normal_facing_min_cos >= self.min_normal_facing_cos and
                poses[cam_b].normal_facing_min_cos >= self.min_normal_facing_cos
            )
            pair_ok = (
                reproj_ok and
                normal_ok and
                rot_err <= self.max_rotation_error_deg and
                trans_err <= self.max_translation_error_mm
            )
            if not pair_ok:
                failed_cameras.update([cam_a, cam_b])

            pair_checks.append(PairCheck(
                cam_a=cam_a,
                cam_b=cam_b,
                rotation_error_deg=rot_err,
                translation_error_mm=trans_err,
                ok=pair_ok,
            ))

        ok = bool(pair_checks) and all(pair.ok for pair in pair_checks)
        if ok:
            message = "KALIBRASYON OK: Session baslatilabilir."
        else:
            message = "HATA: Kalibrasyon bozulmus olabilir. Kayit yapilamaz."

        return CalibrationCheckResult(
            ok=ok,
            camera_poses=poses,
            pair_checks=pair_checks,
            failed_cameras=sorted(failed_cameras),
            message=message,
        )

    @staticmethod
    def print_report(result: CalibrationCheckResult) -> None:
        green = "\033[1;32m"
        red = "\033[1;31m"
        yellow = "\033[1;33m"
        reset = "\033[0m"

        color = green if result.ok else red
        print(f"\n{color}{'=' * 72}{reset}")
        print(f"{color}{result.message}{reset}")
        print(f"{color}{'=' * 72}{reset}")

        if result.camera_poses:
            print("\nCamera cube poses:")
            for cam_id, pose in sorted(result.camera_poses.items()):
                normal_bits = ", ".join(
                    f"{tag_id}:{score:.2f}" for tag_id, score in sorted(pose.normal_checks.items())
                )
                if not normal_bits:
                    normal_bits = "n/a"
                print(
                    f"  cam{cam_id}: tags={pose.tag_ids}, points={pose.num_points}, "
                    f"reproj={pose.reprojection_error_px:.3f}px, "
                    f"normal_min={pose.normal_facing_min_cos:.2f}, normals=[{normal_bits}]"
                )

        if result.pair_checks:
            print("\nCamera-pair deltas:")
            for pair in result.pair_checks:
                status = "OK" if pair.ok else "FAIL"
                status_color = green if pair.ok else red
                print(
                    f"  {status_color}{status}{reset} cam{pair.cam_a}->cam{pair.cam_b}: "
                    f"rot={pair.rotation_error_deg:.3f}deg, "
                    f"trans={pair.translation_error_mm:.1f}mm"
                )

        if result.failed_cameras:
            print(f"\n{yellow}Suspect cameras/pairs include: {result.failed_cameras}{reset}")
