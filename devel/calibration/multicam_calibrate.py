#!/usr/bin/env python3
"""
Graph-Based Multi-Camera Calibration using ChArUco Board

Two-stage calibration:
  Stage 1: Calibrate intrinsic parameters per camera from separate captures
  Stage 2: Build a pairwise extrinsic graph from stereo session directories,
           then compose transformations via shortest-path (Dijkstra) to get
           all cameras into a single global reference frame.

Each session directory corresponds to a stereo capture session (2 cameras).
For a 4-camera setup, run 3 sessions:
  Session 1: cameras 1,3 -> session_cam1_cam3/
  Session 2: cameras 2,4 -> session_cam2_cam4/
  Session 3: cameras 1,2 -> session_cam1_cam2/  (or 3,4)

Output NPZ format:
  K1, dist1, K2, dist2, ..., KN, distN
  R_1_to_ref, t_1_to_ref, R_2_to_ref, t_2_to_ref, ...
  ref_camera, num_cameras
  (plus backward-compatible R_1_to_2, t_1_to_2 if applicable)

Usage:
    python calibration/multicam_calibrate.py \\
        --intrinsic-dir-1 ./intrinsic_cam1 --intrinsic-dir-2 ./intrinsic_cam2 \\
        --intrinsic-dir-3 ./intrinsic_cam3 --intrinsic-dir-4 ./intrinsic_cam4 \\
        --session-dirs ./session_cam1_cam3 ./session_cam2_cam4 ./session_cam1_cam2 \\
        --output multicam_calibration.npz --num-cameras 4 --ref-camera 1
"""

import argparse
import heapq
from itertools import combinations
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import yaml


class MulticamCalibrator:
    def __init__(self, args):
        self.args = args
        self.num_cameras = args.num_cameras
        self.setup_charuco_board()

        self.image_size = None

        # Per-camera intrinsics
        self.intrinsics = {}  # cam_idx -> (K, dist, error)
        self.intrinsic_diagnostics = {}
        self.capture_factory_intrinsics = {}

        # Pairwise extrinsics graph
        # edges[(i,j)] = (R, T, rms, num_pairs)  where P_j = R @ P_i + T
        self.edges = {}
        self.pair_diagnostics = {}
        self.cycle_diagnostics = []
        self.pose_diversity = {}
        self.global_quality = {}

    def setup_charuco_board(self):
        """Initialize ChArUco board."""
        aruco_dict_map = {
            '4X4_50': cv2.aruco.DICT_4X4_50,
            '4X4_100': cv2.aruco.DICT_4X4_100,
            '4X4_250': cv2.aruco.DICT_4X4_250,
            '4X4_1000': cv2.aruco.DICT_4X4_1000,
            '5X5_50': cv2.aruco.DICT_5X5_50,
            '5X5_100': cv2.aruco.DICT_5X5_100,
            '5X5_250': cv2.aruco.DICT_5X5_250,
            '5X5_1000': cv2.aruco.DICT_5X5_1000,
            '6X6_50': cv2.aruco.DICT_6X6_50,
            '6X6_100': cv2.aruco.DICT_6X6_100,
            '6X6_250': cv2.aruco.DICT_6X6_250,
            '6X6_1000': cv2.aruco.DICT_6X6_1000,
        }

        aruco_dict_id = aruco_dict_map.get(self.args.aruco_dict, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)

        self.board = cv2.aruco.CharucoBoard(
            (self.args.squares_x, self.args.squares_y),
            self.args.square_length,
            self.args.marker_length,
            self.aruco_dict
        )

        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.charuco_detector = cv2.aruco.CharucoDetector(
            self.board, cv2.aruco.CharucoParameters(),
            self.detector_params, cv2.aruco.RefineParameters()
        )

        print(f"ChArUco Board: {self.args.squares_x} x {self.args.squares_y}, "
              f"square={self.args.square_length}m, marker={self.args.marker_length}m")

    def detect_charuco_in_image(self, image_path):
        """Detect ChArUco corners in an image file."""
        image = cv2.imread(str(image_path))
        if image is None:
            return None, None

        if self.image_size is None:
            self.image_size = (image.shape[1], image.shape[0])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        charuco_corners, charuco_ids, _, _ = self.charuco_detector.detectBoard(gray)

        if charuco_corners is None or charuco_ids is None:
            return None, None

        charuco_corners = np.asarray(charuco_corners, dtype=np.float32).reshape(
            -1, 1, 2
        )
        charuco_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1)
        if (
            len(charuco_corners) < 4
            or len(charuco_corners) != len(charuco_ids)
            or len(np.unique(charuco_ids)) != len(charuco_ids)
            or not np.all(np.isfinite(charuco_corners))
        ):
            return None, None
        return charuco_corners, charuco_ids

        return None, None

    # --- Stage 1: Intrinsic Calibration ---

    def calibrate_intrinsics(
        self,
        cam_idx,
        intrinsic_dir,
        initial_camera_matrix=None,
    ):
        """Calibrate intrinsics for a single camera."""
        intrinsic_path = Path(intrinsic_dir)
        if not intrinsic_path.exists():
            raise FileNotFoundError(f"Intrinsic directory not found: {intrinsic_path}")

        images = sorted(intrinsic_path.glob("*.png"))
        print(f"\n  Camera {cam_idx + 1}: Processing {len(images)} intrinsic images...")

        all_corners = []
        all_ids = []
        successful = 0

        for img_path in images:
            corners, ids = self.detect_charuco_in_image(img_path)
            if corners is not None:
                all_corners.append(corners)
                all_ids.append(ids)
                successful += 1

        print(f"  Camera {cam_idx + 1}: {successful}/{len(images)} images successful")

        if successful < 5:
            raise ValueError(f"Camera {cam_idx + 1}: Only {successful} images. Need at least 5.")

        # Prepare object and image points
        obj_points = []
        img_points = []
        for corners, ids in zip(all_corners, all_ids):
            obj_pts = self.board.getChessboardCorners()[ids.flatten()]
            obj_points.append(obj_pts.astype(np.float32))
            img_points.append(corners.astype(np.float32))

        def run_calibration(selected_obj_points, selected_img_points):
            flags = 0
            camera_matrix = None
            distortion = None
            if initial_camera_matrix is not None:
                camera_matrix = np.asarray(
                    initial_camera_matrix, dtype=np.float64
                ).copy()
                distortion = np.zeros((5, 1), dtype=np.float64)
                flags = (
                    cv2.CALIB_USE_INTRINSIC_GUESS
                    | cv2.CALIB_FIX_ASPECT_RATIO
                )
            return cv2.calibrateCameraExtended(
                selected_obj_points,
                selected_img_points,
                self.image_size,
                camera_matrix,
                distortion,
                flags=flags,
            )

        calibration = run_calibration(obj_points, img_points)
        ret, K, dist, rvecs, tvecs = calibration[:5]
        per_view_errors = np.asarray(calibration[-1]).reshape(-1)

        median_error = float(np.median(per_view_errors))
        mad_error = float(
            np.median(np.abs(per_view_errors - median_error))
        )
        robust_sigma = 1.4826 * mad_error
        outlier_threshold = max(
            1.0,
            median_error + 3.0 * max(robust_sigma, 0.05),
        )
        keep_indices = np.flatnonzero(per_view_errors <= outlier_threshold)
        minimum_kept = max(12, int(np.ceil(0.75 * len(obj_points))))
        if len(keep_indices) < len(obj_points) and len(keep_indices) >= minimum_kept:
            print(
                f"  Camera {cam_idx + 1}: rejecting "
                f"{len(obj_points) - len(keep_indices)} high-error intrinsic views "
                f"(threshold={outlier_threshold:.3f}px)"
            )
            obj_points = [obj_points[index] for index in keep_indices]
            img_points = [img_points[index] for index in keep_indices]
            calibration = run_calibration(obj_points, img_points)
            ret, K, dist, rvecs, tvecs = calibration[:5]
            per_view_errors = np.asarray(calibration[-1]).reshape(-1)

        image_diagonal = float(np.hypot(*self.image_size))
        normalized_centers = np.asarray(
            [
                np.mean(points.reshape(-1, 2), axis=0) / self.image_size
                for points in img_points
            ],
            dtype=np.float64,
        )
        center_span = float(
            np.linalg.norm(
                np.ptp(normalized_centers, axis=0)
                * np.asarray(self.image_size, dtype=np.float64)
            )
            / image_diagonal
        )
        normals = []
        depths = []
        for rotation_vector, translation_vector in zip(rvecs, tvecs):
            rotation, _ = cv2.Rodrigues(rotation_vector)
            normals.append(rotation[:, 2])
            depths.append(float(np.asarray(translation_vector).reshape(3)[2]))
        normal_span = 0.0
        for first_index in range(len(normals)):
            for second_index in range(first_index + 1, len(normals)):
                cosine = np.clip(
                    np.dot(normals[first_index], normals[second_index]),
                    -1.0,
                    1.0,
                )
                normal_span = max(
                    normal_span,
                    float(np.degrees(np.arccos(cosine))),
                )
        depth_span = (
            float(np.ptp(depths)) if len(depths) > 1 else 0.0
        )

        rms_error = float(ret)
        self.intrinsic_diagnostics[cam_idx] = {
            "accepted_views": len(obj_points),
            "rejected_views": successful - len(obj_points),
            "rms_px": rms_error,
            "per_view_median_px": float(np.median(per_view_errors)),
            "per_view_p95_px": float(np.percentile(per_view_errors, 95)),
            "normal_span_deg": normal_span,
            "image_center_span_ratio": center_span,
            "depth_span_m": depth_span,
        }

        print(
            f"  Camera {cam_idx + 1}: RMS={rms_error:.4f}px, "
            f"view_P95={np.percentile(per_view_errors, 95):.4f}px, "
              f"fx={K[0, 0]:.1f}, fy={K[1, 1]:.1f}")
        print(
            f"    diversity: normal={normal_span:.1f}deg, "
            f"image span={center_span:.2f}, depth span={depth_span:.2f}m"
        )

        self.intrinsics[cam_idx] = (K, dist, rms_error)
        return K, dist

    # --- Stage 2: Pairwise Extrinsics ---

    def load_multicam_captures(self):
        """
        Load capture sets from multiple session directories.

        Each session directory contains camera_X/ subdirectories (e.g.
        camera_1/ and camera_3/ for a session capturing cameras 1 and 3).
        Images within a session are matched by sorted index.

        Returns list of dicts, one per capture set:
          [{cam_idx: (corners, ids), ...}, ...]
        """
        all_capture_sets = []

        for session_dir in self.args.session_dirs:
            session_path = Path(session_dir)
            if not session_path.exists():
                print(f"  Warning: Session dir not found: {session_path}")
                continue

            session_info_path = session_path / "session_info.json"
            if session_info_path.exists():
                with open(
                    session_info_path, "r", encoding="utf-8"
                ) as session_info_file:
                    session_info = json.load(session_info_file)
                for camera_id, profile in session_info.get(
                    "factory_color_intrinsics", {}
                ).items():
                    self.capture_factory_intrinsics[int(camera_id) - 1] = profile

            # Discover which cameras are in this session from directory names
            cam_dirs = {}
            for d in sorted(session_path.iterdir()):
                if d.is_dir() and d.name.startswith('camera_'):
                    try:
                        cam_idx = int(d.name.split('_')[1]) - 1  # 0-indexed
                    except (IndexError, ValueError):
                        continue
                    images = sorted(d.glob("*.png"))
                    if images:
                        cam_dirs[cam_idx] = images

            if len(cam_dirs) < 2:
                print(f"  Warning: Session {session_dir} has < 2 cameras with images, skipping")
                continue

            cam_indices = sorted(cam_dirs.keys())
            cam_names = [str(c + 1) for c in cam_indices]
            num_captures = min(len(cam_dirs[c]) for c in cam_indices)
            print(f"\n  Session {session_path.name}: cameras [{', '.join(cam_names)}], "
                  f"{num_captures} capture sets")

            session_sets = 0
            for cap_idx in range(num_captures):
                detections = {}
                for cam_idx in cam_indices:
                    corners, ids = self.detect_charuco_in_image(cam_dirs[cam_idx][cap_idx])
                    if corners is not None:
                        detections[cam_idx] = (corners, ids)

                if len(detections) >= 2:
                    all_capture_sets.append(detections)
                    session_sets += 1

            print(f"    {session_sets} sets with >= 2 camera detections")

        print(f"\n  Total across all sessions: {len(all_capture_sets)} capture sets")
        self.validate_master_intrinsics_against_capture_profiles()
        return all_capture_sets

    def validate_master_intrinsics_against_capture_profiles(self):
        """Reject a master calibration that is inconsistent with camera profiles."""
        if not self.capture_factory_intrinsics:
            print(
                "  Warning: capture profiles do not contain factory intrinsics; "
                "master-intrinsic plausibility cannot be checked for this old run."
            )
            return

        maximum_focal_deviation = float(
            getattr(self.args, "max_master_focal_deviation", 0.03)
        )
        failures = []
        for camera_index, profile in sorted(
            self.capture_factory_intrinsics.items()
        ):
            if camera_index not in self.intrinsics:
                continue
            camera_matrix = self.intrinsics[camera_index][0]
            focal_deviation = max(
                abs(camera_matrix[0, 0] / profile["fx"] - 1.0),
                abs(camera_matrix[1, 1] / profile["fy"] - 1.0),
            )
            print(
                f"  Camera {camera_index + 1}: master/factory focal delta="
                f"{100.0 * focal_deviation:.2f}%"
            )
            if focal_deviation > maximum_focal_deviation:
                failures.append(
                    f"cam{camera_index + 1} "
                    f"{100.0 * focal_deviation:.2f}%"
                )
        if failures:
            raise RuntimeError(
                "Master intrinsics are implausibly far from the active camera "
                "profiles (limit "
                f"{100.0 * maximum_focal_deviation:.1f}%): "
                + ", ".join(failures)
                + ". Re-run record_intrinsic.py with full image coverage, "
                "large two-axis board tilts, and varied depth before solving "
                "daily extrinsics."
            )

    def calibrate_pairwise_extrinsics(self, capture_sets):
        """
        For each camera pair with enough shared captures, run stereoCalibrate.
        """
        # Collect matched corners per camera pair
        pair_data = {}  # (i, j) -> list of (obj_pts, img_pts_i, img_pts_j)

        for cap_set in capture_sets:
            cam_indices = sorted(cap_set.keys())

            for a_pos in range(len(cam_indices)):
                for b_pos in range(a_pos + 1, len(cam_indices)):
                    i = cam_indices[a_pos]
                    j = cam_indices[b_pos]

                    corners_i, ids_i = cap_set[i]
                    corners_j, ids_j = cap_set[j]

                    # Find common corner IDs
                    ids_i_flat = ids_i.flatten()
                    ids_j_flat = ids_j.flatten()
                    common_ids = np.intersect1d(ids_i_flat, ids_j_flat)

                    if len(common_ids) < 4:
                        continue

                    indices_i = np.array([np.where(ids_i_flat == cid)[0][0] for cid in common_ids])
                    indices_j = np.array([np.where(ids_j_flat == cid)[0][0] for cid in common_ids])

                    matched_corners_i = corners_i[indices_i]
                    matched_corners_j = corners_j[indices_j]
                    obj_pts = self.board.getChessboardCorners()[common_ids]

                    if (i, j) not in pair_data:
                        pair_data[(i, j)] = []

                    pair_data[(i, j)].append((
                        obj_pts.astype(np.float32),
                        matched_corners_i.astype(np.float32),
                        matched_corners_j.astype(np.float32),
                    ))

        # Run stereoCalibrate for each pair with enough data
        min_pairs = self.args.min_pairs
        print(f"\n  Pairwise extrinsic calibration (min {min_pairs} shared captures):")

        for (i, j), data_list in sorted(pair_data.items()):
            if len(data_list) < min_pairs:
                print(f"    Cameras ({i + 1},{j + 1}): Only {len(data_list)} shared captures, skipping")
                continue

            Ki, disti = self.intrinsics[i][0], self.intrinsics[i][1]
            Kj, distj = self.intrinsics[j][0], self.intrinsics[j][1]

            obj_points = [d[0] for d in data_list]
            img_points_i = [d[1] for d in data_list]
            img_points_j = [d[2] for d in data_list]

            ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                obj_points, img_points_i, img_points_j,
                Ki, disti, Kj, distj,
                self.image_size,
                flags=cv2.CALIB_FIX_INTRINSIC,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
            )

            # R, T: P_j = R @ P_i + T
            self.edges[(i, j)] = (R, T.flatten(), ret, len(data_list))

            baseline = np.linalg.norm(T)
            print(f"    Cameras ({i + 1},{j + 1}): RMS={ret:.4f}px, "
                  f"baseline={baseline:.4f}m, {len(data_list)} pairs")

            diagnostics = self._relative_pose_diagnostics(i, j, data_list, R, T.flatten())
            self.pair_diagnostics[(i, j)] = diagnostics
            if diagnostics["valid_pose_samples"]:
                print(
                    "      independent PnP spread: "
                    f"rot median/P95={diagnostics['rotation_median_deg']:.2f}/"
                    f"{diagnostics['rotation_p95_deg']:.2f}deg, "
                    f"trans median/P95={diagnostics['translation_median_mm']:.1f}/"
                    f"{diagnostics['translation_p95_mm']:.1f}mm"
                )

        self.pose_diversity = self._compute_pose_diversity(capture_sets)
        self.cycle_diagnostics = self.evaluate_cycle_consistency()

    @staticmethod
    def _rotation_delta_degrees(rotation_a, rotation_b):
        delta = rotation_a @ rotation_b.T
        value = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(value)))

    def _solve_board_pose(self, object_points, image_points, camera_index):
        camera_matrix, distortion = self.intrinsics[camera_index][0:2]
        object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
        image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        if len(object_points) < 4:
            return None

        candidates = []
        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            ok = False
        if ok:
            rotation, _ = cv2.Rodrigues(rvec)
            camera_points = (rotation @ object_points.T).T + np.asarray(tvec).reshape(1, 3)
            if np.min(camera_points[:, 2]) > 0:
                projected, _ = cv2.projectPoints(
                    object_points, rvec, tvec, camera_matrix, distortion
                )
                errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
                candidates.append(
                    (float(np.median(errors)), rotation, np.asarray(tvec).reshape(3))
                )

        if not candidates:
            try:
                result = cv2.solvePnPGeneric(
                    object_points,
                    image_points,
                    camera_matrix,
                    distortion,
                    flags=cv2.SOLVEPNP_IPPE,
                )
            except cv2.error:
                result = None
            if result is not None and bool(result[0]):
                pose_candidates = zip(result[1], result[2])
            else:
                pose_candidates = ()
            for rvec, tvec in pose_candidates:
                rotation, _ = cv2.Rodrigues(rvec)
                camera_points = (rotation @ object_points.T).T + np.asarray(tvec).reshape(1, 3)
                if np.min(camera_points[:, 2]) <= 0:
                    continue
                projected, _ = cv2.projectPoints(
                    object_points, rvec, tvec, camera_matrix, distortion
                )
                errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
                candidates.append(
                    (float(np.median(errors)), rotation, np.asarray(tvec).reshape(3))
                )

        if not candidates:
            return None
        _error, rotation, translation = min(candidates, key=lambda item: item[0])
        return rotation, translation

    def _relative_pose_diagnostics(self, i, j, data_list, stereo_rotation, stereo_translation):
        rotation_errors = []
        translation_errors = []
        for object_points, image_points_i, image_points_j in data_list:
            pose_i = self._solve_board_pose(object_points, image_points_i, i)
            pose_j = self._solve_board_pose(object_points, image_points_j, j)
            if pose_i is None or pose_j is None:
                continue

            rotation_i, translation_i = pose_i
            rotation_j, translation_j = pose_j
            relative_rotation = rotation_j @ rotation_i.T
            relative_translation = translation_j - relative_rotation @ translation_i
            rotation_errors.append(
                self._rotation_delta_degrees(relative_rotation, stereo_rotation)
            )
            translation_errors.append(
                float(np.linalg.norm(relative_translation - stereo_translation))
            )

        if not rotation_errors:
            return {
                "valid_pose_samples": 0,
                "rotation_median_deg": float("nan"),
                "rotation_p95_deg": float("nan"),
                "translation_median_mm": float("nan"),
                "translation_p95_mm": float("nan"),
            }

        return {
            "valid_pose_samples": len(rotation_errors),
            "rotation_median_deg": float(np.median(rotation_errors)),
            "rotation_p95_deg": float(np.percentile(rotation_errors, 95)),
            "translation_median_mm": float(np.median(translation_errors) * 1000.0),
            "translation_p95_mm": float(np.percentile(translation_errors, 95) * 1000.0),
        }

    def _compute_pose_diversity(self, capture_sets):
        per_camera = {camera_index: [] for camera_index in range(self.num_cameras)}
        board_points = self.board.getChessboardCorners()

        for capture_set in capture_sets:
            for camera_index, (corners, ids) in capture_set.items():
                object_points = board_points[ids.flatten()]
                pose = self._solve_board_pose(object_points, corners, camera_index)
                if pose is None:
                    continue
                rotation, translation = pose
                center = np.mean(np.asarray(corners).reshape(-1, 2), axis=0)
                per_camera[camera_index].append(
                    (rotation[:, 2], translation, center)
                )

        diagnostics = {}
        image_diagonal = (
            float(np.hypot(*self.image_size)) if self.image_size is not None else 1.0
        )
        print("\n  ChArUco pose diversity:")
        for camera_index, samples in per_camera.items():
            if not samples:
                diagnostics[camera_index] = {
                    "views": 0,
                    "normal_span_deg": 0.0,
                    "center_span_ratio": 0.0,
                    "depth_span_m": 0.0,
                }
                continue

            normals = np.asarray([sample[0] for sample in samples])
            translations = np.asarray([sample[1] for sample in samples])
            centers = np.asarray([sample[2] for sample in samples])
            mean_normal = np.mean(normals, axis=0)
            mean_normal /= max(np.linalg.norm(mean_normal), 1e-12)
            normal_angles = np.degrees(
                np.arccos(np.clip(normals @ mean_normal, -1.0, 1.0))
            )
            center_span = np.linalg.norm(
                np.max(centers, axis=0) - np.min(centers, axis=0)
            ) / image_diagonal
            depth_span = float(np.ptp(translations[:, 2]))
            item = {
                "views": len(samples),
                "normal_span_deg": float(np.max(normal_angles)),
                "center_span_ratio": float(center_span),
                "depth_span_m": depth_span,
            }
            diagnostics[camera_index] = item
            warning = (
                item["normal_span_deg"] < 12.0
                or item["center_span_ratio"] < 0.18
                or item["depth_span_m"] < 0.12
            )
            suffix = "  [WEAK DIVERSITY]" if warning else ""
            print(
                f"    Camera {camera_index + 1}: views={item['views']}, "
                f"normal span={item['normal_span_deg']:.1f}deg, "
                f"image span={item['center_span_ratio']:.2f}, "
                f"depth span={item['depth_span_m']:.2f}m{suffix}"
            )
        return diagnostics

    def evaluate_cycle_consistency(self):
        diagnostics = []
        print("\n  Pairwise graph cycle consistency:")
        for i, j, k in combinations(range(self.num_cameras), 3):
            if any(
                self.get_transform(src, dst)[0] is None
                for src, dst in ((i, j), (j, k), (k, i))
            ):
                continue
            rotation_ij, translation_ij = self.get_transform(i, j)
            rotation_jk, translation_jk = self.get_transform(j, k)
            rotation_ki, translation_ki = self.get_transform(k, i)
            rotation_loop = rotation_ki @ rotation_jk @ rotation_ij
            translation_loop = (
                rotation_ki @ (rotation_jk @ translation_ij + translation_jk)
                + translation_ki
            )
            item = {
                "cycle": [i + 1, j + 1, k + 1, i + 1],
                "rotation_error_deg": self._rotation_delta_degrees(
                    rotation_loop, np.eye(3)
                ),
                "translation_error_mm": float(
                    np.linalg.norm(translation_loop) * 1000.0
                ),
            }
            diagnostics.append(item)
            print(
                f"    {i + 1}->{j + 1}->{k + 1}->{i + 1}: "
                f"rot={item['rotation_error_deg']:.2f}deg, "
                f"trans={item['translation_error_mm']:.1f}mm"
            )
        if not diagnostics:
            print("    No complete three-camera cycles available.")
        return diagnostics

    def refine_global_bundle_adjustment(self, capture_sets, initial_transforms, ref_cam):
        try:
            from scipy.optimize import least_squares
            from scipy.sparse import lil_matrix
        except ImportError as exc:
            raise RuntimeError(
                "Global ChArUco bundle adjustment requires scipy."
            ) from exc

        board_points = self.board.getChessboardCorners().astype(np.float64)
        camera_slices = {}
        board_slices = []
        parameters = []

        for camera_index in range(self.num_cameras):
            if camera_index == ref_cam:
                continue
            transform = initial_transforms.get(camera_index)
            if transform is None:
                raise RuntimeError(
                    f"Camera {camera_index + 1} is unreachable from camera {ref_cam + 1}."
                )
            rotation, translation = transform
            rvec, _ = cv2.Rodrigues(rotation)
            start = len(parameters)
            parameters.extend(rvec.reshape(3))
            parameters.extend(np.asarray(translation).reshape(3))
            camera_slices[camera_index] = slice(start, start + 6)

        usable_capture_sets = []
        for capture_set in capture_sets:
            board_to_ref = None
            for camera_index, (corners, ids) in capture_set.items():
                object_points = board_points[ids.flatten()]
                board_to_camera = self._solve_board_pose(
                    object_points, corners, camera_index
                )
                camera_transform = initial_transforms.get(camera_index)
                if board_to_camera is None or camera_transform is None:
                    continue
                rotation_board_camera, translation_board_camera = board_to_camera
                rotation_ref_camera, translation_ref_camera = camera_transform
                rotation_board_ref = (
                    rotation_ref_camera.T @ rotation_board_camera
                )
                translation_board_ref = rotation_ref_camera.T @ (
                    translation_board_camera - translation_ref_camera
                )
                board_to_ref = (rotation_board_ref, translation_board_ref)
                break

            if board_to_ref is None:
                continue
            rotation, translation = board_to_ref
            rvec, _ = cv2.Rodrigues(rotation)
            start = len(parameters)
            parameters.extend(rvec.reshape(3))
            parameters.extend(np.asarray(translation).reshape(3))
            board_slices.append(slice(start, start + 6))
            usable_capture_sets.append(capture_set)

        if len(usable_capture_sets) < self.args.min_pairs:
            raise RuntimeError(
                f"Only {len(usable_capture_sets)} capture sets could initialize bundle adjustment."
            )

        parameters = np.asarray(parameters, dtype=np.float64)
        residual_count = sum(
            2 * len(ids)
            for capture_set in usable_capture_sets
            for _camera_index, (_corners, ids) in capture_set.items()
        )
        sparsity = lil_matrix((residual_count, len(parameters)), dtype=np.int8)
        row = 0
        for capture_index, capture_set in enumerate(usable_capture_sets):
            board_slice = board_slices[capture_index]
            for camera_index, (_corners, ids) in capture_set.items():
                count = 2 * len(ids)
                sparsity[row:row + count, board_slice] = 1
                if camera_index != ref_cam:
                    sparsity[row:row + count, camera_slices[camera_index]] = 1
                row += count

        def unpack_camera(values, camera_index):
            if camera_index == ref_cam:
                return np.eye(3), np.zeros(3)
            chunk = values[camera_slices[camera_index]]
            rotation, _ = cv2.Rodrigues(chunk[:3])
            return rotation, chunk[3:6]

        def residuals(values, return_metadata=False):
            output = []
            metadata = []
            for capture_index, capture_set in enumerate(usable_capture_sets):
                board_chunk = values[board_slices[capture_index]]
                rotation_board_ref, _ = cv2.Rodrigues(board_chunk[:3])
                translation_board_ref = board_chunk[3:6]
                for camera_index, (corners, ids) in capture_set.items():
                    rotation_ref_camera, translation_ref_camera = unpack_camera(
                        values, camera_index
                    )
                    object_points = board_points[ids.flatten()]
                    points_ref = (
                        rotation_board_ref @ object_points.T
                    ).T + translation_board_ref
                    points_camera = (
                        rotation_ref_camera @ points_ref.T
                    ).T + translation_ref_camera
                    camera_matrix, distortion = self.intrinsics[camera_index][0:2]
                    projected, _ = cv2.projectPoints(
                        points_camera,
                        np.zeros(3),
                        np.zeros(3),
                        camera_matrix,
                        distortion,
                    )
                    delta = (
                        projected.reshape(-1, 2)
                        - np.asarray(corners).reshape(-1, 2)
                    )
                    output.extend(delta.reshape(-1))
                    if return_metadata:
                        metadata.extend(
                            (camera_index, float(np.linalg.norm(error)))
                            for error in delta
                        )
            values_out = np.asarray(output, dtype=np.float64)
            if not np.all(np.isfinite(values_out)):
                raise RuntimeError("Non-finite residuals encountered in global calibration.")
            if return_metadata:
                return values_out, metadata
            return values_out

        print(
            f"\n  Global bundle adjustment: {len(usable_capture_sets)} board poses, "
            f"{residual_count // 2} observed corners, {len(parameters)} parameters"
        )
        result = least_squares(
            residuals,
            parameters,
            jac_sparsity=sparsity.tocsr(),
            method="trf",
            loss=getattr(self.args, "charuco_robust_loss", "soft_l1"),
            f_scale=getattr(self.args, "charuco_robust_scale_px", 1.0),
            x_scale="jac",
            max_nfev=getattr(self.args, "charuco_max_nfev", 300),
            verbose=0,
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(
                f"Global ChArUco bundle adjustment failed: {result.message}"
            )

        transforms = {ref_cam: (np.eye(3), np.zeros(3))}
        for camera_index in range(self.num_cameras):
            if camera_index == ref_cam:
                continue
            transforms[camera_index] = unpack_camera(result.x, camera_index)

        _flat_residuals, metadata = residuals(result.x, return_metadata=True)
        all_errors = np.asarray([item[1] for item in metadata], dtype=np.float64)
        if len(all_errors) == 0 or not np.all(np.isfinite(all_errors)):
            raise RuntimeError(
                "Global ChArUco calibration produced no finite reprojection errors."
            )
        quality = {
            "optimizer_message": str(result.message),
            "cost": float(result.cost),
            "observed_corners": int(len(all_errors)),
            "median_reprojection_error_px": float(np.median(all_errors)),
            "p95_reprojection_error_px": float(np.percentile(all_errors, 95)),
            "max_reprojection_error_px": float(np.max(all_errors)),
            "per_camera": {},
        }
        for camera_index in range(self.num_cameras):
            camera_errors = np.asarray(
                [error for cam, error in metadata if cam == camera_index],
                dtype=np.float64,
            )
            if len(camera_errors) == 0:
                raise RuntimeError(
                    "Global ChArUco calibration has no accepted observations for "
                    f"camera {camera_index + 1}."
                )
            quality["per_camera"][camera_index] = {
                "corners": int(len(camera_errors)),
                "median_reprojection_error_px": float(np.median(camera_errors)),
                "p95_reprojection_error_px": float(
                    np.percentile(camera_errors, 95)
                ),
            }

        self.global_quality = quality
        print(
            "  Global BA reprojection: "
            f"median={quality['median_reprojection_error_px']:.3f}px, "
            f"P95={quality['p95_reprojection_error_px']:.3f}px, "
            f"max={quality['max_reprojection_error_px']:.3f}px"
        )
        for camera_index, item in quality["per_camera"].items():
            print(
                f"    Camera {camera_index + 1}: corners={item['corners']}, "
                f"median={item['median_reprojection_error_px']:.3f}px, "
                f"P95={item['p95_reprojection_error_px']:.3f}px"
            )

        max_p95 = getattr(self.args, "max_global_reproj_p95_px", 2.5)
        worst_camera_index, worst_camera_quality = max(
            quality["per_camera"].items(),
            key=lambda item: item[1]["p95_reprojection_error_px"],
        )
        aggregate_p95 = quality["p95_reprojection_error_px"]
        worst_camera_p95 = worst_camera_quality["p95_reprojection_error_px"]
        if aggregate_p95 > max_p95 or worst_camera_p95 > max_p95:
            raise RuntimeError(
                "Global ChArUco calibration failed the reprojection quality gate: "
                f"aggregate P95={aggregate_p95:.3f}px, "
                f"worst camera={worst_camera_index + 1} "
                f"P95={worst_camera_p95:.3f}px, limit={max_p95:.3f}px. "
                "Do not use this calibration."
            )
        return transforms

    # --- Graph-based path composition ---

    def build_adjacency(self):
        """
        Build a weighted adjacency list from pairwise edges.
        Weight = RMS error (lower is better).
        Include both directions (i->j and j->i with inverted transform).
        """
        adj = {i: [] for i in range(self.num_cameras)}

        for (i, j), (R, T, rms, n_pairs) in self.edges.items():
            # Forward: i -> j
            adj[i].append((j, rms))
            # Backward: j -> i
            adj[j].append((i, rms))

        return adj

    def get_transform(self, src, dst):
        """
        Get (R, T) such that P_dst = R @ P_src + T.
        Handles both forward and inverse lookups.
        """
        if (src, dst) in self.edges:
            R, T, _, _ = self.edges[(src, dst)]
            return R, T
        elif (dst, src) in self.edges:
            R_ji, T_ji, _, _ = self.edges[(dst, src)]
            # Invert: P_dst = R_ji^T @ P_src - R_ji^T @ T_ji
            R_inv = R_ji.T
            T_inv = -R_ji.T @ T_ji
            return R_inv, T_inv
        else:
            return None, None

    def dijkstra_path(self, ref_cam):
        """
        Find lowest-RMS path from ref_cam to every other camera.
        Returns: {cam_idx: [ref_cam, ..., cam_idx]}
        """
        adj = self.build_adjacency()

        dist = {i: float('inf') for i in range(self.num_cameras)}
        dist[ref_cam] = 0.0
        prev = {i: None for i in range(self.num_cameras)}
        visited = set()

        pq = [(0.0, ref_cam)]

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            for v, weight in adj[u]:
                new_dist = d + weight
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(pq, (new_dist, v))

        # Reconstruct paths
        paths = {}
        for cam in range(self.num_cameras):
            if cam == ref_cam:
                paths[cam] = [ref_cam]
                continue
            if prev[cam] is None:
                paths[cam] = None  # Unreachable
                continue

            path = []
            node = cam
            while node is not None:
                path.append(node)
                node = prev[node]
            paths[cam] = list(reversed(path))

        return paths, dist

    def compose_transforms(self, ref_cam):
        """
        Compose pairwise transforms along Dijkstra paths to get
        R_ref_to_i, t_ref_to_i for each camera i.

        Dijkstra returns paths [ref, ..., cam_i]. Composing forward along
        that path gives a ref-to-cam transform:
            P_cam_i = R_composed @ P_ref + t_composed

        NOTE: Despite the NPZ key names ``R_i_to_ref`` / ``t_i_to_ref``,
        the stored convention is **ref -> cam**:
            P_cam = R @ P_ref + t
        To convert a point from camera frame to the reference frame, invert:
            P_ref = R^T @ P_cam - R^T @ t
        """
        paths, dists = self.dijkstra_path(ref_cam)

        transforms = {}  # cam_idx -> (R_to_ref, t_to_ref)

        for cam in range(self.num_cameras):
            if cam == ref_cam:
                transforms[cam] = (np.eye(3), np.zeros(3))
                continue

            path = paths[cam]
            if path is None:
                print(f"  WARNING: Camera {cam + 1} is unreachable from reference camera {ref_cam + 1}!")
                transforms[cam] = None
                continue

            # Compose transforms along path
            R_composed = np.eye(3)
            t_composed = np.zeros(3)

            for step in range(len(path) - 1):
                src = path[step]
                dst = path[step + 1]

                R_step, T_step = self.get_transform(src, dst)
                if R_step is None:
                    print(f"  WARNING: No transform between cameras {src + 1} and {dst + 1}")
                    transforms[cam] = None
                    break

                # Compose: P_new = R_step @ P_old + T_step
                # Combined with previous: P_final = R_step @ (R_prev @ P + t_prev) + T_step
                #                                  = R_step @ R_prev @ P + R_step @ t_prev + T_step
                t_composed = R_step @ t_composed + T_step
                R_composed = R_step @ R_composed
            else:
                # Path goes ref -> ... -> cam, so this gives us ref_to_cam
                transforms[cam] = (R_composed, t_composed)

            path_str = " -> ".join(str(p + 1) for p in path)
            if transforms[cam] is not None:
                print(f"  Camera {cam + 1} -> ref: path [{path_str}], "
                      f"total RMS weight={dists[cam]:.4f}")

        return transforms

    # --- Auto-select reference camera ---

    def auto_select_ref_camera(self):
        """Select reference camera with highest connectivity and lowest average RMS."""
        adj = self.build_adjacency()

        best_cam = 0
        best_score = float('inf')

        for cam in range(self.num_cameras):
            neighbors = adj[cam]
            if len(neighbors) == 0:
                continue

            connectivity = len(neighbors)
            avg_rms = np.mean([rms for _, rms in neighbors])

            # Score: lower is better. Penalize low connectivity heavily.
            score = avg_rms / connectivity

            if score < best_score:
                best_score = score
                best_cam = cam

        return best_cam

    # --- Save ---

    def save_calibration(self, transforms, ref_cam):
        """Save calibration results to NPZ file."""
        data = {
            'ref_camera': ref_cam + 1,  # 1-indexed for user-facing
            'num_cameras': self.num_cameras,
            'image_size': np.array(self.image_size) if self.image_size else np.array([0, 0]),
            'transform_convention': np.array(
                'legacy R_i_to_ref/t_i_to_ref keys store ref_to_camera'
            ),
        }

        for cam_idx in range(self.num_cameras):
            cam_num = cam_idx + 1
            K, dist, error = self.intrinsics[cam_idx]
            data[f'K{cam_num}'] = K
            data[f'dist{cam_num}'] = dist

            if transforms.get(cam_idx) is not None:
                R_to_ref, t_to_ref = transforms[cam_idx]
                data[f'R_{cam_num}_to_ref'] = R_to_ref
                data[f't_{cam_num}_to_ref'] = t_to_ref
                transform_ref_to_camera = np.eye(4)
                transform_ref_to_camera[:3, :3] = R_to_ref
                transform_ref_to_camera[:3, 3] = t_to_ref
                data[f'T_ref_to_cam{cam_num}'] = transform_ref_to_camera
                data[f'T_cam{cam_num}_to_ref'] = np.linalg.inv(
                    transform_ref_to_camera
                )

        # Backward-compatible keys for 2-camera pipeline
        if (
            self.num_cameras >= 2
            and transforms.get(0) is not None
            and transforms.get(1) is not None
        ):
            rotation_ref_camera_1, translation_ref_camera_1 = transforms[0]
            rotation_ref_camera_2, translation_ref_camera_2 = transforms[1]
            rotation_1_to_2 = (
                rotation_ref_camera_2 @ rotation_ref_camera_1.T
            )
            translation_1_to_2 = (
                translation_ref_camera_2
                - rotation_1_to_2 @ translation_ref_camera_1
            )
            data['R_1_to_2'] = rotation_1_to_2
            data['t_1_to_2'] = translation_1_to_2

        np.savez(self.args.output, **data)
        print(f"\nSaved calibration to {self.args.output}")

    def save_calibration_yaml(self, transforms, ref_cam):
        """Save calibration to a readable YAML format containing full parameter matrices."""
        yaml_data = {
            'calibration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'image_width': int(self.image_size[0]) if self.image_size else 0,
            'image_height': int(self.image_size[1]) if self.image_size else 0,
            'board_config': {
                'squares_x': int(self.args.squares_x),
                'squares_y': int(self.args.squares_y),
                'square_length': float(self.args.square_length),
                'marker_length': float(self.args.marker_length),
                'aruco_dict': self.args.aruco_dict
            },
            'num_cameras': self.num_cameras,
            'reference_camera': ref_cam + 1,
            'cameras': {}
        }

        for cam_idx in range(self.num_cameras):
            cam_num = cam_idx + 1
            K, dist, error = self.intrinsics[cam_idx]
            
            cam_data = {
                'camera_matrix': K.tolist(),
                'distortion_coefficients': dist.flatten().tolist(),
                'reprojection_error': float(error)
            }

            if cam_idx != ref_cam and transforms.get(cam_idx) is not None:
                R_to_ref, t_to_ref = transforms[cam_idx]
                R_ref_to_cam = R_to_ref
                t_ref_to_cam = t_to_ref
                
                cam_data['transform_from_ref'] = {
                    'rotation_matrix': R_ref_to_cam.tolist(),
                    'translation_vector': t_ref_to_cam.flatten().tolist(),
                    'baseline_meters': float(np.linalg.norm(t_ref_to_cam))
                }
            
            yaml_data['cameras'][f'camera_{cam_num}'] = cam_data

        output_file = Path(self.args.output)
        yaml_path = output_file.with_name(output_file.stem + ".yaml")

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            
        print(f"Saved readable calibration parameters to {yaml_path}")

    def save_calibration_summary(self, transforms, ref_cam):
        """Save calibration metric summary to a JSON file."""
        summary = {
            "num_cameras": self.num_cameras,
            "reference_camera": ref_cam + 1,
            "cameras_calibrated": sum(1 for v in transforms.values() if v is not None),
            "intrinsics": {},
            "pairwise_edges": [],
            "transforms_from_ref": {},
            "pair_pose_diagnostics": {},
            "cycle_consistency": self.cycle_diagnostics,
            "pose_diversity": {},
            "global_bundle_adjustment": self.global_quality,
        }
        
        for cam_idx in range(self.num_cameras):
            _, _, error = self.intrinsics[cam_idx]
            status = "OK" if transforms.get(cam_idx) is not None else "UNREACHABLE"
            summary["intrinsics"][f"camera_{cam_idx + 1}"] = {
                "mean_reprojection_error_px": float(error),
                "status": status
            }
            
        for (i, j), (_, T, rms, n) in sorted(self.edges.items()):
            summary["pairwise_edges"].append({
                "camera_pair": [i + 1, j + 1],
                "rms_error_px": float(rms),
                "baseline_meters": float(np.linalg.norm(T)),
                "num_shared_captures": int(n)
            })
            diagnostics = self.pair_diagnostics.get((i, j))
            if diagnostics:
                summary["pair_pose_diagnostics"][f"camera_{i + 1}_camera_{j + 1}"] = diagnostics

        for camera_index, diagnostics in self.pose_diversity.items():
            summary["pose_diversity"][f"camera_{camera_index + 1}"] = diagnostics
            
        for cam_idx in range(self.num_cameras):
            if cam_idx == ref_cam:
                continue
            if transforms.get(cam_idx) is None:
                continue
                
            R_to_ref, t_to_ref = transforms[cam_idx]
            R_ref_to_cam = R_to_ref
            t_ref_to_cam = t_to_ref
            
            baseline = float(np.linalg.norm(t_ref_to_cam))
            rvec, _ = cv2.Rodrigues(R_ref_to_cam)
            angle = float(np.linalg.norm(rvec) * 180 / np.pi)
            
            summary["transforms_from_ref"][f"camera_{cam_idx + 1}"] = {
                "baseline_meters": baseline,
                "rotation_angle_degrees": angle
            }
            
        output_file = Path(self.args.output)
        summary_path = output_file.with_name(output_file.stem + "_summary.json")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4)
        print(f"\nSaved calibration summary metrics to {summary_path}")

    # --- Main ---

    def run(self):
        print(f"{'=' * 70}")
        print(f"MULTI-CAMERA CALIBRATION ({self.num_cameras} cameras)")
        print(f"{'=' * 70}")

        # Stage 1: Intrinsics
        print(f"\n{'=' * 70}")
        print("STAGE 1: INTRINSIC CALIBRATION")
        print(f"{'=' * 70}")

        for cam_idx in range(self.num_cameras):
            intrinsic_dir = getattr(self.args, f'intrinsic_dir_{cam_idx + 1}', None)
            if intrinsic_dir is None:
                # Fall back: search session dirs for camera_N/
                for session_dir in self.args.session_dirs:
                    candidate = Path(session_dir) / f"camera_{cam_idx + 1}"
                    if candidate.exists() and any(candidate.glob("*.png")):
                        intrinsic_dir = str(candidate)
                        print(f"\n  Camera {cam_idx + 1}: Using session dir for intrinsics "
                              f"(fallback: {session_dir})")
                        break
                if intrinsic_dir is None:
                    raise FileNotFoundError(
                        f"No intrinsic directory for camera {cam_idx + 1}. "
                        f"Provide --intrinsic-dir-{cam_idx + 1} or ensure a session dir "
                        f"contains camera_{cam_idx + 1}/")

            self.calibrate_intrinsics(cam_idx, intrinsic_dir)

        # Stage 2: Pairwise extrinsics
        print(f"\n{'=' * 70}")
        print("STAGE 2: PAIRWISE EXTRINSIC CALIBRATION")
        print(f"{'=' * 70}")

        capture_sets = self.load_multicam_captures()
        self.calibrate_pairwise_extrinsics(capture_sets)

        if len(self.edges) == 0:
            print("\nERROR: No pairwise calibrations succeeded. Check capture data.")
            return

        # Select reference camera
        if self.args.ref_camera is not None:
            ref_cam = self.args.ref_camera - 1  # Convert to 0-indexed
        else:
            ref_cam = self.auto_select_ref_camera()
            print(f"\n  Auto-selected reference camera: {ref_cam + 1}")

        print(f"\n  Reference camera: {ref_cam + 1}")

        # Compose transforms
        print(f"\n{'=' * 70}")
        print("TRANSFORM COMPOSITION (Dijkstra shortest path)")
        print(f"{'=' * 70}")

        transforms = self.compose_transforms(ref_cam)

        # Save
        self.save_calibration(transforms, ref_cam)
        self.save_calibration_yaml(transforms, ref_cam)

        # Quality summary
        print(f"\n{'=' * 70}")
        print("CALIBRATION SUMMARY")
        print(f"{'=' * 70}")

        print(f"  Reference camera: {ref_cam + 1}")
        print(f"  Cameras calibrated: {sum(1 for v in transforms.values() if v is not None)}/{self.num_cameras}")

        for cam_idx in range(self.num_cameras):
            K, dist, error = self.intrinsics[cam_idx]
            status = "OK" if transforms.get(cam_idx) is not None else "UNREACHABLE"
            print(f"  Camera {cam_idx + 1}: intrinsic error={error:.4f}px, status={status}")

        for (i, j), (R, T, rms, n) in sorted(self.edges.items()):
            print(f"  Edge ({i + 1},{j + 1}): RMS={rms:.4f}px, baseline={np.linalg.norm(T):.4f}m, {n} pairs")

        # Report transforms between reference camera and all other cameras
        print(f"\n  Transforms from reference camera {ref_cam + 1} to each camera:")
        for cam_idx in range(self.num_cameras):
            if cam_idx == ref_cam:
                continue
            if transforms.get(cam_idx) is None:
                print(f"\n  Camera {ref_cam + 1} -> Camera {cam_idx + 1}: UNREACHABLE")
                continue

            R_to_ref, t_to_ref = transforms[cam_idx]
            R_ref_to_cam = R_to_ref
            t_ref_to_cam = t_to_ref

            baseline = np.linalg.norm(t_ref_to_cam)
            rvec, _ = cv2.Rodrigues(R_ref_to_cam)
            angle = np.linalg.norm(rvec) * 180 / np.pi
            axis = rvec.flatten() / np.linalg.norm(rvec) if np.linalg.norm(rvec) > 0 else np.zeros(3)

            print(f"\n  Camera {ref_cam + 1} -> Camera {cam_idx + 1}:")
            print(f"    Rotation matrix R:\n      {R_ref_to_cam}")
            print(f"    Translation vector T (meters): {t_ref_to_cam}")
            print(f"    Baseline (distance between cameras): {baseline:.4f} meters")
            print(f"    Rotation: {angle:.2f} degrees around axis [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")

        self.save_calibration_summary(transforms, ref_cam)

        print(f"\n{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Graph-based multi-camera calibration using ChArUco boards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Intrinsic directories (one per camera)
    for i in range(1, 9):
        parser.add_argument(f'--intrinsic-dir-{i}', type=str, default=None,
                            help=f'Directory with intrinsic images for camera {i}')

    parser.add_argument('--session-dirs', type=str, nargs='+', required=True,
                        help='One or more session directories, each containing camera_X/ subdirs '
                             '(e.g. session_cam1_cam3/ session_cam2_cam4/ session_cam1_cam2/)')
    parser.add_argument('--output', type=str, default='multicam_calibration.npz',
                        help='Output calibration file')
    parser.add_argument('--num-cameras', type=int, default=4,
                        help='Number of cameras')
    parser.add_argument('--ref-camera', type=int, default=None,
                        help='Reference camera (1-indexed). Auto-select if not specified.')
    parser.add_argument('--min-pairs', type=int, default=5,
                        help='Minimum shared captures for a pairwise calibration')
    parser.add_argument(
        '--charuco-robust-loss',
        choices=('linear', 'soft_l1', 'huber', 'cauchy', 'arctan'),
        default='soft_l1',
    )
    parser.add_argument('--charuco-robust-scale-px', type=float, default=1.0)
    parser.add_argument('--charuco-max-nfev', type=int, default=300)
    parser.add_argument('--max-global-reproj-p95-px', type=float, default=2.5)

    # ChArUco board parameters
    parser.add_argument('--squares-x', type=int, default=3)
    parser.add_argument('--squares-y', type=int, default=4)
    parser.add_argument('--square-length', type=float, default=0.063)
    parser.add_argument('--marker-length', type=float, default=0.047)
    parser.add_argument('--aruco-dict', type=str, default='4X4_50',
                        choices=['4X4_50', '4X4_100', '4X4_250', '4X4_1000',
                                 '5X5_50', '5X5_100', '5X5_250', '5X5_1000',
                                 '6X6_50', '6X6_100', '6X6_250', '6X6_1000'])

    args = parser.parse_args()

    calibrator = MulticamCalibrator(args)
    calibrator.run()


if __name__ == '__main__':
    main()
