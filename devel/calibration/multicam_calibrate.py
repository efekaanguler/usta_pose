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
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime


class MulticamCalibrator:
    def __init__(self, args):
        self.args = args
        self.num_cameras = args.num_cameras
        self.setup_charuco_board()

        self.image_size = None

        # Per-camera intrinsics
        self.intrinsics = {}  # cam_idx -> (K, dist, error)

        # Pairwise extrinsics graph
        # edges[(i,j)] = (R, T, rms, num_pairs)  where P_j = R @ P_i + T
        self.edges = {}

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
        self.charuco_detector = cv2.aruco.CharucoDetector(self.board, detectorParams=self.detector_params)

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

        if charuco_corners is not None and len(charuco_corners) >= 4:
            return charuco_corners, charuco_ids

        return None, None

    # --- Stage 1: Intrinsic Calibration ---

    def calibrate_intrinsics(self, cam_idx, intrinsic_dir):
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

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, self.image_size, None, None, flags=0
        )

        # Compute mean reprojection error
        total_error = 0
        total_points = 0
        for i, (obj_pts, img_pts) in enumerate(zip(obj_points, img_points)):
            reproj, _ = cv2.projectPoints(obj_pts, rvecs[i], tvecs[i], K, dist)
            error = cv2.norm(img_pts, reproj, cv2.NORM_L2) / len(img_pts)
            total_error += error * len(img_pts)
            total_points += len(img_pts)

        mean_error = total_error / total_points

        print(f"  Camera {cam_idx + 1}: RMS={ret:.4f}px, mean_error={mean_error:.4f}px, "
              f"fx={K[0, 0]:.1f}, fy={K[1, 1]:.1f}")

        self.intrinsics[cam_idx] = (K, dist, mean_error)
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
        return all_capture_sets

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

        # Backward-compatible keys for 2-camera pipeline
        if self.num_cameras >= 2 and 0 in transforms and 1 in transforms:
            # Compute R_1_to_2: transform from cam1 to cam2
            # cam1 -> ref -> cam2_inv is complex; just provide if we have direct edge
            if (0, 1) in self.edges:
                R_12, T_12, _, _ = self.edges[(0, 1)]
                data['R_1_to_2'] = R_12
                data['t_1_to_2'] = T_12
            elif (1, 0) in self.edges:
                R_21, T_21, _, _ = self.edges[(1, 0)]
                data['R_1_to_2'] = R_21.T
                data['t_1_to_2'] = -R_21.T @ T_21

        np.savez(self.args.output, **data)
        print(f"\nSaved calibration to {self.args.output}")

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
            # Invert to get ref -> cam: P_cam = R_to_ref^T @ (P_ref - t_to_ref)
            R_ref_to_cam = R_to_ref.T
            t_ref_to_cam = -R_to_ref.T @ t_to_ref

            baseline = np.linalg.norm(t_ref_to_cam)
            rvec, _ = cv2.Rodrigues(R_ref_to_cam)
            angle = np.linalg.norm(rvec) * 180 / np.pi
            axis = rvec.flatten() / np.linalg.norm(rvec) if np.linalg.norm(rvec) > 0 else np.zeros(3)

            print(f"\n  Camera {ref_cam + 1} -> Camera {cam_idx + 1}:")
            print(f"    Rotation matrix R:\n      {R_ref_to_cam}")
            print(f"    Translation vector T (meters): {t_ref_to_cam}")
            print(f"    Baseline (distance between cameras): {baseline:.4f} meters")
            print(f"    Rotation: {angle:.2f} degrees around axis [{axis[0]:.4f}, {axis[1]:.4f}, {axis[2]:.4f}]")

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
