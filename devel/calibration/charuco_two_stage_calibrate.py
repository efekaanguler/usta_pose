#!/usr/bin/env python3
"""
ChArUco-based Two-Stage Stereo Camera Calibration

Stage 1: Calibrate intrinsic parameters for each camera independently
Stage 2: Calibrate extrinsic parameters (R, T) using stereo image pairs

This approach provides better calibration quality by allowing full FOV coverage
for intrinsic calibration, including edges and corners.

Usage:
    python charuco_two_stage_calibrate.py \
        --intrinsic-dir-1 ./intrinsic_cam1 \
        --intrinsic-dir-2 ./intrinsic_cam2 \
        --stereo-dir ./stereo_captures \
        --output calibration.npz

Or use only stereo captures (fallback to original method):
    python charuco_two_stage_calibrate.py \
        --stereo-dir ./stereo_captures \
        --output calibration.npz
"""

import argparse
import cv2
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime


class CharucoTwoStageCalibrator:
    def __init__(self, args):
        self.args = args
        self.setup_charuco_board()

        self.image_size = None

        # Intrinsic calibration data
        self.intrinsic_corners_1 = []
        self.intrinsic_ids_1 = []
        self.intrinsic_corners_2 = []
        self.intrinsic_ids_2 = []

        # Stereo calibration data
        self.stereo_corners_1 = []
        self.stereo_ids_1 = []
        self.stereo_corners_2 = []
        self.stereo_ids_2 = []

    def setup_charuco_board(self):
        """Initialize ChArUco board"""
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

        print(f"ChArUco Board Configuration:")
        print(f"  Squares: {self.args.squares_x} × {self.args.squares_y}")
        print(f"  Square size: {self.args.square_length} m")
        print(f"  Marker size: {self.args.marker_length} m")

    def detect_charuco_in_image(self, image_path):
        """Detect ChArUco corners in image"""
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

    def load_intrinsic_images(self, intrinsic_dir, camera_name):
        """Load images from intrinsic calibration directory"""
        intrinsic_path = Path(intrinsic_dir)
        if not intrinsic_path.exists():
            raise FileNotFoundError(f"Intrinsic directory not found: {intrinsic_path}")

        images = sorted(intrinsic_path.glob("*.png"))
        print(f"\nProcessing {len(images)} intrinsic images for {camera_name}...")

        all_corners = []
        all_ids = []
        successful = 0

        for i, img_path in enumerate(images):
            corners, ids = self.detect_charuco_in_image(img_path)

            if corners is not None:
                all_corners.append(corners)
                all_ids.append(ids)
                successful += 1
                print(f"  ✓ Image {i+1}/{len(images)}: {len(corners)} corners")
            else:
                print(f"  ✗ Image {i+1}/{len(images)}: detection failed")

        print(f"Successfully processed {successful}/{len(images)} images")

        if successful < 10:
            print(f"Warning: Only {successful} images for {camera_name}. Recommend 20+.")

        return all_corners, all_ids

    def load_stereo_image_pairs(self):
        """Load synchronized stereo image pairs"""
        stereo_path = Path(self.args.stereo_dir)
        if not stereo_path.exists():
            raise FileNotFoundError(f"Stereo directory not found: {stereo_path}")

        cam1_dir = stereo_path / "camera_1"
        cam2_dir = stereo_path / "camera_2"

        if not cam1_dir.exists() or not cam2_dir.exists():
            raise FileNotFoundError(
                f"Expected camera_1 and camera_2 subdirectories in {stereo_path}"
            )

        images1 = sorted(cam1_dir.glob("*.png"))
        images2 = sorted(cam2_dir.glob("*.png"))

        num_pairs = min(len(images1), len(images2))
        print(f"\nProcessing {num_pairs} stereo image pairs...")

        corners1_list = []
        ids1_list = []
        corners2_list = []
        ids2_list = []
        successful = 0

        for i, (img1_path, img2_path) in enumerate(zip(images1, images2)):
            corners1, ids1 = self.detect_charuco_in_image(img1_path)
            corners2, ids2 = self.detect_charuco_in_image(img2_path)

            if corners1 is not None and corners2 is not None:
                corners1_list.append(corners1)
                ids1_list.append(ids1)
                corners2_list.append(corners2)
                ids2_list.append(ids2)
                successful += 1
                print(f"  ✓ Pair {i+1}/{num_pairs}: "
                      f"cam1={len(corners1)} corners, cam2={len(corners2)} corners")
            else:
                status1 = "OK" if corners1 is not None else "FAILED"
                status2 = "OK" if corners2 is not None else "FAILED"
                print(f"  ✗ Pair {i+1}/{num_pairs}: cam1={status1}, cam2={status2}")

        print(f"Successfully processed {successful}/{num_pairs} pairs")

        if successful < 10:
            print(f"Warning: Only {successful} stereo pairs. Recommend 20+.")

        return corners1_list, ids1_list, corners2_list, ids2_list

    def calibrate_camera_intrinsics(self, all_corners, all_ids, camera_name):
        """Calibrate intrinsic parameters for a single camera"""
        print(f"\nCalibrating {camera_name} intrinsics...")

        obj_points = []
        img_points = []

        for corners, ids in zip(all_corners, all_ids):
            obj_pts = self.board.getChessboardCorners()[ids.flatten()]
            obj_points.append(obj_pts.astype(np.float32))
            img_points.append(corners.astype(np.float32))

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, self.image_size,
            None, None, flags=0
        )

        # Compute reprojection error
        total_error = 0
        total_points = 0

        for i, (obj_pts, img_pts) in enumerate(zip(obj_points, img_points)):
            img_pts_reprojected, _ = cv2.projectPoints(
                obj_pts, rvecs[i], tvecs[i], K, dist
            )
            error = cv2.norm(img_pts, img_pts_reprojected, cv2.NORM_L2) / len(img_pts)
            total_error += error * len(img_pts)
            total_points += len(img_pts)

        mean_error = total_error / total_points

        print(f"  RMS reprojection error: {ret:.4f} pixels")
        print(f"  Mean reprojection error: {mean_error:.4f} pixels")
        print(f"  Focal length: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
        print(f"  Principal point: cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")

        return K, dist, mean_error

    def calibrate_stereo_extrinsics(self, K1, dist1, K2, dist2):
        """Calibrate stereo extrinsics (R, T) with fixed intrinsics"""
        print(f"\nCalibrating stereo extrinsics...")

        # Prepare matched object and image points
        obj_points = []
        img_points_1 = []
        img_points_2 = []

        for corners1, ids1, corners2, ids2 in zip(
            self.stereo_corners_1, self.stereo_ids_1,
            self.stereo_corners_2, self.stereo_ids_2
        ):
            # Find common corner IDs
            ids1_flat = ids1.flatten()
            ids2_flat = ids2.flatten()
            common_ids = np.intersect1d(ids1_flat, ids2_flat)

            if len(common_ids) < 4:
                continue

            # Get indices of common IDs
            indices1 = np.array([np.where(ids1_flat == cid)[0][0] for cid in common_ids])
            indices2 = np.array([np.where(ids2_flat == cid)[0][0] for cid in common_ids])

            # Extract matched corners
            matched_corners1 = corners1[indices1]
            matched_corners2 = corners2[indices2]

            # Get 3D object points
            obj_pts = self.board.getChessboardCorners()[common_ids]

            obj_points.append(obj_pts.astype(np.float32))
            img_points_1.append(matched_corners1.astype(np.float32))
            img_points_2.append(matched_corners2.astype(np.float32))

        print(f"  Using {len(obj_points)} image pairs with matched corners")

        # Stereo calibration with fixed intrinsics
        flags = cv2.CALIB_FIX_INTRINSIC

        ret, K1_out, dist1_out, K2_out, dist2_out, R, T, E, F = cv2.stereoCalibrate(
            obj_points, img_points_1, img_points_2,
            K1, dist1, K2, dist2,
            self.image_size,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        )

        print(f"  Stereo RMS reprojection error: {ret:.4f} pixels")
        print(f"  Translation vector T (meters): {T.flatten()}")
        print(f"  Baseline: {np.linalg.norm(T):.4f} meters")

        rvec, _ = cv2.Rodrigues(R)
        angle = np.linalg.norm(rvec) * 180 / np.pi
        print(f"  Rotation: {angle:.2f}°")

        return R, T, ret

    def save_calibration_npz(self, K1, dist1, K2, dist2, R, T):
        """Save calibration in NPZ format"""
        np.savez(
            self.args.output,
            K1=K1,
            dist1=dist1,
            K2=K2,
            dist2=dist2,
            R=R,
            T=T,
            image_size=np.array(self.image_size)
        )
        print(f"\n✓ Saved calibration to {self.args.output}")

    def save_calibration_yaml(self, K1, dist1, K2, dist2, R, T,
                              error1, error2, stereo_error):
        """Save calibration in YAML format"""
        yaml_path = self.args.output.replace('.npz', '.yaml')

        calibration_data = {
            'calibration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'calibration_method': 'two_stage_charuco',
            'image_width': int(self.image_size[0]),
            'image_height': int(self.image_size[1]),
            'board_config': {
                'squares_x': int(self.args.squares_x),
                'squares_y': int(self.args.squares_y),
                'square_length': float(self.args.square_length),
                'marker_length': float(self.args.marker_length),
                'aruco_dict': self.args.aruco_dict
            },
            'camera_1': {
                'camera_matrix': K1.tolist(),
                'distortion_coefficients': dist1.flatten().tolist(),
                'reprojection_error': float(error1)
            },
            'camera_2': {
                'camera_matrix': K2.tolist(),
                'distortion_coefficients': dist2.flatten().tolist(),
                'reprojection_error': float(error2)
            },
            'stereo': {
                'rotation_matrix': R.tolist(),
                'translation_vector': T.flatten().tolist(),
                'baseline_meters': float(np.linalg.norm(T)),
                'reprojection_error': float(stereo_error)
            }
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(calibration_data, f, default_flow_style=False, sort_keys=False)

        print(f"✓ Saved calibration to {yaml_path}")

    def run(self):
        """Main calibration pipeline"""
        print(f"{'='*70}")
        print(f"TWO-STAGE STEREO CAMERA CALIBRATION")
        print(f"{'='*70}")

        # Check if we have separate intrinsic directories or using stereo-only
        use_two_stage = (self.args.intrinsic_dir_1 is not None and
                        self.args.intrinsic_dir_2 is not None)

        if use_two_stage:
            print("\nMode: TWO-STAGE CALIBRATION")
            print("  Stage 1: Intrinsic calibration from separate captures")
            print("  Stage 2: Extrinsic calibration from stereo pairs")

            # Stage 1: Load and calibrate intrinsics separately
            print(f"\n{'='*70}")
            print("STAGE 1: INTRINSIC CALIBRATION")
            print(f"{'='*70}")

            self.intrinsic_corners_1, self.intrinsic_ids_1 = self.load_intrinsic_images(
                self.args.intrinsic_dir_1, "Camera 1"
            )

            self.intrinsic_corners_2, self.intrinsic_ids_2 = self.load_intrinsic_images(
                self.args.intrinsic_dir_2, "Camera 2"
            )

            K1, dist1, error1 = self.calibrate_camera_intrinsics(
                self.intrinsic_corners_1, self.intrinsic_ids_1, "Camera 1"
            )

            K2, dist2, error2 = self.calibrate_camera_intrinsics(
                self.intrinsic_corners_2, self.intrinsic_ids_2, "Camera 2"
            )

            # Stage 2: Load stereo pairs and calibrate extrinsics
            print(f"\n{'='*70}")
            print("STAGE 2: EXTRINSIC CALIBRATION")
            print(f"{'='*70}")

            self.stereo_corners_1, self.stereo_ids_1, \
            self.stereo_corners_2, self.stereo_ids_2 = self.load_stereo_image_pairs()

            R, T, stereo_error = self.calibrate_stereo_extrinsics(K1, dist1, K2, dist2)

        else:
            print("\nMode: SINGLE-STAGE CALIBRATION (fallback)")
            print("  Using only stereo pairs for both intrinsic and extrinsic")

            # Load stereo pairs
            self.stereo_corners_1, self.stereo_ids_1, \
            self.stereo_corners_2, self.stereo_ids_2 = self.load_stereo_image_pairs()

            # Calibrate intrinsics from stereo data
            K1, dist1, error1 = self.calibrate_camera_intrinsics(
                self.stereo_corners_1, self.stereo_ids_1, "Camera 1"
            )

            K2, dist2, error2 = self.calibrate_camera_intrinsics(
                self.stereo_corners_2, self.stereo_ids_2, "Camera 2"
            )

            # Calibrate extrinsics
            R, T, stereo_error = self.calibrate_stereo_extrinsics(K1, dist1, K2, dist2)

        # Save results
        self.save_calibration_npz(K1, dist1, K2, dist2, R, T)
        self.save_calibration_yaml(K1, dist1, K2, dist2, R, T,
                                   error1, error2, stereo_error)

        # Quality assessment
        print(f"\n{'='*70}")
        print(f"CALIBRATION QUALITY ASSESSMENT")
        print(f"{'='*70}")

        if error1 < 0.5 and error2 < 0.5:
            print("✓ Intrinsic calibration: EXCELLENT (< 0.5 pixels)")
        elif error1 < 1.0 and error2 < 1.0:
            print("✓ Intrinsic calibration: GOOD (< 1.0 pixels)")
        else:
            print("⚠ Intrinsic calibration: ACCEPTABLE (> 1.0 pixels)")

        if stereo_error < 0.5:
            print("✓ Stereo calibration: EXCELLENT (< 0.5 pixels)")
        elif stereo_error < 1.0:
            print("✓ Stereo calibration: GOOD (< 1.0 pixels)")
        else:
            print("⚠ Stereo calibration: ACCEPTABLE (> 1.0 pixels)")

        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage stereo camera calibration using ChArUco boards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input directories
    parser.add_argument('--intrinsic-dir-1', type=str, default=None,
                       help='Directory with intrinsic calibration images for camera 1')
    parser.add_argument('--intrinsic-dir-2', type=str, default=None,
                       help='Directory with intrinsic calibration images for camera 2')
    parser.add_argument('--stereo-dir', type=str, required=True,
                       help='Directory with stereo image pairs (camera_1 and camera_2 subdirs)')

    # Output
    parser.add_argument('--output', type=str, default='charuco_calibration.npz',
                       help='Output calibration file path (NPZ format)')

    # ChArUco board parameters
    parser.add_argument('--squares-x', type=int, default=3,
                       help='Number of squares in X direction')
    parser.add_argument('--squares-y', type=int, default=4,
                       help='Number of squares in Y direction')
    parser.add_argument('--square-length', type=float, default=0.063,
                       help='Square side length in meters')
    parser.add_argument('--marker-length', type=float, default=0.047,
                       help='Marker side length in meters')
    parser.add_argument('--aruco-dict', type=str, default='4X4_50',
                       choices=['4X4_50', '4X4_100', '4X4_250', '4X4_1000',
                               '5X5_50', '5X5_100', '5X5_250', '5X5_1000',
                               '6X6_50', '6X6_100', '6X6_250', '6X6_1000'],
                       help='ArUco dictionary')

    args = parser.parse_args()

    # Validate: either both intrinsic dirs provided or neither
    has_intrinsic_1 = args.intrinsic_dir_1 is not None
    has_intrinsic_2 = args.intrinsic_dir_2 is not None

    if has_intrinsic_1 != has_intrinsic_2:
        parser.error("Must provide both --intrinsic-dir-1 and --intrinsic-dir-2, or neither")

    calibrator = CharucoTwoStageCalibrator(args)
    calibrator.run()


if __name__ == '__main__':
    main()
