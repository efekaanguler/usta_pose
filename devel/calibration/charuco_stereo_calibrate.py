#!/usr/bin/env python3
"""
ChArUco-based Stereo Camera Calibration

Processes synchronized image pairs to compute:
1. Intrinsic parameters (K matrix, distortion coefficients) for each camera
2. Extrinsic parameters (R, T) representing the relative pose between cameras

Usage:
    python charuco_stereo_calibrate.py --input-dir ./charuco_captures --output calibration.npz

Outputs:
    - NPZ file with calibration parameters (for process.py)
    - YAML file with calibration parameters (for compatibility with existing scripts)
    - Reprojection error statistics
"""

import argparse
import cv2
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime


class CharucoStereoCalibrator:
    def __init__(self, args):
        self.args = args
        self.input_dir = Path(args.input_dir)

        # Initialize ChArUco board
        self.setup_charuco_board()

        # Data storage
        self.all_corners_1 = []  # List of detected corners for camera 1
        self.all_ids_1 = []      # List of corner IDs for camera 1
        self.all_corners_2 = []  # List of detected corners for camera 2
        self.all_ids_2 = []      # List of corner IDs for camera 2
        self.image_size = None

    def setup_charuco_board(self):
        """Initialize ChArUco board with specified parameters"""
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
        print(f"  Dictionary: {self.args.aruco_dict}")

    def detect_charuco_in_image(self, image_path):
        """Detect ChArUco corners in a single image"""
        image = cv2.imread(str(image_path))
        if image is None:
            return None, None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, _, _ = self.charuco_detector.detectBoard(gray)

        if charuco_corners is not None and len(charuco_corners) >= 4:
            return charuco_corners, charuco_ids

        return None, None

    def load_image_pairs(self):
        """Load and process all synchronized image pairs"""
        cam1_dir = self.input_dir / "camera_1"
        cam2_dir = self.input_dir / "camera_2"

        if not cam1_dir.exists() or not cam2_dir.exists():
            raise FileNotFoundError(
                f"Expected camera_1 and camera_2 subdirectories in {self.input_dir}"
            )

        # Get sorted list of images
        images1 = sorted(cam1_dir.glob("*.png"))
        images2 = sorted(cam2_dir.glob("*.png"))

        if len(images1) != len(images2):
            print(f"Warning: Unequal number of images (cam1: {len(images1)}, cam2: {len(images2)})")

        num_pairs = min(len(images1), len(images2))
        print(f"\nProcessing {num_pairs} image pairs...")

        successful_pairs = 0

        for i, (img1_path, img2_path) in enumerate(zip(images1, images2)):
            # Detect ChArUco in both images
            corners1, ids1 = self.detect_charuco_in_image(img1_path)
            corners2, ids2 = self.detect_charuco_in_image(img2_path)

            if corners1 is not None and corners2 is not None:
                # Get image size from first successful pair
                if self.image_size is None:
                    img = cv2.imread(str(img1_path))
                    self.image_size = (img.shape[1], img.shape[0])  # (width, height)

                self.all_corners_1.append(corners1)
                self.all_ids_1.append(ids1)
                self.all_corners_2.append(corners2)
                self.all_ids_2.append(ids2)
                successful_pairs += 1

                print(f"  ✓ Pair {i+1}/{num_pairs}: "
                      f"cam1={len(corners1)} corners, cam2={len(corners2)} corners")
            else:
                status1 = "OK" if corners1 is not None else "FAILED"
                status2 = "OK" if corners2 is not None else "FAILED"
                print(f"  ✗ Pair {i+1}/{num_pairs}: cam1={status1}, cam2={status2}")

        print(f"\nSuccessfully processed {successful_pairs}/{num_pairs} pairs")

        if successful_pairs < 10:
            print("Warning: Less than 10 successful pairs. Calibration may be unreliable.")
            print("Recommendation: Capture more images with good board visibility.")

        return successful_pairs

    def calibrate_camera_intrinsics(self, all_corners, all_ids, camera_name):
        """Calibrate intrinsic parameters for a single camera"""
        print(f"\nCalibrating {camera_name} intrinsics...")

        # Prepare object points (3D coordinates of ChArUco corners in board space)
        obj_points = []
        img_points = []

        for corners, ids in zip(all_corners, all_ids):
            # Get 3D positions of detected corners from board definition
            obj_pts = self.board.getChessboardCorners()[ids.flatten()]
            obj_points.append(obj_pts.astype(np.float32))
            img_points.append(corners.astype(np.float32))

        # Calibrate
        calibration_flags = 0  # Use default calibration
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, self.image_size,
            None, None, flags=calibration_flags
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
        print(f"  Distortion coefficients: {dist.flatten()}")

        return K, dist, rvecs, tvecs, mean_error

    def calibrate_stereo(self, K1, dist1, K2, dist2):
        """Calibrate stereo extrinsics (R, T between cameras)"""
        print(f"\nCalibrating stereo extrinsics...")

        # Prepare matched object and image points for stereo calibration
        # We need to ensure the same ChArUco corners are used for both cameras
        obj_points = []
        img_points_1 = []
        img_points_2 = []

        for corners1, ids1, corners2, ids2 in zip(
            self.all_corners_1, self.all_ids_1,
            self.all_corners_2, self.all_ids_2
        ):
            # Find common corner IDs between both cameras
            ids1_flat = ids1.flatten()
            ids2_flat = ids2.flatten()
            common_ids = np.intersect1d(ids1_flat, ids2_flat)

            if len(common_ids) < 4:
                continue

            # Get indices of common IDs in each camera's detection
            indices1 = np.array([np.where(ids1_flat == cid)[0][0] for cid in common_ids])
            indices2 = np.array([np.where(ids2_flat == cid)[0][0] for cid in common_ids])

            # Extract matched corners
            matched_corners1 = corners1[indices1]
            matched_corners2 = corners2[indices2]

            # Get 3D object points for these corners
            obj_pts = self.board.getChessboardCorners()[common_ids]

            obj_points.append(obj_pts.astype(np.float32))
            img_points_1.append(matched_corners1.astype(np.float32))
            img_points_2.append(matched_corners2.astype(np.float32))

        print(f"  Using {len(obj_points)} image pairs with matched corners")

        # Stereo calibration
        # Fix intrinsic parameters and only optimize extrinsics
        flags = cv2.CALIB_FIX_INTRINSIC

        ret, K1_out, dist1_out, K2_out, dist2_out, R, T, E, F = cv2.stereoCalibrate(
            obj_points, img_points_1, img_points_2,
            K1, dist1, K2, dist2,
            self.image_size,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        )

        print(f"  Stereo RMS reprojection error: {ret:.4f} pixels")
        print(f"  Rotation matrix R:")
        print(f"    {R}")
        print(f"  Translation vector T (meters):")
        print(f"    {T.flatten()}")
        print(f"  Baseline (distance between cameras): {np.linalg.norm(T):.4f} meters")

        # Convert rotation matrix to Rodrigues vector for verification
        rvec, _ = cv2.Rodrigues(R)
        angle = np.linalg.norm(rvec) * 180 / np.pi
        axis = rvec.flatten() / np.linalg.norm(rvec) if np.linalg.norm(rvec) > 0 else [0, 0, 0]
        print(f"  Rotation: {angle:.2f}° around axis {axis}")

        return R, T, ret

    def save_calibration_npz(self, K1, dist1, K2, dist2, R, T):
        """Save calibration in NPZ format (for process.py)"""
        output_path = self.args.output

        np.savez(
            output_path,
            K1=K1,
            dist1=dist1,
            K2=K2,
            dist2=dist2,
            R=R,
            T=T,
            image_size=np.array(self.image_size)
        )

        print(f"\n✓ Saved calibration to {output_path}")

    def save_calibration_yaml(self, K1, dist1, K2, dist2, R, T, error1, error2, stereo_error):
        """Save calibration in YAML format (for compatibility)"""
        yaml_path = self.args.output.replace('.npz', '.yaml')

        calibration_data = {
            'calibration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
        print(f"STEREO CAMERA CALIBRATION")
        print(f"{'='*70}")

        # Step 1: Load image pairs and detect ChArUco corners
        num_pairs = self.load_image_pairs()

        if num_pairs < 10:
            response = input("\nProceed with calibration anyway? (y/n): ")
            if response.lower() != 'y':
                print("Calibration aborted.")
                return

        # Step 2: Calibrate camera 1 intrinsics
        K1, dist1, rvecs1, tvecs1, error1 = self.calibrate_camera_intrinsics(
            self.all_corners_1, self.all_ids_1, "Camera 1"
        )

        # Step 3: Calibrate camera 2 intrinsics
        K2, dist2, rvecs2, tvecs2, error2 = self.calibrate_camera_intrinsics(
            self.all_corners_2, self.all_ids_2, "Camera 2"
        )

        # Step 4: Calibrate stereo extrinsics
        R, T, stereo_error = self.calibrate_stereo(K1, dist1, K2, dist2)

        # Step 5: Save results
        self.save_calibration_npz(K1, dist1, K2, dist2, R, T)
        self.save_calibration_yaml(K1, dist1, K2, dist2, R, T, error1, error2, stereo_error)

        # Step 6: Quality assessment
        print(f"\n{'='*70}")
        print(f"CALIBRATION QUALITY ASSESSMENT")
        print(f"{'='*70}")

        if error1 < 0.5 and error2 < 0.5:
            print("✓ Intrinsic calibration: EXCELLENT (< 0.5 pixels)")
        elif error1 < 1.0 and error2 < 1.0:
            print("✓ Intrinsic calibration: GOOD (< 1.0 pixels)")
        else:
            print("⚠ Intrinsic calibration: ACCEPTABLE (> 1.0 pixels - consider recapturing)")

        if stereo_error < 0.5:
            print("✓ Stereo calibration: EXCELLENT (< 0.5 pixels)")
        elif stereo_error < 1.0:
            print("✓ Stereo calibration: GOOD (< 1.0 pixels)")
        else:
            print("⚠ Stereo calibration: ACCEPTABLE (> 1.0 pixels - consider recapturing)")

        print(f"\nRecommendations for improvement:")
        print(f"  - Capture more images (current: {num_pairs}, recommended: 30+)")
        print(f"  - Ensure board covers entire field of view in different positions")
        print(f"  - Include tilted views (not just frontal)")
        print(f"  - Vary the distance to the board (near and far)")
        print(f"  - Use a larger ChArUco board if possible (current: {self.args.squares_x}×{self.args.squares_y})")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate stereo camera system using ChArUco board images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/output
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing camera_1 and camera_2 subdirectories with images')
    parser.add_argument('--output', type=str, default='charuco_stereo_calibration.npz',
                       help='Output calibration file path (NPZ format)')

    # ChArUco board parameters (must match the capture parameters)
    parser.add_argument('--squares-x', type=int, default=3,
                       help='Number of squares in X direction (columns)')
    parser.add_argument('--squares-y', type=int, default=4,
                       help='Number of squares in Y direction (rows)')
    parser.add_argument('--square-length', type=float, default=0.063,
                       help='Square side length in meters')
    parser.add_argument('--marker-length', type=float, default=0.047,
                       help='Marker side length in meters')
    parser.add_argument('--aruco-dict', type=str, default='4X4_50',
                       choices=['4X4_50', '4X4_100', '4X4_250', '4X4_1000',
                               '5X5_50', '5X5_100', '5X5_250', '5X5_1000',
                               '6X6_50', '6X6_100', '6X6_250', '6X6_1000'],
                       help='ArUco dictionary (must match board)')

    args = parser.parse_args()

    calibrator = CharucoStereoCalibrator(args)
    calibrator.run()


if __name__ == '__main__':
    main()
