#!/usr/bin/env python3
"""
Multi-Camera ChArUco Calibration Image Capture

Captures synchronized images from a pair of RealSense cameras for
stereo extrinsic calibration using a ChArUco board.

Use --camera-ids to specify which cameras are being captured in this
session (e.g. "1,3"). The output directories are named by camera ID
(camera_1/, camera_3/) so the calibration script can discover pairs.

Recommended 4-camera calibration workflow (3 separate sessions):
  Session 1: --camera-ids 1,3  (tripod cam 1 + table cam 3)
  Session 2: --camera-ids 2,4  (tripod cam 2 + table cam 4)
  Session 3: --camera-ids 1,2  (both tripod cams, or use 3,4 for table cams)

Usage:
    python calibration/multicam_capture.py \\
        --output-dir ./session_cam1_cam3 --camera-ids 1,3 \\
        --num-captures 25 --auto-capture \\
        --squares-x 3 --squares-y 4 --square-length 0.063 --marker-length 0.047

    Serial numbers are read from camera_config.json (see --cam-config).
"""

import argparse
import cv2
import json
import numpy as np
import pyrealsense2 as rs
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class MulticamCaptureApp:
    def __init__(self, args):
        self.args = args

        # Parse camera IDs (e.g. "1,3" means we're capturing cameras 1 and 3)
        if args.camera_ids:
            self.camera_ids = [int(x) for x in args.camera_ids.split(',')]
            self.num_cameras = len(self.camera_ids)
        else:
            self.camera_ids = list(range(1, args.num_cameras + 1))
            self.num_cameras = args.num_cameras

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories named by actual camera ID
        self.cam_dirs = []
        for cam_id in self.camera_ids:
            d = self.output_dir / f"camera_{cam_id}"
            d.mkdir(exist_ok=True)
            self.cam_dirs.append(d)

        self.setup_charuco_board()
        self.setup_cameras()

        self.capture_count = 0
        self.target_captures = args.num_captures

        # Auto-capture state
        self.auto_capture = args.auto_capture
        self.countdown_start_time = None
        self.last_capture_time = 0

    def setup_charuco_board(self):
        """Initialize ChArUco board detector."""
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
        print(f"  Squares: {self.args.squares_x} x {self.args.squares_y}")
        print(f"  Square size: {self.args.square_length} m")
        print(f"  Marker size: {self.args.marker_length} m")
        print(f"  Dictionary: {self.args.aruco_dict}")

    def setup_cameras(self):
        """Initialize N RealSense cameras."""
        ctx = rs.context()
        devices = ctx.query_devices()

        if len(devices) < self.num_cameras:
            raise RuntimeError(
                f"Found only {len(devices)} RealSense camera(s). Need {self.num_cameras}."
            )

        print(f"\nFound {len(devices)} RealSense cameras:")
        for i, dev in enumerate(devices):
            print(f"  Device {i}: {dev.get_info(rs.camera_info.serial_number)}")

        # Load serial mapping from config file
        from utils import load_camera_serials
        config_serials = load_camera_serials(self.args.cam_config)
        print(f"Loaded camera serials from {self.args.cam_config}: {config_serials}")

        # Collect serial numbers: config file > auto-detect
        self.serials = []
        for cam_id in self.camera_ids:
            serial = config_serials.get(cam_id)
            if serial:
                self.serials.append(serial)

        if len(self.serials) < self.num_cameras:
            # Fill remaining with auto-detected serials
            available = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
            for s in available:
                if s not in self.serials and len(self.serials) < self.num_cameras:
                    self.serials.append(s)

        print(f"\nUsing cameras:")
        for i, s in enumerate(self.serials):
            print(f"  Camera {self.camera_ids[i]}: {s}")

        # Start pipelines
        self.pipelines = []
        for i, serial in enumerate(self.serials):
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, self.args.width, self.args.height,
                              rs.format.bgr8, self.args.fps)
            pipe.start(cfg)
            self.pipelines.append(pipe)

        # Warm up
        print(f"\nWarming up {self.num_cameras} cameras...")
        for _ in range(30):
            for pipe in self.pipelines:
                pipe.wait_for_frames()
        print("Cameras ready!")

    def detect_charuco(self, image):
        """Detect ChArUco board in image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)

        detected_image = image.copy()

        num_markers = 0 if marker_ids is None or len(marker_ids) == 0 else len(marker_ids)
        num_corners = 0 if charuco_corners is None or len(charuco_corners) == 0 else len(charuco_corners)

        if num_markers > 0:
            cv2.aruco.drawDetectedMarkers(detected_image, marker_corners, marker_ids)

        if num_corners > 0:
            cv2.aruco.drawDetectedCornersCharuco(detected_image, charuco_corners, charuco_ids)
            return charuco_corners, charuco_ids, detected_image, num_corners

        return None, None, detected_image, 0

    def create_info_overlay(self, image, cam_idx, num_corners, detecting_cameras):
        """Add information overlay to image."""
        overlay = image.copy()
        h, w = overlay.shape[:2]

        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

        cv2.putText(overlay, f"Cam {self.camera_ids[cam_idx]} ({self.serials[cam_idx][-6:]})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if num_corners > 0:
            color = (0, 255, 0) if detecting_cameras >= self.args.min_cameras else (0, 165, 255)
            cv2.putText(overlay, f"Detected: {num_corners} corners", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(overlay, "No board detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Progress
        cv2.putText(overlay, f"{self.capture_count}/{self.target_captures}", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return overlay

    def add_countdown_overlay(self, image, time_remaining):
        """Add countdown timer overlay."""
        overlay = image.copy()
        h, w = overlay.shape[:2]

        if time_remaining > 0:
            countdown_num = int(np.ceil(time_remaining))
            center_x, center_y = w // 2, h // 2
            pulse = 1.0 + 0.2 * np.sin(time_remaining * 3.14159)
            current_radius = int(60 * pulse)

            cv2.circle(overlay, (center_x, center_y), current_radius, (0, 255, 255), -1)
            overlay = cv2.addWeighted(image, 0.3, overlay, 0.7, 0)

            text = str(countdown_num)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 6)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2
            cv2.putText(overlay, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 6)
        else:
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
            overlay = cv2.addWeighted(image, 0.4, overlay, 0.6, 0)

            text = "CAPTURED!"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            cv2.putText(overlay, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

        return overlay

    def run(self):
        """Main capture loop."""
        print(f"\n{'=' * 60}")
        print(f"MULTI-CAMERA CALIBRATION IMAGE CAPTURE")
        print(f"{'=' * 60}")
        print(f"Cameras: {self.num_cameras}")
        print(f"Target: {self.target_captures} capture sets")
        print(f"Min cameras per capture: {self.args.min_cameras}")

        if self.auto_capture:
            print(f"Mode: AUTO-CAPTURE (3s countdown)")
        else:
            print(f"Mode: MANUAL (press SPACE to capture)")

        print(f"\nCapture strategy tips:")
        print(f"  - Move board to cover different camera pair overlaps")
        print(f"  - Some captures for tripod cameras (1,2)")
        print(f"  - Some captures for table cameras (3,4)")
        print(f"  - Some bridge captures (tripod + table)")
        print(f"\nPress Q to quit")
        print(f"{'=' * 60}\n")

        min_corners = max(4, int(0.3 * (self.args.squares_x - 1) * (self.args.squares_y - 1)))

        window_name = "Multi-Camera ChArUco Capture (Q to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                current_time = time.time()

                # Capture frames from all cameras
                images = []
                for pipe in self.pipelines:
                    frames = pipe.wait_for_frames()
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        images.append(np.asanyarray(color_frame.get_data()))
                    else:
                        images.append(None)

                # Detect ChArUco in all images
                detections = []  # (corners, ids, display_image, num_corners)
                for img in images:
                    if img is not None:
                        detections.append(self.detect_charuco(img))
                    else:
                        detections.append((None, None, np.zeros((self.args.height, self.args.width, 3), dtype=np.uint8), 0))

                # Count how many cameras detected the board
                detecting_cameras = sum(
                    1 for corners, _, _, nc in detections
                    if corners is not None and nc >= min_corners
                )

                is_ready = detecting_cameras >= self.args.min_cameras

                # Build display grid
                display_images = []
                for i, (corners, ids, detected_img, nc) in enumerate(detections):
                    display = self.create_info_overlay(detected_img, i, nc, detecting_cameras)
                    display_images.append(display)

                # Auto-capture logic
                should_capture = False
                time_remaining = 0

                if self.auto_capture:
                    if is_ready:
                        if self.countdown_start_time is None:
                            self.countdown_start_time = current_time

                        countdown_duration = 3
                        time_in_countdown = current_time - self.countdown_start_time
                        time_remaining = countdown_duration - time_in_countdown

                        if time_remaining <= 0:
                            should_capture = True
                            self.countdown_start_time = None
                            time_remaining = 0

                        # Add countdown to detecting cameras
                        for i, (corners, _, _, nc) in enumerate(detections):
                            if corners is not None and nc >= min_corners:
                                display_images[i] = self.add_countdown_overlay(display_images[i], time_remaining)
                    else:
                        self.countdown_start_time = None

                # Show which cameras are detecting
                status_text = f"Detecting: {detecting_cameras}/{self.num_cameras} cams"
                status_color = (0, 255, 0) if is_ready else (0, 0, 255)
                for i in range(len(display_images)):
                    h = display_images[i].shape[0]
                    cv2.putText(display_images[i], status_text, (10, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                # Arrange in 2x2 grid (or 2xN)
                rows = []
                cols_per_row = 2
                for row_start in range(0, self.num_cameras, cols_per_row):
                    row_imgs = []
                    for col in range(cols_per_row):
                        idx = row_start + col
                        if idx < self.num_cameras:
                            # Resize for uniform grid
                            resized = cv2.resize(display_images[idx], (640, 360))
                            row_imgs.append(resized)
                        else:
                            row_imgs.append(np.zeros((360, 640, 3), dtype=np.uint8))
                    rows.append(np.hstack(row_imgs))

                combined = np.vstack(rows)
                cv2.imshow(window_name, combined)

                # Handle keys
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:
                    print("\nQuitting...")
                    break

                if key == ord(' ') and not self.auto_capture:
                    if is_ready:
                        should_capture = True
                    else:
                        print(f"Cannot capture - only {detecting_cameras} cameras detect the board (need {self.args.min_cameras})")

                # Perform capture
                if should_capture:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                    # Which cameras see the board
                    cam_status = []
                    for i, (corners, ids, _, nc) in enumerate(detections):
                        if images[i] is not None:
                            img_path = self.cam_dirs[i] / f"capture_{self.capture_count:03d}_{timestamp}.png"
                            cv2.imwrite(str(img_path), images[i])

                        if corners is not None and nc >= min_corners:
                            cam_status.append(f"cam{self.camera_ids[i]}:{nc}c")
                        else:
                            cam_status.append(f"cam{self.camera_ids[i]}:---")

                    self.capture_count += 1
                    self.last_capture_time = current_time

                    print(f"Captured {self.capture_count}/{self.target_captures}  [{', '.join(cam_status)}]")

                    if self.capture_count >= self.target_captures:
                        print(f"\nTarget reached! Captured {self.capture_count} sets.")
                        if self.auto_capture:
                            time.sleep(2)
                            break
                        else:
                            print("Press Q to quit or continue capturing.")

        finally:
            for pipe in self.pipelines:
                pipe.stop()
            cv2.destroyAllWindows()

            # Save session metadata
            session_info = {
                "camera_ids": self.camera_ids,
                "serials": {str(cid): s for cid, s in zip(self.camera_ids, self.serials)},
                "num_captures": self.capture_count,
                "timestamp": datetime.now().isoformat(),
            }
            info_path = self.output_dir / "session_info.json"
            with open(info_path, 'w') as f:
                json.dump(session_info, f, indent=2)

            print(f"\nCapture session complete!")
            print(f"Total captures: {self.capture_count}")
            print(f"Images saved to: {self.output_dir}")
            print(f"Session info saved to: {info_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Capture synchronized images from N RealSense cameras for multi-camera ChArUco calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--cam-config', type=str, default='./camera_config.json',
                        help='Path to camera config JSON file mapping cam IDs to serial numbers')
    parser.add_argument('--output-dir', type=str, default='./multicam_captures',
                        help='Directory to save captured images')
    parser.add_argument('--num-cameras', type=int, default=4,
                        help='Number of cameras (ignored when --camera-ids is given)')
    parser.add_argument('--camera-ids', type=str, default=None,
                        help='Comma-separated camera IDs to capture, e.g. "1,3". '
                             'Directories are named camera_1/, camera_3/, etc. '
                             'Serials are resolved from the config file.')
    parser.add_argument('--num-captures', type=int, default=30,
                        help='Target number of capture sets')
    parser.add_argument('--min-cameras', type=int, default=2,
                        help='Minimum cameras that must see the board to trigger capture')

    # Capture mode
    parser.add_argument('--auto-capture', action='store_true',
                        help='Enable automatic capture with countdown')

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

    # Resolution
    parser.add_argument('--width', type=int, default=1280, help='Image width')
    parser.add_argument('--height', type=int, default=720, help='Image height')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')

    args = parser.parse_args()

    if args.square_length <= args.marker_length:
        parser.error("Square length must be greater than marker length")

    if args.min_cameras < 2:
        parser.error("--min-cameras must be at least 2")

    actual_num = len(args.camera_ids.split(',')) if args.camera_ids else args.num_cameras
    if args.min_cameras > actual_num:
        parser.error("--min-cameras cannot exceed number of cameras")

    app = MulticamCaptureApp(args)
    app.run()


if __name__ == '__main__':
    main()
