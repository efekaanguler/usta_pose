#!/usr/bin/env python3
"""
ChArUco-based Single Camera Intrinsic Calibration Image Capture

Captures images from a single RealSense camera for intrinsic parameter calibration.
This allows you to cover the entire field of view, including edges and corners.

Usage:
    # Capture for camera 1
    python charuco_intrinsic_capture.py --camera-id 1 --output-dir ./intrinsic_cam1

    # Capture for camera 2
    python charuco_intrinsic_capture.py --camera-id 2 --output-dir ./intrinsic_cam2

    # Auto-capture mode (recommended)
    python charuco_intrinsic_capture.py --camera-id 1 --output-dir ./intrinsic_cam1 --auto-capture

    Serial numbers are read from camera_config.json (see --cam-config).
"""

import argparse
import cv2
import numpy as np
import os
import sys
import pyrealsense2 as rs
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class CharucoIntrinsicCaptureApp:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChArUco board
        self.setup_charuco_board()

        # Initialize RealSense camera
        self.setup_camera()

        # Capture counter
        self.capture_count = 0
        self.target_captures = args.num_captures

        # Auto-capture state
        self.auto_capture = args.auto_capture
        self.capture_interval = args.capture_interval
        self.last_capture_time = 0
        self.countdown_start_time = None

    def setup_charuco_board(self):
        """Initialize ChArUco board detector with specified parameters"""
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
        print(f"  Square size: {self.args.square_length * 1000:.1f} mm")
        print(f"  Marker size: {self.args.marker_length * 1000:.1f} mm")
        print(f"  Dictionary: {self.args.aruco_dict}")
        print(f"  Total corners: {(self.args.squares_x-1) * (self.args.squares_y-1)}")
        print(f"\nIMPORTANT: Verify these parameters match your printed board!")

    def setup_camera(self):
        """Initialize single RealSense camera"""
        ctx = rs.context()
        devices = ctx.query_devices()

        if len(devices) == 0:
            raise RuntimeError("No RealSense cameras found")

        print(f"\nFound {len(devices)} RealSense camera(s):")
        for i, dev in enumerate(devices):
            serial = dev.get_info(rs.camera_info.serial_number)
            print(f"  Camera {i}: {serial}")

        # Get serial number: config file > auto-detect
        from utils import load_camera_serials
        config_serials = load_camera_serials(self.args.cam_config)
        config_serial = config_serials.get(self.args.camera_id)
        if config_serial:
            self.serial = config_serial
            print(f"Loaded serial for camera {self.args.camera_id} from {self.args.cam_config}")
        else:
            # Use first detected camera
            self.serial = devices[0].get_info(rs.camera_info.serial_number)

        print(f"\nUsing camera: {self.serial} (Camera {self.args.camera_id})")

        # Configure and start pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.args.width, self.args.height,
                           rs.format.bgr8, self.args.fps)

        # Start streaming
        print(f"Starting camera stream ({self.args.width}x{self.args.height} @ {self.args.fps} FPS)...")
        self.pipeline.start(config)

        # Wait for camera to stabilize
        print("Warming up camera...")
        for _ in range(30):
            self.pipeline.wait_for_frames()

        print("Camera ready!")

    def detect_charuco(self, image):
        """Detect ChArUco board in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(gray)

        detected_image = image.copy()

        if marker_ids is not None and len(marker_ids) > 0:
            num_markers = len(marker_ids)
            cv2.aruco.drawDetectedMarkers(detected_image, marker_corners, marker_ids)
        else:
            num_markers = 0

        if charuco_corners is not None and len(charuco_corners) > 0:
            num_corners = len(charuco_corners)
            cv2.aruco.drawDetectedCornersCharuco(detected_image, charuco_corners, charuco_ids)
            print(f"DEBUG: Detected {num_corners} ChArUco corners from {num_markers} markers")
            return charuco_corners, charuco_ids, detected_image, num_corners, num_markers
        else:
            print(f"DEBUG: No corners detected. Markers: {num_markers}, charuco_corners type: {type(charuco_corners)}, value: {charuco_corners}")
            return None, None, detected_image, 0, num_markers

    def create_info_overlay(self, image, num_corners, num_markers, is_good=False):
        """Add information overlay to image"""
        overlay = image.copy()
        h, w = overlay.shape[:2]

        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

        cv2.putText(overlay, f"Camera {self.args.camera_id} - Intrinsic Calibration", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if num_corners > 0:
            color = (0, 255, 0) if is_good else (0, 165, 255)
            text = f"Markers: {num_markers} | Corners: {num_corners}"
            cv2.putText(overlay, text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            if num_markers > 0:
                cv2.putText(overlay, f"Markers: {num_markers} | No corners detected", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                cv2.putText(overlay, "No board detected", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        progress_text = f"Captures: {self.capture_count}/{self.target_captures}"
        cv2.putText(overlay, progress_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return overlay

    def add_countdown_overlay(self, image, time_remaining):
        """Add countdown timer overlay"""
        overlay = image.copy()
        h, w = overlay.shape[:2]

        if time_remaining > 0:
            countdown_num = int(np.ceil(time_remaining))
            center_x, center_y = w // 2, h // 2
            radius = 80

            pulse = 1.0 + 0.2 * np.sin(time_remaining * 3.14159)
            current_radius = int(radius * pulse)

            cv2.circle(overlay, (center_x, center_y), current_radius, (0, 255, 255), -1)
            overlay = cv2.addWeighted(image, 0.3, overlay, 0.7, 0)

            font_scale = 3.0
            thickness = 8
            text = str(countdown_num)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2
            cv2.putText(overlay, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

            instruction = "HOLD STEADY"
            cv2.putText(overlay, instruction, (center_x - 120, center_y + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        else:
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
            overlay = cv2.addWeighted(image, 0.4, overlay, 0.6, 0)

            text = "CAPTURING!"
            font_scale = 2.5
            thickness = 6
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            cv2.putText(overlay, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        return overlay

    def add_coverage_guide(self, image):
        """Add visual guide for FOV coverage"""
        overlay = image.copy()
        h, w = overlay.shape[:2]

        # Draw grid to guide coverage
        grid_color = (100, 100, 100)
        line_thickness = 1

        # Divide into 3x3 grid
        for i in range(1, 3):
            x = int(w * i / 3)
            cv2.line(overlay, (x, 0), (x, h), grid_color, line_thickness)

        for i in range(1, 3):
            y = int(h * i / 3)
            cv2.line(overlay, (0, y), (w, y), grid_color, line_thickness)

        # Add corner markers
        marker_size = 30
        corner_color = (0, 255, 255)
        corners = [(0, 0), (w, 0), (0, h), (w, h)]

        for cx, cy in corners:
            x = max(5, min(cx - marker_size//2, w - marker_size - 5))
            y = max(5, min(cy - marker_size//2, h - marker_size - 5))
            cv2.rectangle(overlay, (x, y), (x + marker_size, y + marker_size),
                         corner_color, 2)

        overlay = cv2.addWeighted(image, 0.9, overlay, 0.1, 0)

        # Add instruction at bottom
        instruction = "Cover all grid regions including corners and edges"
        text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = (w - text_size[0]) // 2
        cv2.rectangle(overlay, (text_x - 10, h - 35), (text_x + text_size[0] + 10, h - 5),
                     (0, 0, 0), -1)
        cv2.putText(overlay, instruction, (text_x, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return overlay

    def run(self):
        """Main capture loop"""
        print(f"\n{'='*60}")
        print(f"INTRINSIC CALIBRATION IMAGE CAPTURE - CAMERA {self.args.camera_id}")
        print(f"{'='*60}")
        print(f"Target: {self.target_captures} images")

        if self.auto_capture:
            print(f"Mode: AUTO-CAPTURE (interval: {self.capture_interval}s)")
        else:
            print(f"Mode: MANUAL")

        print(f"\nIMPORTANT - For good intrinsic calibration:")
        print(f"  - Cover the ENTIRE field of view")
        print(f"  - Position board at ALL edges and corners")
        print(f"  - Include center, left, right, top, bottom")
        print(f"  - Vary distance (near and far)")
        print(f"  - Tilt board at different angles")
        print(f"{'='*60}\n")

        mode_str = "AUTO" if self.auto_capture else "MANUAL"
        window_name = f"Camera {self.args.camera_id} Intrinsic Calibration - {mode_str} MODE"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                current_time = time.time()

                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

                image = np.asanyarray(color_frame.get_data())

                corners, ids, detected, num_corners, num_markers = self.detect_charuco(image)

                min_corners = max(3, int(0.25 * (self.args.squares_x-1) * (self.args.squares_y-1)))
                is_good = corners is not None and num_corners >= min_corners

                display = self.create_info_overlay(detected, num_corners, num_markers, is_good)

                # Add coverage guide
                display = self.add_coverage_guide(display)

                # Auto-capture logic
                should_capture = False
                time_remaining = 0

                if self.auto_capture:
                    time_since_last = current_time - self.last_capture_time

                    if is_good:
                        if self.countdown_start_time is None:
                            self.countdown_start_time = current_time

                        countdown_duration = 3
                        time_in_countdown = current_time - self.countdown_start_time
                        time_remaining = countdown_duration - time_in_countdown

                        if time_remaining <= 0:
                            should_capture = True
                            self.countdown_start_time = None
                            time_remaining = 0

                        display = self.add_countdown_overlay(display, time_remaining)
                    else:
                        self.countdown_start_time = None
                else:
                    if is_good:
                        instruction = "READY - Press SPACE to capture"
                        h, w = display.shape[:2]
                        cv2.putText(display, instruction, (w//2 - 200, h - 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:
                    print("\nQuitting...")
                    break

                if key == ord(' ') and not self.auto_capture:
                    if is_good:
                        should_capture = True
                    else:
                        print("✗ Cannot capture - board not detected well")

                if should_capture:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    img_path = self.output_dir / f"capture_{self.capture_count:03d}_{timestamp}.png"

                    cv2.imwrite(str(img_path), image)

                    self.capture_count += 1
                    self.last_capture_time = current_time

                    print(f"✓ Captured {self.capture_count}/{self.target_captures} "
                          f"({num_corners} corners)")

                    if self.capture_count >= self.target_captures:
                        print(f"\n{'='*60}")
                        print(f"Target reached! Captured {self.capture_count} images.")
                        if self.auto_capture:
                            print(f"Waiting 2 seconds before exit...")
                            time.sleep(2)
                            break
                        else:
                            print(f"Press Q to quit or continue capturing.")
                        print(f"{'='*60}\n")

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

            print(f"\nCapture session complete!")
            print(f"Total captures: {self.capture_count}")
            print(f"Images saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Capture images from single RealSense camera for intrinsic calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Camera identification
    parser.add_argument('--cam-config', type=str, default='./camera_config.json',
                       help='Path to camera config JSON file mapping cam IDs to serial numbers')
    parser.add_argument('--camera-id', type=int, required=True,
                       help='Camera identifier (1-4) for labeling and config lookup')

    # Output settings
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Directory to save captured images')
    parser.add_argument('--num-captures', type=int, default=30,
                       help='Target number of images to capture')

    # Capture mode
    parser.add_argument('--auto-capture', action='store_true',
                       help='Enable automatic capture mode with countdown timer')
    parser.add_argument('--capture-interval', type=float, default=4.0,
                       help='Time interval between captures in auto mode (seconds)')

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

    # Camera settings
    parser.add_argument('--width', type=int, default=1280,
                       help='Image width')
    parser.add_argument('--height', type=int, default=720,
                       help='Image height')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second')

    args = parser.parse_args()

    if args.square_length <= args.marker_length:
        parser.error("Square length must be greater than marker length")

    app = CharucoIntrinsicCaptureApp(args)
    app.run()


if __name__ == '__main__':
    main()
