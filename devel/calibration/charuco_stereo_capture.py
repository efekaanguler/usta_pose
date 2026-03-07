#!/usr/bin/env python3
"""
ChArUco-based Stereo Camera Image Capture

Simultaneously captures synchronized image pairs from two RealSense cameras
for stereo calibration using a ChArUco board.

Usage:
    # Auto-capture mode (recommended for single person)
    python charuco_stereo_capture.py --output-dir ./charuco_images --num-captures 30 --auto-capture

    # Manual mode
    python charuco_stereo_capture.py --output-dir ./charuco_images --num-captures 30

Controls (manual mode):
    SPACE: Capture synchronized pair from both cameras
    Q: Quit

Auto-capture mode:
    - Captures automatically every N seconds (default: 4s)
    - Shows countdown timer (3...2...1...CAPTURING!)
    - Only captures when board is detected in BOTH cameras
    - Press Q to quit anytime
"""

import argparse
import cv2
import numpy as np
import pyrealsense2 as rs
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class CharucoStereoCaptureApp:
    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each camera
        self.cam1_dir = self.output_dir / "camera_1"
        self.cam2_dir = self.output_dir / "camera_2"
        self.cam1_dir.mkdir(exist_ok=True)
        self.cam2_dir.mkdir(exist_ok=True)

        # Initialize ChArUco board
        self.setup_charuco_board()

        # Initialize RealSense cameras
        self.setup_cameras()

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
        # Select ArUco dictionary
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

        # Create ChArUco board
        # Note: squares_x and squares_y are the number of squares, not corners
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
        print(f"  Square size: {self.args.square_length} mm")
        print(f"  Marker size: {self.args.marker_length} mm")
        print(f"  Dictionary: {self.args.aruco_dict}")
        print(f"  Expected corners: {(self.args.squares_x-1) * (self.args.squares_y-1)}")

    def setup_cameras(self):
        """Initialize two RealSense cameras"""
        ctx = rs.context()
        devices = ctx.query_devices()

        if len(devices) < 2:
            raise RuntimeError(f"Found only {len(devices)} RealSense camera(s). Need 2 cameras.")

        print(f"\nFound {len(devices)} RealSense cameras:")
        for i, dev in enumerate(devices):
            print(f"  Camera {i}: {dev.get_info(rs.camera_info.serial_number)}")

        # Get serial numbers: config file > auto-detect
        from utils import load_camera_serials
        config_serials = load_camera_serials(self.args.cam_config)
        self.serial1 = config_serials.get(1)
        self.serial2 = config_serials.get(2)

        if not self.serial1 or not self.serial2:
            # Fall back to first two detected cameras
            if not self.serial1:
                self.serial1 = devices[0].get_info(rs.camera_info.serial_number)
            if not self.serial2:
                self.serial2 = devices[1].get_info(rs.camera_info.serial_number)

        print(f"\nUsing cameras:")
        print(f"  Camera 1: {self.serial1}")
        print(f"  Camera 2: {self.serial2}")

        # Configure and start pipelines
        self.pipeline1 = rs.pipeline()
        self.pipeline2 = rs.pipeline()

        config1 = rs.config()
        config2 = rs.config()

        config1.enable_device(self.serial1)
        config2.enable_device(self.serial2)

        # Enable color streams
        config1.enable_stream(rs.stream.color, self.args.width, self.args.height,
                             rs.format.bgr8, self.args.fps)
        config2.enable_stream(rs.stream.color, self.args.width, self.args.height,
                             rs.format.bgr8, self.args.fps)

        # Start streaming
        print(f"\nStarting camera streams ({self.args.width}x{self.args.height} @ {self.args.fps} FPS)...")
        self.pipeline1.start(config1)
        self.pipeline2.start(config2)

        # Wait for cameras to stabilize
        print("Warming up cameras...")
        for _ in range(30):
            self.pipeline1.wait_for_frames()
            self.pipeline2.wait_for_frames()

        print("Cameras ready!")

    def detect_charuco(self, image):
        """Detect ChArUco board in image and return corners"""
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

    def create_info_overlay(self, image, camera_name, num_corners, is_good=False):
        """Add information overlay to image"""
        overlay = image.copy()
        h, w = overlay.shape[:2]

        # Background for text
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

        # Camera name
        cv2.putText(overlay, camera_name, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Detection status
        if num_corners > 0:
            color = (0, 255, 0) if is_good else (0, 165, 255)  # Green if good, orange otherwise
            text = f"Detected: {num_corners} corners"
            cv2.putText(overlay, text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(overlay, "No board detected", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return overlay

    def add_countdown_overlay(self, image, time_remaining):
        """Add countdown timer overlay in center of image"""
        overlay = image.copy()
        h, w = overlay.shape[:2]

        if time_remaining > 0:
            # Countdown number
            countdown_num = int(np.ceil(time_remaining))

            # Large countdown circle and number
            center_x, center_y = w // 2, h // 2
            radius = 80

            # Pulsing effect based on remaining time
            pulse = 1.0 + 0.2 * np.sin(time_remaining * 3.14159)
            current_radius = int(radius * pulse)

            # Circle background
            cv2.circle(overlay, (center_x, center_y), current_radius, (0, 255, 255), -1)
            overlay = cv2.addWeighted(image, 0.3, overlay, 0.7, 0)

            # Countdown number
            font_scale = 3.0
            thickness = 8
            text = str(countdown_num)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2
            cv2.putText(overlay, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

            # "Hold steady" text below
            instruction = "HOLD STEADY"
            cv2.putText(overlay, instruction, (center_x - 120, center_y + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        else:
            # Capturing flash
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
            overlay = cv2.addWeighted(image, 0.4, overlay, 0.6, 0)

            # "CAPTURING!" text
            text = "CAPTURING!"
            font_scale = 2.5
            thickness = 6
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            cv2.putText(overlay, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        return overlay

    def run(self):
        """Main capture loop"""
        print(f"\n{'='*60}")
        print(f"STEREO CALIBRATION IMAGE CAPTURE")
        print(f"{'='*60}")
        print(f"Target: {self.target_captures} synchronized image pairs")

        if self.auto_capture:
            print(f"Mode: AUTO-CAPTURE (interval: {self.capture_interval}s)")
            print(f"\nControls:")
            print(f"  Q: Quit")
            print(f"\nHow it works:")
            print(f"  - Countdown timer appears when board is detected in BOTH cameras")
            print(f"  - Hold board steady during countdown")
            print(f"  - Image captures automatically")
            print(f"  - Move to next position and repeat")
        else:
            print(f"Mode: MANUAL")
            print(f"\nControls:")
            print(f"  SPACE: Capture synchronized pair (when board detected in BOTH cameras)")
            print(f"  Q: Quit")

        print(f"\nTips for good calibration:")
        print(f"  - Move board to different positions (left, right, top, bottom, center)")
        print(f"  - Vary distance (near and far)")
        print(f"  - Tilt board at different angles (not just frontal)")
        print(f"  - Ensure board is fully visible in BOTH cameras")
        print(f"  - Hold steady when capturing (avoid motion blur)")
        print(f"{'='*60}\n")

        mode_str = "AUTO" if self.auto_capture else "MANUAL"
        window_name = f"Stereo ChArUco Capture - {mode_str} MODE (Press Q to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                current_time = time.time()

                # Capture frames from both cameras simultaneously
                frames1 = self.pipeline1.wait_for_frames()
                frames2 = self.pipeline2.wait_for_frames()

                color_frame1 = frames1.get_color_frame()
                color_frame2 = frames2.get_color_frame()

                if not color_frame1 or not color_frame2:
                    continue

                # Convert to numpy arrays
                image1 = np.asanyarray(color_frame1.get_data())
                image2 = np.asanyarray(color_frame2.get_data())

                # Detect ChArUco in both images
                corners1, ids1, detected1, num_corners1 = self.detect_charuco(image1)
                corners2, ids2, detected2, num_corners2 = self.detect_charuco(image2)

                # Check if detection is good in both cameras
                min_corners = max(4, int(0.3 * (self.args.squares_x-1) * (self.args.squares_y-1)))
                is_good = (corners1 is not None and corners2 is not None and
                          num_corners1 >= min_corners and num_corners2 >= min_corners)

                # Add overlays
                display1 = self.create_info_overlay(detected1, f"Camera 1 ({self.serial1})",
                                                   num_corners1, is_good)
                display2 = self.create_info_overlay(detected2, f"Camera 2 ({self.serial2})",
                                                   num_corners2, is_good)

                # Add capture progress
                progress_text = f"Captures: {self.capture_count}/{self.target_captures}"
                h1, w1 = display1.shape[:2]
                cv2.putText(display1, progress_text, (w1-250, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display2, progress_text, (w1-250, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Auto-capture logic
                should_capture = False
                time_remaining = 0

                if self.auto_capture:
                    # Check if enough time has passed since last capture
                    time_since_last = current_time - self.last_capture_time

                    if is_good:
                        # Start countdown if not already started
                        if self.countdown_start_time is None:
                            self.countdown_start_time = current_time

                        # Calculate time remaining in countdown
                        countdown_duration = 3  # 3 second countdown
                        time_in_countdown = current_time - self.countdown_start_time
                        time_remaining = countdown_duration - time_in_countdown

                        if time_remaining <= 0:
                            # Countdown finished - capture!
                            should_capture = True
                            self.countdown_start_time = None
                            time_remaining = 0  # For flash effect

                        # Add countdown overlay
                        display1 = self.add_countdown_overlay(display1, time_remaining)
                        display2 = self.add_countdown_overlay(display2, time_remaining)
                    else:
                        # Board not detected - reset countdown
                        self.countdown_start_time = None

                        # Show waiting message
                        if time_since_last >= self.capture_interval:
                            instruction = "Waiting for board detection..."
                            cv2.putText(display1, instruction, (10, h1-20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                            cv2.putText(display2, instruction, (10, h1-20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                else:
                    # Manual mode - show instruction
                    if is_good:
                        instruction = "READY - Press SPACE to capture"
                        color = (0, 255, 0)
                        cv2.putText(display1, instruction, (10, h1-20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(display2, instruction, (10, h1-20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Combine images side by side
                combined = np.hstack([display1, display2])

                # Resize for display if too large
                display_height = 720
                if combined.shape[0] > display_height:
                    scale = display_height / combined.shape[0]
                    new_width = int(combined.shape[1] * scale)
                    combined = cv2.resize(combined, (new_width, display_height))

                cv2.imshow(window_name, combined)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # Q or ESC
                    print("\nQuitting...")
                    break

                # Manual capture with SPACE (works in both modes)
                if key == ord(' ') and not self.auto_capture:
                    if is_good:
                        should_capture = True
                    else:
                        print("✗ Cannot capture - board not detected well in both cameras")
                        if corners1 is None:
                            print(f"  Camera 1: No detection")
                        elif num_corners1 < min_corners:
                            print(f"  Camera 1: Only {num_corners1} corners (need {min_corners})")
                        if corners2 is None:
                            print(f"  Camera 2: No detection")
                        elif num_corners2 < min_corners:
                            print(f"  Camera 2: Only {num_corners2} corners (need {min_corners})")

                # Perform capture if triggered
                if should_capture:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                    img1_path = self.cam1_dir / f"capture_{self.capture_count:03d}_{timestamp}.png"
                    img2_path = self.cam2_dir / f"capture_{self.capture_count:03d}_{timestamp}.png"

                    cv2.imwrite(str(img1_path), image1)
                    cv2.imwrite(str(img2_path), image2)

                    self.capture_count += 1
                    self.last_capture_time = current_time

                    print(f"✓ Captured pair {self.capture_count}/{self.target_captures} "
                          f"(cam1: {num_corners1} corners, cam2: {num_corners2} corners)")

                    if self.capture_count >= self.target_captures:
                        print(f"\n{'='*60}")
                        print(f"Target reached! Captured {self.capture_count} pairs.")
                        if self.auto_capture:
                            print(f"Waiting 3 seconds before exit...")
                            time.sleep(3)
                            break
                        else:
                            print(f"Press Q to quit or continue capturing more images.")
                        print(f"{'='*60}\n")

        finally:
            # Cleanup
            self.pipeline1.stop()
            self.pipeline2.stop()
            cv2.destroyAllWindows()

            print(f"\nCapture session complete!")
            print(f"Total captures: {self.capture_count}")
            print(f"Images saved to: {self.output_dir}")
            print(f"  Camera 1: {self.cam1_dir}")
            print(f"  Camera 2: {self.cam2_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Capture synchronized image pairs from two RealSense cameras for ChArUco stereo calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Output settings
    parser.add_argument('--output-dir', type=str, default='./charuco_captures',
                       help='Directory to save captured images')
    parser.add_argument('--num-captures', type=int, default=30,
                       help='Target number of image pairs to capture')

    # Capture mode
    parser.add_argument('--auto-capture', action='store_true',
                       help='Enable automatic capture mode with countdown timer (recommended for single person)')
    parser.add_argument('--capture-interval', type=float, default=4.0,
                       help='Time interval between captures in auto mode (seconds). Includes 3s countdown.')

    # ChArUco board parameters
    parser.add_argument('--squares-x', type=int, default=3,
                       help='Number of squares in X direction (columns)')
    parser.add_argument('--squares-y', type=int, default=4,
                       help='Number of squares in Y direction (rows)')
    parser.add_argument('--square-length', type=float, default=0.063,
                       help='Square side length in meters (e.g., 0.063 for 63mm)')
    parser.add_argument('--marker-length', type=float, default=0.047,
                       help='Marker side length in meters (e.g., 0.047 for 47mm)')
    parser.add_argument('--aruco-dict', type=str, default='4X4_50',
                       choices=['4X4_50', '4X4_100', '4X4_250', '4X4_1000',
                               '5X5_50', '5X5_100', '5X5_250', '5X5_1000',
                               '6X6_50', '6X6_100', '6X6_250', '6X6_1000'],
                       help='ArUco dictionary to use')

    # Camera settings
    parser.add_argument('--cam-config', type=str, default='./camera_config.json',
                       help='Path to camera config JSON file mapping cam IDs to serial numbers')
    parser.add_argument('--width', type=int, default=1280,
                       help='Image width')
    parser.add_argument('--height', type=int, default=720,
                       help='Image height')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second')

    args = parser.parse_args()

    # Validate board parameters
    if args.square_length <= args.marker_length:
        parser.error("Square length must be greater than marker length")

    app = CharucoStereoCaptureApp(args)
    app.run()


if __name__ == '__main__':
    main()
