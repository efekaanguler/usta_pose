#!/usr/bin/env python3
"""
Daily extrinsic-only capture for the 4-camera RealSense rig.

Default method records all four cameras observing the known five-face
AprilTag cube under:

    ../record/recordings/calib_data/extrinsic/current/

The legacy pairwise ChArUco workflow remains available with --method charuco.
This script deliberately does not calculate intrinsics.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEVEL_DIR = SCRIPT_DIR.parent
RECORD_DIR = DEVEL_DIR / "record"
RECORDINGS_DIR = RECORD_DIR / "recordings"
DEFAULT_CALIB_DIR = RECORDINGS_DIR / "calib_data"
DEFAULT_EXTRINSIC_DIR = DEFAULT_CALIB_DIR / "extrinsic"
DEFAULT_CAM_CONFIG = RECORD_DIR / "camera_config.json"
DEFAULT_CUBE_LAYOUT = RECORD_DIR / "apriltag_cube_layout.json"

DEFAULT_PAIRS = ["1,2", "1,3", "2,4", "2,3", "1,4"]


def setup_realsense_cameras(args):
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise ImportError(
            "pyrealsense2 is required for RealSense cube capture."
        ) from exc

    sys.path.insert(0, str(DEVEL_DIR))
    from utils import load_camera_serials

    serial_mapping = load_camera_serials(args.cam_config)
    missing = [camera_id for camera_id in range(1, args.num_cameras + 1)
               if not serial_mapping.get(camera_id)]
    if missing:
        raise RuntimeError(
            f"Missing camera serials in {args.cam_config}: {missing}"
        )

    stream_configs = [
        (args.width, args.height, args.fps),
        (args.width, args.height, 15),
        (848, 480, 30),
        (848, 480, 15),
        (640, 480, 30),
        (640, 480, 15),
    ]
    pipelines = []
    actual_stream = None
    for width, height, fps in stream_configs:
        candidate_pipelines = []
        try:
            for camera_id in range(1, args.num_cameras + 1):
                serial = serial_mapping[camera_id]
                candidate = rs.pipeline()
                config = rs.config()
                config.enable_device(serial)
                config.enable_stream(
                    rs.stream.color,
                    width,
                    height,
                    rs.format.bgr8,
                    fps,
                )
                candidate.start(config)
                candidate_pipelines.append(candidate)
        except RuntimeError:
            for candidate in candidate_pipelines:
                candidate.stop()
            continue

        pipelines = candidate_pipelines
        actual_stream = (width, height, fps)
        break

    if not pipelines:
        raise RuntimeError(
            "Could not start all configured cameras with a common color profile."
        )

    for camera_id, pipeline in enumerate(pipelines, start=1):
        serial = serial_mapping[camera_id]
        width, height, fps = actual_stream
        print(
            f"[record_extrinsic] Camera {camera_id} ({serial}) "
            f"started at {width}x{height}@{fps}."
        )

    for _warmup_index in range(30):
        for pipeline in pipelines:
            pipeline.wait_for_frames()
    return pipelines, actual_stream


def capture_color_frames(pipelines):
    images = []
    for pipeline in pipelines:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if color_frame is None:
            images.append(None)
        else:
            images.append(np.asanyarray(color_frame.get_data()))
    return images


def build_cube_preview(images, detections_by_camera, capture_count, target_count):
    previews = []
    for camera_index, image in enumerate(images):
        camera_id = camera_index + 1
        if image is None:
            image = np.zeros((360, 640, 3), dtype=np.uint8)
        overlay = image.copy()
        detections = detections_by_camera.get(camera_id, [])
        for detection in detections:
            points = np.round(detection.corners).astype(np.int32).reshape(4, 2)
            cv2.polylines(overlay, [points], True, (0, 255, 0), 2, cv2.LINE_AA)
            center = tuple(np.round(points.mean(axis=0)).astype(int))
            cv2.putText(
                overlay,
                f"ID {detection.tag_id}",
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        ready_color = (0, 210, 0) if detections else (0, 0, 255)
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 72), (0, 0, 0), -1)
        cv2.putText(
            overlay,
            f"CAM{camera_id} | tags={[item.tag_id for item in detections]}",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            ready_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"accepted {capture_count}/{target_count}",
            (18, 59),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        previews.append(cv2.resize(overlay, (640, 360)))

    while len(previews) < 4:
        previews.append(np.zeros((360, 640, 3), dtype=np.uint8))
    return np.vstack([np.hstack(previews[:2]), np.hstack(previews[2:4])])


def save_cube_capture(run_dir, capture_index, images):
    capture_dir = run_dir / f"capture_{capture_index:03d}"
    capture_dir.mkdir(parents=True, exist_ok=False)
    for camera_index, image in enumerate(images):
        if image is None:
            continue
        image_path = capture_dir / f"camera_{camera_index + 1}.png"
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError(f"Could not write calibration image: {image_path}")
    return capture_dir


def cube_pose_changed(
    detections_by_camera,
    previous_detections_by_camera,
    images,
    min_normalized_change,
):
    if previous_detections_by_camera is None:
        return True

    camera_changes = []
    for camera_id, detections in detections_by_camera.items():
        previous = {
            detection.tag_id: detection
            for detection in previous_detections_by_camera.get(camera_id, [])
        }
        current = {detection.tag_id: detection for detection in detections}
        if set(previous) != set(current):
            camera_changes.append(float("inf"))
            continue

        common_ids = sorted(set(previous) & set(current))
        if not common_ids:
            continue
        image = images[camera_id - 1]
        diagonal = float(np.hypot(image.shape[1], image.shape[0]))
        corner_changes = [
            np.mean(np.linalg.norm(
                current[tag_id].corners - previous[tag_id].corners,
                axis=1,
            )) / diagonal
            for tag_id in common_ids
        ]
        camera_changes.append(float(np.median(corner_changes)))

    if not camera_changes:
        return False
    return float(np.median(camera_changes)) >= min_normalized_change


def run_cube_capture(args):
    from cube_calibration import (
        create_apriltag_detector,
        detect_known_cube_tags,
        load_cube_layout,
    )

    cube_layout = load_cube_layout(args.cube_layout)
    detector = create_apriltag_detector(args.apriltag_family)
    pipelines, actual_stream = setup_realsense_cameras(args)
    run_dir = args.output_dir / "current"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    stale_manifest = args.output_dir / "latest_extrinsic_run.json"
    stale_manifest.unlink(missing_ok=True)

    capture_count = 0
    last_capture_time = time.monotonic()
    previous_detections_by_camera = None
    window_name = "4-Camera AprilTag Cube Extrinsic Capture"
    if not args.no_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n[record_extrinsic] AprilTag cube mode")
    print(f"  Cube layout: {args.cube_layout}")
    print(f"  Accepted capture target: {args.num_captures}")
    print(f"  Minimum cameras per capture: {args.min_cameras}")
    print("  Keep the cube's untagged face on the table.")
    print("  Keep the cube still while a capture is taken.")
    if args.manual:
        print("  SPACE: capture when ready | Q: quit")
    else:
        print(f"  Auto capture interval: {args.capture_interval:.1f}s | Q: quit")

    try:
        while capture_count < args.num_captures:
            images = capture_color_frames(pipelines)
            detections_by_camera = {}
            for camera_index, image in enumerate(images):
                if image is None:
                    continue
                camera_id = camera_index + 1
                detections_by_camera[camera_id] = detect_known_cube_tags(
                    image,
                    detector,
                    cube_layout.keys(),
                    min_decision_margin=args.min_decision_margin,
                    max_hamming=args.max_hamming,
                    min_tag_edge_px=args.min_tag_edge_px,
                )

            ready_cameras = sum(bool(items) for items in detections_by_camera.values())
            ready = ready_cameras >= args.min_cameras
            pose_changed = ready and cube_pose_changed(
                detections_by_camera,
                previous_detections_by_camera,
                images,
                args.min_pose_change,
            )
            capture_ready = ready and pose_changed
            key = -1
            if not args.no_gui:
                preview = build_cube_preview(
                    images,
                    detections_by_camera,
                    capture_count,
                    args.num_captures,
                )
                if capture_ready:
                    status = f"READY ({ready_cameras}/{args.num_cameras})"
                    color = (0, 255, 0)
                elif ready:
                    status = "MOVE/ROTATE CUBE TO A NEW POSE"
                    color = (0, 190, 255)
                else:
                    status = (
                        f"NEED {args.min_cameras} CAMERAS "
                        f"({ready_cameras}/{args.num_cameras})"
                    )
                    color = (0, 0, 255)
                cv2.putText(
                    preview,
                    status,
                    (24, preview.shape[0] - 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    color,
                    3,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            should_capture = False
            if args.manual:
                should_capture = key == ord(" ") and capture_ready
            elif (
                capture_ready
                and time.monotonic() - last_capture_time >= args.capture_interval
            ):
                should_capture = True

            if not should_capture:
                continue

            capture_dir = save_cube_capture(run_dir, capture_count, images)
            capture_count += 1
            last_capture_time = time.monotonic()
            previous_detections_by_camera = {
                camera_id: list(detections)
                for camera_id, detections in detections_by_camera.items()
            }
            print(
                f"  Captured {capture_count}/{args.num_captures}: "
                f"{capture_dir.name} ({ready_cameras} cameras with valid tags)"
            )
    finally:
        for pipeline in pipelines:
            try:
                pipeline.stop()
            except RuntimeError:
                pass
        cv2.destroyAllWindows()

    if capture_count < args.num_captures:
        raise RuntimeError(
            f"Capture stopped at {capture_count}/{args.num_captures}; calibration not run."
        )

    print(f"\n[record_extrinsic] Cube captures ready: {run_dir}")
    print(f"[record_extrinsic] Stream: {actual_stream}")
    return run_dir


def board_args(args):
    return [
        "--squares-x", str(args.squares_x),
        "--squares-y", str(args.squares_y),
        "--square-length", str(args.square_length),
        "--marker-length", str(args.marker_length),
        "--aruco-dict", args.aruco_dict,
    ]


def normalize_pair(pair):
    parts = [p.strip() for p in pair.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid camera pair '{pair}'. Expected format like 1,2.")
    cam_a, cam_b = (int(parts[0]), int(parts[1]))
    if cam_a == cam_b:
        raise ValueError(f"Invalid camera pair '{pair}'. Cameras must be different.")
    return cam_a, cam_b


def run_pair_capture(args, run_dir, pair):
    cam_a, cam_b = normalize_pair(pair)
    session_dir = run_dir / f"session_cam{cam_a}_cam{cam_b}"
    session_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "multicam_capture.py"),
        "--cam-config", str(args.cam_config),
        "--output-dir", str(session_dir),
        "--camera-ids", f"{cam_a},{cam_b}",
        "--num-captures", str(args.num_captures),
        "--min-cameras", "2",
        "--capture-interval", str(args.capture_interval),
        "--width", str(args.width),
        "--height", str(args.height),
        "--fps", str(args.fps),
        *board_args(args),
    ]
    if not args.manual:
        cmd.append("--auto-capture")

    print(f"\n[record_extrinsic] Pair cam{cam_a}+cam{cam_b}: {session_dir}")
    subprocess.run(cmd, check=True)
    return session_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Capture routine extrinsic-only observations using an AprilTag "
            "cube or the legacy pairwise ChArUco workflow."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", choices=("cube", "charuco"), default="cube")
    parser.add_argument("--cam-config", type=Path, default=DEFAULT_CAM_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRINSIC_DIR)
    parser.add_argument("--cube-layout", type=Path, default=DEFAULT_CUBE_LAYOUT)
    parser.add_argument("--apriltag-family", type=str, default="tag36h11")
    parser.add_argument("--num-cameras", type=int, default=4)
    parser.add_argument("--min-cameras", type=int, default=4)
    parser.add_argument("--min-decision-margin", type=float, default=20.0)
    parser.add_argument("--max-hamming", type=int, default=0)
    parser.add_argument("--min-tag-edge-px", type=float, default=45.0)
    parser.add_argument(
        "--min-pose-change",
        type=float,
        default=0.015,
        help="Minimum median tag-corner displacement as an image-diagonal ratio.",
    )
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS,
                        help="Two-camera sessions to capture, e.g. --pairs 1,2 1,3 2,4")
    parser.add_argument("--num-captures", type=int, default=30)
    parser.add_argument("--capture-interval", type=float, default=4.0)
    parser.add_argument("--manual", action="store_true", help="Use manual SPACE capture instead of auto-capture.")
    parser.add_argument("--squares-x", type=int, default=4)
    parser.add_argument("--squares-y", type=int, default=3)
    parser.add_argument("--square-length", type=float, default=0.063)
    parser.add_argument("--marker-length", type=float, default=0.047)
    parser.add_argument("--aruco-dict", type=str, default="4X4_50")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if args.num_captures < 10:
        parser.error("--num-captures must be at least 10.")
    if args.num_cameras < 2:
        parser.error("--num-cameras must be at least 2.")
    if not 2 <= args.min_cameras <= args.num_cameras:
        parser.error("--min-cameras must be between 2 and --num-cameras.")
    if args.method == "cube":
        if not args.cube_layout.exists():
            parser.error(f"Cube layout does not exist: {args.cube_layout}")
        if args.manual and args.no_gui:
            parser.error("--manual cannot be combined with --no-gui.")
    else:
        if args.square_length <= args.marker_length:
            parser.error("--square-length must be greater than --marker-length.")
        for pair in args.pairs:
            normalize_pair(pair)
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "cube":
        run_cube_capture(args)
        return

    run_dir = args.output_dir / "current"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    session_dirs = []
    for pair in args.pairs:
        session_dirs.append(run_pair_capture(args, run_dir, pair))

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "pairs": args.pairs,
        "session_dirs": [str(path) for path in session_dirs],
        "num_captures": args.num_captures,
        "note": "Extrinsic-only capture; intrinsics are not calculated here.",
    }

    manifest_path = run_dir / "extrinsic_run.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    latest_path = args.output_dir / "latest_extrinsic_run.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[record_extrinsic] Latest extrinsic run: {run_dir}")
    print(f"[record_extrinsic] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
