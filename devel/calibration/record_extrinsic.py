#!/usr/bin/env python3
"""
Daily extrinsic-only capture for the 4-camera RealSense rig.

Default method records the known five-face AprilTag cube in the legacy
pairwise camera graph under:

    ../record/recordings/calib_data/extrinsic/current/

Each pair is captured separately, so the cube can be moved through a much
larger useful field of view than an all-four-camera capture would permit.
The legacy pairwise ChArUco workflow remains available with --method charuco.
This script deliberately does not calculate intrinsics.
"""

import argparse
import json
import os
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
DEFAULT_IDEAL_CUBE_LAYOUT = RECORD_DIR / "apriltag_cube_layout.json"
DEFAULT_CALIBRATED_CUBE_LAYOUT = (
    RECORD_DIR / "apriltag_cube_layout_calibrated.json"
)
DEFAULT_CUBE_LAYOUT = (
    DEFAULT_CALIBRATED_CUBE_LAYOUT
    if DEFAULT_CALIBRATED_CUBE_LAYOUT.exists()
    else DEFAULT_IDEAL_CUBE_LAYOUT
)

DEFAULT_PAIRS = ["1,2", "1,3", "2,4", "2,3", "1,4"]
_NATIVE_RESOURCES = []


def setup_realsense_cameras(args, camera_ids):
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise ImportError(
            "pyrealsense2 is required for RealSense cube capture."
        ) from exc

    sys.path.insert(0, str(DEVEL_DIR))
    from utils import load_camera_serials

    serial_mapping = load_camera_serials(args.cam_config)
    camera_ids = list(camera_ids)
    missing = [camera_id for camera_id in camera_ids
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
            for camera_id in camera_ids:
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
            f"Could not start cameras {camera_ids} with a common color profile."
        )

    for camera_id, pipeline in zip(camera_ids, pipelines):
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


def build_cube_preview(
    images,
    camera_ids,
    detections_by_camera,
    capture_count,
    target_count,
):
    previews = []
    for camera_id, image in zip(camera_ids, images):
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

    if len(previews) <= 2:
        return np.hstack(previews)

    while len(previews) < 4:
        previews.append(np.zeros((360, 640, 3), dtype=np.uint8))
    return np.vstack([np.hstack(previews[:2]), np.hstack(previews[2:4])])


def add_cube_countdown_overlay(preview, time_remaining):
    overlay = preview.copy()
    height, width = overlay.shape[:2]
    center = (width // 2, height // 2)

    if time_remaining > 0:
        pulse = 1.0 + 0.2 * np.sin(time_remaining * np.pi)
        radius = int(78 * pulse)
        cv2.circle(overlay, center, radius, (0, 255, 255), -1, cv2.LINE_AA)
        overlay = cv2.addWeighted(preview, 0.35, overlay, 0.65, 0)

        countdown_text = str(int(np.ceil(time_remaining)))
        text_size = cv2.getTextSize(
            countdown_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            3.0,
            8,
        )[0]
        text_origin = (
            center[0] - text_size[0] // 2,
            center[1] + text_size[1] // 2,
        )
        cv2.putText(
            overlay,
            countdown_text,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            3.0,
            (0, 0, 0),
            8,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "HOLD CUBE STILL",
            (center[0] - 190, center[1] + 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
    else:
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 255, 0), -1)
        overlay = cv2.addWeighted(preview, 0.40, overlay, 0.60, 0)
        capture_text = "CAPTURED!"
        text_size = cv2.getTextSize(
            capture_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            6,
        )[0]
        cv2.putText(
            overlay,
            capture_text,
            (
                center[0] - text_size[0] // 2,
                center[1] + text_size[1] // 2,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (255, 255, 255),
            6,
            cv2.LINE_AA,
        )
    return overlay


def save_cube_capture(run_dir, capture_index, images, camera_ids):
    capture_dir = run_dir / f"capture_{capture_index:03d}"
    capture_dir.mkdir(parents=True, exist_ok=False)
    for camera_id, image in zip(camera_ids, images):
        if image is None:
            continue
        image_path = capture_dir / f"camera_{camera_id}.png"
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError(f"Could not write calibration image: {image_path}")
    return capture_dir


def cube_pose_changed(
    detections_by_camera,
    previous_detections_by_camera,
    images_by_camera,
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
        image = images_by_camera[camera_id]
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


def cube_pose_stable(
    detections_by_camera,
    previous_detections_by_camera,
    images_by_camera,
    max_normalized_change,
):
    if previous_detections_by_camera is None:
        return False

    camera_changes = []
    for camera_id, detections in detections_by_camera.items():
        previous = {
            detection.tag_id: detection
            for detection in previous_detections_by_camera.get(camera_id, [])
        }
        current = {detection.tag_id: detection for detection in detections}
        if not current or set(previous) != set(current):
            return False

        image = images_by_camera[camera_id]
        diagonal = float(np.hypot(image.shape[1], image.shape[0]))
        corner_changes = [
            np.mean(np.linalg.norm(
                current[tag_id].corners - previous[tag_id].corners,
                axis=1,
            )) / diagonal
            for tag_id in sorted(current)
        ]
        camera_changes.append(float(np.max(corner_changes)))

    if not camera_changes:
        return False
    return float(np.max(camera_changes)) <= max_normalized_change


def run_cube_camera_group_capture(
    args,
    run_dir,
    camera_ids,
    detector,
    cube_layout,
):
    from cube_calibration import detect_known_cube_tags

    camera_ids = list(camera_ids)
    pipelines, actual_stream = setup_realsense_cameras(args, camera_ids)
    required_cameras = (
        len(camera_ids)
        if args.cube_capture_mode == "pairwise"
        else args.min_cameras
    )
    capture_count = 0
    countdown_start_time = None
    previous_detections_by_camera = None
    previous_live_detections_by_camera = None
    camera_label = "+".join(f"CAM{camera_id}" for camera_id in camera_ids)
    window_name = f"AprilTag Cube Extrinsic Capture | {camera_label}"
    if not args.no_gui:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"\n[record_extrinsic] AprilTag cube session: {camera_label}")
    print(f"  Cube layout: {args.cube_layout}")
    print(f"  Accepted capture target: {args.num_captures}")
    print(f"  Cameras required per capture: {required_cameras}")
    print("  Keep the cube's untagged face on the table.")
    print("  Move and rotate the cube between captures; keep it still at capture time.")
    print("  The two cameras may observe different tagged faces of the same cube.")
    if args.manual:
        print("  SPACE: capture when ready | Q: quit")
    else:
        print(f"  Auto countdown: {args.capture_interval:.1f}s | Q: quit")

    try:
        while capture_count < args.num_captures:
            images = capture_color_frames(pipelines)
            images_by_camera = dict(zip(camera_ids, images))
            detections_by_camera = {}
            for camera_id, image in images_by_camera.items():
                if image is None:
                    continue
                detections_by_camera[camera_id] = detect_known_cube_tags(
                    image,
                    detector,
                    cube_layout.keys(),
                    min_decision_margin=args.min_decision_margin,
                    max_hamming=args.max_hamming,
                    min_tag_edge_px=args.min_tag_edge_px,
                )

            ready_cameras = sum(bool(items) for items in detections_by_camera.values())
            ready = ready_cameras >= required_cameras
            pose_changed = ready and cube_pose_changed(
                detections_by_camera,
                previous_detections_by_camera,
                images_by_camera,
                args.min_pose_change,
            )
            pose_stable = ready and cube_pose_stable(
                detections_by_camera,
                previous_live_detections_by_camera,
                images_by_camera,
                args.max_stability_change,
            )
            previous_live_detections_by_camera = {
                camera_id: list(detections)
                for camera_id, detections in detections_by_camera.items()
            }
            capture_ready = ready and pose_changed and pose_stable
            should_capture = False
            time_remaining = None
            current_time = time.monotonic()
            if not args.manual:
                if capture_ready:
                    if countdown_start_time is None:
                        countdown_start_time = current_time
                    elapsed = current_time - countdown_start_time
                    time_remaining = args.capture_interval - elapsed
                    if time_remaining <= 0:
                        time_remaining = 0.0
                        should_capture = True
                else:
                    countdown_start_time = None

            key = -1
            if not args.no_gui:
                preview = build_cube_preview(
                    images,
                    camera_ids,
                    detections_by_camera,
                    capture_count,
                    args.num_captures,
                )
                if capture_ready:
                    status = f"READY ({ready_cameras}/{len(camera_ids)})"
                    color = (0, 255, 0)
                elif ready:
                    if not pose_changed:
                        status = "MOVE/ROTATE CUBE TO A NEW POSE"
                        color = (0, 190, 255)
                    else:
                        status = "HOLD CUBE STILL"
                        color = (0, 255, 255)
                else:
                    status = (
                        f"NEED {required_cameras} CAMERAS "
                        f"({ready_cameras}/{len(camera_ids)})"
                    )
                    color = (0, 0, 255)
                if time_remaining is not None:
                    preview = add_cube_countdown_overlay(
                        preview,
                        time_remaining,
                    )
                    if time_remaining > 0:
                        status = f"HOLD STILL - {time_remaining:.1f}s"
                        color = (0, 255, 255)
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

            if args.manual:
                should_capture = key == ord(" ") and capture_ready

            if not should_capture:
                continue

            capture_dir = save_cube_capture(
                run_dir,
                capture_count,
                images,
                camera_ids,
            )
            capture_count += 1
            countdown_start_time = None
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

    print(f"\n[record_extrinsic] Cube session ready: {run_dir}")
    print(f"[record_extrinsic] {camera_label} stream: {actual_stream}")
    return run_dir


def run_cube_capture(args):
    from cube_calibration import create_apriltag_detector, load_cube_layout

    cube_layout = load_cube_layout(args.cube_layout)
    detector = create_apriltag_detector(args.apriltag_family)
    _NATIVE_RESOURCES.append(detector)
    run_dir = args.output_dir / "current"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    stale_manifest = args.output_dir / "latest_extrinsic_run.json"
    stale_manifest.unlink(missing_ok=True)

    if args.cube_capture_mode == "pairwise":
        for pair in args.pairs:
            cam_a, cam_b = normalize_pair(pair)
            session_dir = run_dir / f"session_cam{cam_a}_cam{cam_b}"
            session_dir.mkdir(parents=True, exist_ok=False)
            run_cube_camera_group_capture(
                args,
                session_dir,
                (cam_a, cam_b),
                detector,
                cube_layout,
            )
    else:
        run_cube_camera_group_capture(
            args,
            run_dir,
            range(1, args.num_cameras + 1),
            detector,
            cube_layout,
        )

    print(f"\n[record_extrinsic] All cube captures ready: {run_dir}")
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
        "--min-pose-change", str(args.min_pose_change),
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
    parser.add_argument(
        "--cube-capture-mode",
        choices=("pairwise", "all"),
        default="pairwise",
        help=(
            "Capture each legacy camera pair separately, or require a shared "
            "all-camera cube view."
        ),
    )
    parser.add_argument("--num-cameras", type=int, default=4)
    parser.add_argument(
        "--min-cameras",
        type=int,
        default=2,
        help="Minimum cube-visible cameras in --cube-capture-mode all.",
    )
    parser.add_argument("--min-decision-margin", type=float, default=20.0)
    parser.add_argument("--max-hamming", type=int, default=0)
    parser.add_argument("--min-tag-edge-px", type=float, default=45.0)
    parser.add_argument(
        "--min-pose-change",
        type=float,
        default=0.015,
        help="Minimum median tag-corner displacement as an image-diagonal ratio.",
    )
    parser.add_argument(
        "--max-stability-change",
        type=float,
        default=0.002,
        help=(
            "Maximum frame-to-frame tag-corner displacement as an image-diagonal "
            "ratio while the automatic countdown is running."
        ),
    )
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=DEFAULT_PAIRS,
        help="Two-camera sessions for cube pairwise or legacy ChArUco capture.",
    )
    parser.add_argument("--num-captures", type=int, default=30)
    parser.add_argument(
        "--capture-interval",
        type=float,
        default=1.0,
        help="Seconds shown in the automatic pre-capture countdown.",
    )
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
    if args.capture_interval <= 0:
        parser.error("--capture-interval must be greater than zero.")
    if args.max_stability_change <= 0:
        parser.error("--max-stability-change must be greater than zero.")
    if args.num_cameras < 2:
        parser.error("--num-cameras must be at least 2.")
    if not 2 <= args.min_cameras <= args.num_cameras:
        parser.error("--min-cameras must be between 2 and --num-cameras.")
    if args.method == "cube":
        if not args.cube_layout.exists():
            parser.error(f"Cube layout does not exist: {args.cube_layout}")
        if args.manual and args.no_gui:
            parser.error("--manual cannot be combined with --no-gui.")
        if args.cube_capture_mode == "pairwise":
            for pair in args.pairs:
                cam_a, cam_b = normalize_pair(pair)
                if not 1 <= cam_a <= args.num_cameras:
                    parser.error(f"Camera {cam_a} in --pairs exceeds --num-cameras.")
                if not 1 <= cam_b <= args.num_cameras:
                    parser.error(f"Camera {cam_b} in --pairs exceeds --num-cameras.")
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
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

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
