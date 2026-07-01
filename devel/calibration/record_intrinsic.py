#!/usr/bin/env python3
"""
One-time high precision ChArUco intrinsic calibration for 4 RealSense cameras.

This script captures at least 50 valid color frames per camera and writes the
fixed lens parameters to:

    ../record/recordings/calib_data/master_intrinsics.npz

Daily calibration should not run this script. Use record_extrinsic.py +
calculate.py for the routine extrinsic-only flow.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from multicam_calibrate import MulticamCalibrator


SCRIPT_DIR = Path(__file__).resolve().parent
DEVEL_DIR = SCRIPT_DIR.parent
RECORD_DIR = DEVEL_DIR / "record"
RECORDINGS_DIR = RECORD_DIR / "recordings"
DEFAULT_CALIB_DIR = RECORDINGS_DIR / "calib_data"
DEFAULT_CAM_CONFIG = RECORD_DIR / "camera_config.json"
DEFAULT_MASTER_INTRINSICS = DEFAULT_CALIB_DIR / "master_intrinsics.npz"


def board_args(args):
    return [
        "--squares-x", str(args.squares_x),
        "--squares-y", str(args.squares_y),
        "--square-length", str(args.square_length),
        "--marker-length", str(args.marker_length),
        "--aruco-dict", args.aruco_dict,
    ]


def capture_intrinsic_images(args, run_dir):
    capture_script = SCRIPT_DIR / "charuco_intrinsic_capture.py"
    for cam_id in range(1, args.num_cameras + 1):
        cam_dir = run_dir / f"cam{cam_id}"
        cam_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(capture_script),
            "--cam-config", str(args.cam_config),
            "--camera-id", str(cam_id),
            "--output-dir", str(cam_dir),
            "--num-captures", str(args.num_captures),
            "--capture-interval", str(args.capture_interval),
            "--width", str(args.width),
            "--height", str(args.height),
            "--fps", str(args.fps),
            *board_args(args),
        ]
        if not args.manual:
            cmd.append("--auto-capture")

        print(f"\n[record_intrinsic] Camera {cam_id}: capturing into {cam_dir}")
        subprocess.run(cmd, check=True)

        image_count = len(list(cam_dir.glob("*.png")))
        if image_count < args.num_captures:
            raise RuntimeError(
                f"Camera {cam_id} captured only {image_count} images; "
                f"expected at least {args.num_captures}."
            )


def calculate_master_intrinsics(args, run_dir, output_path):
    calibrator_args = argparse.Namespace(
        num_cameras=args.num_cameras,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        aruco_dict=args.aruco_dict,
        session_dirs=[],
        min_pairs=5,
        output=str(output_path),
        ref_camera=1,
    )
    calibrator = MulticamCalibrator(calibrator_args)

    for cam_idx in range(args.num_cameras):
        cam_dir = run_dir / f"cam{cam_idx + 1}"
        image_count = len(list(cam_dir.glob("*.png")))
        if image_count < args.num_captures:
            raise RuntimeError(
                f"{cam_dir} has {image_count} images; "
                f"need at least {args.num_captures} for master intrinsics."
            )
        calibrator.calibrate_intrinsics(cam_idx, cam_dir)

    created_at = datetime.now().isoformat(timespec="seconds")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "created_at": np.array(created_at),
        "source_run": np.array(str(run_dir)),
        "num_cameras": np.array(args.num_cameras),
        "image_size": np.array(calibrator.image_size if calibrator.image_size else [0, 0]),
        "squares_x": np.array(args.squares_x),
        "squares_y": np.array(args.squares_y),
        "square_length": np.array(args.square_length),
        "marker_length": np.array(args.marker_length),
        "aruco_dict": np.array(args.aruco_dict),
    }

    summary = {
        "created_at": created_at,
        "source_run": str(run_dir),
        "num_cameras": args.num_cameras,
        "image_size": list(calibrator.image_size) if calibrator.image_size else [0, 0],
        "board": {
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_length": args.square_length,
            "marker_length": args.marker_length,
            "aruco_dict": args.aruco_dict,
        },
        "cameras": {},
    }

    for cam_idx in range(args.num_cameras):
        cam_num = cam_idx + 1
        K, dist, error = calibrator.intrinsics[cam_idx]
        data[f"K{cam_num}"] = K
        data[f"dist{cam_num}"] = dist
        data[f"intrinsic_error{cam_num}"] = np.array(error)

        summary["cameras"][f"camera_{cam_num}"] = {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "distortion": dist.flatten().astype(float).tolist(),
            "mean_reprojection_error_px": float(error),
        }

    np.savez(output_path, **data)

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    latest_marker = output_path.parent / "latest_intrinsic_run.json"
    with open(latest_marker, "w", encoding="utf-8") as f:
        json.dump({"source_run": str(run_dir), "master_intrinsics": str(output_path)}, f, indent=2)

    print(f"\n[record_intrinsic] Master intrinsics saved: {output_path}")
    print(f"[record_intrinsic] Human-readable summary: {json_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture and calculate fixed master intrinsics for all RealSense cameras.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cam-config", type=Path, default=DEFAULT_CAM_CONFIG)
    parser.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_MASTER_INTRINSICS)
    parser.add_argument("--num-cameras", type=int, default=4)
    parser.add_argument("--num-captures", type=int, default=50)
    parser.add_argument("--capture-interval", type=float, default=4.0)
    parser.add_argument("--manual", action="store_true", help="Use manual SPACE capture instead of auto-capture.")
    parser.add_argument("--skip-capture", action="store_true", help="Recalculate from an existing --input-run.")
    parser.add_argument("--input-run", type=Path, default=None, help="Existing intrinsic run directory with cam1..cam4.")
    parser.add_argument("--squares-x", type=int, default=4)
    parser.add_argument("--squares-y", type=int, default=3)
    parser.add_argument("--square-length", type=float, default=0.063)
    parser.add_argument("--marker-length", type=float, default=0.047)
    parser.add_argument("--aruco-dict", type=str, default="4X4_50")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if args.num_captures < 50:
        parser.error("--num-captures must be at least 50 for master intrinsics.")
    if args.square_length <= args.marker_length:
        parser.error("--square-length must be greater than --marker-length.")
    if args.skip_capture and args.input_run is None:
        parser.error("--skip-capture requires --input-run.")
    return args


def main():
    args = parse_args()
    args.calib_dir.mkdir(parents=True, exist_ok=True)

    if args.input_run is not None:
        run_dir = args.input_run
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = args.calib_dir / "intrinsic" / f"run_{timestamp}"

    if not args.skip_capture:
        capture_intrinsic_images(args, run_dir)

    calculate_master_intrinsics(args, run_dir, args.output)


if __name__ == "__main__":
    main()
