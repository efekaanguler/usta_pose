#!/usr/bin/env python3
"""
Daily extrinsic-only ChArUco capture for the 4-camera RealSense rig.

This script records synchronized two-camera image sets under:

    ../record/recordings/calib_data/extrinsic/run_YYYYMMDD_HHMMSS/

It deliberately does not calculate intrinsics. Fixed lens parameters must come
from master_intrinsics.npz, created once by record_intrinsic.py.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEVEL_DIR = SCRIPT_DIR.parent
RECORD_DIR = DEVEL_DIR / "record"
RECORDINGS_DIR = RECORD_DIR / "recordings"
DEFAULT_CALIB_DIR = RECORDINGS_DIR / "calib_data"
DEFAULT_EXTRINSIC_DIR = DEFAULT_CALIB_DIR / "extrinsic"
DEFAULT_CAM_CONFIG = RECORD_DIR / "camera_config.json"

DEFAULT_PAIRS = ["1,2", "1,3", "2,4", "2,3", "1,4"]


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
        description="Capture routine extrinsic-only ChArUco image pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cam-config", type=Path, default=DEFAULT_CAM_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRINSIC_DIR)
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS,
                        help="Two-camera sessions to capture, e.g. --pairs 1,2 1,3 2,4")
    parser.add_argument("--num-captures", type=int, default=20)
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

    if args.num_captures < 5:
        parser.error("--num-captures must be at least 5 for stereo extrinsics.")
    if args.square_length <= args.marker_length:
        parser.error("--square-length must be greater than --marker-length.")
    for pair in args.pairs:
        normalize_pair(pair)
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"run_{timestamp}"
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
