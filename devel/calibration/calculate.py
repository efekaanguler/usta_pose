#!/usr/bin/env python3
"""
Combine fixed master intrinsics with the latest daily extrinsic captures.

Input:
    ../record/recordings/calib_data/master_intrinsics.npz
    ../record/recordings/calib_data/extrinsic/latest_extrinsic_run.json

Output, overwritten on every successful run:
    ../record/recordings/multicam_calibration.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np

from multicam_calibrate import MulticamCalibrator


SCRIPT_DIR = Path(__file__).resolve().parent
DEVEL_DIR = SCRIPT_DIR.parent
RECORD_DIR = DEVEL_DIR / "record"
RECORDINGS_DIR = RECORD_DIR / "recordings"
DEFAULT_CALIB_DIR = RECORDINGS_DIR / "calib_data"
DEFAULT_MASTER_INTRINSICS = DEFAULT_CALIB_DIR / "master_intrinsics.npz"
DEFAULT_EXTRINSIC_DIR = DEFAULT_CALIB_DIR / "extrinsic"
DEFAULT_OUTPUT = RECORDINGS_DIR / "multicam_calibration.npz"


def load_master_intrinsics(path, num_cameras):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing master intrinsics: {path}\n"
            "Run record_intrinsic.py once before the daily calibration flow."
        )

    intrinsics = {}
    with np.load(path) as data:
        image_size = np.array(data["image_size"], dtype=int) if "image_size" in data else np.array([0, 0])
        for cam_num in range(1, num_cameras + 1):
            k_key = f"K{cam_num}"
            d_key = f"dist{cam_num}"
            if k_key not in data or d_key not in data:
                raise KeyError(f"{path} does not contain {k_key}/{d_key}.")

            error_key = f"intrinsic_error{cam_num}"
            error = float(data[error_key]) if error_key in data else 0.0
            intrinsics[cam_num - 1] = (
                np.array(data[k_key], dtype=np.float64),
                np.array(data[d_key], dtype=np.float64),
                error,
            )

    return intrinsics, tuple(int(v) for v in image_size.tolist())


def session_dirs_from_latest_manifest(extrinsic_dir):
    latest_path = extrinsic_dir / "latest_extrinsic_run.json"
    if not latest_path.exists():
        return None

    with open(latest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    session_dirs = [Path(p) for p in manifest.get("session_dirs", [])]
    existing = [p for p in session_dirs if p.exists()]
    if existing:
        return existing
    return None


def discover_latest_session_dirs(extrinsic_dir):
    from_manifest = session_dirs_from_latest_manifest(extrinsic_dir)
    if from_manifest:
        return from_manifest

    runs = sorted(
        [p for p in extrinsic_dir.glob("run_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError(
            f"No extrinsic runs found under {extrinsic_dir}. "
            "Run record_extrinsic.py first."
        )

    session_dirs = sorted(
        [p for p in runs[0].iterdir() if p.is_dir() and p.name.startswith("session_cam")]
    )
    if not session_dirs:
        raise FileNotFoundError(f"No session_cam* directories found in {runs[0]}.")
    return session_dirs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Solve daily extrinsics with fixed master intrinsics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--master-intrinsics", type=Path, default=DEFAULT_MASTER_INTRINSICS)
    parser.add_argument("--extrinsic-dir", type=Path, default=DEFAULT_EXTRINSIC_DIR)
    parser.add_argument("--session-dirs", type=Path, nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-cameras", type=int, default=4)
    parser.add_argument("--ref-camera", type=int, default=1)
    parser.add_argument("--min-pairs", type=int, default=5)
    parser.add_argument("--squares-x", type=int, default=4)
    parser.add_argument("--squares-y", type=int, default=3)
    parser.add_argument("--square-length", type=float, default=0.063)
    parser.add_argument("--marker-length", type=float, default=0.047)
    parser.add_argument("--aruco-dict", type=str, default="4X4_50")
    args = parser.parse_args()

    if args.square_length <= args.marker_length:
        parser.error("--square-length must be greater than --marker-length.")
    return args


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fixed_intrinsics, master_image_size = load_master_intrinsics(
        args.master_intrinsics, args.num_cameras
    )
    session_dirs = args.session_dirs or discover_latest_session_dirs(args.extrinsic_dir)

    calibrator_args = argparse.Namespace(
        num_cameras=args.num_cameras,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        aruco_dict=args.aruco_dict,
        session_dirs=[str(p) for p in session_dirs],
        output=str(args.output),
        ref_camera=args.ref_camera,
        min_pairs=args.min_pairs,
    )

    calibrator = MulticamCalibrator(calibrator_args)
    calibrator.intrinsics = fixed_intrinsics

    print("=" * 70)
    print("MULTI-CAMERA DAILY CALIBRATION")
    print("=" * 70)
    print(f"Master intrinsics: {args.master_intrinsics}")
    print(f"Output overwrite:  {args.output}")
    print("Extrinsic sessions:")
    for session_dir in session_dirs:
        print(f"  - {session_dir}")

    print("\nSTAGE 1: fixed intrinsics loaded; no intrinsic calibration is performed.")
    for cam_idx in range(args.num_cameras):
        K, _dist, error = fixed_intrinsics[cam_idx]
        print(
            f"  Camera {cam_idx + 1}: fx={K[0, 0]:.2f}, fy={K[1, 1]:.2f}, "
            f"cx={K[0, 2]:.2f}, cy={K[1, 2]:.2f}, err={error:.4f}px"
        )

    print("\nSTAGE 2: solving pairwise extrinsics with CALIB_FIX_INTRINSIC.")
    capture_sets = calibrator.load_multicam_captures()

    if master_image_size != (0, 0) and calibrator.image_size is not None:
        if tuple(calibrator.image_size) != master_image_size:
            raise ValueError(
                "Extrinsic image size does not match master intrinsics: "
                f"extrinsic={calibrator.image_size}, master={master_image_size}"
            )

    calibrator.calibrate_pairwise_extrinsics(capture_sets)
    if not calibrator.edges:
        raise RuntimeError("No pairwise extrinsic calibration succeeded.")

    ref_cam = args.ref_camera - 1 if args.ref_camera is not None else calibrator.auto_select_ref_camera()
    print(f"\nReference camera: {ref_cam + 1}")

    transforms = calibrator.compose_transforms(ref_cam)
    calibrator.save_calibration(transforms, ref_cam)
    calibrator.save_calibration_yaml(transforms, ref_cam)
    calibrator.save_calibration_summary(transforms, ref_cam)

    print("\nDaily calibration complete.")
    print(f"Final calibration file: {args.output}")


if __name__ == "__main__":
    main()
