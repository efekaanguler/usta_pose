#!/usr/bin/env python3
"""Combine fixed master intrinsics with the latest daily extrinsic capture."""

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
DEFAULT_IDEAL_CUBE_LAYOUT = RECORD_DIR / "apriltag_cube_layout.json"
DEFAULT_CALIBRATED_CUBE_LAYOUT = (
    RECORD_DIR / "apriltag_cube_layout_calibrated.json"
)
DEFAULT_CUBE_LAYOUT = (
    DEFAULT_CALIBRATED_CUBE_LAYOUT
    if DEFAULT_CALIBRATED_CUBE_LAYOUT.exists()
    else DEFAULT_IDEAL_CUBE_LAYOUT
)


def load_master_intrinsics(path, num_cameras):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing master intrinsics: {path}\n"
            "Run record_intrinsic.py once before the daily calibration flow."
        )

    intrinsics = {}
    with np.load(path) as data:
        image_size = (
            np.asarray(data["image_size"], dtype=int).reshape(-1)
            if "image_size" in data
            else np.array([0, 0], dtype=int)
        )
        if image_size.size != 2:
            raise ValueError(
                f"{path} contains invalid image_size shape: {image_size.shape}."
            )
        if np.any(image_size < 0) or (
            np.any(image_size == 0) and not np.all(image_size == 0)
        ):
            raise ValueError(
                f"{path} contains invalid image_size values: {image_size.tolist()}."
            )
        for cam_num in range(1, num_cameras + 1):
            k_key = f"K{cam_num}"
            d_key = f"dist{cam_num}"
            if k_key not in data or d_key not in data:
                raise KeyError(f"{path} does not contain {k_key}/{d_key}.")

            camera_matrix = np.asarray(data[k_key], dtype=np.float64)
            distortion = np.asarray(data[d_key], dtype=np.float64)
            if camera_matrix.shape != (3, 3):
                raise ValueError(
                    f"{path}:{k_key} must be 3x3, got {camera_matrix.shape}."
                )
            if not np.all(np.isfinite(camera_matrix)):
                raise ValueError(f"{path}:{k_key} contains non-finite values.")
            if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
                raise ValueError(f"{path}:{k_key} contains non-positive focal lengths.")
            if abs(float(np.linalg.det(camera_matrix))) < 1e-12:
                raise ValueError(f"{path}:{k_key} is singular.")
            if distortion.size < 4 or not np.all(np.isfinite(distortion)):
                raise ValueError(
                    f"{path}:{d_key} must contain at least four finite coefficients."
                )

            error_key = f"intrinsic_error{cam_num}"
            error = float(data[error_key]) if error_key in data else 0.0
            if not np.isfinite(error) or error < 0:
                raise ValueError(f"{path}:{error_key} is invalid: {error}.")
            intrinsics[cam_num - 1] = (
                camera_matrix,
                distortion,
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
    parser.add_argument("--method", choices=("cube", "charuco"), default="cube")
    parser.add_argument("--master-intrinsics", type=Path, default=DEFAULT_MASTER_INTRINSICS)
    parser.add_argument("--extrinsic-dir", type=Path, default=DEFAULT_EXTRINSIC_DIR)
    parser.add_argument("--captures-dir", type=Path, default=None)
    parser.add_argument("--session-dirs", type=Path, nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-cameras", type=int, default=4)
    parser.add_argument("--ref-camera", type=int, default=1)
    parser.add_argument("--cube-layout", type=Path, default=DEFAULT_CUBE_LAYOUT)
    parser.add_argument("--apriltag-family", type=str, default="tag36h11")
    parser.add_argument(
        "--cube-solver",
        choices=("pairwise", "joint-ba"),
        default="pairwise",
        help=(
            "Fast robust pairwise pose graph, or the optional full joint "
            "camera/cube bundle adjustment."
        ),
    )
    parser.add_argument("--min-decision-margin", type=float, default=20.0)
    parser.add_argument("--max-hamming", type=int, default=0)
    parser.add_argument("--min-tag-edge-px", type=float, default=45.0)
    parser.add_argument("--max-initial-reproj-px", type=float, default=3.0)
    parser.add_argument("--min-views-per-camera", type=int, default=10)
    parser.add_argument(
        "--robust-loss",
        choices=("linear", "soft_l1", "huber", "cauchy", "arctan"),
        default="huber",
    )
    parser.add_argument("--robust-scale-px", type=float, default=1.0)
    parser.add_argument("--max-final-p95-px", type=float, default=2.5)
    parser.add_argument("--min-pairs", type=int, default=5)
    parser.add_argument(
        "--charuco-solver",
        choices=("global-ba", "pairwise"),
        default="global-ba",
        help=(
            "Robust global camera/board bundle adjustment is the safe default. "
            "Pairwise preserves the legacy shortest-path solver for comparison only."
        ),
    )
    parser.add_argument(
        "--charuco-robust-loss",
        choices=("linear", "soft_l1", "huber", "cauchy", "arctan"),
        default="soft_l1",
    )
    parser.add_argument("--charuco-robust-scale-px", type=float, default=1.0)
    parser.add_argument("--charuco-max-nfev", type=int, default=300)
    parser.add_argument("--max-global-reproj-p95-px", type=float, default=2.5)
    parser.add_argument(
        "--max-master-focal-deviation",
        type=float,
        default=0.03,
        help=(
            "Maximum allowed focal-length deviation between master intrinsics "
            "and the active RealSense color-stream profile."
        ),
    )
    parser.add_argument("--squares-x", type=int, default=4)
    parser.add_argument("--squares-y", type=int, default=3)
    parser.add_argument("--square-length", type=float, default=0.063)
    parser.add_argument("--marker-length", type=float, default=0.047)
    parser.add_argument("--aruco-dict", type=str, default="4X4_50")
    args = parser.parse_args()

    if args.method == "charuco" and args.square_length <= args.marker_length:
        parser.error("--square-length must be greater than --marker-length.")
    if not 1 <= args.ref_camera <= args.num_cameras:
        parser.error("--ref-camera must be within --num-cameras.")
    if args.method == "cube" and not args.cube_layout.exists():
        parser.error(f"Cube layout does not exist: {args.cube_layout}")
    return args


def print_fixed_intrinsics(master_path, fixed_intrinsics):
    print(f"Master intrinsics: {master_path}")
    for cam_idx in range(len(fixed_intrinsics)):
        camera_matrix, _distortion, error = fixed_intrinsics[cam_idx]
        print(
            f"  Camera {cam_idx + 1}: "
            f"fx={camera_matrix[0, 0]:.2f}, fy={camera_matrix[1, 1]:.2f}, "
            f"cx={camera_matrix[0, 2]:.2f}, cy={camera_matrix[1, 2]:.2f}, "
            f"intrinsic err={error:.4f}px"
        )


def run_cube_calibration(args, fixed_intrinsics, master_image_size):
    from cube_calibration import CubeMulticamCalibrator, save_cube_calibration

    cube_intrinsics = {
        camera_index + 1: values
        for camera_index, values in fixed_intrinsics.items()
    }
    captures_dir = args.captures_dir or (args.extrinsic_dir / "current")
    calibrator = CubeMulticamCalibrator(
        cube_intrinsics,
        args.cube_layout,
        reference_camera=args.ref_camera,
        families=args.apriltag_family,
        min_decision_margin=args.min_decision_margin,
        max_hamming=args.max_hamming,
        min_tag_edge_px=args.min_tag_edge_px,
        max_initial_reprojection_error_px=args.max_initial_reproj_px,
        min_views_per_camera=args.min_views_per_camera,
        robust_loss=args.robust_loss,
        robust_scale_px=args.robust_scale_px,
        max_final_p95_px=args.max_final_p95_px,
    )

    print("=" * 70)
    print("APRILTAG CUBE MULTI-CAMERA DAILY CALIBRATION")
    print("=" * 70)
    print_fixed_intrinsics(args.master_intrinsics, fixed_intrinsics)
    print(f"Cube layout:       {args.cube_layout}")
    print(f"Capture directory: {captures_dir}")
    print(f"Cube solver:       {args.cube_solver}")
    print(f"Output overwrite:  {args.output}")
    print("\nSTAGE 1: detecting known cube corners and initializing PnP poses.")
    observations = calibrator.load_capture_directory(captures_dir)

    if master_image_size != (0, 0) and calibrator.image_size != master_image_size:
        raise ValueError(
            "Extrinsic image size does not match master intrinsics: "
            f"extrinsic={calibrator.image_size}, master={master_image_size}"
        )

    per_camera = {
        camera_id: sum(item.camera_id == camera_id for item in observations)
        for camera_id in cube_intrinsics
    }
    print(f"  Accepted observations: {len(observations)} | per camera: {per_camera}")
    if args.cube_solver == "pairwise":
        print("\nSTAGE 2: robust pairwise transforms and camera pose graph.")
        result = calibrator.calibrate_pairwise_pose_graph()
    else:
        print("\nSTAGE 2: full joint camera/cube bundle adjustment.")
        result = calibrator.calibrate_joint_bundle_adjustment()
    save_cube_calibration(args.output, cube_intrinsics, result)

    print(f"  Optimizer: {result.optimizer_message}")
    print(f"  Rejected corners: {result.rejected_corners}")
    for camera_id in sorted(result.camera_quality):
        quality = result.camera_quality[camera_id]
        print(
            f"  Camera {camera_id}: views={quality.observations}, "
            f"corners={quality.corners}, "
            f"median={quality.median_reprojection_error_px:.3f}px, "
            f"P95={quality.p95_reprojection_error_px:.3f}px, "
            f"max={quality.max_reprojection_error_px:.3f}px"
        )


def run_charuco_calibration(args, fixed_intrinsics, master_image_size):
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
        charuco_robust_loss=args.charuco_robust_loss,
        charuco_robust_scale_px=args.charuco_robust_scale_px,
        charuco_max_nfev=args.charuco_max_nfev,
        max_global_reproj_p95_px=args.max_global_reproj_p95_px,
        max_master_focal_deviation=args.max_master_focal_deviation,
    )

    calibrator = MulticamCalibrator(calibrator_args)
    calibrator.intrinsics = fixed_intrinsics

    print("=" * 70)
    print("LEGACY CHARUCO MULTI-CAMERA DAILY CALIBRATION")
    print("=" * 70)
    print_fixed_intrinsics(args.master_intrinsics, fixed_intrinsics)
    print(f"Output overwrite:  {args.output}")
    print("Extrinsic sessions:")
    for session_dir in session_dirs:
        print(f"  - {session_dir}")

    print("\nInitializing pairwise extrinsics with CALIB_FIX_INTRINSIC.")
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

    initial_transforms = calibrator.compose_transforms(ref_cam)
    if args.charuco_solver == "global-ba":
        print(
            "\nRefining all cameras and all board poses in one robust global "
            "bundle adjustment."
        )
        transforms = calibrator.refine_global_bundle_adjustment(
            capture_sets, initial_transforms, ref_cam
        )
    else:
        print(
            "\nWARNING: Using legacy pairwise shortest-path transforms. "
            "This mode can report low stereo RMS while producing poor global 3D fusion."
        )
        transforms = initial_transforms
    calibrator.save_calibration(transforms, ref_cam)
    calibrator.save_calibration_yaml(transforms, ref_cam)
    calibrator.save_calibration_summary(transforms, ref_cam)



def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fixed_intrinsics, master_image_size = load_master_intrinsics(
        args.master_intrinsics, args.num_cameras
    )

    if args.method == "cube":
        run_cube_calibration(args, fixed_intrinsics, master_image_size)
    else:
        run_charuco_calibration(args, fixed_intrinsics, master_image_size)

    print("\nDaily calibration complete.")
    print(f"Final calibration file: {args.output}")


if __name__ == "__main__":
    main()
