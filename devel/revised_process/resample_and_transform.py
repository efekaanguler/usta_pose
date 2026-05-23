#!/usr/bin/env python3
"""
Step 3: Resample, smooth, transform, and export a person-based ML dataset.

1. Reads independent raw camX CSVs.
2. Creates a synchronized 30 FPS overlap timeline.
3. Interpolates missing pose/gaze values and applies Savitzky-Golay smoothing.
4. Converts camera-local coordinates to the Cam1 reference frame.
5. Exports root-relative pose, gaze directions, and provenance metadata.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


NUM_KEYPOINTS = 133
TARGET_FPS = 30.0
TARGET_STEP_MS = 1000.0 / TARGET_FPS
MIN_KEYPOINT_SCORE = 0.3

POSE_CAM_TO_PERSON = {1: 2, 2: 1}
GAZE_CAM_TO_PERSON = {3: 2, 4: 1}
PERSON_TO_POSE_CAM = {1: 2, 2: 1}
PERSON_TO_GAZE_CAM = {1: 4, 2: 3}


def smooth_series(values):
    """Apply Savitzky-Golay smoothing when the series is long and finite."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) >= 11 and np.all(np.isfinite(values)):
        return savgol_filter(values, window_length=11, polyorder=3)
    return values


def read_numeric_csv(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["hw_timestamp_ms"])
    df = df.replace(r"^\s*$", np.nan, regex=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("hw_timestamp_ms").reset_index(drop=True)
    return df


def nearest_valid_on_timeline(source_times, source_valid, target_times):
    source_times = np.asarray(source_times, dtype=np.float64)
    source_valid = np.asarray(source_valid, dtype=bool)
    target_times = np.asarray(target_times, dtype=np.float64)

    valid_times = source_times[source_valid]
    observed = np.zeros(len(target_times), dtype=bool)
    if len(valid_times) == 0:
        return observed

    insert_idx = np.searchsorted(valid_times, target_times)
    candidates = [
        np.clip(insert_idx, 0, len(valid_times) - 1),
        np.clip(insert_idx - 1, 0, len(valid_times) - 1),
    ]
    nearest_delta = np.full(len(target_times), np.inf, dtype=np.float64)
    for candidate in candidates:
        nearest_delta = np.minimum(nearest_delta, np.abs(target_times - valid_times[candidate]))

    return nearest_delta <= 1e-4


def nearest_true_on_timeline(source_times, source_true, target_times, max_delta_ms=TARGET_STEP_MS / 2.0):
    source_times = np.asarray(source_times, dtype=np.float64)
    source_true = np.asarray(source_true, dtype=bool)
    target_times = np.asarray(target_times, dtype=np.float64)

    true_times = source_times[source_true]
    out = np.zeros(len(target_times), dtype=bool)
    if len(true_times) == 0:
        return out

    insert_idx = np.searchsorted(true_times, target_times)
    candidates = [
        np.clip(insert_idx, 0, len(true_times) - 1),
        np.clip(insert_idx - 1, 0, len(true_times) - 1),
    ]
    nearest_delta = np.full(len(target_times), np.inf, dtype=np.float64)
    for candidate in candidates:
        nearest_delta = np.minimum(nearest_delta, np.abs(target_times - true_times[candidate]))

    return nearest_delta <= max_delta_ms


def interpolate_series(source_times, values, target_times, *, smooth=False):
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    resampled = np.full(len(target_times), np.nan, dtype=np.float64)

    if np.sum(valid) == 1:
        resampled[:] = values[valid][0]
    elif np.sum(valid) > 1:
        valid_values = values[valid]
        f = interp1d(
            source_times[valid],
            valid_values,
            kind="linear",
            bounds_error=False,
            fill_value=(valid_values[0], valid_values[-1]),
        )
        resampled = f(target_times)

    if smooth:
        resampled = smooth_series(resampled)

    observed = nearest_valid_on_timeline(source_times, valid, target_times)
    interpolated = np.isfinite(resampled) & ~observed
    return resampled, observed, interpolated


def resample_pose_camera(raw_csv_path, target_times):
    df = read_numeric_csv(raw_csv_path)
    source_times = df["hw_timestamp_ms"].to_numpy(dtype=np.float64)
    out = {}

    for kpt_i in range(NUM_KEYPOINTS):
        coord_cols = [f"kpt{kpt_i}_{axis}" for axis in ("x", "y", "z")]
        score_col = f"kpt{kpt_i}_score"

        score = (
            df[score_col].to_numpy(dtype=np.float64)
            if score_col in df.columns
            else np.full(len(df), np.nan, dtype=np.float64)
        )

        raw_coords = []
        for col in coord_cols:
            values = (
                df[col].to_numpy(dtype=np.float64)
                if col in df.columns
                else np.full(len(df), np.nan, dtype=np.float64)
            )
            raw_coords.append(values)

        joint_valid = (
            np.isfinite(score)
            & (score >= MIN_KEYPOINT_SCORE)
            & np.isfinite(raw_coords[0])
            & np.isfinite(raw_coords[1])
            & np.isfinite(raw_coords[2])
        )

        resampled_coords = []
        for axis, values in zip(("x", "y", "z"), raw_coords):
            masked = values.copy()
            masked[~joint_valid] = np.nan
            resampled, _, _ = interpolate_series(source_times, masked, target_times, smooth=True)
            out[f"kpt{kpt_i}_{axis}"] = resampled
            resampled_coords.append(resampled)

        resampled_score, _, _ = interpolate_series(source_times, score, target_times, smooth=False)
        out[f"kpt{kpt_i}_score"] = resampled_score

        coord_available = (
            np.isfinite(resampled_coords[0])
            & np.isfinite(resampled_coords[1])
            & np.isfinite(resampled_coords[2])
        )
        observed = nearest_valid_on_timeline(source_times, joint_valid, target_times) & coord_available
        out[f"kpt{kpt_i}_observed"] = observed
        out[f"kpt{kpt_i}_interpolated"] = coord_available & ~observed

    return out


def resample_gaze_camera(raw_csv_path, target_times):
    df = read_numeric_csv(raw_csv_path)
    source_times = df["hw_timestamp_ms"].to_numpy(dtype=np.float64)
    out = {}

    vector_cols = ["gaze_x", "gaze_y", "gaze_z"]
    vector_values = []
    for col in vector_cols:
        values = (
            df[col].to_numpy(dtype=np.float64)
            if col in df.columns
            else np.full(len(df), np.nan, dtype=np.float64)
        )
        vector_values.append(values)

    gaze_valid = np.isfinite(vector_values[0]) & np.isfinite(vector_values[1]) & np.isfinite(vector_values[2])

    resampled_vectors = []
    for col, values in zip(vector_cols, vector_values):
        masked = values.copy()
        masked[~gaze_valid] = np.nan
        resampled, _, _ = interpolate_series(source_times, masked, target_times, smooth=True)
        out[col] = resampled
        resampled_vectors.append(resampled)

    for col in ("gaze_yaw", "gaze_pitch"):
        values = (
            df[col].to_numpy(dtype=np.float64)
            if col in df.columns
            else np.full(len(df), np.nan, dtype=np.float64)
        )
        masked = values.copy()
        masked[~gaze_valid] = np.nan
        resampled, _, _ = interpolate_series(source_times, masked, target_times, smooth=True)
        out[col] = resampled

    vector_available = (
        np.isfinite(resampled_vectors[0])
        & np.isfinite(resampled_vectors[1])
        & np.isfinite(resampled_vectors[2])
    )
    observed = nearest_valid_on_timeline(source_times, gaze_valid, target_times) & vector_available
    out["gaze_observed"] = observed
    out["gaze_interpolated"] = vector_available & ~observed

    if "face_detected" in df.columns:
        face_raw = df["face_detected"].fillna(0).to_numpy(dtype=np.float64) > 0
        out["face_detected"] = nearest_true_on_timeline(source_times, face_raw, target_times)
    else:
        out["face_detected"] = np.zeros(len(target_times), dtype=bool)

    return out


def load_camera_to_ref_transform(calib_data, cam_id):
    """
    Return transform for P_ref = R_cam_to_ref @ P_cam + t_cam_to_ref.

    The calibration NPZ keys are named R_{cam}_to_ref/t_{cam}_to_ref, but
    calibrate.sh/multicam_calibrate.py stores ref -> cam transforms there:
    P_cam = R_ref_to_cam @ P_ref + t_ref_to_cam.
    """
    if cam_id == 1:
        return np.eye(3), np.zeros(3)

    r_key = f"R_{cam_id}_to_ref"
    t_key = f"t_{cam_id}_to_ref"
    if r_key not in calib_data or t_key not in calib_data:
        print(f"Warning: Calibration for cam {cam_id} missing. Using identity.")
        return np.eye(3), np.zeros(3)

    r_ref_to_cam = np.asarray(calib_data[r_key], dtype=np.float64)
    t_ref_to_cam = np.asarray(calib_data[t_key], dtype=np.float64).reshape(3)
    r_cam_to_ref = r_ref_to_cam.T
    t_cam_to_ref = -r_cam_to_ref @ t_ref_to_cam
    return r_cam_to_ref, t_cam_to_ref


def transform_points(points_cam, r_cam_to_ref, t_cam_to_ref):
    return points_cam @ r_cam_to_ref.T + t_cam_to_ref


def rotate_vectors(vectors_cam, r_cam_to_ref):
    vectors_ref = vectors_cam @ r_cam_to_ref.T
    norms = np.linalg.norm(vectors_ref, axis=1)
    finite = np.isfinite(norms) & (norms > 0)
    vectors_ref[finite] = vectors_ref[finite] / norms[finite, np.newaxis]
    return vectors_ref


def append_pose_person(final_data, pose_data, calib_data, cam_id, person_id, target_len):
    r_cam_to_ref, t_cam_to_ref = load_camera_to_ref_transform(calib_data, cam_id)
    world = {}

    for kpt_i in range(NUM_KEYPOINTS):
        points_cam = np.column_stack([
            pose_data[f"kpt{kpt_i}_x"],
            pose_data[f"kpt{kpt_i}_y"],
            pose_data[f"kpt{kpt_i}_z"],
        ])
        world[kpt_i] = transform_points(points_cam, r_cam_to_ref, t_cam_to_ref)

    left_hip = world[11]
    right_hip = world[12]
    left_valid = np.all(np.isfinite(left_hip), axis=1)
    right_valid = np.all(np.isfinite(right_hip), axis=1)
    root_valid = left_valid & right_valid

    root = (left_hip + right_hip) / 2.0
    root[~root_valid] = np.nan

    prefix = f"p{person_id}"
    final_data[f"{prefix}_root_x"] = root[:, 0]
    final_data[f"{prefix}_root_y"] = root[:, 1]
    final_data[f"{prefix}_root_z"] = root[:, 2]
    final_data[f"{prefix}_root_valid"] = root_valid
    final_data[f"{prefix}_left_hip_observed"] = pose_data["kpt11_observed"]
    final_data[f"{prefix}_right_hip_observed"] = pose_data["kpt12_observed"]
    final_data[f"{prefix}_left_hip_interpolated"] = pose_data["kpt11_interpolated"]
    final_data[f"{prefix}_right_hip_interpolated"] = pose_data["kpt12_interpolated"]
    final_data[f"{prefix}_pose_cam_id"] = np.full(target_len, cam_id, dtype=np.int16)

    for kpt_i in range(NUM_KEYPOINTS):
        rel = world[kpt_i] - root
        rel[~root_valid] = np.nan
        final_data[f"{prefix}_kpt{kpt_i}_rel_x"] = rel[:, 0]
        final_data[f"{prefix}_kpt{kpt_i}_rel_y"] = rel[:, 1]
        final_data[f"{prefix}_kpt{kpt_i}_rel_z"] = rel[:, 2]
        final_data[f"{prefix}_kpt{kpt_i}_score"] = pose_data[f"kpt{kpt_i}_score"]
        final_data[f"{prefix}_kpt{kpt_i}_observed"] = pose_data[f"kpt{kpt_i}_observed"]
        final_data[f"{prefix}_kpt{kpt_i}_interpolated"] = pose_data[f"kpt{kpt_i}_interpolated"]


def append_missing_pose_person(final_data, person_id, target_len):
    prefix = f"p{person_id}"
    nan = np.full(target_len, np.nan, dtype=np.float64)
    false = np.zeros(target_len, dtype=bool)
    final_data[f"{prefix}_root_x"] = nan.copy()
    final_data[f"{prefix}_root_y"] = nan.copy()
    final_data[f"{prefix}_root_z"] = nan.copy()
    final_data[f"{prefix}_root_valid"] = false.copy()
    final_data[f"{prefix}_left_hip_observed"] = false.copy()
    final_data[f"{prefix}_right_hip_observed"] = false.copy()
    final_data[f"{prefix}_left_hip_interpolated"] = false.copy()
    final_data[f"{prefix}_right_hip_interpolated"] = false.copy()
    final_data[f"{prefix}_pose_cam_id"] = np.full(target_len, PERSON_TO_POSE_CAM[person_id], dtype=np.int16)
    for kpt_i in range(NUM_KEYPOINTS):
        final_data[f"{prefix}_kpt{kpt_i}_rel_x"] = nan.copy()
        final_data[f"{prefix}_kpt{kpt_i}_rel_y"] = nan.copy()
        final_data[f"{prefix}_kpt{kpt_i}_rel_z"] = nan.copy()
        final_data[f"{prefix}_kpt{kpt_i}_score"] = nan.copy()
        final_data[f"{prefix}_kpt{kpt_i}_observed"] = false.copy()
        final_data[f"{prefix}_kpt{kpt_i}_interpolated"] = false.copy()


def append_gaze_person(final_data, gaze_data, calib_data, cam_id, person_id, target_len):
    r_cam_to_ref, _ = load_camera_to_ref_transform(calib_data, cam_id)
    vectors_cam = np.column_stack([gaze_data["gaze_x"], gaze_data["gaze_y"], gaze_data["gaze_z"]])
    vectors_ref = rotate_vectors(vectors_cam, r_cam_to_ref)

    prefix = f"p{person_id}"
    final_data[f"{prefix}_gaze_dir_x"] = vectors_ref[:, 0]
    final_data[f"{prefix}_gaze_dir_y"] = vectors_ref[:, 1]
    final_data[f"{prefix}_gaze_dir_z"] = vectors_ref[:, 2]
    final_data[f"{prefix}_face_detected"] = gaze_data["face_detected"]
    final_data[f"{prefix}_gaze_observed"] = gaze_data["gaze_observed"]
    final_data[f"{prefix}_gaze_interpolated"] = gaze_data["gaze_interpolated"]
    final_data[f"{prefix}_gaze_yaw"] = gaze_data["gaze_yaw"]
    final_data[f"{prefix}_gaze_pitch"] = gaze_data["gaze_pitch"]
    final_data[f"{prefix}_gaze_cam_id"] = np.full(target_len, cam_id, dtype=np.int16)


def append_missing_gaze_person(final_data, person_id, target_len):
    prefix = f"p{person_id}"
    nan = np.full(target_len, np.nan, dtype=np.float64)
    false = np.zeros(target_len, dtype=bool)
    final_data[f"{prefix}_gaze_dir_x"] = nan.copy()
    final_data[f"{prefix}_gaze_dir_y"] = nan.copy()
    final_data[f"{prefix}_gaze_dir_z"] = nan.copy()
    final_data[f"{prefix}_face_detected"] = false.copy()
    final_data[f"{prefix}_gaze_observed"] = false.copy()
    final_data[f"{prefix}_gaze_interpolated"] = false.copy()
    final_data[f"{prefix}_gaze_yaw"] = nan.copy()
    final_data[f"{prefix}_gaze_pitch"] = nan.copy()
    final_data[f"{prefix}_gaze_cam_id"] = np.full(target_len, PERSON_TO_GAZE_CAM[person_id], dtype=np.int16)


def find_raw_files(session_dir):
    raw_files = {"pose": {}, "gaze": {}}

    for cam_id in POSE_CAM_TO_PERSON:
        path = os.path.join(session_dir, f"cam{cam_id}", f"cam{cam_id}_pose_raw.csv")
        if os.path.exists(path):
            raw_files["pose"][cam_id] = path

    for cam_id in GAZE_CAM_TO_PERSON:
        path = os.path.join(session_dir, f"cam{cam_id}", f"cam{cam_id}_gaze_raw.csv")
        if os.path.exists(path):
            raw_files["gaze"][cam_id] = path

    return raw_files


def build_overlap_timeline(raw_files):
    min_time, max_time = -np.inf, np.inf
    paths = list(raw_files["pose"].values()) + list(raw_files["gaze"].values())
    for path in paths:
        ts = pd.read_csv(path, usecols=["hw_timestamp_ms"])["hw_timestamp_ms"].dropna().to_numpy(dtype=np.float64)
        if len(ts) == 0:
            continue
        ts.sort()
        min_time = max(min_time, ts[0])
        max_time = min(max_time, ts[-1])

    if not np.isfinite(min_time) or not np.isfinite(max_time) or min_time >= max_time:
        return None

    return np.arange(min_time, max_time, TARGET_STEP_MS)


def ordered_final_columns(columns):
    ordered = ["timestamp_ms"]
    for person_id in (1, 2):
        prefix = f"p{person_id}"
        ordered.extend([
            f"{prefix}_root_x",
            f"{prefix}_root_y",
            f"{prefix}_root_z",
            f"{prefix}_root_valid",
            f"{prefix}_left_hip_observed",
            f"{prefix}_right_hip_observed",
            f"{prefix}_left_hip_interpolated",
            f"{prefix}_right_hip_interpolated",
            f"{prefix}_pose_cam_id",
            f"{prefix}_gaze_cam_id",
        ])
        for kpt_i in range(NUM_KEYPOINTS):
            ordered.extend([
                f"{prefix}_kpt{kpt_i}_rel_x",
                f"{prefix}_kpt{kpt_i}_rel_y",
                f"{prefix}_kpt{kpt_i}_rel_z",
                f"{prefix}_kpt{kpt_i}_score",
                f"{prefix}_kpt{kpt_i}_observed",
                f"{prefix}_kpt{kpt_i}_interpolated",
            ])
        ordered.extend([
            f"{prefix}_gaze_dir_x",
            f"{prefix}_gaze_dir_y",
            f"{prefix}_gaze_dir_z",
            f"{prefix}_face_detected",
            f"{prefix}_gaze_observed",
            f"{prefix}_gaze_interpolated",
            f"{prefix}_gaze_yaw",
            f"{prefix}_gaze_pitch",
        ])
    return [col for col in ordered if col in columns]


def build_dataset(session_dir, calib_path):
    calib_data = np.load(calib_path) if calib_path and os.path.exists(calib_path) else {}
    raw_files = find_raw_files(session_dir)

    if not raw_files["pose"] and not raw_files["gaze"]:
        raise FileNotFoundError("No raw pose or gaze CSV files found.")

    target_times = build_overlap_timeline(raw_files)
    if target_times is None or len(target_times) == 0:
        raise ValueError("No overlapping timestamps between available cameras.")

    target_len = len(target_times)
    print(
        f"Global timeline: {target_times[0]:.1f}ms to {target_times[-1]:.1f}ms "
        f"({target_len} frames @ {TARGET_FPS:.0f}fps)"
    )

    final_data = {"timestamp_ms": target_times}

    pose_by_cam = {}
    for cam_id, path in sorted(raw_files["pose"].items()):
        print(f"Resampling pose cam{cam_id}...")
        pose_by_cam[cam_id] = resample_pose_camera(path, target_times)

    gaze_by_cam = {}
    for cam_id, path in sorted(raw_files["gaze"].items()):
        print(f"Resampling gaze cam{cam_id}...")
        gaze_by_cam[cam_id] = resample_gaze_camera(path, target_times)

    print("Applying calibration and root-relative transformations...")
    for person_id in (1, 2):
        pose_cam_id = PERSON_TO_POSE_CAM[person_id]
        if pose_cam_id in pose_by_cam:
            append_pose_person(final_data, pose_by_cam[pose_cam_id], calib_data, pose_cam_id, person_id, target_len)
        else:
            append_missing_pose_person(final_data, person_id, target_len)

        gaze_cam_id = PERSON_TO_GAZE_CAM[person_id]
        if gaze_cam_id in gaze_by_cam:
            append_gaze_person(final_data, gaze_by_cam[gaze_cam_id], calib_data, gaze_cam_id, person_id, target_len)
        else:
            append_missing_gaze_person(final_data, person_id, target_len)

    df_ml = pd.DataFrame(final_data)
    return df_ml[ordered_final_columns(df_ml.columns)]


def resolve_calib_path(session_dir, explicit_path):
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates.append(os.path.join(session_dir, "calib_data", "multicam_calibration.npz"))
    candidates.append(os.path.join(session_dir, "multicam_calibration.npz"))
    candidates.append(os.path.join(os.path.dirname(session_dir), "multicam_calibration.npz"))

    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", type=str, required=True)
    parser.add_argument("--calib-npz", type=str, default=None)
    args = parser.parse_args()

    calib_path = resolve_calib_path(args.session_dir, args.calib_npz)
    if calib_path is None:
        print("Warning: Calibration file not found. Identity transforms will be used.")

    df_ml = build_dataset(args.session_dir, calib_path)

    out_path = os.path.join(args.session_dir, "session_ml_dataset.parquet")
    df_ml.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"\nSaved ML-ready tabular dataset to: {out_path}")
    print(f"Shape: {df_ml.shape}")


if __name__ == "__main__":
    main()
