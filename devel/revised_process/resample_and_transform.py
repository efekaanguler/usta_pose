#!/usr/bin/env python3
"""
Step 3: Resample, smooth, transform, and export a person-based ML dataset.

1. Reads independent raw camX CSVs.
2. Creates a synchronized overlap timeline at an estimated session FPS.
3. Interpolates only short pose/gaze gaps and smooths finite segments.
4. Converts camera-local coordinates to the Cam1 reference frame.
5. Exports root-relative pose, gaze directions, and provenance metadata.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


NUM_KEYPOINTS = 133
TARGET_FPS = 30.0  # compatibility constant; runtime defaults to adaptive FPS.
TARGET_STEP_MS = 1000.0 / TARGET_FPS
MIN_TARGET_FPS = 15.0
MAX_TARGET_FPS = 30.0
DEFAULT_MAX_INTERP_GAP_MS = 150.0
MIN_KEYPOINT_SCORE = 0.3

POSE_CAM_TO_PERSON = {1: 2, 2: 1}
GAZE_CAM_TO_PERSON = {3: 2, 4: 1}
PERSON_TO_POSE_CAM = {1: 2, 2: 1}
PERSON_TO_GAZE_CAM = {1: 4, 2: 3}

ROOT_SOURCE_MISSING = 0
ROOT_SOURCE_HIPS = 1
ROOT_SOURCE_TORSO = 2
ROOT_SOURCE_SHOULDERS = 3
ROOT_SOURCE_NAMES = {
    ROOT_SOURCE_MISSING: "missing",
    ROOT_SOURCE_HIPS: "hips",
    ROOT_SOURCE_TORSO: "torso",
    ROOT_SOURCE_SHOULDERS: "shoulders",
}

MIN_HIP_WIDTH_M = 0.12
MAX_HIP_WIDTH_M = 0.65
MIN_SHOULDER_WIDTH_M = 0.18
MAX_SHOULDER_WIDTH_M = 0.80
MIN_TORSO_LENGTH_M = 0.18
MAX_TORSO_LENGTH_M = 1.05
MAX_ROOT_SPEED_M_S = 4.0


# ---------------------------------------------------------------------------
# CSV/timeline helpers
# ---------------------------------------------------------------------------

def finite_runs(mask):
    runs = []
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        if start is not None and (not value or idx == len(mask) - 1):
            end = idx if value and idx == len(mask) - 1 else idx - 1
            runs.append((start, end + 1))
            start = None
    return runs


def smooth_series(values):
    """Apply Savitzky-Golay smoothing only inside finite contiguous segments."""
    values = np.asarray(values, dtype=np.float64)
    out = values.copy()
    finite = np.isfinite(values)
    for start, end in finite_runs(finite):
        run_len = end - start
        if run_len < 5:
            continue
        window = min(11, run_len if run_len % 2 == 1 else run_len - 1)
        if window <= 3:
            continue
        out[start:end] = savgol_filter(values[start:end], window_length=window, polyorder=3)
    return out


def read_numeric_csv(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["hw_timestamp_ms"])
    df = df.replace(r"^\s*$", np.nan, regex=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("hw_timestamp_ms").reset_index(drop=True)
    return df


def read_unique_timestamps(path):
    ts = pd.read_csv(path, usecols=["hw_timestamp_ms"])["hw_timestamp_ms"]
    ts = pd.to_numeric(ts, errors="coerce").dropna().to_numpy(dtype=np.float64)
    if len(ts) == 0:
        return ts
    return np.unique(np.sort(ts))


def estimate_target_fps(raw_files, explicit_fps=None):
    if explicit_fps is not None:
        if explicit_fps <= 0.0:
            raise ValueError("target_fps must be positive.")
        return float(explicit_fps)

    paths = list(raw_files["pose"].values()) + list(raw_files["gaze"].values())
    rates = []
    for path in paths:
        ts = read_unique_timestamps(path)
        if len(ts) < 2:
            continue
        duration_s = (ts[-1] - ts[0]) / 1000.0
        if duration_s <= 0.0:
            continue
        rates.append((len(ts) - 1) / duration_s)

    if not rates:
        return TARGET_FPS

    estimated = float(np.median(rates))
    estimated = float(np.clip(np.ceil(estimated), MIN_TARGET_FPS, MAX_TARGET_FPS))
    return estimated


def nearest_valid_on_timeline(source_times, source_valid, target_times, max_delta_ms):
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

    return nearest_delta <= max_delta_ms


def nearest_true_on_timeline(source_times, source_true, target_times, max_delta_ms):
    return nearest_valid_on_timeline(source_times, source_true, target_times, max_delta_ms)


def interpolate_series(
    source_times,
    values,
    target_times,
    *,
    smooth=False,
    max_gap_ms=DEFAULT_MAX_INTERP_GAP_MS,
    observed_tolerance_ms=None,
):
    """Interpolate only between bracketing observations no farther apart than max_gap_ms."""
    source_times = np.asarray(source_times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    valid = np.isfinite(source_times) & np.isfinite(values)
    resampled = np.full(len(target_times), np.nan, dtype=np.float64)

    if observed_tolerance_ms is None:
        if len(target_times) > 1:
            observed_tolerance_ms = float(np.nanmedian(np.diff(target_times))) / 2.0
        else:
            observed_tolerance_ms = 1e-4

    valid_times = source_times[valid]
    valid_values = values[valid]
    if len(valid_times) > 0:
        order = np.argsort(valid_times)
        valid_times = valid_times[order]
        valid_values = valid_values[order]

    if len(valid_times) == 1:
        near = np.abs(target_times - valid_times[0]) <= observed_tolerance_ms
        resampled[near] = valid_values[0]
    elif len(valid_times) > 1:
        prev_idx = np.searchsorted(valid_times, target_times, side="right") - 1
        next_idx = np.searchsorted(valid_times, target_times, side="left")
        in_bounds = (prev_idx >= 0) & (next_idx < len(valid_times))
        bracket_gap = np.full(len(target_times), np.inf, dtype=np.float64)
        bracket_gap[in_bounds] = valid_times[next_idx[in_bounds]] - valid_times[prev_idx[in_bounds]]
        fill = in_bounds & (bracket_gap <= max_gap_ms)
        if np.any(fill):
            resampled[fill] = np.interp(target_times[fill], valid_times, valid_values)

    if smooth:
        resampled = smooth_series(resampled)

    observed = nearest_valid_on_timeline(source_times, valid, target_times, observed_tolerance_ms)
    interpolated = np.isfinite(resampled) & ~observed
    return resampled, observed, interpolated


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample_pose_camera(raw_csv_path, target_times, target_step_ms, max_interp_gap_ms):
    df = read_numeric_csv(raw_csv_path)
    source_times = df["hw_timestamp_ms"].to_numpy(dtype=np.float64)
    observed_tolerance_ms = target_step_ms / 2.0
    out = {}

    if "pose_rejected" in df.columns:
        pose_rejected = df["pose_rejected"].fillna(0).to_numpy(dtype=np.float64) > 0
    else:
        pose_rejected = np.zeros(len(df), dtype=bool)

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
            ~pose_rejected
            & np.isfinite(score)
            & (score >= MIN_KEYPOINT_SCORE)
            & np.isfinite(raw_coords[0])
            & np.isfinite(raw_coords[1])
            & np.isfinite(raw_coords[2])
        )

        resampled_coords = []
        for axis, values in zip(("x", "y", "z"), raw_coords):
            masked = values.copy()
            masked[~joint_valid] = np.nan
            resampled, _, _ = interpolate_series(
                source_times,
                masked,
                target_times,
                smooth=True,
                max_gap_ms=max_interp_gap_ms,
                observed_tolerance_ms=observed_tolerance_ms,
            )
            out[f"kpt{kpt_i}_{axis}"] = resampled
            resampled_coords.append(resampled)

        masked_score = score.copy()
        masked_score[~joint_valid] = np.nan
        resampled_score, _, _ = interpolate_series(
            source_times,
            masked_score,
            target_times,
            smooth=False,
            max_gap_ms=max_interp_gap_ms,
            observed_tolerance_ms=observed_tolerance_ms,
        )
        out[f"kpt{kpt_i}_score"] = resampled_score

        coord_available = (
            np.isfinite(resampled_coords[0])
            & np.isfinite(resampled_coords[1])
            & np.isfinite(resampled_coords[2])
        )
        observed = nearest_valid_on_timeline(source_times, joint_valid, target_times, observed_tolerance_ms) & coord_available
        out[f"kpt{kpt_i}_observed"] = observed
        out[f"kpt{kpt_i}_interpolated"] = coord_available & ~observed

    return out


def resample_gaze_camera(raw_csv_path, target_times, target_step_ms, max_interp_gap_ms):
    df = read_numeric_csv(raw_csv_path)
    source_times = df["hw_timestamp_ms"].to_numpy(dtype=np.float64)
    observed_tolerance_ms = target_step_ms / 2.0
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
        resampled, _, _ = interpolate_series(
            source_times,
            masked,
            target_times,
            smooth=True,
            max_gap_ms=max_interp_gap_ms,
            observed_tolerance_ms=observed_tolerance_ms,
        )
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
        resampled, _, _ = interpolate_series(
            source_times,
            masked,
            target_times,
            smooth=True,
            max_gap_ms=max_interp_gap_ms,
            observed_tolerance_ms=observed_tolerance_ms,
        )
        out[col] = resampled

    vector_available = (
        np.isfinite(resampled_vectors[0])
        & np.isfinite(resampled_vectors[1])
        & np.isfinite(resampled_vectors[2])
    )
    observed = nearest_valid_on_timeline(source_times, gaze_valid, target_times, observed_tolerance_ms) & vector_available
    out["gaze_observed"] = observed
    out["gaze_interpolated"] = vector_available & ~observed

    if "face_detected" in df.columns:
        face_raw = df["face_detected"].fillna(0).to_numpy(dtype=np.float64) > 0
        out["face_detected"] = nearest_true_on_timeline(source_times, face_raw, target_times, observed_tolerance_ms)
    else:
        out["face_detected"] = np.zeros(len(target_times), dtype=bool)

    return out


# ---------------------------------------------------------------------------
# Calibration and root handling
# ---------------------------------------------------------------------------

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


def finite_point(point):
    return np.asarray(point).shape == (3,) and np.all(np.isfinite(point))


def distance(a, b):
    if not finite_point(a) or not finite_point(b):
        return np.nan
    return float(np.linalg.norm(a - b))


def plausible_range(value, lower, upper):
    return np.isfinite(value) and lower <= value <= upper


def candidate_flags(pose_data, joints, frame_idx):
    observed = all(bool(pose_data[f"kpt{joint}_observed"][frame_idx]) for joint in joints)
    interpolated = any(bool(pose_data[f"kpt{joint}_interpolated"][frame_idx]) for joint in joints)
    return observed, interpolated


def torso_length_ok(shoulder_mid, lower_mid):
    value = distance(shoulder_mid, lower_mid)
    return plausible_range(value, MIN_TORSO_LENGTH_M, MAX_TORSO_LENGTH_M)


def compute_robust_root(world, pose_data, target_times):
    target_len = len(target_times)
    root = np.full((target_len, 3), np.nan, dtype=np.float64)
    root_valid = np.zeros(target_len, dtype=bool)
    root_source = np.full(target_len, ROOT_SOURCE_MISSING, dtype=np.int8)
    root_observed = np.zeros(target_len, dtype=bool)
    root_interpolated = np.zeros(target_len, dtype=bool)

    for idx in range(target_len):
        left_shoulder = world[5][idx]
        right_shoulder = world[6][idx]
        left_hip = world[11][idx]
        right_hip = world[12][idx]

        left_shoulder_valid = finite_point(left_shoulder)
        right_shoulder_valid = finite_point(right_shoulder)
        left_hip_valid = finite_point(left_hip)
        right_hip_valid = finite_point(right_hip)

        shoulder_mid = None
        shoulder_ok = False
        if left_shoulder_valid and right_shoulder_valid:
            shoulder_width = distance(left_shoulder, right_shoulder)
            # RTMW3D + depth can produce compressed metric skeletons. Treat
            # upper bounds as outlier guards, not minimum-size requirements.
            shoulder_ok = np.isfinite(shoulder_width) and shoulder_width <= MAX_SHOULDER_WIDTH_M
            if shoulder_ok:
                shoulder_mid = (left_shoulder + right_shoulder) / 2.0

        hip_mid = None
        hip_ok = False
        if left_hip_valid and right_hip_valid:
            hip_width = distance(left_hip, right_hip)
            hip_ok = np.isfinite(hip_width) and hip_width <= MAX_HIP_WIDTH_M
            if hip_ok:
                hip_mid = (left_hip + right_hip) / 2.0

        torso_points = []
        torso_joints = []
        if shoulder_ok:
            torso_points.extend([left_shoulder, right_shoulder])
            torso_joints.extend([5, 6])
        if left_hip_valid:
            torso_points.append(left_hip)
            torso_joints.append(11)
        if right_hip_valid:
            torso_points.append(right_hip)
            torso_joints.append(12)

        if shoulder_ok and len(torso_points) >= 3:
            lower_points = [point for point, joint in zip(torso_points, torso_joints) if joint in (11, 12)]
            lower_mid = np.mean(lower_points, axis=0)
            torso_len = distance(shoulder_mid, lower_mid)
            if np.isfinite(torso_len) and torso_len <= MAX_TORSO_LENGTH_M:
                root[idx] = np.mean(torso_points, axis=0)
                root_valid[idx] = True
                root_source[idx] = ROOT_SOURCE_TORSO
                root_observed[idx], root_interpolated[idx] = candidate_flags(pose_data, torso_joints, idx)
                continue

        if hip_ok:
            root[idx] = hip_mid
            root_valid[idx] = True
            root_source[idx] = ROOT_SOURCE_HIPS
            root_observed[idx], root_interpolated[idx] = candidate_flags(pose_data, [11, 12], idx)
            continue

        if shoulder_ok:
            root[idx] = shoulder_mid
            root_valid[idx] = True
            root_source[idx] = ROOT_SOURCE_SHOULDERS
            root_observed[idx], root_interpolated[idx] = candidate_flags(pose_data, [5, 6], idx)

    last_idx = None
    last_root = None
    for idx in range(target_len):
        if not root_valid[idx]:
            continue
        if last_idx is None:
            last_idx = idx
            last_root = root[idx].copy()
            continue
        dt_s = (target_times[idx] - target_times[last_idx]) / 1000.0
        if dt_s <= 0.0:
            root_valid[idx] = False
        else:
            speed = np.linalg.norm(root[idx] - last_root) / dt_s
            if np.isfinite(speed) and speed <= MAX_ROOT_SPEED_M_S:
                last_idx = idx
                last_root = root[idx].copy()
            else:
                root_valid[idx] = False
        if not root_valid[idx]:
            root[idx] = np.nan
            root_source[idx] = ROOT_SOURCE_MISSING
            root_observed[idx] = False
            root_interpolated[idx] = False

    return root, root_valid, root_source, root_observed, root_interpolated


# ---------------------------------------------------------------------------
# Export assembly
# ---------------------------------------------------------------------------

def append_pose_person(final_data, pose_data, calib_data, cam_id, person_id, target_times):
    target_len = len(target_times)
    r_cam_to_ref, t_cam_to_ref = load_camera_to_ref_transform(calib_data, cam_id)
    world = {}

    for kpt_i in range(NUM_KEYPOINTS):
        points_cam = np.column_stack([
            pose_data[f"kpt{kpt_i}_x"],
            pose_data[f"kpt{kpt_i}_y"],
            pose_data[f"kpt{kpt_i}_z"],
        ])
        world[kpt_i] = transform_points(points_cam, r_cam_to_ref, t_cam_to_ref)

    root, root_valid, root_source, root_observed, root_interpolated = compute_robust_root(
        world,
        pose_data,
        target_times,
    )

    prefix = f"p{person_id}"
    final_data[f"{prefix}_root_x"] = root[:, 0]
    final_data[f"{prefix}_root_y"] = root[:, 1]
    final_data[f"{prefix}_root_z"] = root[:, 2]
    final_data[f"{prefix}_root_valid"] = root_valid
    final_data[f"{prefix}_root_source"] = root_source
    final_data[f"{prefix}_root_observed"] = root_observed
    final_data[f"{prefix}_root_interpolated"] = root_interpolated
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
    final_data[f"{prefix}_root_source"] = np.full(target_len, ROOT_SOURCE_MISSING, dtype=np.int8)
    final_data[f"{prefix}_root_observed"] = false.copy()
    final_data[f"{prefix}_root_interpolated"] = false.copy()
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


def build_overlap_timeline(raw_files, target_fps=None):
    fps = estimate_target_fps(raw_files, target_fps)
    step_ms = 1000.0 / fps
    min_time, max_time = -np.inf, np.inf
    paths = list(raw_files["pose"].values()) + list(raw_files["gaze"].values())
    for path in paths:
        ts = read_unique_timestamps(path)
        if len(ts) == 0:
            continue
        min_time = max(min_time, ts[0])
        max_time = min(max_time, ts[-1])

    if not np.isfinite(min_time) or not np.isfinite(max_time) or min_time >= max_time:
        return None, fps, step_ms

    return np.arange(min_time, max_time, step_ms), fps, step_ms


def ordered_final_columns(columns):
    ordered = ["timestamp_ms"]
    for person_id in (1, 2):
        prefix = f"p{person_id}"
        ordered.extend([
            f"{prefix}_root_x",
            f"{prefix}_root_y",
            f"{prefix}_root_z",
            f"{prefix}_root_valid",
            f"{prefix}_root_source",
            f"{prefix}_root_observed",
            f"{prefix}_root_interpolated",
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


def build_dataset(session_dir, calib_path, target_fps=None, max_interp_gap_ms=DEFAULT_MAX_INTERP_GAP_MS):
    calib_data = np.load(calib_path) if calib_path and os.path.exists(calib_path) else {}
    raw_files = find_raw_files(session_dir)

    if not raw_files["pose"] and not raw_files["gaze"]:
        raise FileNotFoundError("No raw pose or gaze CSV files found.")

    target_times, resolved_fps, target_step_ms = build_overlap_timeline(raw_files, target_fps)
    if target_times is None or len(target_times) == 0:
        raise ValueError("No overlapping timestamps between available cameras.")

    target_len = len(target_times)
    print(
        f"Global timeline: {target_times[0]:.1f}ms to {target_times[-1]:.1f}ms "
        f"({target_len} frames @ {resolved_fps:.0f}fps, max interp gap {max_interp_gap_ms:.0f}ms)"
    )

    final_data = {"timestamp_ms": target_times}

    pose_by_cam = {}
    for cam_id, path in sorted(raw_files["pose"].items()):
        print(f"Resampling pose cam{cam_id}...")
        pose_by_cam[cam_id] = resample_pose_camera(path, target_times, target_step_ms, max_interp_gap_ms)

    gaze_by_cam = {}
    for cam_id, path in sorted(raw_files["gaze"].items()):
        print(f"Resampling gaze cam{cam_id}...")
        gaze_by_cam[cam_id] = resample_gaze_camera(path, target_times, target_step_ms, max_interp_gap_ms)

    print("Applying calibration and robust root-relative transformations...")
    for person_id in (1, 2):
        pose_cam_id = PERSON_TO_POSE_CAM[person_id]
        if pose_cam_id in pose_by_cam:
            append_pose_person(final_data, pose_by_cam[pose_cam_id], calib_data, pose_cam_id, person_id, target_times)
        else:
            append_missing_pose_person(final_data, person_id, target_len)

        gaze_cam_id = PERSON_TO_GAZE_CAM[person_id]
        if gaze_cam_id in gaze_by_cam:
            append_gaze_person(final_data, gaze_by_cam[gaze_cam_id], calib_data, gaze_cam_id, person_id, target_len)
        else:
            append_missing_gaze_person(final_data, person_id, target_len)

    df_ml = pd.DataFrame(final_data)
    return df_ml[ordered_final_columns(df_ml.columns)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def normalized_dir(path):
    return os.path.normpath(os.path.abspath(os.path.expanduser(path)))


def normalized_path(path):
    return os.path.normpath(os.path.abspath(os.path.expanduser(path)))


def resolve_calib_path(session_dir, explicit_path):
    session_dir = normalized_dir(session_dir)
    if explicit_path:
        explicit_path = normalized_path(explicit_path)
        return explicit_path if os.path.exists(explicit_path) else None

    candidates = []
    candidates.append(os.path.join(session_dir, "calib_data", "multicam_calibration.npz"))
    candidates.append(os.path.join(session_dir, "multicam_calibration.npz"))
    candidates.append(os.path.join(os.path.dirname(session_dir), "multicam_calibration.npz"))
    candidates.append(os.path.join(os.path.dirname(session_dir), "calib_data", "multicam_calibration.npz"))

    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-dir", type=str, required=True)
    parser.add_argument("--calib-npz", type=str, default=None)
    parser.add_argument("--target-fps", type=float, default=None, help="Override adaptive output FPS.")
    parser.add_argument(
        "--max-interp-gap-ms",
        type=float,
        default=DEFAULT_MAX_INTERP_GAP_MS,
        help="Interpolate only between observations separated by at most this gap.",
    )
    parser.add_argument(
        "--allow-missing-calib",
        action="store_true",
        help="Allow identity camera transforms when no calibration NPZ is available.",
    )
    args = parser.parse_args()

    session_dir = normalized_dir(args.session_dir)
    calib_path = resolve_calib_path(args.session_dir, args.calib_npz)
    if calib_path is None and not args.allow_missing_calib:
        parser.error(
            "Calibration file not found. Pass --calib-npz, place multicam_calibration.npz "
            "next to the session directory, or use --allow-missing-calib for debugging."
        )
    if calib_path is None and args.allow_missing_calib:
        print("Warning: Calibration file not found. Identity transforms will be used.")

    df_ml = build_dataset(
        session_dir,
        calib_path,
        target_fps=args.target_fps,
        max_interp_gap_ms=args.max_interp_gap_ms,
    )

    out_path = os.path.join(session_dir, "session_ml_dataset.parquet")
    df_ml.to_parquet(out_path, engine="pyarrow", index=False)
    print(f"\nSaved ML-ready tabular dataset to: {out_path}")
    print(f"Shape: {df_ml.shape}")
    print("Root source codes:", ROOT_SOURCE_NAMES)


if __name__ == "__main__":
    main()
