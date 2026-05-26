#!/usr/bin/env python3
"""
Create an RViz-aligned dyadic interaction parquet from session_ml_dataset.parquet.

The source parquet stores person roots in the calibrated reference frame and
keypoints relative to each root. This postprocess reconstructs absolute
keypoints, applies the same display rotation used by revised_visualize_rviz.py,
and exports a richer tabular dataset for interaction analysis/model training.
"""

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from revised_visualize_rviz import (
    NUM_KEYPOINTS,
    RVIZ_DISPLAY_ROTATION,
    resolve_calib_path,
    resolve_parquet_path,
    validate_revised_schema,
)


SCHEMA_VERSION = "interaction_v1"
OUTPUT_BASENAME = "session_interaction_dataset.parquet"
BODY_KEYPOINTS = tuple(range(17))
HAND_KEYPOINTS = tuple(range(91, 133))
LEFT_WRIST = 9
RIGHT_WRIST = 10
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2


def finite_rows(values):
    values = np.asarray(values, dtype=np.float64)
    return np.all(np.isfinite(values), axis=1)


def normalize_rows(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    out = np.full_like(vectors, np.nan, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(norms) & (norms > 0.0) & finite_rows(vectors)
    out[valid] = vectors[valid] / norms[valid, np.newaxis]
    return out


def rotate_points(points):
    points = np.asarray(points, dtype=np.float64)
    return points @ RVIZ_DISPLAY_ROTATION.T


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_column(df, column, default=np.nan):
    if column in df.columns:
        return df[column].to_numpy()
    return np.full(len(df), default)


def bool_column(df, column):
    values = safe_column(df, column, False)
    out = np.zeros(len(df), dtype=bool)
    for idx, value in enumerate(values):
        if value is None:
            continue
        if isinstance(value, str):
            out[idx] = value.strip().lower() in {"1", "true", "t", "yes", "y"}
            continue
        try:
            if np.isnan(value):
                continue
        except TypeError:
            pass
        out[idx] = bool(value)
    return out


def numeric_column(df, column):
    values = safe_column(df, column, np.nan)
    return np.asarray(values, dtype=np.float64)


def source_session_dir(parquet_path, session_dir=None):
    if session_dir:
        return os.path.abspath(os.path.expanduser(session_dir))
    return os.path.dirname(os.path.abspath(os.path.expanduser(parquet_path)))


def default_output_path(parquet_path, session_dir=None):
    return os.path.join(source_session_dir(parquet_path, session_dir), OUTPUT_BASENAME)


def flatten_matrix(prefix, matrix, out):
    matrix = np.asarray(matrix, dtype=np.float64)
    for row in range(3):
        for col in range(3):
            out[f"{prefix}_{row}{col}"] = float(matrix[row, col])


def load_calibration_metadata(calib_path):
    metadata = {}
    if not calib_path:
        return metadata

    calib = np.load(calib_path)
    metadata["calib_path"] = os.path.abspath(os.path.expanduser(calib_path))
    metadata["calib_sha256"] = sha256_file(calib_path)
    metadata["calib_ref_camera"] = int(np.asarray(calib["ref_camera"]).item()) if "ref_camera" in calib else np.nan
    num_cameras = int(np.asarray(calib["num_cameras"]).item()) if "num_cameras" in calib else 4
    metadata["calib_num_cameras"] = num_cameras

    if "image_size" in calib:
        image_size = np.asarray(calib["image_size"]).reshape(-1)
        metadata["calib_image_width"] = int(image_size[0]) if len(image_size) > 0 else np.nan
        metadata["calib_image_height"] = int(image_size[1]) if len(image_size) > 1 else np.nan
    else:
        metadata["calib_image_width"] = np.nan
        metadata["calib_image_height"] = np.nan

    for cam_id in range(1, num_cameras + 1):
        k_key = f"K{cam_id}"
        if k_key in calib:
            k_mat = np.asarray(calib[k_key], dtype=np.float64)
            metadata[f"calib_cam{cam_id}_K_fx"] = float(k_mat[0, 0])
            metadata[f"calib_cam{cam_id}_K_fy"] = float(k_mat[1, 1])
            metadata[f"calib_cam{cam_id}_K_cx"] = float(k_mat[0, 2])
            metadata[f"calib_cam{cam_id}_K_cy"] = float(k_mat[1, 2])

        dist_key = f"dist{cam_id}"
        dist = np.asarray(calib[dist_key], dtype=np.float64).reshape(-1) if dist_key in calib else np.array([])
        for idx in range(5):
            metadata[f"calib_cam{cam_id}_dist_{idx}"] = float(dist[idx]) if idx < len(dist) else np.nan

        r_key = f"R_{cam_id}_to_ref"
        t_key = f"t_{cam_id}_to_ref"
        if r_key not in calib or t_key not in calib:
            continue

        # The NPZ key names are historical. Existing processing/visualization
        # treats these stored matrices as ref -> cam transforms.
        r_ref_to_cam = np.asarray(calib[r_key], dtype=np.float64)
        t_ref_to_cam = np.asarray(calib[t_key], dtype=np.float64).reshape(3)
        r_cam_to_ref = r_ref_to_cam.T
        t_cam_to_ref = -r_cam_to_ref @ t_ref_to_cam

        flatten_matrix(f"calib_cam{cam_id}_R_ref_to_cam", r_ref_to_cam, metadata)
        metadata[f"calib_cam{cam_id}_t_ref_to_cam_x"] = float(t_ref_to_cam[0])
        metadata[f"calib_cam{cam_id}_t_ref_to_cam_y"] = float(t_ref_to_cam[1])
        metadata[f"calib_cam{cam_id}_t_ref_to_cam_z"] = float(t_ref_to_cam[2])
        flatten_matrix(f"calib_cam{cam_id}_R_cam_to_ref", r_cam_to_ref, metadata)
        metadata[f"calib_cam{cam_id}_t_cam_to_ref_x"] = float(t_cam_to_ref[0])
        metadata[f"calib_cam{cam_id}_t_cam_to_ref_y"] = float(t_cam_to_ref[1])
        metadata[f"calib_cam{cam_id}_t_cam_to_ref_z"] = float(t_cam_to_ref[2])

    flatten_matrix("rviz_rotation", RVIZ_DISPLAY_ROTATION, metadata)
    return metadata


def person_root(df, person_id):
    prefix = f"p{person_id}"
    root = np.column_stack([
        numeric_column(df, f"{prefix}_root_x"),
        numeric_column(df, f"{prefix}_root_y"),
        numeric_column(df, f"{prefix}_root_z"),
    ])
    valid = bool_column(df, f"{prefix}_root_valid") & finite_rows(root)
    root[~valid] = np.nan
    return rotate_points(root), valid


def person_keypoints(df, person_id, root_rviz):
    n_frames = len(df)
    keypoints = np.full((n_frames, NUM_KEYPOINTS, 3), np.nan, dtype=np.float64)
    person_rel = np.full_like(keypoints, np.nan)
    scores = np.full((n_frames, NUM_KEYPOINTS), np.nan, dtype=np.float64)
    observed = np.zeros((n_frames, NUM_KEYPOINTS), dtype=bool)
    interpolated = np.zeros((n_frames, NUM_KEYPOINTS), dtype=bool)

    for keypoint_idx in range(NUM_KEYPOINTS):
        prefix = f"p{person_id}_kpt{keypoint_idx}"
        rel_source = np.column_stack([
            numeric_column(df, f"{prefix}_rel_x"),
            numeric_column(df, f"{prefix}_rel_y"),
            numeric_column(df, f"{prefix}_rel_z"),
        ])
        rel_rviz = rotate_points(rel_source)
        world_rviz = root_rviz + rel_rviz
        rel_valid = finite_rows(rel_rviz) & finite_rows(root_rviz)
        rel_rviz[~rel_valid] = np.nan
        world_rviz[~rel_valid] = np.nan
        keypoints[:, keypoint_idx, :] = world_rviz
        person_rel[:, keypoint_idx, :] = rel_rviz
        scores[:, keypoint_idx] = numeric_column(df, f"{prefix}_score")
        observed[:, keypoint_idx] = bool_column(df, f"{prefix}_observed")
        interpolated[:, keypoint_idx] = bool_column(df, f"{prefix}_interpolated")

    return keypoints, person_rel, scores, observed, interpolated


def gaze_vectors(df, person_id):
    prefix = f"p{person_id}"
    gaze_source = np.column_stack([
        numeric_column(df, f"{prefix}_gaze_dir_x"),
        numeric_column(df, f"{prefix}_gaze_dir_y"),
        numeric_column(df, f"{prefix}_gaze_dir_z"),
    ])
    return normalize_rows(rotate_points(gaze_source))


def head_points(keypoints, roots):
    eyes_valid = finite_rows(keypoints[:, LEFT_EYE, :]) & finite_rows(keypoints[:, RIGHT_EYE, :])
    nose_valid = finite_rows(keypoints[:, NOSE, :])
    out = roots.copy()
    out[nose_valid] = keypoints[nose_valid, NOSE, :]
    out[eyes_valid] = (keypoints[eyes_valid, LEFT_EYE, :] + keypoints[eyes_valid, RIGHT_EYE, :]) / 2.0
    out[~finite_rows(out)] = np.nan
    return out


def vector_cos_and_angle(source, target, direction):
    target_unit = normalize_rows(target - source)
    direction_unit = normalize_rows(direction)
    valid = finite_rows(target_unit) & finite_rows(direction_unit)
    cos = np.full(len(source), np.nan, dtype=np.float64)
    angle = np.full(len(source), np.nan, dtype=np.float64)
    cos[valid] = np.sum(target_unit[valid] * direction_unit[valid], axis=1)
    cos[valid] = np.clip(cos[valid], -1.0, 1.0)
    angle[valid] = np.degrees(np.arccos(cos[valid]))
    return cos, angle


def distance_series(a, b):
    valid = finite_rows(a) & finite_rows(b)
    out = np.full(len(a), np.nan, dtype=np.float64)
    out[valid] = np.linalg.norm(a[valid] - b[valid], axis=1)
    return out


def speed_series(points, timestamps_ms):
    points = np.asarray(points, dtype=np.float64)
    out = np.full(len(points), np.nan, dtype=np.float64)
    if len(points) < 2:
        return out
    dt = np.diff(np.asarray(timestamps_ms, dtype=np.float64)) / 1000.0
    valid = finite_rows(points[1:]) & finite_rows(points[:-1]) & np.isfinite(dt) & (dt > 0.0)
    out[1:][valid] = np.linalg.norm(points[1:][valid] - points[:-1][valid], axis=1) / dt[valid]
    return out


def motion_energy_series(keypoints, timestamps_ms, keypoint_indices):
    out = np.full(len(keypoints), np.nan, dtype=np.float64)
    if len(keypoints) < 2:
        return out
    dt = np.diff(np.asarray(timestamps_ms, dtype=np.float64)) / 1000.0
    for frame_idx in range(1, len(keypoints)):
        if not np.isfinite(dt[frame_idx - 1]) or dt[frame_idx - 1] <= 0.0:
            continue
        prev_points = keypoints[frame_idx - 1, keypoint_indices, :]
        curr_points = keypoints[frame_idx, keypoint_indices, :]
        valid = finite_rows(prev_points) & finite_rows(curr_points)
        if np.any(valid):
            speeds = np.linalg.norm(curr_points[valid] - prev_points[valid], axis=1) / dt[frame_idx - 1]
            out[frame_idx] = float(np.nanmean(speeds))
    return out


def add_xyz_columns(out, prefix, values):
    out[f"{prefix}_x"] = values[:, 0]
    out[f"{prefix}_y"] = values[:, 1]
    out[f"{prefix}_z"] = values[:, 2]


def add_keypoint_columns(out, person_id, keypoints, person_rel, dyad_ref, scores, observed, interpolated):
    dyad_rel = keypoints - dyad_ref[:, np.newaxis, :]
    dyad_rel[~finite_rows(dyad_ref), :, :] = np.nan
    for keypoint_idx in range(NUM_KEYPOINTS):
        prefix = f"p{person_id}_kpt{keypoint_idx}"
        add_xyz_columns(out, f"{prefix}_world", keypoints[:, keypoint_idx, :])
        add_xyz_columns(out, f"{prefix}_person_rel", person_rel[:, keypoint_idx, :])
        add_xyz_columns(out, f"{prefix}_dyad_rel", dyad_rel[:, keypoint_idx, :])
        out[f"{prefix}_score"] = scores[:, keypoint_idx]
        out[f"{prefix}_observed"] = observed[:, keypoint_idx]
        out[f"{prefix}_interpolated"] = interpolated[:, keypoint_idx]


def ratio(counts, denominator):
    return counts.astype(np.float64) / float(denominator)


def build_interaction_dataset(
    input_parquet,
    calib_path,
    *,
    session_dir=None,
    pose_model="rtmpose2d",
    gaze_model="puregaze",
):
    if pd is None:
        raise RuntimeError("pandas is required to create interaction parquet files.")

    input_parquet = os.path.abspath(os.path.expanduser(input_parquet))
    session_dir = source_session_dir(input_parquet, session_dir)
    df = pd.read_parquet(input_parquet)
    validate_revised_schema(df)

    timestamps = numeric_column(df, "timestamp_ms")
    n_frames = len(df)
    out = {
        "schema_version": np.full(n_frames, SCHEMA_VERSION, dtype=object),
        "session_id": np.full(n_frames, os.path.basename(session_dir), dtype=object),
        "timestamp_ms": timestamps,
        "frame_idx": np.arange(n_frames, dtype=np.int64),
        "source_parquet_path": np.full(n_frames, input_parquet, dtype=object),
        "coordinate_frame": np.full(n_frames, "rviz_world", dtype=object),
        "pose_model": np.full(n_frames, pose_model, dtype=object),
        "gaze_model": np.full(n_frames, gaze_model, dtype=object),
    }

    for key, value in load_calibration_metadata(calib_path).items():
        out[key] = np.full(n_frames, value, dtype=object if isinstance(value, str) else None)

    p1_ref, p1_ref_valid = person_root(df, 1)
    p2_ref, p2_ref_valid = person_root(df, 2)
    dyad_valid = p1_ref_valid & p2_ref_valid
    dyad_ref = np.full_like(p1_ref, np.nan)
    dyad_ref[dyad_valid] = (p1_ref[dyad_valid] + p2_ref[dyad_valid]) / 2.0

    add_xyz_columns(out, "dyad_ref", dyad_ref)
    out["dyad_ref_valid"] = dyad_valid
    root_delta = p2_ref - p1_ref
    root_distance = distance_series(p1_ref, p2_ref)
    out["dyad_root_distance"] = root_distance
    out["dyad_root_horizontal_distance"] = np.sqrt(root_delta[:, 0] ** 2 + root_delta[:, 1] ** 2)
    out["dyad_root_horizontal_distance"][~dyad_valid] = np.nan
    out["dyad_vertical_offset_p1_minus_p2"] = p1_ref[:, 2] - p2_ref[:, 2]
    out["dyad_vertical_offset_p1_minus_p2"][~dyad_valid] = np.nan
    add_xyz_columns(out, "dyad_p1_to_p2_unit", normalize_rows(root_delta))
    add_xyz_columns(out, "dyad_p2_to_p1_unit", normalize_rows(-root_delta))

    people = {}
    for person_id, ref, ref_valid in ((1, p1_ref, p1_ref_valid), (2, p2_ref, p2_ref_valid)):
        prefix = f"p{person_id}"
        keypoints, person_rel, scores, observed, interpolated = person_keypoints(df, person_id, ref)
        gaze = gaze_vectors(df, person_id)
        people[person_id] = {
            "ref": ref,
            "ref_valid": ref_valid,
            "keypoints": keypoints,
            "person_rel": person_rel,
            "scores": scores,
            "observed": observed,
            "interpolated": interpolated,
            "gaze": gaze,
            "head": head_points(keypoints, ref),
        }

        add_xyz_columns(out, f"{prefix}_ref", ref)
        out[f"{prefix}_ref_valid"] = ref_valid
        out[f"{prefix}_ref_source"] = safe_column(df, f"{prefix}_root_source", 0)
        out[f"{prefix}_ref_observed"] = bool_column(df, f"{prefix}_root_observed")
        out[f"{prefix}_ref_interpolated"] = bool_column(df, f"{prefix}_root_interpolated")
        add_xyz_columns(out, f"{prefix}_ref_to_dyad", ref - dyad_ref)
        out[f"{prefix}_pose_cam_id"] = safe_column(df, f"{prefix}_pose_cam_id", np.nan)
        out[f"{prefix}_gaze_cam_id"] = safe_column(df, f"{prefix}_gaze_cam_id", np.nan)

        add_keypoint_columns(out, person_id, keypoints, person_rel, dyad_ref, scores, observed, interpolated)
        add_xyz_columns(out, f"{prefix}_gaze_dir", gaze)
        out[f"{prefix}_gaze_observed"] = bool_column(df, f"{prefix}_gaze_observed")
        out[f"{prefix}_gaze_interpolated"] = bool_column(df, f"{prefix}_gaze_interpolated")
        out[f"{prefix}_face_detected"] = bool_column(df, f"{prefix}_face_detected")
        out[f"{prefix}_gaze_yaw"] = numeric_column(df, f"{prefix}_gaze_yaw")
        out[f"{prefix}_gaze_pitch"] = numeric_column(df, f"{prefix}_gaze_pitch")

        body_valid = finite_rows(keypoints[:, BODY_KEYPOINTS, :].reshape(n_frames * len(BODY_KEYPOINTS), 3))
        body_valid = body_valid.reshape(n_frames, len(BODY_KEYPOINTS))
        hand_valid = finite_rows(keypoints[:, HAND_KEYPOINTS, :].reshape(n_frames * len(HAND_KEYPOINTS), 3))
        hand_valid = hand_valid.reshape(n_frames, len(HAND_KEYPOINTS))
        out[f"{prefix}_body_valid_keypoint_count"] = body_valid.sum(axis=1).astype(np.int16)
        out[f"{prefix}_body_valid_keypoint_ratio"] = ratio(out[f"{prefix}_body_valid_keypoint_count"], len(BODY_KEYPOINTS))
        out[f"{prefix}_hand_valid_keypoint_count"] = hand_valid.sum(axis=1).astype(np.int16)
        out[f"{prefix}_hand_valid_keypoint_ratio"] = ratio(out[f"{prefix}_hand_valid_keypoint_count"], len(HAND_KEYPOINTS))
        out[f"{prefix}_any_pose_interpolated"] = interpolated.any(axis=1)
        out[f"{prefix}_any_gaze_interpolated"] = out[f"{prefix}_gaze_interpolated"]
        out[f"{prefix}_motion_speed"] = speed_series(ref, timestamps)
        out[f"{prefix}_motion_energy_body"] = motion_energy_series(keypoints, timestamps, BODY_KEYPOINTS)

    p1 = people[1]
    p2 = people[2]
    p1_to_p2_head_cos, p1_to_p2_head_angle = vector_cos_and_angle(p1["head"], p2["head"], p1["gaze"])
    p2_to_p1_head_cos, p2_to_p1_head_angle = vector_cos_and_angle(p2["head"], p1["head"], p2["gaze"])
    p1_to_dyad_cos, p1_to_dyad_angle = vector_cos_and_angle(p1["head"], dyad_ref, p1["gaze"])
    p2_to_dyad_cos, p2_to_dyad_angle = vector_cos_and_angle(p2["head"], dyad_ref, p2["gaze"])

    out["p1_gaze_to_other_head_cos"] = p1_to_p2_head_cos
    out["p1_gaze_to_other_head_angle_deg"] = p1_to_p2_head_angle
    out["p1_gaze_to_dyad_ref_cos"] = p1_to_dyad_cos
    out["p1_gaze_to_dyad_ref_angle_deg"] = p1_to_dyad_angle
    out["p2_gaze_to_other_head_cos"] = p2_to_p1_head_cos
    out["p2_gaze_to_other_head_angle_deg"] = p2_to_p1_head_angle
    out["p2_gaze_to_dyad_ref_cos"] = p2_to_dyad_cos
    out["p2_gaze_to_dyad_ref_angle_deg"] = p2_to_dyad_angle
    out["mutual_gaze_cos_min"] = np.fmin(p1_to_p2_head_cos, p2_to_p1_head_cos)

    out["p1_head_to_p2_head_distance"] = distance_series(p1["head"], p2["head"])
    out["p1_left_wrist_to_p2_right_wrist_distance"] = distance_series(
        p1["keypoints"][:, LEFT_WRIST, :],
        p2["keypoints"][:, RIGHT_WRIST, :],
    )
    out["p1_right_wrist_to_p2_left_wrist_distance"] = distance_series(
        p1["keypoints"][:, RIGHT_WRIST, :],
        p2["keypoints"][:, LEFT_WRIST, :],
    )
    out["p1_left_wrist_to_p2_head_distance"] = distance_series(p1["keypoints"][:, LEFT_WRIST, :], p2["head"])
    out["p1_right_wrist_to_p2_head_distance"] = distance_series(p1["keypoints"][:, RIGHT_WRIST, :], p2["head"])
    out["p2_left_wrist_to_p1_head_distance"] = distance_series(p2["keypoints"][:, LEFT_WRIST, :], p1["head"])
    out["p2_right_wrist_to_p1_head_distance"] = distance_series(p2["keypoints"][:, RIGHT_WRIST, :], p1["head"])

    out["motion_energy_ratio_p1_over_p2"] = (
        out["p1_motion_energy_body"] / out["p2_motion_energy_body"]
    )
    invalid_ratio = ~np.isfinite(out["motion_energy_ratio_p1_over_p2"]) | (out["p2_motion_energy_body"] <= 0.0)
    out["motion_energy_ratio_p1_over_p2"][invalid_ratio] = np.nan

    out["frame_pose_valid"] = (
        out["p1_ref_valid"]
        & out["p2_ref_valid"]
        & (out["p1_body_valid_keypoint_count"] >= 5)
        & (out["p2_body_valid_keypoint_count"] >= 5)
    )
    out["frame_gaze_valid"] = (
        finite_rows(p1["gaze"])
        & finite_rows(p2["gaze"])
        & out["p1_gaze_observed"]
        & out["p2_gaze_observed"]
    )
    out["frame_interaction_valid"] = out["frame_pose_valid"] & dyad_valid

    return pd.DataFrame(out)


def write_interaction_dataset(input_parquet, output_parquet, calib_path, **kwargs):
    df = build_interaction_dataset(input_parquet, calib_path, **kwargs)
    output_parquet = os.path.abspath(os.path.expanduser(output_parquet))
    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, engine="pyarrow", index=False)
    return output_parquet, df


def main():
    parser = argparse.ArgumentParser(
        description="Create an RViz-aligned dyadic interaction parquet from session_ml_dataset.parquet."
    )
    parser.add_argument("--session-dir", type=str, default=None)
    parser.add_argument("--input-parquet", "--parquet", dest="input_parquet", type=str, default=None)
    parser.add_argument("--output-parquet", type=str, default=None)
    parser.add_argument("--calib", type=str, default=None)
    parser.add_argument("--pose-model", type=str, default=os.environ.get("POSE_MODEL", "rtmpose2d"))
    parser.add_argument("--gaze-model", type=str, default=os.environ.get("GAZE_MODEL", "puregaze"))
    args = parser.parse_args()

    input_parquet = resolve_parquet_path(args.input_parquet, args.session_dir)
    if not os.path.exists(input_parquet):
        parser.error(f"Input parquet not found: {input_parquet}")

    calib_path = resolve_calib_path(args.calib, args.session_dir, input_parquet)
    if calib_path is None:
        parser.error("Calibration NPZ not found. Provide --calib or keep multicam_calibration.npz near the session.")

    output_parquet = args.output_parquet or default_output_path(input_parquet, args.session_dir)
    output_path, df = write_interaction_dataset(
        input_parquet,
        output_parquet,
        calib_path,
        session_dir=args.session_dir,
        pose_model=args.pose_model,
        gaze_model=args.gaze_model,
    )
    print(f"Saved interaction parquet: {output_path}")
    print(f"Shape: {df.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
