#!/usr/bin/env python3
"""Parquet-only HRI analysis for dyadic pose/gaze interaction sessions.

This script implements the goal in usta_pose/testing/goal.md as a practical,
reproducible analysis pipeline. It reads interaction parquet files, infers dyads
from chronological four-order blocks, extracts person-swap-invariant features,
runs paired dyad-level statistics, discovers a first nonverbal vocabulary, trains
simple group-validated models when the input has enough dyads, and writes a
markdown report.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover
    pq = None
    _PYARROW_IMPORT_ERROR = exc
else:
    _PYARROW_IMPORT_ERROR = None

from scipy import stats
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FILENAME_RE = re.compile(r"^(?P<date>\d{8})_(?P<time>\d{6})_order(?P<order>[1-4])\.parquet$")
RANDOM_SEED = 42
EPS = 1e-9

BODY_KEYPOINTS = tuple(range(17))
HAND_KEYPOINTS = tuple(range(91, 133))
LEFT_WRIST = 9
RIGHT_WRIST = 10
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2

QUALITY_COLUMNS = [
    "frame_pose_valid",
    "frame_gaze_valid",
    "frame_interaction_valid",
    "p1_body_valid_keypoint_ratio",
    "p2_body_valid_keypoint_ratio",
    "p1_hand_valid_keypoint_ratio",
    "p2_hand_valid_keypoint_ratio",
    "p1_body_valid_keypoint_count",
    "p2_body_valid_keypoint_count",
    "p1_hand_valid_keypoint_count",
    "p2_hand_valid_keypoint_count",
    "p1_any_pose_interpolated",
    "p2_any_pose_interpolated",
    "p1_any_gaze_interpolated",
    "p2_any_gaze_interpolated",
]

BASE_COLUMNS = [
    "schema_version",
    "session_id",
    "timestamp_ms",
    "frame_idx",
    "dyad_root_distance",
    "dyad_root_horizontal_distance",
    "dyad_vertical_offset_p1_minus_p2",
    "p1_motion_speed",
    "p2_motion_speed",
    "p1_motion_energy_body",
    "p2_motion_energy_body",
    "motion_energy_ratio_p1_over_p2",
    "p1_gaze_to_other_head_angle_deg",
    "p2_gaze_to_other_head_angle_deg",
    "p1_gaze_to_other_head_cos",
    "p2_gaze_to_other_head_cos",
    "p1_gaze_to_dyad_ref_angle_deg",
    "p2_gaze_to_dyad_ref_angle_deg",
    "p1_gaze_to_dyad_ref_cos",
    "p2_gaze_to_dyad_ref_cos",
    "mutual_gaze_cos_min",
    "p1_head_to_p2_head_distance",
    "p1_left_wrist_to_p2_right_wrist_distance",
    "p1_right_wrist_to_p2_left_wrist_distance",
    "p1_left_wrist_to_p2_head_distance",
    "p1_right_wrist_to_p2_head_distance",
    "p2_left_wrist_to_p1_head_distance",
    "p2_right_wrist_to_p1_head_distance",
]

FALLBACK_KEYPOINT_COLUMNS = []
for person in ("p1", "p2"):
    FALLBACK_KEYPOINT_COLUMNS.extend([f"{person}_ref_{axis}" for axis in ("x", "y", "z")])
    for kpt in (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_WRIST, RIGHT_WRIST):
        FALLBACK_KEYPOINT_COLUMNS.extend([f"{person}_kpt{kpt}_world_{axis}" for axis in ("x", "y", "z")])

PRIMARY_ANALYSIS_FEATURES = [
    "root_distance_mean",
    "root_distance_delta",
    "root_horizontal_distance_mean",
    "vertical_offset_abs_mean",
    "pair_motion_energy_mean",
    "pair_motion_energy_max",
    "motion_energy_absdiff_mean",
    "motion_balance_mean",
    "both_active_ratio",
    "one_active_ratio",
    "neither_active_ratio",
    "activity_state_entropy",
    "motion_corr_zero_lag",
    "motion_crosscorr_absmax",
    "motion_crosscorr_abs_lag_s",
    "closest_hand_to_other_head_mean",
    "closest_hand_to_other_head_min",
    "hand_to_hand_min_mean",
    "hand_reach_delta_mean",
    "partner_gaze_ratio_mean",
    "partner_gaze_ratio_any",
    "partner_gaze_ratio_both",
    "partner_gaze_asym_abs",
    "task_gaze_ratio_mean",
    "mutual_gaze_ratio",
    "mutual_gaze_cos_mean",
    "gaze_switch_rate_mean",
    "valid_pose_ratio",
    "valid_gaze_ratio",
    "valid_interaction_ratio",
]

VOCAB_FEATURES = [
    "pair_motion_energy_mean",
    "motion_energy_absdiff_mean",
    "both_active_ratio",
    "one_active_ratio",
    "activity_state_entropy",
    "root_distance_mean",
    "root_distance_delta",
    "motion_crosscorr_absmax",
    "motion_crosscorr_abs_lag_s",
    "closest_hand_to_other_head_mean",
    "hand_to_hand_min_mean",
    "partner_gaze_ratio_mean",
    "partner_gaze_ratio_both",
    "task_gaze_ratio_mean",
    "mutual_gaze_ratio",
]

STAT_FEATURES = [
    "root_distance_mean_mean",
    "pair_motion_energy_mean_mean",
    "pair_motion_energy_max_mean",
    "motion_energy_absdiff_mean_mean",
    "motion_balance_mean_mean",
    "both_active_ratio_mean",
    "one_active_ratio_mean",
    "activity_state_entropy_mean",
    "motion_crosscorr_absmax_mean",
    "motion_crosscorr_abs_lag_s_mean",
    "closest_hand_to_other_head_mean_mean",
    "closest_hand_to_other_head_min_mean",
    "hand_to_hand_min_mean_mean",
    "partner_gaze_ratio_mean_mean",
    "partner_gaze_ratio_both_mean",
    "task_gaze_ratio_mean_mean",
    "mutual_gaze_ratio_mean",
    "valid_pose_ratio_mean",
    "valid_gaze_ratio_mean",
    "valid_interaction_ratio_mean",
]


@dataclass
class RunConfig:
    input_dir: Path
    output_dir: Path
    window_s: float = 1.0
    stride_s: float = 0.5
    min_window_valid_ratio: float = 0.5
    max_lag_s: float = 2.0
    random_seed: int = RANDOM_SEED
    vocabulary_k: int = 8
    max_model_windows: int = 6000


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return np.nan
    return float(np.nanmean(arr))


def finite_std(values: Iterable[float]) -> float:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    if arr.size <= 1 or not np.isfinite(arr).any():
        return np.nan
    return float(np.nanstd(arr))


def finite_median(values: Iterable[float]) -> float:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return np.nan
    return float(np.nanmedian(arr))


def finite_ratio(mask: Iterable[bool]) -> float:
    arr = np.asarray(mask)
    if arr.size == 0:
        return np.nan
    return float(np.nanmean(arr.astype(float)))


def bool_series(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df:
        return pd.Series(np.full(len(df), default, dtype=bool), index=df.index)
    s = df[column]
    if s.dtype == bool:
        return s.fillna(default)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float).astype(bool)
    return s.astype(str).str.lower().isin({"1", "true", "t", "yes", "y"})


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(np.full(len(df), np.nan), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def parse_session_file(path: Path) -> dict:
    match = FILENAME_RE.match(path.name)
    if not match:
        return {
            "file_name": path.name,
            "session_label": path.stem,
            "recording_date": None,
            "recording_time": None,
            "order": np.nan,
            "parse_ok": False,
        }
    order = int(match.group("order"))
    return {
        "file_name": path.name,
        "session_label": path.stem,
        "recording_date": match.group("date"),
        "recording_time": match.group("time"),
        "recording_datetime_key": f"{match.group('date')}_{match.group('time')}",
        "order": order,
        "is_competitive_practiced": bool(order > 1),
        "parse_ok": True,
    }


def list_parquets(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*.parquet"))


def infer_dyads(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.sort_values(["recording_date", "recording_time", "file_name"], na_position="last").reset_index(drop=True)
    dyad_ids = []
    block_indices = []
    block_valid = []
    block_notes = []
    for start in range(0, len(meta), 4):
        block = meta.iloc[start : start + 4]
        dyad_num = start // 4 + 1
        orders = sorted(pd.to_numeric(block["order"], errors="coerce").dropna().astype(int).tolist())
        valid = len(block) == 4 and orders == [1, 2, 3, 4]
        note = "complete chronological four-order block" if valid else f"incomplete/ambiguous block; orders={orders}"
        for _ in range(len(block)):
            dyad_ids.append(f"dyad_{dyad_num:03d}")
            block_indices.append(dyad_num)
            block_valid.append(valid)
            block_notes.append(note)
    meta["dyad_id"] = dyad_ids
    meta["dyad_block_index"] = block_indices
    meta["dyad_block_valid"] = block_valid
    meta["dyad_block_note"] = block_notes
    return meta


def parquet_schema_columns(path: Path) -> list[str]:
    if pq is None:
        raise RuntimeError(f"pyarrow is required to inspect parquet files: {_PYARROW_IMPORT_ERROR}")
    return list(pq.ParquetFile(path).schema_arrow.names)


def parquet_row_count(path: Path) -> int:
    if pq is None:
        raise RuntimeError(f"pyarrow is required to inspect parquet files: {_PYARROW_IMPORT_ERROR}")
    return int(pq.ParquetFile(path).metadata.num_rows)


def available_columns(path: Path, requested: Iterable[str]) -> list[str]:
    cols = set(parquet_schema_columns(path))
    return [c for c in requested if c in cols]


def read_relevant_parquet(path: Path) -> pd.DataFrame:
    requested = list(dict.fromkeys(BASE_COLUMNS + QUALITY_COLUMNS + FALLBACK_KEYPOINT_COLUMNS))
    cols = available_columns(path, requested)
    if "timestamp_ms" not in cols:
        # Fall back to full read only when timestamp is missing from the inspected schema.
        return pd.read_parquet(path)
    return pd.read_parquet(path, columns=cols)


def point_array(df: pd.DataFrame, prefix: str) -> np.ndarray:
    return np.column_stack([numeric_series(df, f"{prefix}_{axis}").to_numpy(float) for axis in ("x", "y", "z")])


def distance_from_points(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full(len(a), np.nan, dtype=float)
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    out[valid] = np.linalg.norm(a[valid] - b[valid], axis=1)
    return out


def speed_from_points(points: np.ndarray, timestamps_ms: np.ndarray) -> np.ndarray:
    out = np.full(len(points), np.nan, dtype=float)
    if len(points) < 2:
        return out
    dt = np.diff(timestamps_ms) / 1000.0
    valid = (
        np.isfinite(points[1:]).all(axis=1)
        & np.isfinite(points[:-1]).all(axis=1)
        & np.isfinite(dt)
        & (dt > 0)
    )
    out[1:][valid] = np.linalg.norm(points[1:][valid] - points[:-1][valid], axis=1) / dt[valid]
    return out


def robust_activity_threshold(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return 0.03
    q60 = np.nanpercentile(arr, 60)
    med = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - med))
    return float(max(0.03, min(q60, med + 1.5 * mad if np.isfinite(mad) else q60)))


def compute_crosscorr(a: np.ndarray, b: np.ndarray, timestamps_ms: np.ndarray, max_lag_s: float) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 4:
        return np.nan, np.nan, np.nan
    a = a[valid]
    b = b[valid]
    t = np.asarray(timestamps_ms, dtype=float)[valid]
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)
    if np.nanstd(a) <= EPS or np.nanstd(b) <= EPS:
        return np.nan, np.nan, np.nan
    dt = np.nanmedian(np.diff(t)) / 1000.0 if len(t) > 2 else np.nan
    if not np.isfinite(dt) or dt <= 0:
        dt = 1.0 / 30.0
    max_lag = int(max(1, round(max_lag_s / dt)))
    zero_corr = float(np.corrcoef(a, b)[0, 1])
    best_corr = np.nan
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x, y = a[:lag], b[-lag:]
        elif lag > 0:
            x, y = a[lag:], b[:-lag]
        else:
            x, y = a, b
        if len(x) < 4 or np.nanstd(x) <= EPS or np.nanstd(y) <= EPS:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if not np.isfinite(best_corr) or abs(c) > abs(best_corr):
            best_corr = c
            best_lag = lag
    return zero_corr, best_corr, abs(best_lag * dt)


def shannon_entropy_from_counts(counts: np.ndarray) -> float:
    total = np.sum(counts)
    if total <= 0:
        return np.nan
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


def prepare_frame_features(df: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["timestamp_ms"] = numeric_series(df, "timestamp_ms")
    if out["timestamp_ms"].isna().all():
        out["timestamp_ms"] = np.arange(len(df), dtype=float) * (1000.0 / 30.0)
    out["frame_idx"] = numeric_series(df, "frame_idx") if "frame_idx" in df else np.arange(len(df))

    out["frame_pose_valid"] = bool_series(df, "frame_pose_valid", True)
    out["frame_gaze_valid"] = bool_series(df, "frame_gaze_valid", False)
    out["frame_interaction_valid"] = bool_series(df, "frame_interaction_valid", True)

    out["root_distance"] = numeric_series(df, "dyad_root_distance")
    out["root_horizontal_distance"] = numeric_series(df, "dyad_root_horizontal_distance")
    out["vertical_offset_abs"] = numeric_series(df, "dyad_vertical_offset_p1_minus_p2").abs()

    for person in ("p1", "p2"):
        out[f"{person}_motion_speed"] = numeric_series(df, f"{person}_motion_speed")
        if out[f"{person}_motion_speed"].isna().all() and all(f"{person}_ref_{axis}" in df for axis in ("x", "y", "z")):
            out[f"{person}_motion_speed"] = speed_from_points(point_array(df, f"{person}_ref"), out["timestamp_ms"].to_numpy(float))
        out[f"{person}_motion_energy"] = numeric_series(df, f"{person}_motion_energy_body")
        if out[f"{person}_motion_energy"].isna().all():
            out[f"{person}_motion_energy"] = out[f"{person}_motion_speed"]

        partner_angle = numeric_series(df, f"{person}_gaze_to_other_head_angle_deg")
        task_angle = numeric_series(df, f"{person}_gaze_to_dyad_ref_angle_deg")
        out[f"{person}_partner_gaze"] = (partner_angle <= 25.0) & np.isfinite(partner_angle)
        out[f"{person}_task_gaze"] = (task_angle <= 35.0) & np.isfinite(task_angle)
        out[f"{person}_partner_gaze_angle"] = partner_angle
        out[f"{person}_task_gaze_angle"] = task_angle

    if "mutual_gaze_cos_min" in df:
        out["mutual_gaze_cos"] = numeric_series(df, "mutual_gaze_cos_min")
        out["mutual_gaze"] = out["mutual_gaze_cos"] >= math.cos(math.radians(25.0))
    else:
        out["mutual_gaze_cos"] = np.nan
        out["mutual_gaze"] = out["p1_partner_gaze"] & out["p2_partner_gaze"]

    p1_hand_cols = ["p1_left_wrist_to_p2_head_distance", "p1_right_wrist_to_p2_head_distance"]
    p2_hand_cols = ["p2_left_wrist_to_p1_head_distance", "p2_right_wrist_to_p1_head_distance"]
    if all(c in df for c in p1_hand_cols):
        out["p1_closest_hand_to_other_head"] = pd.concat([numeric_series(df, c) for c in p1_hand_cols], axis=1).min(axis=1)
    elif all(f"p1_kpt{k}_world_x" in df for k in (LEFT_WRIST, RIGHT_WRIST)) and all(f"p2_kpt{k}_world_x" in df for k in (NOSE, LEFT_EYE, RIGHT_EYE)):
        p2_head = np.nanmean(
            np.stack([point_array(df, f"p2_kpt{k}_world") for k in (NOSE, LEFT_EYE, RIGHT_EYE)], axis=0),
            axis=0,
        )
        out["p1_closest_hand_to_other_head"] = np.fmin(
            distance_from_points(point_array(df, f"p1_kpt{LEFT_WRIST}_world"), p2_head),
            distance_from_points(point_array(df, f"p1_kpt{RIGHT_WRIST}_world"), p2_head),
        )
    else:
        out["p1_closest_hand_to_other_head"] = np.nan

    if all(c in df for c in p2_hand_cols):
        out["p2_closest_hand_to_other_head"] = pd.concat([numeric_series(df, c) for c in p2_hand_cols], axis=1).min(axis=1)
    elif all(f"p2_kpt{k}_world_x" in df for k in (LEFT_WRIST, RIGHT_WRIST)) and all(f"p1_kpt{k}_world_x" in df for k in (NOSE, LEFT_EYE, RIGHT_EYE)):
        p1_head = np.nanmean(
            np.stack([point_array(df, f"p1_kpt{k}_world") for k in (NOSE, LEFT_EYE, RIGHT_EYE)], axis=0),
            axis=0,
        )
        out["p2_closest_hand_to_other_head"] = np.fmin(
            distance_from_points(point_array(df, f"p2_kpt{LEFT_WRIST}_world"), p1_head),
            distance_from_points(point_array(df, f"p2_kpt{RIGHT_WRIST}_world"), p1_head),
        )
    else:
        out["p2_closest_hand_to_other_head"] = np.nan

    hand_to_hand_cols = ["p1_left_wrist_to_p2_right_wrist_distance", "p1_right_wrist_to_p2_left_wrist_distance"]
    if all(c in df for c in hand_to_hand_cols):
        out["hand_to_hand_min"] = pd.concat([numeric_series(df, c) for c in hand_to_hand_cols], axis=1).min(axis=1)
    else:
        out["hand_to_hand_min"] = np.nan

    p1_thr = robust_activity_threshold(out["p1_motion_energy"].to_numpy(float))
    p2_thr = robust_activity_threshold(out["p2_motion_energy"].to_numpy(float))
    out["p1_active"] = out["p1_motion_energy"] > p1_thr
    out["p2_active"] = out["p2_motion_energy"] > p2_thr
    out["both_active"] = out["p1_active"] & out["p2_active"]
    out["one_active"] = out["p1_active"] ^ out["p2_active"]
    out["neither_active"] = ~(out["p1_active"] | out["p2_active"])

    return out.sort_values("timestamp_ms").reset_index(drop=True)


def summarize_window(w: pd.DataFrame, session_meta: pd.Series, window_index: int, config: RunConfig) -> dict:
    p1_energy = w["p1_motion_energy"].to_numpy(float)
    p2_energy = w["p2_motion_energy"].to_numpy(float)
    timestamps = w["timestamp_ms"].to_numpy(float)
    zero_corr, best_corr, best_lag_s = compute_crosscorr(p1_energy, p2_energy, timestamps, min(config.max_lag_s, config.window_s / 2.0))

    p1_hand = w["p1_closest_hand_to_other_head"].to_numpy(float)
    p2_hand = w["p2_closest_hand_to_other_head"].to_numpy(float)
    p1_hand_delta = p1_hand[-1] - p1_hand[0] if len(p1_hand) > 1 and np.isfinite([p1_hand[0], p1_hand[-1]]).all() else np.nan
    p2_hand_delta = p2_hand[-1] - p2_hand[0] if len(p2_hand) > 1 and np.isfinite([p2_hand[0], p2_hand[-1]]).all() else np.nan

    state_ids = (w["p1_active"].astype(int).to_numpy() * 2) + w["p2_active"].astype(int).to_numpy()
    state_counts = np.bincount(state_ids, minlength=4)

    root = w["root_distance"].to_numpy(float)
    root_delta = root[-1] - root[0] if len(root) > 1 and np.isfinite([root[0], root[-1]]).all() else np.nan

    p1_gaze = w["p1_partner_gaze"].to_numpy(bool)
    p2_gaze = w["p2_partner_gaze"].to_numpy(bool)
    p1_task = w["p1_task_gaze"].to_numpy(bool)
    p2_task = w["p2_task_gaze"].to_numpy(bool)

    def switch_rate(mask: np.ndarray) -> float:
        valid_len = len(mask)
        if valid_len < 2:
            return np.nan
        duration_s = max((timestamps[-1] - timestamps[0]) / 1000.0, EPS)
        return float(np.sum(mask[1:] != mask[:-1]) / duration_s)

    p1_mean = finite_mean(p1_energy)
    p2_mean = finite_mean(p2_energy)
    pair_mean = finite_mean([p1_mean, p2_mean])
    pair_max = finite_mean([np.nanmax(p1_energy) if np.isfinite(p1_energy).any() else np.nan,
                            np.nanmax(p2_energy) if np.isfinite(p2_energy).any() else np.nan])
    absdiff = abs(p1_mean - p2_mean) if np.isfinite([p1_mean, p2_mean]).all() else np.nan
    balance = absdiff / (p1_mean + p2_mean + EPS) if np.isfinite([p1_mean, p2_mean]).all() else np.nan

    valid_pose = finite_ratio(w["frame_pose_valid"])
    valid_gaze = finite_ratio(w["frame_gaze_valid"])
    valid_interaction = finite_ratio(w["frame_interaction_valid"])

    out = {
        "session_label": session_meta["session_label"],
        "file_name": session_meta["file_name"],
        "dyad_id": session_meta["dyad_id"],
        "dyad_block_valid": bool(session_meta["dyad_block_valid"]),
        "order": int(session_meta["order"]) if pd.notna(session_meta["order"]) else np.nan,
        "is_competitive_practiced": bool(session_meta.get("is_competitive_practiced", False)),
        "window_index": window_index,
        "window_start_ms": float(w["timestamp_ms"].iloc[0]),
        "window_end_ms": float(w["timestamp_ms"].iloc[-1]),
        "window_mid_ms": float((w["timestamp_ms"].iloc[0] + w["timestamp_ms"].iloc[-1]) / 2.0),
        "n_frames": int(len(w)),
        "valid_pose_ratio": valid_pose,
        "valid_gaze_ratio": valid_gaze,
        "valid_interaction_ratio": valid_interaction,
        "passes_quality": bool(valid_interaction >= config.min_window_valid_ratio),
        "root_distance_mean": finite_mean(root),
        "root_distance_std": finite_std(root),
        "root_distance_delta": root_delta,
        "root_horizontal_distance_mean": finite_mean(w["root_horizontal_distance"].to_numpy(float)),
        "vertical_offset_abs_mean": finite_mean(w["vertical_offset_abs"].to_numpy(float)),
        "pair_motion_energy_mean": pair_mean,
        "pair_motion_energy_max": pair_max,
        "motion_energy_absdiff_mean": absdiff,
        "motion_balance_mean": balance,
        "both_active_ratio": finite_ratio(w["both_active"]),
        "one_active_ratio": finite_ratio(w["one_active"]),
        "neither_active_ratio": finite_ratio(w["neither_active"]),
        "activity_state_entropy": shannon_entropy_from_counts(state_counts),
        "motion_corr_zero_lag": zero_corr,
        "motion_crosscorr_absmax": abs(best_corr) if np.isfinite(best_corr) else np.nan,
        "motion_crosscorr_abs_lag_s": best_lag_s,
        "closest_hand_to_other_head_mean": finite_mean(np.fmin(p1_hand, p2_hand)),
        "closest_hand_to_other_head_min": np.nanmin(np.fmin(p1_hand, p2_hand)) if np.isfinite(np.fmin(p1_hand, p2_hand)).any() else np.nan,
        "hand_to_hand_min_mean": finite_mean(w["hand_to_hand_min"].to_numpy(float)),
        "hand_reach_delta_mean": finite_mean([p1_hand_delta, p2_hand_delta]),
        "partner_gaze_ratio_mean": finite_mean([finite_ratio(p1_gaze), finite_ratio(p2_gaze)]),
        "partner_gaze_ratio_any": finite_ratio(p1_gaze | p2_gaze),
        "partner_gaze_ratio_both": finite_ratio(p1_gaze & p2_gaze),
        "partner_gaze_asym_abs": abs(finite_ratio(p1_gaze) - finite_ratio(p2_gaze)),
        "task_gaze_ratio_mean": finite_mean([finite_ratio(p1_task), finite_ratio(p2_task)]),
        "mutual_gaze_ratio": finite_ratio(w["mutual_gaze"]),
        "mutual_gaze_cos_mean": finite_mean(w["mutual_gaze_cos"].to_numpy(float)),
        "gaze_switch_rate_mean": finite_mean([switch_rate(p1_gaze), switch_rate(p2_gaze)]),
        # Side diagnostics are retained for auditing, but not used as primary analysis/model features.
        "side_p1_motion_energy_mean": p1_mean,
        "side_p2_motion_energy_mean": p2_mean,
        "side_p1_partner_gaze_ratio": finite_ratio(p1_gaze),
        "side_p2_partner_gaze_ratio": finite_ratio(p2_gaze),
    }
    return out


def extract_windows_for_session(path: Path, session_meta: pd.Series, config: RunConfig) -> tuple[pd.DataFrame, dict]:
    df = read_relevant_parquet(path)
    frame = prepare_frame_features(df, config)
    if frame.empty:
        return pd.DataFrame(), {}

    t = frame["timestamp_ms"].to_numpy(float)
    start_t = np.nanmin(t)
    end_t = np.nanmax(t)
    duration_s = (end_t - start_t) / 1000.0 if np.isfinite([start_t, end_t]).all() else np.nan
    if not np.isfinite(duration_s) or duration_s <= 0:
        duration_s = len(frame) / 30.0
        frame["timestamp_ms"] = np.arange(len(frame), dtype=float) * (1000.0 / 30.0)
        start_t = float(frame["timestamp_ms"].iloc[0])
        end_t = float(frame["timestamp_ms"].iloc[-1])

    rows = []
    win_ms = config.window_s * 1000.0
    stride_ms = config.stride_s * 1000.0
    if end_t - start_t < win_ms:
        windows = [start_t]
    else:
        windows = list(np.arange(start_t, end_t - win_ms + stride_ms, stride_ms))
    for idx, ws in enumerate(windows):
        we = ws + win_ms
        w = frame[(frame["timestamp_ms"] >= ws) & (frame["timestamp_ms"] < we)]
        if len(w) < 3:
            continue
        rows.append(summarize_window(w, session_meta, idx, config))

    quality = {
        "session_label": session_meta["session_label"],
        "file_name": session_meta["file_name"],
        "dyad_id": session_meta["dyad_id"],
        "order": session_meta["order"],
        "n_rows_loaded": int(len(frame)),
        "duration_s": float(duration_s),
        "fps_median": float(1000.0 / np.nanmedian(np.diff(t))) if len(t) > 2 and np.nanmedian(np.diff(t)) > 0 else np.nan,
        "frame_pose_valid_ratio": finite_ratio(frame["frame_pose_valid"]),
        "frame_gaze_valid_ratio": finite_ratio(frame["frame_gaze_valid"]),
        "frame_interaction_valid_ratio": finite_ratio(frame["frame_interaction_valid"]),
        "window_count": int(len(rows)),
        "quality_window_count": int(sum(r["passes_quality"] for r in rows)),
    }
    return pd.DataFrame(rows), quality


def build_metadata(input_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    warnings_out = []
    files = list_parquets(input_dir)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")
    rows = []
    first_schema = None
    for path in files:
        row = parse_session_file(path)
        row["path"] = str(path.resolve())
        try:
            row["row_count"] = parquet_row_count(path)
            cols = parquet_schema_columns(path)
            row["column_count"] = len(cols)
            row["has_interaction_schema"] = "dyad_root_distance" in cols and "frame_interaction_valid" in cols
            if first_schema is None:
                first_schema = set(cols)
                row["schema_matches_first"] = True
            else:
                row["schema_matches_first"] = set(cols) == first_schema
        except Exception as exc:
            row["row_count"] = np.nan
            row["column_count"] = np.nan
            row["has_interaction_schema"] = False
            row["schema_matches_first"] = False
            row["schema_error"] = str(exc)
        rows.append(row)
    meta = pd.DataFrame(rows)
    meta = infer_dyads(meta)
    if len(meta) < 4:
        warnings_out.append(
            f"Input directory has only {len(meta)} parquet file(s); dyad-level order statistics and group-validated models will be limited or skipped."
        )
    invalid_blocks = meta.loc[~meta["dyad_block_valid"], "dyad_id"].unique().tolist()
    if invalid_blocks:
        warnings_out.append(f"Incomplete or ambiguous dyad blocks detected: {', '.join(invalid_blocks)}")
    if not bool(meta["has_interaction_schema"].all()):
        warnings_out.append("At least one parquet does not look like an interaction parquet with derived dyad/gaze features.")
    if not bool(meta["schema_matches_first"].all()):
        warnings_out.append("Not all parquet schemas match the first file exactly; feature extraction uses available-column fallbacks.")
    return meta, warnings_out


def extract_feature_tables(meta: pd.DataFrame, config: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    all_windows = []
    quality_rows = []
    warnings_out = []
    for _, row in meta.iterrows():
        path = Path(row["path"])
        try:
            wdf, q = extract_windows_for_session(path, row, config)
            if not wdf.empty:
                all_windows.append(wdf)
            if q:
                quality_rows.append(q)
        except Exception as exc:
            warnings_out.append(f"Failed to extract features for {path.name}: {exc}")
    window_df = pd.concat(all_windows, ignore_index=True) if all_windows else pd.DataFrame()
    quality_df = pd.DataFrame(quality_rows)
    session_df = aggregate_session_features(window_df, quality_df) if not window_df.empty else pd.DataFrame()
    return window_df, session_df, quality_df, warnings_out


def aggregate_session_features(window_df: pd.DataFrame, quality_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feature_cols = [c for c in PRIMARY_ANALYSIS_FEATURES if c in window_df.columns]
    for (session_label, dyad_id, order), group in window_df.groupby(["session_label", "dyad_id", "order"], dropna=False):
        qgroup = group[group["passes_quality"]].copy()
        if qgroup.empty:
            qgroup = group.copy()
        out = {
            "session_label": session_label,
            "dyad_id": dyad_id,
            "order": int(order) if pd.notna(order) else np.nan,
            "is_competitive_practiced": bool(order > 1) if pd.notna(order) else False,
            "window_count": int(len(group)),
            "quality_window_count": int(group["passes_quality"].sum()) if "passes_quality" in group else int(len(group)),
        }
        for col in feature_cols:
            arr = pd.to_numeric(qgroup[col], errors="coerce").to_numpy(float)
            out[f"{col}_mean"] = finite_mean(arr)
            out[f"{col}_std"] = finite_std(arr)
            out[f"{col}_median"] = finite_median(arr)
        rows.append(out)
    session_df = pd.DataFrame(rows)
    if not quality_df.empty:
        qcols = ["session_label", "duration_s", "fps_median", "frame_pose_valid_ratio", "frame_gaze_valid_ratio", "frame_interaction_valid_ratio"]
        session_df = session_df.merge(quality_df[[c for c in qcols if c in quality_df.columns]], on="session_label", how="left")
    return session_df.sort_values(["dyad_id", "order", "session_label"]).reset_index(drop=True)


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = RANDOM_SEED) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = np.nanmean(sample)
    return float(np.nanpercentile(means, 100 * alpha / 2)), float(np.nanpercentile(means, 100 * (1 - alpha / 2)))


def wilcoxon_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3 or np.allclose(values, 0):
        return np.nan
    try:
        return float(stats.wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        return np.nan


def bh_fdr(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    q_ranked = ranked * m / np.arange(1, m + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_valid = np.empty_like(q_ranked)
    q_valid[order] = np.clip(q_ranked, 0, 1)
    q[valid] = q_valid
    return q


def confidence_label(n: int, q_value: float, ci_low: float, ci_high: float, robust: bool) -> str:
    ci_excludes_zero = np.isfinite(ci_low) and np.isfinite(ci_high) and (ci_low > 0 or ci_high < 0)
    if n >= 8 and np.isfinite(q_value) and q_value < 0.05 and ci_excludes_zero and robust:
        return "strong"
    if n >= 6 and ci_excludes_zero and (not np.isfinite(q_value) or q_value < 0.15):
        return "moderate"
    return "weak/exploratory"


def run_statistics(session_df: pd.DataFrame) -> pd.DataFrame:
    if session_df.empty or "dyad_id" not in session_df or session_df["dyad_id"].nunique() < 2:
        return pd.DataFrame(
            columns=[
                "contrast",
                "feature",
                "n_dyads",
                "mean_diff",
                "median_diff",
                "effect_size_dz",
                "ci95_low",
                "ci95_high",
                "p_value",
                "q_value",
                "confidence",
                "interpretation_note",
            ]
        )

    feature_cols = [c for c in STAT_FEATURES if c in session_df.columns]
    rows = []
    for feature in feature_cols:
        pivot = session_df.pivot_table(index="dyad_id", columns="order", values=feature, aggfunc="mean")
        if 1 in pivot.columns and any(c in pivot.columns for c in (2, 3, 4)):
            comp_cols = [c for c in (2, 3, 4) if c in pivot.columns]
            diff = pivot[comp_cols].mean(axis=1) - pivot[1]
            rows.append(stat_row("order1_vs_orders2_4", feature, diff, "Later orders minus order1; competition and practice are confounded."))
        if 2 in pivot.columns and 4 in pivot.columns:
            diff = pivot[4] - pivot[2]
            rows.append(stat_row("order4_vs_order2", feature, diff, "Late competitive order minus early competitive order."))
        if all(c in pivot.columns for c in (2, 3, 4)):
            slopes = []
            for _, r in pivot[[2, 3, 4]].iterrows():
                y = r.to_numpy(float)
                if np.isfinite(y).sum() == 3:
                    slopes.append(float(np.polyfit(np.array([2, 3, 4], dtype=float), y, 1)[0]))
            rows.append(stat_row("competitive_order2_4_linear_slope", feature, np.asarray(slopes), "Per-dyad linear slope across orders 2, 3, and 4."))

    result = pd.DataFrame(rows)
    if not result.empty:
        result["q_value"] = bh_fdr(result["p_value"].to_numpy(float))
        result["confidence"] = [
            confidence_label(int(r.n_dyads), float(r.q_value), float(r.ci95_low), float(r.ci95_high), True)
            for r in result.itertuples()
        ]
        result = result.sort_values(["confidence", "q_value", "feature"], na_position="last").reset_index(drop=True)
    return result


def stat_row(contrast: str, feature: str, values: np.ndarray, note: str) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = finite_mean(values)
    std = finite_std(values)
    ci_low, ci_high = bootstrap_ci(values, n_boot=3000) if len(values) else (np.nan, np.nan)
    return {
        "contrast": contrast,
        "feature": feature,
        "n_dyads": int(len(values)),
        "mean_diff": mean,
        "median_diff": finite_median(values),
        "effect_size_dz": float(mean / std) if np.isfinite(mean) and np.isfinite(std) and std > EPS else np.nan,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "p_value": wilcoxon_p(values),
        "q_value": np.nan,
        "confidence": "pending",
        "interpretation_note": note,
    }


def select_quality_windows(window_df: pd.DataFrame, min_valid_ratio: float) -> pd.DataFrame:
    if window_df.empty:
        return window_df
    q = window_df[window_df.get("passes_quality", True)].copy()
    if len(q) < max(20, 0.2 * len(window_df)):
        q = window_df[window_df["valid_interaction_ratio"] >= min(0.3, min_valid_ratio)].copy()
    if q.empty:
        q = window_df.copy()
    return q.reset_index(drop=True)


def discover_vocabulary(window_df: pd.DataFrame, config: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if window_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {"status": "skipped", "reason": "no windows"}
    vocab_df = select_quality_windows(window_df, config.min_window_valid_ratio).copy()
    features = [c for c in VOCAB_FEATURES if c in vocab_df.columns]
    usable = vocab_df[features].apply(pd.to_numeric, errors="coerce")
    nonempty_features = [c for c in features if usable[c].notna().sum() >= max(5, int(0.05 * len(usable)))]
    if len(nonempty_features) < 3 or len(vocab_df) < 8:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "status": "skipped",
            "reason": "not enough usable windows/features for clustering",
            "n_windows": int(len(vocab_df)),
            "n_features": int(len(nonempty_features)),
        }

    Xraw = vocab_df[nonempty_features].apply(pd.to_numeric, errors="coerce")
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    X = pipe.fit_transform(Xraw)
    n = len(vocab_df)
    k = int(min(config.vocabulary_k, max(2, n // 20))) if n < 160 else int(config.vocabulary_k)
    k = max(2, min(k, n - 1))
    clusterer = MiniBatchKMeans(n_clusters=k, random_state=config.random_seed, n_init=20, batch_size=512)
    labels = clusterer.fit_predict(X)
    vocab_df["token_id"] = labels
    vocab_df["token"] = [f"NV{label:02d}" for label in labels]

    sil = np.nan
    if len(np.unique(labels)) > 1 and len(labels) > len(np.unique(labels)):
        sample_size = min(3000, len(labels))
        try:
            sil = float(silhouette_score(X, labels, sample_size=sample_size, random_state=config.random_seed))
        except Exception:
            sil = np.nan

    centers_scaled = clusterer.cluster_centers_
    scaler = pipe.named_steps["scaler"]
    imputer = pipe.named_steps["imputer"]
    centers_imputed = scaler.inverse_transform(centers_scaled)
    center_df = pd.DataFrame(centers_imputed, columns=nonempty_features)
    global_medians = Xraw.median(numeric_only=True)

    summary_rows = []
    for token_id in sorted(np.unique(labels)):
        group = vocab_df[vocab_df["token_id"] == token_id]
        center = center_df.iloc[token_id]
        desc = describe_token(center, global_medians)
        row = {
            "token_id": int(token_id),
            "token": f"NV{token_id:02d}",
            "description": desc,
            "window_count": int(len(group)),
            "prevalence": float(len(group) / len(vocab_df)),
            "mean_duration_windows_per_run": token_run_length(group),
            "top_session": group["session_label"].value_counts().index[0] if "session_label" in group and not group.empty else None,
        }
        for order in sorted(vocab_df["order"].dropna().unique()):
            denom = max(1, int((vocab_df["order"] == order).sum()))
            row[f"prevalence_order{int(order)}"] = float(((vocab_df["order"] == order) & (vocab_df["token_id"] == token_id)).sum() / denom)
        for feat in nonempty_features:
            row[f"centroid_{feat}"] = float(center[feat]) if np.isfinite(center[feat]) else np.nan
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    transition_df = token_transitions(vocab_df)
    info = {
        "status": "ok",
        "n_windows": int(len(vocab_df)),
        "n_features": int(len(nonempty_features)),
        "k": int(k),
        "silhouette": sil,
        "features": nonempty_features,
    }
    return vocab_df, summary_df, transition_df, info


def describe_token(center: pd.Series, med: pd.Series) -> str:
    parts = []
    motion = center.get("pair_motion_energy_mean", np.nan)
    motion_med = med.get("pair_motion_energy_mean", np.nan)
    both = center.get("both_active_ratio", np.nan)
    one = center.get("one_active_ratio", np.nan)
    gaze = center.get("partner_gaze_ratio_mean", np.nan)
    mutual = center.get("mutual_gaze_ratio", np.nan)
    task = center.get("task_gaze_ratio_mean", np.nan)
    dist = center.get("root_distance_mean", np.nan)
    dist_med = med.get("root_distance_mean", np.nan)
    asym = center.get("motion_energy_absdiff_mean", np.nan)
    asym_med = med.get("motion_energy_absdiff_mean", np.nan)

    if np.isfinite(motion) and np.isfinite(motion_med):
        parts.append("high motion" if motion > motion_med * 1.25 else "low motion" if motion < motion_med * 0.75 else "moderate motion")
    if np.isfinite(both) and both > 0.45:
        parts.append("simultaneous activity")
    elif np.isfinite(one) and one > 0.45:
        parts.append("alternating/one-sided activity")
    else:
        parts.append("limited activity")
    if np.isfinite(gaze) and gaze > 0.25:
        parts.append("partner gaze")
    elif np.isfinite(mutual) and mutual > 0.15:
        parts.append("mutual gaze")
    elif np.isfinite(task) and task > 0.35:
        parts.append("task-focused gaze")
    else:
        parts.append("low/unknown gaze engagement")
    if np.isfinite(dist) and np.isfinite(dist_med):
        parts.append("close posture" if dist < dist_med * 0.95 else "separated posture" if dist > dist_med * 1.05 else "typical distance")
    if np.isfinite(asym) and np.isfinite(asym_med) and asym > asym_med * 1.25:
        parts.append("motion-asymmetric")
    return "; ".join(parts)


def token_run_length(group: pd.DataFrame) -> float:
    if group.empty:
        return np.nan
    lengths = []
    for _, sgroup in group.sort_values(["session_label", "window_index"]).groupby("session_label"):
        idx = sgroup["window_index"].to_numpy(int)
        if len(idx) == 0:
            continue
        current = 1
        for a, b in zip(idx[:-1], idx[1:]):
            if b == a + 1:
                current += 1
            else:
                lengths.append(current)
                current = 1
        lengths.append(current)
    return finite_mean(lengths)


def token_transitions(tokens_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if tokens_df.empty:
        return pd.DataFrame()
    for session_label, group in tokens_df.sort_values(["session_label", "window_index"]).groupby("session_label"):
        toks = group["token"].tolist()
        order = int(group["order"].iloc[0]) if "order" in group and pd.notna(group["order"].iloc[0]) else np.nan
        dyad_id = group["dyad_id"].iloc[0] if "dyad_id" in group else None
        for a, b in zip(toks[:-1], toks[1:]):
            rows.append({"session_label": session_label, "dyad_id": dyad_id, "order": order, "from_token": a, "to_token": b, "count": 1})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby(["session_label", "dyad_id", "order", "from_token", "to_token"], dropna=False)["count"].sum().reset_index()


def run_models(window_df: pd.DataFrame, tokens_df: pd.DataFrame, config: RunConfig) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    result_rows = []
    matrices = {}
    if window_df.empty or window_df["dyad_id"].nunique() < 3:
        return pd.DataFrame([{"model": "all", "target": "all", "status": "skipped", "reason": "fewer than 3 dyads/windows"}]), matrices

    model_df = select_quality_windows(window_df, config.min_window_valid_ratio).copy()
    if len(model_df) > config.max_model_windows:
        model_df = model_df.sample(config.max_model_windows, random_state=config.random_seed).sort_index().reset_index(drop=True)
    feature_cols = [c for c in PRIMARY_ANALYSIS_FEATURES if c in model_df.columns and c not in {"valid_pose_ratio", "valid_gaze_ratio", "valid_interaction_ratio"}]
    feature_cols = [c for c in feature_cols if model_df[c].notna().sum() >= max(10, int(0.05 * len(model_df)))]
    if len(feature_cols) < 3:
        return pd.DataFrame([{"model": "all", "target": "all", "status": "skipped", "reason": "not enough model features"}]), matrices

    X = model_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    groups = model_df["dyad_id"].astype(str).to_numpy()
    logo = LeaveOneGroupOut()

    targets = {
        "competitive_practiced_vs_order1": model_df["order"].astype(int).gt(1).astype(int),
        "order_1_2_3_4": model_df["order"].astype(int),
    }
    model_specs = {
        "logistic_l2": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.random_seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, max_leaf_nodes=15, random_state=config.random_seed),
        "random_forest_shallow": RandomForestClassifier(n_estimators=160, max_depth=4, min_samples_leaf=10, class_weight="balanced", random_state=config.random_seed, n_jobs=-1),
    }

    for target_name, y in targets.items():
        y = y.to_numpy()
        if len(np.unique(y)) < 2:
            result_rows.append({"model": "all", "target": target_name, "status": "skipped", "reason": "single class target"})
            continue
        majority = majority_class_predictions(y, groups)
        result_rows.append(score_predictions("majority_by_train_fold", target_name, y, majority, "ok", "grouped baseline"))
        for model_name, estimator in model_specs.items():
            preds = np.full(len(y), -999, dtype=int)
            ok = True
            for train_idx, test_idx in logo.split(X, y, groups):
                if len(np.unique(y[train_idx])) < 2:
                    ok = False
                    break
                clf = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", estimator),
                    ]
                )
                try:
                    clf.fit(X.iloc[train_idx], y[train_idx])
                    preds[test_idx] = clf.predict(X.iloc[test_idx])
                except Exception as exc:
                    result_rows.append({"model": model_name, "target": target_name, "status": "failed", "reason": str(exc)})
                    ok = False
                    break
            if ok and np.all(preds != -999):
                result_rows.append(score_predictions(model_name, target_name, y, preds, "ok", "leave-one-dyad-out"))
                matrices[f"{model_name}__{target_name}"] = confusion_df(y, preds)

    markov_rows = run_markov_next_token(tokens_df)
    result_rows.extend(markov_rows)
    return pd.DataFrame(result_rows), matrices


def majority_class_predictions(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    preds = np.full(len(y), -999, dtype=int)
    logo = LeaveOneGroupOut()
    dummy_X = np.zeros((len(y), 1))
    for train_idx, test_idx in logo.split(dummy_X, y, groups):
        values, counts = np.unique(y[train_idx], return_counts=True)
        preds[test_idx] = values[np.argmax(counts)]
    return preds


def score_predictions(model: str, target: str, y_true: np.ndarray, y_pred: np.ndarray, status: str, validation: str) -> dict:
    valid = y_pred != -999
    return {
        "model": model,
        "target": target,
        "status": status,
        "validation": validation,
        "n_samples": int(valid.sum()),
        "accuracy": float(accuracy_score(y_true[valid], y_pred[valid])) if valid.any() else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true[valid], y_pred[valid])) if valid.any() else np.nan,
        "macro_f1": float(f1_score(y_true[valid], y_pred[valid], average="macro", zero_division=0)) if valid.any() else np.nan,
        "swap_consistency": "invariant by construction: only symmetric dyad features are used",
    }


def confusion_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    mat = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(mat, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])


def run_markov_next_token(tokens_df: pd.DataFrame) -> list[dict]:
    if tokens_df.empty or tokens_df["dyad_id"].nunique() < 3:
        return [{"model": "markov_next_token", "target": "next_vocabulary_token", "status": "skipped", "reason": "not enough tokenized dyads"}]
    rows = []
    y_true_all = []
    y_pred_all = []
    y_uni_all = []
    dyads = sorted(tokens_df["dyad_id"].dropna().unique())
    for test_dyad in dyads:
        train = tokens_df[tokens_df["dyad_id"] != test_dyad]
        test = tokens_df[tokens_df["dyad_id"] == test_dyad]
        trans = {}
        unigram = train["token"].value_counts()
        default = unigram.index[0] if not unigram.empty else None
        for _, group in train.sort_values(["session_label", "window_index"]).groupby("session_label"):
            toks = group["token"].tolist()
            for a, b in zip(toks[:-1], toks[1:]):
                trans.setdefault(a, {}).setdefault(b, 0)
                trans[a][b] += 1
        for _, group in test.sort_values(["session_label", "window_index"]).groupby("session_label"):
            toks = group["token"].tolist()
            for a, b in zip(toks[:-1], toks[1:]):
                pred = default
                if a in trans:
                    pred = max(trans[a].items(), key=lambda kv: kv[1])[0]
                if pred is not None:
                    y_true_all.append(b)
                    y_pred_all.append(pred)
                    y_uni_all.append(default)
    if not y_true_all:
        return [{"model": "markov_next_token", "target": "next_vocabulary_token", "status": "skipped", "reason": "no token transitions"}]
    labels = sorted(set(y_true_all) | set(y_pred_all) | set(y_uni_all))
    rows.append(
        {
            "model": "unigram_next_token",
            "target": "next_vocabulary_token",
            "status": "ok",
            "validation": "leave-one-dyad-out",
            "n_samples": int(len(y_true_all)),
            "accuracy": float(accuracy_score(y_true_all, y_uni_all)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_all, y_uni_all)),
            "macro_f1": float(f1_score(y_true_all, y_uni_all, labels=labels, average="macro", zero_division=0)),
            "swap_consistency": "tokenization uses symmetric dyad features",
        }
    )
    rows.append(
        {
            "model": "markov_next_token",
            "target": "next_vocabulary_token",
            "status": "ok",
            "validation": "leave-one-dyad-out",
            "n_samples": int(len(y_true_all)),
            "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_all, y_pred_all)),
            "macro_f1": float(f1_score(y_true_all, y_pred_all, labels=labels, average="macro", zero_division=0)),
            "swap_consistency": "tokenization uses symmetric dyad features",
        }
    )
    return rows


def make_figures(session_df: pd.DataFrame, stats_df: pd.DataFrame, vocab_summary: pd.DataFrame, output_dir: Path) -> list[str]:
    fig_dir = output_dir / "figures"
    ensure_dir(fig_dir)
    paths = []
    if not session_df.empty:
        candidates = [
            "pair_motion_energy_mean_mean",
            "both_active_ratio_mean",
            "partner_gaze_ratio_mean_mean",
            "root_distance_mean_mean",
        ]
        available = [c for c in candidates if c in session_df.columns]
        for col in available:
            plt.figure(figsize=(7, 4))
            data = [session_df.loc[session_df["order"] == o, col].dropna().to_numpy(float) for o in sorted(session_df["order"].dropna().unique())]
            labels = [str(int(o)) for o in sorted(session_df["order"].dropna().unique())]
            plt.boxplot(data, tick_labels=labels, showmeans=True)
            plt.title(col)
            plt.xlabel("Order")
            plt.ylabel(col)
            plt.tight_layout()
            path = fig_dir / f"box_{col}.png"
            plt.savefig(path, dpi=150)
            plt.close()
            paths.append(str(path))
    if not vocab_summary.empty:
        prev_cols = [c for c in vocab_summary.columns if c.startswith("prevalence_order")]
        if prev_cols:
            heat = vocab_summary.set_index("token")[prev_cols]
            plt.figure(figsize=(max(6, len(prev_cols) * 1.2), max(4, len(heat) * 0.45)))
            plt.imshow(heat.to_numpy(float), aspect="auto", cmap="viridis")
            plt.colorbar(label="Prevalence")
            plt.xticks(range(len(prev_cols)), [c.replace("prevalence_", "") for c in prev_cols], rotation=45, ha="right")
            plt.yticks(range(len(heat.index)), heat.index.tolist())
            plt.title("Vocabulary token prevalence by order")
            plt.tight_layout()
            path = fig_dir / "vocabulary_prevalence_by_order.png"
            plt.savefig(path, dpi=150)
            plt.close()
            paths.append(str(path))
    return paths


def evidence_sentence(row: pd.Series) -> str:
    direction = "increased" if row["mean_diff"] > 0 else "decreased"
    return (
        f"{row['feature']} {direction} for {row['contrast']} "
        f"(mean diff={row['mean_diff']:.4g}, 95% CI [{row['ci95_low']:.4g}, {row['ci95_high']:.4g}], "
        f"q={row['q_value']:.4g}, n={int(row['n_dyads'])} dyads; confidence={row['confidence']})."
    )


def write_report(
    output_dir: Path,
    config: RunConfig,
    meta: pd.DataFrame,
    quality_df: pd.DataFrame,
    window_df: pd.DataFrame,
    session_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    vocab_summary: pd.DataFrame,
    vocab_info: dict,
    model_df: pd.DataFrame,
    warnings_out: list[str],
    figure_paths: list[str],
) -> None:
    lines = []
    lines.append("# HRI Parquet Analysis Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This report was generated from tabular pose/gaze parquets only. It uses person-label-invariant dyadic features for primary analysis and models; `p1`/`p2` side diagnostics are retained only for auditing.")
    lines.append("")
    lines.append("## Input and Inventory")
    lines.append("")
    lines.append(f"- Input directory: `{config.input_dir}`")
    lines.append(f"- Parquet files found: {len(meta)}")
    lines.append(f"- Total parquet rows: {int(pd.to_numeric(meta['row_count'], errors='coerce').sum()) if 'row_count' in meta else 'unknown'}")
    lines.append(f"- Inferred dyads: {meta['dyad_id'].nunique() if not meta.empty else 0}")
    lines.append(f"- Complete four-order dyad blocks: {meta.loc[meta['dyad_block_valid'], 'dyad_id'].nunique() if 'dyad_block_valid' in meta else 0}")
    if warnings_out:
        lines.append("")
        lines.append("## Warnings and Limitations")
        lines.append("")
        for warning in warnings_out:
            lines.append(f"- {warning}")
    lines.append("")
    lines.append("## Data Quality")
    lines.append("")
    if not quality_df.empty:
        lines.append(f"- Total extracted windows: {len(window_df)}")
        lines.append(f"- Quality-passing windows: {int(window_df['passes_quality'].sum()) if 'passes_quality' in window_df else len(window_df)}")
        for col in ["frame_pose_valid_ratio", "frame_gaze_valid_ratio", "frame_interaction_valid_ratio"]:
            if col in quality_df:
                lines.append(f"- Mean {col}: {quality_df[col].mean():.3f}")
    else:
        lines.append("No quality table was produced.")
    lines.append("")
    lines.append("## Statistical Findings")
    lines.append("")
    if stats_df.empty:
        lines.append("Dyad-level statistics were skipped or produced no valid contrasts. This usually means the input directory does not contain enough complete dyad/order blocks.")
    else:
        top = stats_df.sort_values(["confidence", "q_value"], na_position="last").head(12)
        for _, row in top.iterrows():
            if pd.notna(row.get("mean_diff")):
                lines.append(f"- {evidence_sentence(row)}")
        strong_count = int((stats_df["confidence"] == "strong").sum()) if "confidence" in stats_df else 0
        moderate_count = int((stats_df["confidence"] == "moderate").sum()) if "confidence" in stats_df else 0
        lines.append("")
        lines.append(f"Summary: {strong_count} strong and {moderate_count} moderate evidence rows after FDR correction. Treat all other rows as exploratory.")
    lines.append("")
    lines.append("## Nonverbal Vocabulary")
    lines.append("")
    if vocab_info.get("status") == "ok":
        lines.append(f"- Vocabulary size: {vocab_info.get('k')}")
        lines.append(f"- Tokenized windows: {vocab_info.get('n_windows')}")
        lines.append(f"- Clustering silhouette: {vocab_info.get('silhouette'):.3f}" if np.isfinite(vocab_info.get("silhouette", np.nan)) else "- Clustering silhouette: unavailable")
        lines.append("- Token descriptions:")
        for _, row in vocab_summary.iterrows():
            lines.append(f"  - `{row['token']}`: {row['description']} (prevalence={row['prevalence']:.3f}, windows={int(row['window_count'])})")
    else:
        lines.append(f"Vocabulary discovery skipped: {vocab_info.get('reason', 'unknown reason')}")
    lines.append("")
    lines.append("## Modeling")
    lines.append("")
    if model_df.empty:
        lines.append("No model results were produced.")
    else:
        ok = model_df[model_df["status"] == "ok"] if "status" in model_df else pd.DataFrame()
        if ok.empty:
            for _, row in model_df.iterrows():
                lines.append(f"- {row.get('model', 'model')} / {row.get('target', 'target')}: skipped ({row.get('reason', 'no reason recorded')}).")
        else:
            for _, row in ok.sort_values("balanced_accuracy", ascending=False, na_position="last").head(10).iterrows():
                lines.append(
                    f"- {row['model']} on `{row['target']}`: balanced accuracy={row['balanced_accuracy']:.3f}, "
                    f"macro F1={row['macro_f1']:.3f}, validation={row.get('validation', 'unknown')}."
                )
        lines.append("- Model features exclude session labels, file names, timestamps as identifiers, and direct order metadata.")
        lines.append("- Swap consistency is guaranteed by using symmetric dyad features only for models and vocabulary.")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    if figure_paths:
        for path in figure_paths:
            rel = Path(path).relative_to(output_dir) if Path(path).is_relative_to(output_dir) else Path(path)
            lines.append(f"- `{rel}`")
    else:
        lines.append("No figures were generated.")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    for name in [
        "metadata.csv",
        "quality_summary.csv",
        "window_features.parquet",
        "session_features.parquet",
        "statistical_results.csv",
        "vocabulary_tokens.parquet",
        "vocabulary_summary.csv",
        "token_transitions.csv",
        "model_results.csv",
        "run_config.json",
    ]:
        lines.append(f"- `{name}`")
    lines.append("")
    lines.append("## Interpretation Guardrails")
    lines.append("")
    lines.append("- `order1` versus later orders is a baseline/noncompetitive first-exposure versus competitive/practiced contrast; competition is confounded with experience.")
    lines.append("- Frames and windows are not treated as independent evidence for final claims; inferential statistics are paired at dyad/session level.")
    lines.append("- `p1` and `p2` are table-side labels, not stable identities.")
    lines.append("- Gaze findings should be weighted by gaze validity coverage.")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_method_notes(base_dir: Path) -> None:
    text = """# What Was Implemented

This directory contains a parquet-only HRI analysis implementation for the goal described in `usta_pose/testing/goal.md`.

## Main Script

- `run_hri_analysis.py` is the reproducible entry point.
- It accepts `--input-dir`, `--output-dir`, window size/stride, quality threshold, lag window, and vocabulary size.
- The default input is repository-root `final_dataset_parquets`, matching the latest user request.

## Pipeline Stages

1. Metadata and validation: parses filenames, counts rows/columns, infers chronological dyad blocks, and flags incomplete order groups.
2. Feature extraction: reads only required parquet columns, computes frame-level motion/gaze/activity proxies, and aggregates them into sliding windows.
3. Person-label invariance: primary features are symmetric across the two people, using pair means, absolute differences, min/max style proximity, activity-state ratios, and non-directional lag magnitudes.
4. Session aggregation: creates session-level summaries from quality-passing windows.
5. Statistics: runs paired dyad-level contrasts with bootstrap confidence intervals, Wilcoxon p-values, and Benjamini-Hochberg FDR correction.
6. Vocabulary discovery: clusters quality windows into interpretable nonverbal tokens and exports token timelines plus transition counts.
7. Modeling: when enough dyads exist, trains simple group-validated models and a Markov next-token baseline using leave-one-dyad-out validation.
8. Reporting: writes `outputs/report.md` with findings, quality limits, vocabulary descriptions, model results, and interpretation guardrails.

## Output Directory

The default output location is `usta_pose/analiz/efe/outputs`. Important files include:

- `metadata.csv`
- `quality_summary.csv`
- `window_features.parquet`
- `session_features.parquet`
- `statistical_results.csv`
- `vocabulary_tokens.parquet`
- `vocabulary_summary.csv`
- `token_transitions.csv`
- `model_results.csv`
- `report.md`
- `figures/`

## Important Caveat

On this workspace, repository-root `final_dataset_parquets` currently contains only one parquet file. The implementation therefore supports that path, but dyad-level statistics and group-validated models require the full multi-session dataset. A full 44-file set is visible elsewhere in the repository under `usta_pose/testing/final_dataset_parquets` and `22kekecdataset/22kekecdataset/default_parquets`; run the script with `--input-dir` pointing to one of those directories for the complete academic analysis.

## Example Commands

Run exactly on the user-specified directory:

```bash
/home/efekaan/Desktop/torch/torch_env/bin/python usta_pose/analiz/efe/run_hri_analysis.py \
  --input-dir final_dataset_parquets \
  --output-dir usta_pose/analiz/efe/outputs
```

Run on the complete testing parquet set:

```bash
/home/efekaan/Desktop/torch/torch_env/bin/python usta_pose/analiz/efe/run_hri_analysis.py \
  --input-dir usta_pose/testing/final_dataset_parquets \
  --output-dir usta_pose/analiz/efe/outputs_full
```
"""
    (base_dir / "WHAT_I_IMPLEMENTED.md").write_text(text, encoding="utf-8")


def save_outputs(
    output_dir: Path,
    config: RunConfig,
    meta: pd.DataFrame,
    quality_df: pd.DataFrame,
    window_df: pd.DataFrame,
    session_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    vocab_tokens: pd.DataFrame,
    vocab_summary: pd.DataFrame,
    token_transitions_df: pd.DataFrame,
    model_df: pd.DataFrame,
    confusion_matrices: dict[str, pd.DataFrame],
    vocab_info: dict,
    warnings_out: list[str],
) -> list[str]:
    ensure_dir(output_dir)
    ensure_dir(output_dir / "figures")
    meta.to_csv(output_dir / "metadata.csv", index=False)
    quality_df.to_csv(output_dir / "quality_summary.csv", index=False)
    if not window_df.empty:
        window_df.to_parquet(output_dir / "window_features.parquet", index=False)
    else:
        pd.DataFrame().to_parquet(output_dir / "window_features.parquet", index=False)
    if not session_df.empty:
        session_df.to_parquet(output_dir / "session_features.parquet", index=False)
    else:
        pd.DataFrame().to_parquet(output_dir / "session_features.parquet", index=False)
    stats_df.to_csv(output_dir / "statistical_results.csv", index=False)
    if not vocab_tokens.empty:
        vocab_tokens.to_parquet(output_dir / "vocabulary_tokens.parquet", index=False)
    else:
        pd.DataFrame().to_parquet(output_dir / "vocabulary_tokens.parquet", index=False)
    vocab_summary.to_csv(output_dir / "vocabulary_summary.csv", index=False)
    token_transitions_df.to_csv(output_dir / "token_transitions.csv", index=False)
    model_df.to_csv(output_dir / "model_results.csv", index=False)
    for name, matrix in confusion_matrices.items():
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        matrix.to_csv(output_dir / f"confusion_{safe}.csv")
    config_dict = {
        "input_dir": str(config.input_dir),
        "output_dir": str(config.output_dir),
        "window_s": config.window_s,
        "stride_s": config.stride_s,
        "min_window_valid_ratio": config.min_window_valid_ratio,
        "max_lag_s": config.max_lag_s,
        "random_seed": config.random_seed,
        "vocabulary_k": config.vocabulary_k,
        "warnings": warnings_out,
        "vocabulary_info": vocab_info,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config_dict, indent=2), encoding="utf-8")
    figure_paths = make_figures(session_df, stats_df, vocab_summary, output_dir)
    return figure_paths


def run(config: RunConfig) -> None:
    warnings.simplefilter("ignore", RuntimeWarning)
    ensure_dir(config.output_dir)
    meta, warnings_out = build_metadata(config.input_dir)
    window_df, session_df, quality_df, feature_warnings = extract_feature_tables(meta, config)
    warnings_out.extend(feature_warnings)
    stats_df = run_statistics(session_df)
    vocab_tokens, vocab_summary, token_transitions_df, vocab_info = discover_vocabulary(window_df, config)
    model_df, confusion_matrices = run_models(window_df, vocab_tokens, config)
    figure_paths = save_outputs(
        config.output_dir,
        config,
        meta,
        quality_df,
        window_df,
        session_df,
        stats_df,
        vocab_tokens,
        vocab_summary,
        token_transitions_df,
        model_df,
        confusion_matrices,
        vocab_info,
        warnings_out,
    )
    write_report(
        config.output_dir,
        config,
        meta,
        quality_df,
        window_df,
        session_df,
        stats_df,
        vocab_summary,
        vocab_info,
        model_df,
        warnings_out,
        figure_paths,
    )
    write_method_notes(config.output_dir.parent if config.output_dir.name == "outputs" else config.output_dir.parent)
    print(f"Wrote analysis outputs to {config.output_dir}")
    print(f"Parquets: {len(meta)}, windows: {len(window_df)}, sessions: {len(session_df)}")
    if warnings_out:
        print("Warnings:")
        for warning in warnings_out:
            print(f"- {warning}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parquet-only dyadic HRI analysis.")
    parser.add_argument("--input-dir", type=Path, default=Path("final_dataset_parquets"), help="Directory containing *_orderN.parquet files.")
    parser.add_argument("--output-dir", type=Path, default=Path("usta_pose/analiz/efe/outputs"), help="Directory for generated analysis outputs.")
    parser.add_argument("--window-s", type=float, default=1.0, help="Sliding window duration in seconds.")
    parser.add_argument("--stride-s", type=float, default=0.5, help="Sliding window stride in seconds.")
    parser.add_argument("--min-window-valid-ratio", type=float, default=0.5, help="Minimum interaction-valid ratio for quality windows.")
    parser.add_argument("--max-lag-s", type=float, default=2.0, help="Maximum lag for cross-correlation features.")
    parser.add_argument("--vocabulary-k", type=int, default=8, help="Target number of nonverbal vocabulary tokens.")
    parser.add_argument("--max-model-windows", type=int, default=6000, help="Maximum sampled windows for model training/evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        window_s=args.window_s,
        stride_s=args.stride_s,
        min_window_valid_ratio=args.min_window_valid_ratio,
        max_lag_s=args.max_lag_s,
        vocabulary_k=args.vocabulary_k,
        max_model_windows=args.max_model_windows,
    )
    run(config)


if __name__ == "__main__":
    main()
