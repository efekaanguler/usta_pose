#!/usr/bin/env python3
"""CENG488 final-project dyadic interaction analysis pipeline.

This script implements a reproducible parquet-only analysis pipeline for the
processed dyadic pose/gaze datasets. It intentionally separates validation,
feature engineering/windowing, within-pair statistics, behavioral vocabulary,
token sequence analysis, classification, ablations, explainability, and
robustness checks.

Default paths are relative to the repository root but every path is configurable.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover
    pq = None
    _PYARROW_IMPORT_ERROR = exc
else:
    _PYARROW_IMPORT_ERROR = None

from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneGroupOut as LeaveOneSessionOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover
    umap = None

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover
    hdbscan = None

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None

from run_hri_analysis import (
    BASE_COLUMNS,
    FALLBACK_KEYPOINT_COLUMNS,
    QUALITY_COLUMNS,
    RunConfig,
    available_columns,
    build_metadata,
    compute_crosscorr,
    ensure_dir,
    finite_mean,
    finite_ratio,
    finite_std,
    numeric_series,
    point_array,
    prepare_frame_features,
    read_relevant_parquet,
)

RANDOM_SEED = 42
EPS = 1e-9
BODY_ORIENTATION_KPTS = (5, 6, 11, 12)  # shoulders and hips in the whole-body layout used by the existing pipeline


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    window_sizes_s: tuple[float, ...]
    stride_ratio: float = 1.0
    max_model_windows: int = 8000
    max_plot_points: int = 3000
    cluster_k_min: int = 2
    cluster_k_max: int = 10
    random_seed: int = RANDOM_SEED
    min_valid_interaction_ratio: float = 0.3


RESULT_SUBDIRS = [
    "validation",
    "features",
    "statistics",
    "clustering",
    "tokens",
    "classification",
    "figures",
    "tables",
]

META_COLUMNS = {
    "pair_id",
    "dyad_id",
    "session_id",
    "session_label",
    "file_name",
    "order",
    "condition_label",
    "window_size_s",
    "window_index",
    "start_time_s",
    "end_time_s",
    "participant_id",
    "camera_id",
}


def make_result_dirs(output_dir: Path) -> dict[str, Path]:
    ensure_dir(output_dir)
    dirs = {name: output_dir / name for name in RESULT_SUBDIRS}
    for p in dirs.values():
        ensure_dir(p)
    return dirs


def safe_to_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=index)


def savefig(path: Path) -> None:
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def condition_from_order(order: int | float) -> str:
    try:
        return "baseline" if int(order) == 1 else "competitive"
    except Exception:
        return "unknown"


def parse_window_sizes(value: str) -> tuple[float, ...]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise ValueError("At least one window size is required")
    return tuple(out)


def parquet_columns(path: Path) -> list[str]:
    if pq is None:
        raise RuntimeError(f"pyarrow is required to inspect parquet files: {_PYARROW_IMPORT_ERROR}")
    return list(pq.ParquetFile(path).schema_arrow.names)


def is_numeric_dtype_safe(dtype) -> bool:
    try:
        return pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
    except Exception:
        return False


def is_pose_or_gaze_feature(col: str) -> bool:
    low = col.lower()
    if low.startswith("calib_"):
        return False
    return any(key in low for key in ["gaze", "kpt", "pose", "motion", "distance", "angle", "yaw", "pitch", "root"])


def is_continuous_outlier_candidate(col: str, dtype) -> bool:
    low = col.lower()
    if not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
        return False
    if any(flag in low for flag in ["observed", "interpolated", "valid", "score", "count", "schema", "id"]):
        return False
    return is_pose_or_gaze_feature(col)


def participant_from_feature(col: str) -> str:
    if col.startswith("p1_"):
        return "p1"
    if col.startswith("p2_"):
        return "p2"
    return "dyad_or_metadata"


def camera_from_feature(col: str) -> str:
    match = re.search(r"cam(\d+)", col)
    return f"cam{match.group(1)}" if match else "fused_or_unknown"


def finite_stats(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }


def bool_transition_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) < 2:
        return 0
    return int(np.sum(mask[1:] != mask[:-1]))


def shannon_entropy(tokens: list[str] | np.ndarray) -> float:
    if len(tokens) == 0:
        return np.nan
    _, counts = np.unique(tokens, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs + EPS)).sum())


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


# ---------------------------------------------------------------------------
# 1. Dataset validation
# ---------------------------------------------------------------------------


def run_validation(meta: pd.DataFrame, dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    validation_dir = dirs["validation"]
    figures_dir = dirs["figures"] / "validation"

    feature_missing_acc: dict[str, dict[str, float]] = {}
    session_feature_rows = []
    session_rows = []
    participant_rows = []
    pair_feature_rows = []
    camera_rows = []
    timestamp_rows = []
    outlier_feature_acc: dict[str, dict[str, float]] = {}
    outlier_session_rows = []

    for _, mrow in meta.iterrows():
        path = Path(mrow["path"])
        df = pd.read_parquet(path)
        session_label = str(mrow["session_label"])
        pair_id = str(mrow["dyad_id"])
        order = int(mrow["order"]) if pd.notna(mrow["order"]) else np.nan
        condition = condition_from_order(order)

        miss = df.isna().mean()
        for col, ratio in miss.items():
            row = {
                "pair_id": pair_id,
                "session_label": session_label,
                "order": order,
                "condition_label": condition,
                "feature": col,
                "missing_ratio": float(ratio),
                "participant_id": participant_from_feature(col),
                "camera_id": camera_from_feature(col),
            }
            session_feature_rows.append(row)
            acc = feature_missing_acc.setdefault(col, {"missing": 0.0, "total": 0.0})
            acc["missing"] += float(df[col].isna().sum())
            acc["total"] += float(len(df))
            pair_feature_rows.append({**row, "n_rows": len(df)})

        session_rows.append(
            {
                "pair_id": pair_id,
                "session_label": session_label,
                "order": order,
                "condition_label": condition,
                "n_rows": int(len(df)),
                "n_features": int(df.shape[1]),
                "overall_missing_ratio": float(df.isna().to_numpy().mean()),
            }
        )

        for participant in ["p1", "p2", "dyad_or_metadata"]:
            cols = [c for c in df.columns if participant_from_feature(c) == participant]
            if not cols:
                continue
            participant_rows.append(
                {
                    "pair_id": pair_id,
                    "session_label": session_label,
                    "order": order,
                    "condition_label": condition,
                    "participant_id": participant,
                    "n_features": len(cols),
                    "missing_ratio": float(df[cols].isna().to_numpy().mean()),
                }
            )

        for camera in sorted({camera_from_feature(c) for c in df.columns}):
            cols = [c for c in df.columns if camera_from_feature(c) == camera]
            camera_rows.append(
                {
                    "pair_id": pair_id,
                    "session_label": session_label,
                    "order": order,
                    "condition_label": condition,
                    "camera_id": camera,
                    "n_features": len(cols),
                    "missing_ratio": float(df[cols].isna().to_numpy().mean()) if cols else np.nan,
                    "note": "Final parquet is a fused interaction table; per-camera raw streams are not preserved except calibration/gaze camera id fields.",
                }
            )
        for cam_col in ["p1_gaze_cam_id", "p2_gaze_cam_id"]:
            if cam_col in df.columns:
                vc = df[cam_col].astype("Int64").value_counts(dropna=False).reset_index()
                vc.columns = ["camera_value", "count"]
                for _, vcrow in vc.iterrows():
                    camera_rows.append(
                        {
                            "pair_id": pair_id,
                            "session_label": session_label,
                            "order": order,
                            "condition_label": condition,
                            "camera_id": cam_col,
                            "n_features": 1,
                            "missing_ratio": float(df[cam_col].isna().mean()),
                            "camera_value": vcrow["camera_value"],
                            "count": int(vcrow["count"]),
                            "note": "Gaze camera-id distribution from fused table.",
                        }
                    )

        timestamp_rows.append(timestamp_consistency_row(df, mrow))

        numeric_cols = [c for c in df.columns if is_continuous_outlier_candidate(c, df[c].dtype)]
        for col in numeric_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            arr = s.to_numpy(float)
            valid = np.isfinite(arr)
            if valid.sum() < 20:
                continue
            v = arr[valid]
            mean = np.nanmean(v)
            std = np.nanstd(v)
            z_mask = np.zeros_like(arr, dtype=bool)
            if std > EPS:
                z_mask[valid] = np.abs((v - mean) / std) > 4.0
            q1, q3 = np.nanpercentile(v, [25, 75])
            iqr = q3 - q1
            iqr_mask = np.zeros_like(arr, dtype=bool)
            if iqr > EPS:
                iqr_mask[valid] = (v < q1 - 3.0 * iqr) | (v > q3 + 3.0 * iqr)
            z_count = int(z_mask.sum())
            iqr_count = int(iqr_mask.sum())
            outlier_session_rows.append(
                {
                    "pair_id": pair_id,
                    "session_label": session_label,
                    "order": order,
                    "condition_label": condition,
                    "feature": col,
                    "participant_id": participant_from_feature(col),
                    "camera_id": camera_from_feature(col),
                    "valid_count": int(valid.sum()),
                    "z_outlier_count": z_count,
                    "z_outlier_ratio": float(z_count / max(1, valid.sum())),
                    "iqr_outlier_count": iqr_count,
                    "iqr_outlier_ratio": float(iqr_count / max(1, valid.sum())),
                }
            )
            acc = outlier_feature_acc.setdefault(col, {"valid": 0.0, "z": 0.0, "iqr": 0.0})
            acc["valid"] += float(valid.sum())
            acc["z"] += float(z_count)
            acc["iqr"] += float(iqr_count)

    feature_missing_df = pd.DataFrame(
        [
            {
                "feature": col,
                "missing_ratio": acc["missing"] / max(acc["total"], 1.0),
                "missing_count": int(acc["missing"]),
                "total_count": int(acc["total"]),
                "participant_id": participant_from_feature(col),
                "camera_id": camera_from_feature(col),
            }
            for col, acc in feature_missing_acc.items()
        ]
    ).sort_values("missing_ratio", ascending=False)

    session_feature_df = pd.DataFrame(session_feature_rows)
    session_df = pd.DataFrame(session_rows)
    participant_df = pd.DataFrame(participant_rows)
    camera_df = pd.DataFrame(camera_rows)
    timestamp_df = pd.DataFrame(timestamp_rows)
    outlier_feature_df = pd.DataFrame(
        [
            {
                "feature": col,
                "valid_count": int(acc["valid"]),
                "z_outlier_count": int(acc["z"]),
                "z_outlier_ratio": acc["z"] / max(acc["valid"], 1.0),
                "iqr_outlier_count": int(acc["iqr"]),
                "iqr_outlier_ratio": acc["iqr"] / max(acc["valid"], 1.0),
                "participant_id": participant_from_feature(col),
                "camera_id": camera_from_feature(col),
            }
            for col, acc in outlier_feature_acc.items()
        ]
    ).sort_values(["z_outlier_ratio", "iqr_outlier_ratio"], ascending=False)
    outlier_session_df = pd.DataFrame(outlier_session_rows)

    pair_missing_df = (
        session_feature_df.groupby(["pair_id", "feature", "participant_id", "camera_id"], dropna=False)["missing_ratio"]
        .mean()
        .reset_index()
        .sort_values(["pair_id", "missing_ratio"], ascending=[True, False])
    )

    safe_to_csv(feature_missing_df, validation_dir / "missing_by_feature.csv")
    safe_to_csv(session_feature_df, validation_dir / "missing_by_session_feature.csv")
    safe_to_csv(session_df, validation_dir / "missing_by_session.csv")
    safe_to_csv(participant_df, validation_dir / "missing_by_participant.csv")
    safe_to_csv(pair_missing_df, validation_dir / "missing_by_pair_feature.csv")
    safe_to_csv(camera_df, validation_dir / "missing_by_camera.csv")
    safe_to_csv(timestamp_df, validation_dir / "timestamp_frame_consistency.csv")
    safe_to_csv(outlier_feature_df, validation_dir / "outliers_by_feature.csv")
    safe_to_csv(outlier_session_df, validation_dir / "outliers_by_session_feature.csv")

    plot_top_missing(feature_missing_df, figures_dir / "top_missing_features.png")
    plot_session_missing(session_df, figures_dir / "session_missing_ratio.png")
    plot_timestamp_gaps(timestamp_df, figures_dir / "timestamp_gap_summary.png")
    plot_top_outliers(outlier_feature_df, figures_dir / "top_outlier_features.png")

    return {
        "missing_by_feature": feature_missing_df,
        "missing_by_session": session_df,
        "timestamp": timestamp_df,
        "outliers_by_feature": outlier_feature_df,
    }


def timestamp_consistency_row(df: pd.DataFrame, mrow: pd.Series) -> dict:
    t = pd.to_numeric(df.get("timestamp_ms", pd.Series(np.arange(len(df)) * (1000 / 30))), errors="coerce").to_numpy(float)
    frame = pd.to_numeric(df.get("frame_idx", pd.Series(np.arange(len(df)))), errors="coerce").to_numpy(float)
    dt = np.diff(t)
    dframe = np.diff(frame)
    positive_dt = dt[np.isfinite(dt) & (dt > 0)]
    expected_dt = float(np.nanmedian(positive_dt)) if positive_dt.size else np.nan
    gap_threshold = max(100.0, 1.5 * expected_dt) if np.isfinite(expected_dt) else 100.0
    gaps = dt[np.isfinite(dt) & (dt > gap_threshold)]
    return {
        "pair_id": mrow["dyad_id"],
        "session_label": mrow["session_label"],
        "order": int(mrow["order"]) if pd.notna(mrow["order"]) else np.nan,
        "condition_label": condition_from_order(mrow["order"]),
        "n_rows": int(len(df)),
        "timestamp_missing_ratio": float(np.mean(~np.isfinite(t))) if len(t) else np.nan,
        "frame_idx_missing_ratio": float(np.mean(~np.isfinite(frame))) if len(frame) else np.nan,
        "median_dt_ms": expected_dt,
        "min_dt_ms": float(np.nanmin(dt)) if dt.size and np.isfinite(dt).any() else np.nan,
        "max_dt_ms": float(np.nanmax(dt)) if dt.size and np.isfinite(dt).any() else np.nan,
        "nonpositive_timestamp_steps": int(np.sum(np.isfinite(dt) & (dt <= 0))),
        "temporal_gap_count": int(len(gaps)),
        "temporal_gap_threshold_ms": gap_threshold,
        "largest_temporal_gap_ms": float(np.nanmax(gaps)) if len(gaps) else 0.0,
        "frame_gap_count": int(np.sum(np.isfinite(dframe) & (dframe > 1))),
        "nonpositive_frame_steps": int(np.sum(np.isfinite(dframe) & (dframe <= 0))),
    }


def plot_top_missing(df: pd.DataFrame, path: Path) -> None:
    top = df.head(25).iloc[::-1]
    if top.empty:
        return
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["missing_ratio"])
    plt.xlabel("Missing ratio")
    plt.title("Top missing features")
    savefig(path)


def plot_session_missing(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    tmp = df.sort_values(["pair_id", "order"])
    labels = tmp["pair_id"].astype(str) + ":S" + tmp["order"].astype(str)
    plt.figure(figsize=(11, 4.5))
    plt.bar(range(len(tmp)), tmp["overall_missing_ratio"])
    plt.xticks(range(len(tmp)), labels, rotation=90, fontsize=7)
    plt.ylabel("Overall missing ratio")
    plt.title("Missingness by session")
    savefig(path)


def plot_timestamp_gaps(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    tmp = df.sort_values(["pair_id", "order"])
    labels = tmp["pair_id"].astype(str) + ":S" + tmp["order"].astype(str)
    plt.figure(figsize=(11, 4.5))
    plt.bar(range(len(tmp)), tmp["temporal_gap_count"])
    plt.xticks(range(len(tmp)), labels, rotation=90, fontsize=7)
    plt.ylabel("Detected temporal gaps")
    plt.title("Timestamp gap count by session")
    savefig(path)


def plot_top_outliers(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    top = df.sort_values("z_outlier_ratio", ascending=False).head(25).iloc[::-1]
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["z_outlier_ratio"], label="|z|>4")
    plt.xlabel("Outlier ratio")
    plt.title("Top pose/gaze outlier features")
    savefig(path)


# ---------------------------------------------------------------------------
# 2-3. Feature engineering and windowing
# ---------------------------------------------------------------------------


def engineering_columns(path: Path) -> list[str]:
    extra = []
    for person in ("p1", "p2"):
        extra.extend([f"{person}_gaze_cam_id", f"{person}_gaze_yaw", f"{person}_gaze_pitch"])
        for kpt in BODY_ORIENTATION_KPTS:
            extra.extend([f"{person}_kpt{kpt}_world_{axis}" for axis in ("x", "y", "z")])
    requested = list(dict.fromkeys(BASE_COLUMNS + QUALITY_COLUMNS + FALLBACK_KEYPOINT_COLUMNS + extra))
    return available_columns(path, requested)


def read_engineering_parquet(path: Path) -> pd.DataFrame:
    cols = engineering_columns(path)
    if "timestamp_ms" not in cols:
        return read_relevant_parquet(path)
    return pd.read_parquet(path, columns=cols)


def body_axis_angle(df: pd.DataFrame, person: str) -> np.ndarray:
    left_candidates = [5, 11]
    right_candidates = [6, 12]
    angles = []
    for left, right in zip(left_candidates, right_candidates):
        needed = [f"{person}_kpt{left}_world_{axis}" for axis in ("x", "y", "z")] + [f"{person}_kpt{right}_world_{axis}" for axis in ("x", "y", "z")]
        if not all(c in df.columns for c in needed):
            continue
        a = point_array(df, f"{person}_kpt{left}_world")
        b = point_array(df, f"{person}_kpt{right}_world")
        vec = b - a
        angle = np.arctan2(vec[:, 1], vec[:, 0])
        angles.append(angle)
    if not angles:
        return np.full(len(df), np.nan)
    return np.nanmean(np.vstack(angles), axis=0)


def circular_absdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = np.angle(np.exp(1j * (a - b)))
    return np.abs(diff)


def prepare_engineered_frames(path: Path, session_meta: pd.Series, base_config: RunConfig) -> pd.DataFrame:
    raw = read_engineering_parquet(path)
    frame = prepare_frame_features(raw, base_config)
    for person in ("p1", "p2"):
        frame[f"{person}_gaze_yaw"] = numeric_series(raw, f"{person}_gaze_yaw").to_numpy(float) if f"{person}_gaze_yaw" in raw else np.nan
        frame[f"{person}_gaze_pitch"] = numeric_series(raw, f"{person}_gaze_pitch").to_numpy(float) if f"{person}_gaze_pitch" in raw else np.nan
        frame[f"{person}_body_axis_angle"] = body_axis_angle(raw, person)
        frame[f"{person}_gaze_partner_int"] = frame[f"{person}_partner_gaze"].astype(int)
        frame[f"{person}_gaze_task_int"] = frame[f"{person}_task_gaze"].astype(int)
        partner = frame[f"{person}_partner_gaze"].to_numpy(bool)
        task = frame[f"{person}_task_gaze"].to_numpy(bool)
        valid_gaze = frame["frame_gaze_valid"].to_numpy(bool)
        cats = np.full(len(frame), "away_unknown", dtype=object)
        cats[task & ~partner & valid_gaze] = "table_task"
        cats[partner & valid_gaze] = "partner"
        cats[~valid_gaze] = "unknown"
        frame[f"{person}_gaze_category"] = cats
    frame["joint_attention_table"] = frame["p1_task_gaze"].to_numpy(bool) & frame["p2_task_gaze"].to_numpy(bool)
    frame["any_partner_gaze"] = frame["p1_partner_gaze"].to_numpy(bool) | frame["p2_partner_gaze"].to_numpy(bool)
    frame["body_orientation_difference_rad"] = circular_absdiff(
        frame["p1_body_axis_angle"].to_numpy(float), frame["p2_body_axis_angle"].to_numpy(float)
    )
    frame["gaze_yaw_difference_rad"] = circular_absdiff(
        frame["p1_gaze_yaw"].to_numpy(float), frame["p2_gaze_yaw"].to_numpy(float)
    )
    frame["pair_motion_energy"] = np.nanmean(
        np.vstack([frame["p1_motion_energy"].to_numpy(float), frame["p2_motion_energy"].to_numpy(float)]), axis=0
    )
    frame["activity_asymmetry_abs"] = np.abs(frame["p1_motion_energy"].to_numpy(float) - frame["p2_motion_energy"].to_numpy(float)) / (
        frame["p1_motion_energy"].to_numpy(float) + frame["p2_motion_energy"].to_numpy(float) + EPS
    )
    frame["session_label"] = session_meta["session_label"]
    return frame


NUMERIC_FRAME_FEATURES = [
    "root_distance",
    "root_horizontal_distance",
    "vertical_offset_abs",
    "p1_motion_energy",
    "p2_motion_energy",
    "pair_motion_energy",
    "p1_motion_speed",
    "p2_motion_speed",
    "activity_asymmetry_abs",
    "p1_closest_hand_to_other_head",
    "p2_closest_hand_to_other_head",
    "hand_to_hand_min",
    "mutual_gaze_cos",
    "p1_partner_gaze_angle",
    "p2_partner_gaze_angle",
    "p1_task_gaze_angle",
    "p2_task_gaze_angle",
    "p1_gaze_yaw",
    "p2_gaze_yaw",
    "p1_gaze_pitch",
    "p2_gaze_pitch",
    "body_orientation_difference_rad",
    "gaze_yaw_difference_rad",
]

BOOLEAN_FRAME_FEATURES = [
    "frame_pose_valid",
    "frame_gaze_valid",
    "frame_interaction_valid",
    "p1_active",
    "p2_active",
    "both_active",
    "one_active",
    "neither_active",
    "p1_partner_gaze",
    "p2_partner_gaze",
    "any_partner_gaze",
    "mutual_gaze",
    "p1_task_gaze",
    "p2_task_gaze",
    "joint_attention_table",
]


def summarize_engineered_window(w: pd.DataFrame, session_meta: pd.Series, window_size_s: float, window_index: int) -> dict:
    timestamps = w["timestamp_ms"].to_numpy(float)
    order = int(session_meta["order"]) if pd.notna(session_meta["order"]) else np.nan
    row: dict[str, float | int | str | bool] = {
        "pair_id": session_meta["dyad_id"],
        "dyad_id": session_meta["dyad_id"],
        "session_id": session_meta["session_label"],
        "session_label": session_meta["session_label"],
        "file_name": session_meta["file_name"],
        "order": order,
        "condition_label": condition_from_order(order),
        "participant_id": "dyad",
        "camera_id": "fused",
        "window_size_s": float(window_size_s),
        "window_index": int(window_index),
        "start_time_s": float(timestamps[0] / 1000.0),
        "end_time_s": float(timestamps[-1] / 1000.0),
        "n_frames": int(len(w)),
    }

    for col in NUMERIC_FRAME_FEATURES:
        if col not in w.columns:
            continue
        stats_dict = finite_stats(pd.to_numeric(w[col], errors="coerce").to_numpy(float))
        for stat_name, value in stats_dict.items():
            row[f"{col}_{stat_name}"] = value

    for col in BOOLEAN_FRAME_FEATURES:
        if col not in w.columns:
            continue
        mask = w[col].fillna(False).astype(bool).to_numpy()
        row[f"{col}_ratio"] = finite_ratio(mask)
        row[f"{col}_count"] = int(mask.sum())
        row[f"{col}_transition_count"] = bool_transition_count(mask)

    for person in ("p1", "p2"):
        cats = w[f"{person}_gaze_category"].astype(str).to_numpy() if f"{person}_gaze_category" in w else np.array([], dtype=str)
        for cat in ["partner", "table_task", "away_unknown", "unknown"]:
            row[f"{person}_gaze_category_{cat}_ratio"] = float(np.mean(cats == cat)) if len(cats) else np.nan
        row[f"{person}_gaze_category_switch_count"] = int(np.sum(cats[1:] != cats[:-1])) if len(cats) > 1 else 0
        duration_s = max(float((timestamps[-1] - timestamps[0]) / 1000.0), EPS) if len(timestamps) > 1 else window_size_s
        row[f"{person}_gaze_switch_rate"] = row[f"{person}_gaze_category_switch_count"] / max(duration_s, EPS)

    row["gaze_switch_rate_mean"] = finite_mean([row.get("p1_gaze_switch_rate"), row.get("p2_gaze_switch_rate")])
    row["partner_gaze_ratio_mean"] = finite_mean([row.get("p1_gaze_category_partner_ratio"), row.get("p2_gaze_category_partner_ratio")])
    row["table_gaze_ratio_mean"] = finite_mean([row.get("p1_gaze_category_table_task_ratio"), row.get("p2_gaze_category_table_task_ratio")])
    row["unknown_gaze_ratio_mean"] = finite_mean([row.get("p1_gaze_category_unknown_ratio"), row.get("p2_gaze_category_unknown_ratio")])
    row["mutual_gaze_event_ratio"] = row.get("mutual_gaze_ratio", np.nan)
    row["joint_attention_table_ratio"] = row.get("joint_attention_table_ratio", np.nan)
    row["movement_intensity_mean"] = row.get("pair_motion_energy_mean", np.nan)
    row["asymmetric_activity_score"] = row.get("activity_asymmetry_abs_mean", np.nan)

    p1_energy = pd.to_numeric(w["p1_motion_energy"], errors="coerce").to_numpy(float) if "p1_motion_energy" in w else np.array([])
    p2_energy = pd.to_numeric(w["p2_motion_energy"], errors="coerce").to_numpy(float) if "p2_motion_energy" in w else np.array([])
    if len(p1_energy) and len(p2_energy):
        zero, best, lag = compute_crosscorr(p1_energy, p2_energy, timestamps, max_lag_s=min(2.0, window_size_s / 2.0))
    else:
        zero, best, lag = np.nan, np.nan, np.nan
    row["dyadic_motion_synchrony_zero_lag"] = zero
    row["dyadic_motion_synchrony_absmax"] = abs(best) if np.isfinite(best) else np.nan
    row["dyadic_motion_synchrony_lag_s"] = lag

    root = pd.to_numeric(w["root_distance"], errors="coerce").to_numpy(float) if "root_distance" in w else np.array([])
    if len(root) > 1 and np.isfinite([root[0], root[-1]]).all():
        row["lean_in_proxy_root_distance_delta"] = float(root[-1] - root[0])
        row["lean_in_proxy_decreasing_distance"] = float(root[-1] < root[0])
    else:
        row["lean_in_proxy_root_distance_delta"] = np.nan
        row["lean_in_proxy_decreasing_distance"] = np.nan

    return row


def build_window_tables(meta: pd.DataFrame, config: PipelineConfig, dirs: dict[str, Path]) -> dict[float, pd.DataFrame]:
    features_dir = dirs["features"]
    all_window_tables: dict[float, list[pd.DataFrame]] = {w: [] for w in config.window_sizes_s}
    base_config = RunConfig(input_dir=config.input_dir, output_dir=config.output_dir, min_window_valid_ratio=config.min_valid_interaction_ratio)

    for _, mrow in meta.iterrows():
        path = Path(mrow["path"])
        frame = prepare_engineered_frames(path, mrow, base_config)
        if frame.empty:
            continue
        t = frame["timestamp_ms"].to_numpy(float)
        start_t = float(np.nanmin(t))
        end_t = float(np.nanmax(t))
        for window_size in config.window_sizes_s:
            win_ms = window_size * 1000.0
            stride_ms = win_ms * config.stride_ratio
            starts = [start_t] if end_t - start_t < win_ms else list(np.arange(start_t, end_t - win_ms + EPS, stride_ms))
            rows = []
            for idx, ws in enumerate(starts):
                we = ws + win_ms
                w = frame[(frame["timestamp_ms"] >= ws) & (frame["timestamp_ms"] < we)]
                if len(w) < 3:
                    continue
                rows.append(summarize_engineered_window(w, mrow, window_size, idx))
            if rows:
                all_window_tables[window_size].append(pd.DataFrame(rows))

    out: dict[float, pd.DataFrame] = {}
    for window_size, tables in all_window_tables.items():
        df = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        out[window_size] = df
        suffix = window_suffix(window_size)
        df.to_parquet(features_dir / f"window_features_{suffix}.parquet", index=False)
        safe_to_csv(df, features_dir / f"window_features_{suffix}.csv")
        session_df = aggregate_session_table(df)
        safe_to_csv(session_df, features_dir / f"session_features_{suffix}.csv")
        plot_core_feature_distributions(df, dirs["figures"] / "features" / f"core_feature_distributions_{suffix}.png")
    return out


def aggregate_session_table(window_df: pd.DataFrame) -> pd.DataFrame:
    if window_df.empty:
        return pd.DataFrame()
    feature_cols = model_feature_columns(window_df)
    rows = []
    for keys, group in window_df.groupby(["pair_id", "session_id", "order", "condition_label", "window_size_s"], dropna=False):
        pair_id, session_id, order, condition, window_size = keys
        row = {
            "pair_id": pair_id,
            "session_id": session_id,
            "order": int(order) if pd.notna(order) else np.nan,
            "condition_label": condition,
            "window_size_s": float(window_size),
            "n_windows": int(len(group)),
        }
        for col in feature_cols:
            arr = pd.to_numeric(group[col], errors="coerce").to_numpy(float)
            row[f"{col}_session_mean"] = finite_mean(arr)
            row[f"{col}_session_std"] = finite_std(arr)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["pair_id", "order", "session_id"]).reset_index(drop=True)


def window_suffix(window_size: float) -> str:
    return f"{int(window_size)}s" if float(window_size).is_integer() else f"{window_size:g}s"


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in META_COLUMNS or c in {"n_frames", "file_name"}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            valid = pd.to_numeric(df[c], errors="coerce").notna().sum()
            if valid >= max(10, int(0.05 * len(df))):
                cols.append(c)
    return cols


def feature_subset_columns(df: pd.DataFrame, subset: str) -> list[str]:
    cols = model_feature_columns(df)
    if subset == "combined":
        return cols
    gaze_keys = ["gaze", "mutual", "joint_attention", "yaw", "pitch"]
    pose_keys = ["motion", "active", "activity", "distance", "hand", "body", "lean", "root", "synchrony", "vertical", "orientation"]
    if subset == "gaze":
        return [c for c in cols if any(k in c.lower() for k in gaze_keys)]
    if subset == "pose":
        return [c for c in cols if any(k in c.lower() for k in pose_keys) and not any(k in c.lower() for k in gaze_keys)]
    raise ValueError(f"Unknown feature subset: {subset}")


def plot_core_feature_distributions(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    cols = [
        "movement_intensity_mean",
        "partner_gaze_ratio_mean",
        "table_gaze_ratio_mean",
        "mutual_gaze_event_ratio",
        "joint_attention_table_ratio",
        "dyadic_motion_synchrony_absmax",
        "asymmetric_activity_score",
        "lean_in_proxy_root_distance_delta",
    ]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return
    n = len(cols)
    fig, axes = plt.subplots(math.ceil(n / 2), 2, figsize=(11, 3.0 * math.ceil(n / 2)))
    axes = np.asarray(axes).reshape(-1)
    for ax, col in zip(axes, cols):
        baseline = pd.to_numeric(df.loc[df["condition_label"] == "baseline", col], errors="coerce")
        comp = pd.to_numeric(df.loc[df["condition_label"] == "competitive", col], errors="coerce")
        ax.boxplot([baseline.dropna(), comp.dropna()], tick_labels=["baseline", "competitive"], showfliers=False)
        ax.set_title(col)
    for ax in axes[len(cols):]:
        ax.axis("off")
    savefig(path)


# ---------------------------------------------------------------------------
# 4. Within-pair session statistics
# ---------------------------------------------------------------------------


def run_session_statistics(window_tables: dict[float, pd.DataFrame], dirs: dict[str, Path], config: PipelineConfig) -> pd.DataFrame:
    rows = []
    for window_size, window_df in window_tables.items():
        session_df = aggregate_session_table(window_df)
        suffix = window_suffix(window_size)
        if session_df.empty:
            continue
        session_features = [c for c in session_df.columns if c.endswith("_session_mean")]
        for feature in session_features:
            pivot = session_df.pivot_table(index="pair_id", columns="order", values=feature, aggfunc="mean")
            if 1 in pivot.columns and any(c in pivot.columns for c in [2, 3, 4]):
                comp_cols = [c for c in [2, 3, 4] if c in pivot.columns]
                diff = pivot[comp_cols].mean(axis=1) - pivot[1]
                rows.append(paired_stat_row(window_size, "S1_vs_S2S3S4_mean", feature, diff))
            for order in [2, 3, 4]:
                if 1 in pivot.columns and order in pivot.columns:
                    rows.append(paired_stat_row(window_size, f"S1_vs_S{order}", feature, pivot[order] - pivot[1]))
            if all(c in pivot.columns for c in [1, 2, 3, 4]):
                rows.append(friedman_stat_row(window_size, "S1_S2_S3_S4", feature, pivot[[1, 2, 3, 4]]))
        stat_df = pd.DataFrame([r for r in rows if r["window_size_s"] == window_size])
        if not stat_df.empty:
            stat_df["q_value"] = bh_fdr(stat_df["p_value"])
            stat_df["direction"] = np.where(stat_df["mean_difference"] > 0, "higher_in_competitive_or_later", "lower_in_competitive_or_later")
            safe_to_csv(stat_df.sort_values("p_value", na_position="last"), dirs["statistics"] / f"session_statistics_{suffix}.csv")
            plot_top_stat_effects(stat_df, dirs["figures"] / "statistics" / f"top_effects_{suffix}.png")
            leave_one_pair = leave_one_pair_effects(session_df, stat_df, top_n=20)
            safe_to_csv(leave_one_pair, dirs["statistics"] / f"leave_one_pair_influence_{suffix}.csv")
    all_stats = pd.DataFrame(rows)
    if not all_stats.empty:
        all_stats["q_value"] = bh_fdr(all_stats["p_value"])
        all_stats["direction"] = np.where(all_stats["mean_difference"] > 0, "higher_in_competitive_or_later", "lower_in_competitive_or_later")
        safe_to_csv(all_stats.sort_values(["window_size_s", "p_value"], na_position="last"), dirs["statistics"] / "session_statistics_all_windows.csv")
        safe_to_csv(all_stats.sort_values(["window_size_s", "p_value"], na_position="last"), dirs["tables"] / "session_statistics_all_windows.csv")
    return all_stats


def paired_stat_row(window_size: float, contrast: str, feature: str, diff: pd.Series) -> dict:
    values = pd.to_numeric(diff, errors="coerce").dropna().to_numpy(float)
    n = len(values)
    mean_diff = finite_mean(values)
    std_diff = finite_std(values)
    p_wilcoxon = np.nan
    if n >= 3 and not np.allclose(values, 0):
        try:
            p_wilcoxon = float(stats.wilcoxon(values, zero_method="wilcox").pvalue)
        except Exception:
            p_wilcoxon = np.nan
    return {
        "window_size_s": window_size,
        "test": "wilcoxon_signed_rank_plus_sign_permutation",
        "contrast": contrast,
        "feature": feature,
        "n_pairs": n,
        "mean_difference": mean_diff,
        "median_difference": float(np.nanmedian(values)) if n else np.nan,
        "cohen_dz": float(mean_diff / std_diff) if np.isfinite(mean_diff) and np.isfinite(std_diff) and std_diff > EPS else np.nan,
        "rank_biserial": matched_pairs_rank_biserial(values),
        "p_value": p_wilcoxon,
        "permutation_p_value": sign_permutation_p(values),
        "q_value": np.nan,
    }


def friedman_stat_row(window_size: float, contrast: str, feature: str, values_df: pd.DataFrame) -> dict:
    complete = values_df.dropna()
    n = len(complete)
    p_value = np.nan
    statistic = np.nan
    if n >= 3:
        try:
            statistic, p_value = stats.friedmanchisquare(*(complete[c].to_numpy(float) for c in complete.columns))
            p_value = float(p_value)
            statistic = float(statistic)
        except Exception:
            pass
    return {
        "window_size_s": window_size,
        "test": "friedman_S1_to_S4",
        "contrast": contrast,
        "feature": feature,
        "n_pairs": n,
        "mean_difference": float(complete[4].mean() - complete[1].mean()) if n and 4 in complete and 1 in complete else np.nan,
        "median_difference": float((complete[4] - complete[1]).median()) if n and 4 in complete and 1 in complete else np.nan,
        "cohen_dz": np.nan,
        "rank_biserial": np.nan,
        "friedman_statistic": statistic,
        "p_value": p_value,
        "permutation_p_value": np.nan,
        "q_value": np.nan,
    }


def sign_permutation_p(values: np.ndarray, n_perm: int = 10000) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return np.nan
    observed = abs(np.nanmean(values))
    if len(values) <= 14:
        signs = np.array(list(itertools.product([-1, 1], repeat=len(values))), dtype=float)
        sims = np.abs((signs * values).mean(axis=1))
    else:
        rng = np.random.default_rng(RANDOM_SEED)
        signs = rng.choice([-1.0, 1.0], size=(n_perm, len(values)))
        sims = np.abs((signs * values).mean(axis=1))
    return float((np.sum(sims >= observed) + 1) / (len(sims) + 1))


def matched_pairs_rank_biserial(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (np.abs(values) > EPS)]
    if len(values) == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(values))
    pos = ranks[values > 0].sum()
    neg = ranks[values < 0].sum()
    total = ranks.sum()
    return float((pos - neg) / total) if total > 0 else np.nan


def leave_one_pair_effects(session_df: pd.DataFrame, stat_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if session_df.empty or stat_df.empty:
        return pd.DataFrame()
    top = stat_df[stat_df["test"].str.contains("wilcoxon", na=False)].sort_values("p_value", na_position="last").head(top_n)
    rows = []
    for _, srow in top.iterrows():
        feature = srow["feature"]
        contrast = srow["contrast"]
        for held_out in session_df["pair_id"].dropna().unique():
            sub = session_df[session_df["pair_id"] != held_out]
            pivot = sub.pivot_table(index="pair_id", columns="order", values=feature, aggfunc="mean")
            if contrast == "S1_vs_S2S3S4_mean" and 1 in pivot.columns:
                comp_cols = [c for c in [2, 3, 4] if c in pivot.columns]
                if comp_cols:
                    diff = pivot[comp_cols].mean(axis=1) - pivot[1]
                else:
                    continue
            else:
                match = re.search(r"S1_vs_S(\d)", contrast)
                if not match:
                    continue
                order = int(match.group(1))
                if 1 not in pivot.columns or order not in pivot.columns:
                    continue
                diff = pivot[order] - pivot[1]
            rows.append(
                {
                    "held_out_pair_id": held_out,
                    "contrast": contrast,
                    "feature": feature,
                    "mean_difference_without_pair": finite_mean(pd.to_numeric(diff, errors="coerce").dropna().to_numpy(float)),
                    "full_mean_difference": srow["mean_difference"],
                }
            )
    return pd.DataFrame(rows)


def plot_top_stat_effects(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    tmp = df[df["test"].str.contains("wilcoxon", na=False)].copy()
    if tmp.empty:
        return
    tmp["abs_d"] = tmp["cohen_dz"].abs()
    top = tmp.sort_values("abs_d", ascending=False).head(20).iloc[::-1]
    plt.figure(figsize=(9, 7))
    plt.barh(top["feature"], top["cohen_dz"])
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("Cohen's dz")
    plt.title("Largest paired effects")
    savefig(path)


# ---------------------------------------------------------------------------
# 5. Unsupervised vocabulary
# ---------------------------------------------------------------------------


def run_clustering(window_tables: dict[float, pd.DataFrame], dirs: dict[str, Path], config: PipelineConfig) -> dict[float, pd.DataFrame]:
    token_tables = {}
    all_eval_rows = []
    for window_size, df in window_tables.items():
        suffix = window_suffix(window_size)
        if df.empty:
            continue
        features = model_feature_columns(df)
        Xdf = df[features].apply(pd.to_numeric, errors="coerce")
        Xdf = Xdf.loc[:, Xdf.notna().sum() >= max(20, int(0.05 * len(Xdf)))]
        if Xdf.shape[1] < 3 or len(Xdf) < 20:
            continue
        X = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]).fit_transform(Xdf)
        pca = PCA(n_components=min(10, X.shape[1]), random_state=config.random_seed)
        pcs = pca.fit_transform(X)
        pca_df = pd.DataFrame(pcs[:, : min(5, pcs.shape[1])], columns=[f"PC{i+1}" for i in range(min(5, pcs.shape[1]))])
        pca_df = pd.concat([df[["pair_id", "session_id", "order", "condition_label", "window_index"]].reset_index(drop=True), pca_df], axis=1)
        safe_to_csv(pca_df, dirs["clustering"] / f"pca_scores_{suffix}.csv")
        safe_to_csv(
            pd.DataFrame({"component": np.arange(1, len(pca.explained_variance_ratio_) + 1), "explained_variance_ratio": pca.explained_variance_ratio_}),
            dirs["clustering"] / f"pca_explained_variance_{suffix}.csv",
        )
        plot_pca(pca_df, dirs["figures"] / "clustering" / f"pca_condition_{suffix}.png")
        run_optional_embeddings(X, df, suffix, dirs, config)

        eval_df, label_sets = evaluate_clusterers(X, config)
        eval_df["window_size_s"] = window_size
        all_eval_rows.append(eval_df)
        safe_to_csv(eval_df, dirs["clustering"] / f"cluster_quality_{suffix}.csv")
        plot_cluster_quality(eval_df, dirs["figures"] / "clustering" / f"cluster_quality_{suffix}.png")

        best = choose_best_cluster_config(eval_df)
        if best is None:
            continue
        labels = label_sets[best]
        token_df = df.copy()
        token_df["cluster_id"] = labels
        token_df["behavior_token"] = [f"BT{int(l):02d}" if int(l) >= 0 else "NOISE" for l in labels]
        profiles = summarize_cluster_profiles(token_df, Xdf, labels)
        safe_to_csv(profiles, dirs["clustering"] / f"cluster_profiles_{suffix}.csv")
        safe_to_csv(token_df[["pair_id", "session_id", "order", "condition_label", "window_size_s", "window_index", "start_time_s", "end_time_s", "behavior_token", "cluster_id"]], dirs["tokens"] / f"token_assignments_{suffix}.csv")
        token_tables[window_size] = token_df
        plot_token_profile_heatmap(profiles, dirs["figures"] / "clustering" / f"cluster_profiles_{suffix}.png")

    if all_eval_rows:
        all_eval = pd.concat(all_eval_rows, ignore_index=True)
        safe_to_csv(all_eval, dirs["clustering"] / "cluster_quality_all_windows.csv")
        safe_to_csv(all_eval, dirs["tables"] / "cluster_quality_all_windows.csv")
    return token_tables


def evaluate_clusterers(X: np.ndarray, config: PipelineConfig) -> tuple[pd.DataFrame, dict[tuple[str, int | str], np.ndarray]]:
    rows = []
    label_sets: dict[tuple[str, int | str], np.ndarray] = {}
    max_k = min(config.cluster_k_max, max(config.cluster_k_min, len(X) - 1))
    for k in range(config.cluster_k_min, max_k + 1):
        seed_labels = []
        for seed in [0, 1, 2, 3, 4]:
            labels = KMeans(n_clusters=k, random_state=config.random_seed + seed, n_init=20).fit_predict(X)
            seed_labels.append(labels)
        label_sets[("kmeans", k)] = seed_labels[0]
        rows.append(cluster_metric_row("kmeans", k, seed_labels[0], X, stability=pairwise_ari(seed_labels)))

        try:
            gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=config.random_seed, n_init=5)
            labels = gmm.fit_predict(X)
            label_sets[("gmm", k)] = labels
            rows.append(cluster_metric_row("gmm", k, labels, X, stability=np.nan))
        except Exception as exc:
            rows.append({"algorithm": "gmm", "k_or_param": k, "status": "failed", "reason": str(exc)})

        try:
            labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
            label_sets[("agglomerative", k)] = labels
            rows.append(cluster_metric_row("agglomerative", k, labels, X, stability=np.nan))
        except Exception as exc:
            rows.append({"algorithm": "agglomerative", "k_or_param": k, "status": "failed", "reason": str(exc)})

    for eps in [0.7, 1.0, 1.3, 1.6, 2.0]:
        labels = DBSCAN(eps=eps, min_samples=10).fit_predict(X)
        key = ("dbscan", f"eps={eps}")
        label_sets[key] = labels
        rows.append(cluster_metric_row("dbscan", f"eps={eps}", labels, X, stability=np.nan))

    if hdbscan is not None:
        for min_cluster_size in [15, 30, 50]:
            labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X)
            key = ("hdbscan", f"min_cluster_size={min_cluster_size}")
            label_sets[key] = labels
            rows.append(cluster_metric_row("hdbscan", f"min_cluster_size={min_cluster_size}", labels, X, stability=np.nan))
    else:
        rows.append({"algorithm": "hdbscan", "k_or_param": "optional", "status": "skipped", "reason": "hdbscan is not installed"})
    return pd.DataFrame(rows), label_sets


def cluster_metric_row(algorithm: str, param, labels: np.ndarray, X: np.ndarray, stability: float) -> dict:
    labels = np.asarray(labels)
    non_noise = labels >= 0
    unique = sorted(set(labels[non_noise]))
    row = {
        "algorithm": algorithm,
        "k_or_param": param,
        "status": "ok",
        "n_clusters": int(len(unique)),
        "noise_ratio": float(np.mean(labels < 0)),
        "stability_ari_mean": stability,
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
    }
    if len(unique) >= 2 and non_noise.sum() > len(unique):
        sample_size = min(3000, int(non_noise.sum()))
        try:
            row["silhouette"] = float(silhouette_score(X[non_noise], labels[non_noise], sample_size=sample_size, random_state=RANDOM_SEED))
            row["davies_bouldin"] = float(davies_bouldin_score(X[non_noise], labels[non_noise]))
            row["calinski_harabasz"] = float(calinski_harabasz_score(X[non_noise], labels[non_noise]))
        except Exception as exc:
            row["status"] = "metric_failed"
            row["reason"] = str(exc)
    return row


def pairwise_ari(labels_list: list[np.ndarray]) -> float:
    vals = []
    for a, b in itertools.combinations(labels_list, 2):
        vals.append(adjusted_rand_score(a, b))
    return finite_mean(vals)


def choose_best_cluster_config(eval_df: pd.DataFrame) -> tuple[str, int | str] | None:
    ok = eval_df[(eval_df["status"] == "ok") & (eval_df["n_clusters"] >= 2) & (eval_df["noise_ratio"] < 0.6)].copy()
    if ok.empty:
        return None
    ok["score"] = ok["silhouette"].fillna(-1) - 0.05 * ok["noise_ratio"].fillna(0)
    best = ok.sort_values(["score", "calinski_harabasz"], ascending=False).iloc[0]
    return (str(best["algorithm"]), best["k_or_param"])


def summarize_cluster_profiles(token_df: pd.DataFrame, feature_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    global_mean = feature_df.mean(numeric_only=True)
    global_std = feature_df.std(numeric_only=True).replace(0, np.nan)
    rows = []
    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        label = f"BT{int(cluster_id):02d}" if int(cluster_id) >= 0 else "NOISE"
        group_features = feature_df.loc[mask]
        z = ((group_features.mean(numeric_only=True) - global_mean) / global_std).sort_values(key=lambda s: s.abs(), ascending=False)
        top_features = z.head(8)
        row = {
            "cluster_id": int(cluster_id),
            "behavior_token": label,
            "window_count": int(mask.sum()),
            "prevalence": float(mask.mean()),
            "tentative_name": tentative_token_name(group_features.mean(numeric_only=True), global_mean),
            "top_profile_features": "; ".join([f"{idx}:{val:.2f}z" for idx, val in top_features.items() if np.isfinite(val)]),
        }
        for feat, val in top_features.items():
            row[f"z_{feat}"] = float(val)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("cluster_id")


def tentative_token_name(mean: pd.Series, global_mean: pd.Series) -> str:
    parts = []
    motion = mean.get("movement_intensity_mean", mean.get("pair_motion_energy_mean", np.nan))
    motion_base = global_mean.get("movement_intensity_mean", global_mean.get("pair_motion_energy_mean", np.nan))
    partner_gaze = mean.get("partner_gaze_ratio_mean", np.nan)
    table_gaze = mean.get("table_gaze_ratio_mean", np.nan)
    mutual = mean.get("mutual_gaze_event_ratio", np.nan)
    joint = mean.get("joint_attention_table_ratio", np.nan)
    sync = mean.get("dyadic_motion_synchrony_absmax", np.nan)
    asym = mean.get("asymmetric_activity_score", np.nan)
    if np.isfinite(motion) and np.isfinite(motion_base):
        parts.append("high-motion" if motion > 1.25 * motion_base else "low-motion" if motion < 0.75 * motion_base else "moderate-motion")
    if np.isfinite(joint) and joint > 0.35:
        parts.append("joint-table-attention")
    elif np.isfinite(mutual) and mutual > 0.2:
        parts.append("mutual-gaze")
    elif np.isfinite(partner_gaze) and partner_gaze > 0.25:
        parts.append("partner-monitoring")
    elif np.isfinite(table_gaze) and table_gaze > 0.35:
        parts.append("task-gaze")
    else:
        parts.append("low-gaze")
    if np.isfinite(sync) and sync > global_mean.get("dyadic_motion_synchrony_absmax", np.inf):
        parts.append("synchronous")
    if np.isfinite(asym) and asym > global_mean.get("asymmetric_activity_score", np.inf):
        parts.append("asymmetric")
    return "+".join(parts)


def plot_pca(pca_df: pd.DataFrame, path: Path) -> None:
    if pca_df.empty or not {"PC1", "PC2"}.issubset(pca_df.columns):
        return
    plt.figure(figsize=(7, 5))
    colors = {"baseline": "tab:blue", "competitive": "tab:orange"}
    for cond, group in pca_df.groupby("condition_label"):
        plt.scatter(group["PC1"], group["PC2"], s=8, alpha=0.55, label=str(cond), c=colors.get(cond, None))
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of window features")
    plt.legend()
    savefig(path)


def run_optional_embeddings(X: np.ndarray, df: pd.DataFrame, suffix: str, dirs: dict[str, Path], config: PipelineConfig) -> None:
    n = len(X)
    rng = np.random.default_rng(config.random_seed)
    idx = np.arange(n)
    if n > config.max_plot_points:
        idx = np.sort(rng.choice(idx, size=config.max_plot_points, replace=False))
    Xs = X[idx]
    meta = df.iloc[idx][["pair_id", "session_id", "order", "condition_label", "window_index"]].reset_index(drop=True)
    if umap is not None:
        try:
            emb = umap.UMAP(n_neighbors=20, min_dist=0.1, random_state=config.random_seed).fit_transform(Xs)
            out = meta.copy()
            out["UMAP1"] = emb[:, 0]
            out["UMAP2"] = emb[:, 1]
            safe_to_csv(out, dirs["clustering"] / f"umap_scores_{suffix}.csv")
            plot_embedding(out, "UMAP1", "UMAP2", dirs["figures"] / "clustering" / f"umap_condition_{suffix}.png")
        except Exception as exc:
            (dirs["clustering"] / f"umap_skipped_{suffix}.txt").write_text(str(exc), encoding="utf-8")
    else:
        (dirs["clustering"] / f"umap_skipped_{suffix}.txt").write_text("umap-learn is not installed", encoding="utf-8")
    try:
        if len(Xs) >= 50:
            emb = TSNE(n_components=2, perplexity=min(30, max(5, (len(Xs) - 1) // 3)), init="pca", learning_rate="auto", random_state=config.random_seed).fit_transform(Xs)
            out = meta.copy()
            out["TSNE1"] = emb[:, 0]
            out["TSNE2"] = emb[:, 1]
            safe_to_csv(out, dirs["clustering"] / f"tsne_scores_{suffix}.csv")
            plot_embedding(out, "TSNE1", "TSNE2", dirs["figures"] / "clustering" / f"tsne_condition_{suffix}.png")
    except Exception as exc:
        (dirs["clustering"] / f"tsne_skipped_{suffix}.txt").write_text(str(exc), encoding="utf-8")


def plot_embedding(df: pd.DataFrame, x: str, y: str, path: Path) -> None:
    plt.figure(figsize=(7, 5))
    for cond, group in df.groupby("condition_label"):
        plt.scatter(group[x], group[y], s=8, alpha=0.6, label=str(cond))
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{x[:-1]} embedding")
    plt.legend()
    savefig(path)


def plot_cluster_quality(df: pd.DataFrame, path: Path) -> None:
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return
    ok["label"] = ok["algorithm"].astype(str) + ":" + ok["k_or_param"].astype(str)
    ok = ok.sort_values("silhouette", ascending=False).head(25).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(ok["label"], ok["silhouette"])
    plt.xlabel("Silhouette score")
    plt.title("Cluster quality candidates")
    savefig(path)


def plot_token_profile_heatmap(profiles: pd.DataFrame, path: Path) -> None:
    if profiles.empty:
        return
    zcols = [c for c in profiles.columns if c.startswith("z_")]
    if not zcols:
        return
    data = profiles.set_index("behavior_token")[zcols].fillna(0)
    # Keep the most variable profile dimensions for readability.
    keep = data.var(axis=0).sort_values(ascending=False).head(18).index
    data = data[keep]
    plt.figure(figsize=(max(8, 0.5 * len(keep)), max(4, 0.45 * len(data))))
    plt.imshow(data.to_numpy(float), aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    plt.yticks(range(len(data)), data.index)
    plt.xticks(range(len(data.columns)), [c[2:] for c in data.columns], rotation=90, fontsize=7)
    plt.colorbar(label="Cluster z-profile")
    plt.title("Behavioral token profiles")
    savefig(path)


# ---------------------------------------------------------------------------
# 6. Token sequence analysis
# ---------------------------------------------------------------------------


def run_token_sequence_analysis(token_tables: dict[float, pd.DataFrame], dirs: dict[str, Path]) -> pd.DataFrame:
    all_summary = []
    for window_size, df in token_tables.items():
        suffix = window_suffix(window_size)
        if df.empty or "behavior_token" not in df:
            continue
        freq = token_frequencies(df)
        trans = token_transitions(df)
        entropy = token_entropy(df)
        ngrams = token_ngrams(df)
        sim = session_similarity(df)
        divergence = transition_divergence(trans)
        safe_to_csv(freq, dirs["tokens"] / f"token_frequencies_{suffix}.csv")
        safe_to_csv(trans, dirs["tokens"] / f"token_transitions_{suffix}.csv")
        safe_to_csv(entropy, dirs["tokens"] / f"sequence_entropy_{suffix}.csv")
        safe_to_csv(ngrams, dirs["tokens"] / f"frequent_ngrams_{suffix}.csv")
        safe_to_csv(sim, dirs["tokens"] / f"session_similarity_{suffix}.csv")
        safe_to_csv(divergence, dirs["tokens"] / f"baseline_competitive_transition_divergence_{suffix}.csv")
        plot_token_frequencies(freq, dirs["figures"] / "tokens" / f"token_frequencies_{suffix}.png")
        all_summary.append(pd.DataFrame({"window_size_s": [window_size], "transition_js_divergence": [divergence["js_divergence"].iloc[0] if not divergence.empty else np.nan]}))
    summary = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    safe_to_csv(summary, dirs["tokens"] / "token_sequence_summary_all_windows.csv")
    return summary


def token_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby(["window_size_s", "pair_id", "session_id", "order", "condition_label", "behavior_token"]).size().reset_index(name="count")
    totals = out.groupby(["window_size_s", "pair_id", "session_id"])["count"].transform("sum")
    out["frequency"] = out["count"] / totals
    return out


def token_transitions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in df.sort_values(["pair_id", "session_id", "window_index"]).groupby(["pair_id", "session_id", "order", "condition_label", "window_size_s"], dropna=False):
        pair_id, session_id, order, condition, window_size = keys
        toks = group["behavior_token"].astype(str).tolist()
        for a, b in zip(toks[:-1], toks[1:]):
            rows.append({"pair_id": pair_id, "session_id": session_id, "order": order, "condition_label": condition, "window_size_s": window_size, "from_token": a, "to_token": b, "count": 1})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby(["pair_id", "session_id", "order", "condition_label", "window_size_s", "from_token", "to_token"], dropna=False)["count"].sum().reset_index()


def token_entropy(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(["pair_id", "session_id", "order", "condition_label", "window_size_s"], dropna=False):
        pair_id, session_id, order, condition, window_size = keys
        toks = group.sort_values("window_index")["behavior_token"].astype(str).tolist()
        rows.append({"pair_id": pair_id, "session_id": session_id, "order": order, "condition_label": condition, "window_size_s": window_size, "sequence_entropy": shannon_entropy(toks), "n_tokens": len(toks), "n_unique_tokens": len(set(toks))})
    return pd.DataFrame(rows)


def token_ngrams(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(["window_size_s", "condition_label"], dropna=False):
        window_size, condition = keys
        counts: dict[tuple[int, tuple[str, ...]], int] = {}
        for _, session in group.sort_values(["session_id", "window_index"]).groupby("session_id"):
            toks = session["behavior_token"].astype(str).tolist()
            for n in (2, 3):
                for gram in zip(*(toks[i:] for i in range(n))):
                    counts[(n, tuple(gram))] = counts.get((n, tuple(gram)), 0) + 1
        for (n, gram), count in counts.items():
            rows.append({"window_size_s": window_size, "condition_label": condition, "n": n, "ngram": "->".join(gram), "count": count})
    return pd.DataFrame(rows).sort_values(["window_size_s", "condition_label", "n", "count"], ascending=[True, True, True, False]).groupby(["window_size_s", "condition_label", "n"]).head(30).reset_index(drop=True) if rows else pd.DataFrame()


def session_similarity(df: pd.DataFrame) -> pd.DataFrame:
    freq = token_frequencies(df)
    if freq.empty:
        return pd.DataFrame()
    mat = freq.pivot_table(index="session_id", columns="behavior_token", values="frequency", fill_value=0)
    rows = []
    for a, b in itertools.combinations(mat.index, 2):
        va = mat.loc[a].to_numpy(float)
        vb = mat.loc[b].to_numpy(float)
        cos = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + EPS))
        rows.append({"session_a": a, "session_b": b, "cosine_similarity": cos})
    return pd.DataFrame(rows)


def transition_divergence(trans: pd.DataFrame) -> pd.DataFrame:
    if trans.empty:
        return pd.DataFrame()
    cond_counts = trans.groupby(["condition_label", "from_token", "to_token"])["count"].sum().reset_index()
    labels = sorted(set(cond_counts["from_token"] + "->" + cond_counts["to_token"]))
    vectors = {}
    for condition, group in cond_counts.groupby("condition_label"):
        pair_labels = group["from_token"] + "->" + group["to_token"]
        vec = pd.Series(group["count"].to_numpy(float), index=pair_labels).groupby(level=0).sum().reindex(labels, fill_value=0).to_numpy(float)
        vectors[condition] = vec / max(vec.sum(), EPS)
    if "baseline" in vectors and "competitive" in vectors:
        js = float(jensenshannon(vectors["baseline"], vectors["competitive"], base=2.0) ** 2)
        return pd.DataFrame([{"comparison": "baseline_vs_competitive", "js_divergence": js, "n_transition_types": len(labels)}])
    return pd.DataFrame()


def plot_token_frequencies(freq: pd.DataFrame, path: Path) -> None:
    if freq.empty:
        return
    agg = freq.groupby(["condition_label", "behavior_token"])["count"].sum().reset_index()
    pivot = agg.pivot_table(index="behavior_token", columns="condition_label", values="count", fill_value=0)
    pivot = pivot.div(pivot.sum(axis=0), axis=1).fillna(0)
    pivot.plot(kind="bar", figsize=(10, 4.5))
    plt.ylabel("Frequency")
    plt.title("Behavioral token frequency by condition")
    savefig(path)


# ---------------------------------------------------------------------------
# 7-8. Classification and ablation
# ---------------------------------------------------------------------------


def classifier_specs(random_seed: int) -> dict[str, object]:
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2500, class_weight="balanced", random_state=random_seed),
        "random_forest": RandomForestClassifier(n_estimators=220, max_depth=8, min_samples_leaf=8, class_weight="balanced", random_state=random_seed, n_jobs=-1),
        "linear_svm": LinearSVC(class_weight="balanced", random_state=random_seed, max_iter=5000),
    }


def run_classification(window_tables: dict[float, pd.DataFrame], dirs: dict[str, Path], config: PipelineConfig) -> tuple[pd.DataFrame, dict]:
    rows = []
    best_info = {"score": -np.inf}
    for window_size, df in window_tables.items():
        suffix = window_suffix(window_size)
        if df.empty:
            continue
        for subset in ["combined", "gaze", "pose"]:
            features = feature_subset_columns(df, subset)
            if len(features) < 3:
                rows.append({"window_size_s": window_size, "feature_set": subset, "target": "condition", "model": "all", "status": "skipped", "reason": "fewer than 3 usable features"})
                continue
            clf_rows, confusion_outputs, candidate_best = classify_condition_and_order(df, features, subset, window_size, config)
            rows.extend(clf_rows)
            for name, cdf in confusion_outputs.items():
                safe_to_csv(cdf, dirs["classification"] / f"confusion_{name}_{suffix}_{subset}.csv")
                plot_confusion(cdf, dirs["figures"] / "classification" / f"confusion_{name}_{suffix}_{subset}.png")
            if candidate_best and candidate_best["score"] > best_info.get("score", -np.inf):
                best_info = candidate_best
        # Pair identity cannot be evaluated with leave-one-pair-out because the held-out class is unseen during training.
        rows.append(
            {
                "window_size_s": window_size,
                "feature_set": "combined",
                "target": "pair_identity",
                "model": "not_applicable_lopo",
                "status": "not_applicable",
                "reason": "Leave-One-Pair-Out holds out the entire target class, so pair identity classification is undefined under the requested leakage-safe split.",
            }
        )
        pair_control = classify_pair_identity_session_control(df, window_size, config)
        rows.extend(pair_control)
        shuffled = shuffled_label_sanity(df, window_size, config)
        rows.extend(shuffled)
    results = pd.DataFrame(rows)
    safe_to_csv(results, dirs["classification"] / "classification_results_all_windows.csv")
    safe_to_csv(results, dirs["tables"] / "classification_results_all_windows.csv")
    plot_classification_summary(results, dirs["figures"] / "classification" / "classification_summary.png")
    return results, best_info


def classify_condition_and_order(df: pd.DataFrame, features: list[str], subset: str, window_size: float, config: PipelineConfig) -> tuple[list[dict], dict[str, pd.DataFrame], dict | None]:
    data = df.dropna(subset=["pair_id", "order"]).copy()
    if len(data) > config.max_model_windows:
        data = data.sample(config.max_model_windows, random_state=config.random_seed).sort_values(["pair_id", "session_id", "window_index"])
    X = data[features].apply(pd.to_numeric, errors="coerce")
    groups = data["pair_id"].astype(str).to_numpy()
    targets = {
        "condition": (data["order"].astype(int).gt(1)).astype(int).to_numpy(),
        "session_number": data["order"].astype(int).to_numpy(),
    }
    rows = []
    confusions = {}
    best = None
    for target_name, y in targets.items():
        for model_name, estimator in classifier_specs(config.random_seed).items():
            pred, score_mat, classes, status, reason = group_cv_predict(X, y, groups, estimator)
            if status != "ok":
                rows.append({"window_size_s": window_size, "feature_set": subset, "target": target_name, "model": model_name, "status": status, "reason": reason})
                continue
            metrics = classification_metrics(y, pred, score_mat, classes)
            row = {"window_size_s": window_size, "feature_set": subset, "target": target_name, "model": model_name, "status": "ok", **metrics}
            rows.append(row)
            cdf = confusion_to_df(y, pred, labels=sorted(np.unique(y)))
            confusions[f"{target_name}_{model_name}"] = cdf
            if target_name == "condition" and subset == "combined" and model_name != "dummy_most_frequent":
                score = metrics.get("macro_f1", np.nan)
                if np.isfinite(score) and (best is None or score > best["score"]):
                    best = {"score": score, "window_size_s": window_size, "model_name": model_name, "estimator": estimator, "features": features, "data": data, "target": y}
    return rows, confusions, best


def group_cv_predict(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, estimator) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, str, str]:
    logo = LeaveOneGroupOut()
    classes = np.array(sorted(np.unique(y)))
    pred = np.full(len(y), fill_value=classes[0], dtype=classes.dtype)
    score_mat = np.full((len(y), len(classes)), np.nan, dtype=float)
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            return pred, None, classes, "failed", "training fold has fewer than two classes"
        pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", clone(estimator))])
        try:
            pipe.fit(X.iloc[train_idx], y[train_idx])
            pred[test_idx] = pipe.predict(X.iloc[test_idx])
            fold_scores = prediction_scores(pipe, X.iloc[test_idx], classes)
            if fold_scores is not None:
                score_mat[test_idx, :] = fold_scores
        except Exception as exc:
            return pred, None, classes, "failed", str(exc)
    if np.isnan(score_mat).all():
        score_mat = None
    return pred, score_mat, classes, "ok", ""


def prediction_scores(pipe: Pipeline, X: pd.DataFrame, classes: np.ndarray) -> np.ndarray | None:
    model = pipe.named_steps["model"]
    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X)
            model_classes = model.classes_
            out = np.zeros((len(X), len(classes)), dtype=float)
            for i, cls in enumerate(model_classes):
                if cls in classes:
                    out[:, np.where(classes == cls)[0][0]] = proba[:, i]
            return out
        except Exception:
            pass
    if hasattr(pipe, "decision_function"):
        try:
            scores = pipe.decision_function(X)
            if scores.ndim == 1:
                return np.column_stack([-scores, scores]) if len(classes) == 2 else None
            out = np.zeros((len(X), len(classes)), dtype=float)
            model_classes = model.classes_
            for i, cls in enumerate(model_classes):
                if cls in classes:
                    out[:, np.where(classes == cls)[0][0]] = scores[:, i]
            return out
        except Exception:
            return None
    return None


def classification_metrics(y: np.ndarray, pred: np.ndarray, score_mat: np.ndarray | None, classes: np.ndarray) -> dict:
    out = {
        "n_samples": int(len(y)),
        "n_classes": int(len(np.unique(y))),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "roc_auc": np.nan,
    }
    if score_mat is not None and not np.isnan(score_mat).all():
        try:
            if len(classes) == 2:
                out["roc_auc"] = float(roc_auc_score(y, score_mat[:, 1]))
            else:
                out["roc_auc"] = float(roc_auc_score(y, score_mat, multi_class="ovr", average="macro", labels=classes))
        except Exception:
            pass
    return out


def confusion_to_df(y: np.ndarray, pred: np.ndarray, labels: list[int]) -> pd.DataFrame:
    mat = confusion_matrix(y, pred, labels=labels)
    rows = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            rows.append({"true_label": true_label, "pred_label": pred_label, "count": int(mat[i, j])})
    return pd.DataFrame(rows)


def plot_confusion(cdf: pd.DataFrame, path: Path) -> None:
    if cdf.empty:
        return
    pivot = cdf.pivot_table(index="true_label", columns="pred_label", values="count", fill_value=0)
    plt.figure(figsize=(5, 4))
    plt.imshow(pivot.to_numpy(), cmap="Blues")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar(label="Count")
    plt.title("Confusion matrix")
    savefig(path)


def classify_pair_identity_session_control(df: pd.DataFrame, window_size: float, config: PipelineConfig) -> list[dict]:
    data = df.dropna(subset=["pair_id", "session_id"]).copy()
    features = feature_subset_columns(data, "combined")
    if len(features) < 3:
        return []
    if len(data) > config.max_model_windows:
        data = data.sample(config.max_model_windows, random_state=config.random_seed).sort_values(["pair_id", "session_id", "window_index"])
    X = data[features].apply(pd.to_numeric, errors="coerce")
    y = pd.factorize(data["pair_id"].astype(str))[0]
    groups = data["session_id"].astype(str).to_numpy()
    rows = []
    for model_name, estimator in {"dummy_most_frequent": DummyClassifier(strategy="most_frequent"), "random_forest": RandomForestClassifier(n_estimators=160, max_depth=8, min_samples_leaf=5, random_state=config.random_seed, n_jobs=-1)}.items():
        pred, score_mat, classes, status, reason = group_cv_predict(X, y, groups, estimator)
        row = {"window_size_s": window_size, "feature_set": "combined", "target": "pair_identity_session_group_control", "model": model_name, "status": status, "reason": reason}
        if status == "ok":
            row.update(classification_metrics(y, pred, score_mat, classes))
        rows.append(row)
    return rows


def shuffled_label_sanity(df: pd.DataFrame, window_size: float, config: PipelineConfig) -> list[dict]:
    data = df.dropna(subset=["pair_id", "order"]).copy()
    features = feature_subset_columns(data, "combined")
    if len(features) < 3:
        return []
    if len(data) > config.max_model_windows:
        data = data.sample(config.max_model_windows, random_state=config.random_seed).sort_values(["pair_id", "session_id", "window_index"])
    X = data[features].apply(pd.to_numeric, errors="coerce")
    y = data["order"].astype(int).gt(1).astype(int).to_numpy()
    rng = np.random.default_rng(config.random_seed)
    shuffled = rng.permutation(y)
    groups = data["pair_id"].astype(str).to_numpy()
    rows = []
    for model_name, estimator in {"logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.random_seed), "random_forest": RandomForestClassifier(n_estimators=160, max_depth=8, min_samples_leaf=8, random_state=config.random_seed, n_jobs=-1)}.items():
        pred, score_mat, classes, status, reason = group_cv_predict(X, shuffled, groups, estimator)
        row = {"window_size_s": window_size, "feature_set": "combined", "target": "condition_shuffled_label_sanity", "model": model_name, "status": status, "reason": reason}
        if status == "ok":
            row.update(classification_metrics(shuffled, pred, score_mat, classes))
        rows.append(row)
    return rows


def plot_classification_summary(results: pd.DataFrame, path: Path) -> None:
    ok = results[(results["status"] == "ok") & (results["target"] == "condition")].copy()
    if ok.empty:
        return
    ok["label"] = ok["window_size_s"].astype(str) + "s/" + ok["feature_set"] + "/" + ok["model"]
    ok = ok.sort_values("macro_f1", ascending=False).head(25).iloc[::-1]
    plt.figure(figsize=(11, 7))
    plt.barh(ok["label"], ok["macro_f1"])
    plt.xlabel("Macro F1")
    plt.title("Condition classification performance")
    savefig(path)


# ---------------------------------------------------------------------------
# 9. Explainability
# ---------------------------------------------------------------------------


def run_explainability(best_info: dict, dirs: dict[str, Path], config: PipelineConfig) -> pd.DataFrame:
    if not best_info or best_info.get("score", -np.inf) == -np.inf:
        out = pd.DataFrame([{"status": "skipped", "reason": "No successful condition classifier"}])
        safe_to_csv(out, dirs["classification"] / "permutation_feature_importance.csv")
        return out
    data = best_info["data"]
    features = best_info["features"]
    y = best_info["target"]
    estimator = clone(best_info["estimator"])
    X = data[features].apply(pd.to_numeric, errors="coerce")
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", estimator)])
    pipe.fit(X, y)
    result = permutation_importance(pipe, X, y, n_repeats=12, random_state=config.random_seed, scoring="f1_macro", n_jobs=-1)
    imp = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "window_size_s": best_info["window_size_s"],
            "model": best_info["model_name"],
        }
    ).sort_values("importance_mean", ascending=False)
    safe_to_csv(imp, dirs["classification"] / "permutation_feature_importance.csv")
    safe_to_csv(imp.head(30), dirs["tables"] / "top_discriminative_features.csv")
    plot_feature_importance(imp, dirs["figures"] / "classification" / "permutation_feature_importance.png")
    plot_top_feature_condition_differences(data, imp.head(8)["feature"].tolist(), dirs["figures"] / "classification" / "top_feature_condition_differences.png")

    if shap is not None and best_info["model_name"] == "random_forest":
        try:
            # SHAP is optional and computed on a sample to keep the pipeline lightweight.
            sample = X.sample(min(1000, len(X)), random_state=config.random_seed)
            transformed = pipe.named_steps["scaler"].transform(pipe.named_steps["imputer"].transform(sample))
            explainer = shap.TreeExplainer(pipe.named_steps["model"])
            vals = explainer.shap_values(transformed)
            arr = vals[1] if isinstance(vals, list) and len(vals) > 1 else vals
            shap_imp = pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(arr).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
            safe_to_csv(shap_imp, dirs["classification"] / "shap_feature_importance.csv")
        except Exception as exc:
            (dirs["classification"] / "shap_skipped.txt").write_text(str(exc), encoding="utf-8")
    else:
        (dirs["classification"] / "shap_skipped.txt").write_text("SHAP unavailable or best model is not random forest; permutation importance was used.", encoding="utf-8")
    return imp


def plot_feature_importance(imp: pd.DataFrame, path: Path) -> None:
    if imp.empty:
        return
    top = imp.head(25).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"].fillna(0))
    plt.xlabel("Permutation importance, macro F1 decrease")
    plt.title("Top discriminative features")
    savefig(path)


def plot_top_feature_condition_differences(data: pd.DataFrame, features: list[str], path: Path) -> None:
    if not features:
        return
    n = len(features)
    fig, axes = plt.subplots(math.ceil(n / 2), 2, figsize=(11, 3.0 * math.ceil(n / 2)))
    axes = np.asarray(axes).reshape(-1)
    for ax, feat in zip(axes, features):
        vals = [pd.to_numeric(data.loc[data["condition_label"] == cond, feat], errors="coerce").dropna() for cond in ["baseline", "competitive"]]
        ax.boxplot(vals, tick_labels=["baseline", "competitive"], showfliers=False)
        ax.set_title(feat, fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    savefig(path)


# ---------------------------------------------------------------------------
# 10-11. Summary
# ---------------------------------------------------------------------------


def write_final_summary(
    config: PipelineConfig,
    meta: pd.DataFrame,
    validation_outputs: dict[str, pd.DataFrame],
    stats_df: pd.DataFrame,
    classification_df: pd.DataFrame,
    token_summary: pd.DataFrame,
    importance_df: pd.DataFrame,
    dirs: dict[str, Path],
) -> None:
    lines = ["# CENG488 Final Project Analysis Summary", ""]
    lines.append("## Dataset")
    lines.append(f"- Input directory: `{config.input_dir}`")
    lines.append(f"- Sessions/parquets: {len(meta)}")
    lines.append(f"- Inferred pairs: {meta['dyad_id'].nunique() if 'dyad_id' in meta else 'unknown'}")
    lines.append(f"- Window sizes: {', '.join(str(w) + 's' for w in config.window_sizes_s)}")
    if "row_count" in meta:
        lines.append(f"- Total frame rows from parquet metadata: {int(pd.to_numeric(meta['row_count'], errors='coerce').sum())}")
    lines.append("")

    missing = validation_outputs.get("missing_by_session", pd.DataFrame())
    if not missing.empty:
        lines.append("## Validation Findings")
        lines.append(f"- Mean overall missing ratio across sessions: {missing['overall_missing_ratio'].mean():.4f}")
        lines.append(f"- Worst session missing ratio: {missing['overall_missing_ratio'].max():.4f}")
    timestamp = validation_outputs.get("timestamp", pd.DataFrame())
    if not timestamp.empty:
        lines.append(f"- Total detected temporal gaps: {int(timestamp['temporal_gap_count'].sum())}")
        lines.append(f"- Largest temporal gap: {timestamp['largest_temporal_gap_ms'].max():.2f} ms")
    outliers = validation_outputs.get("outliers_by_feature", pd.DataFrame())
    if not outliers.empty:
        top_out = outliers.sort_values("z_outlier_ratio", ascending=False).head(5)
        lines.append("- Highest z-score outlier features:")
        for _, row in top_out.iterrows():
            lines.append(f"  - `{row['feature']}`: {row['z_outlier_ratio']:.4f}")
    lines.append("")

    lines.append("## Main Session Statistics")
    if stats_df.empty:
        lines.append("- No statistical results were produced.")
    else:
        paired = stats_df[stats_df["test"].astype(str).str.contains("wilcoxon", na=False)].copy()
        paired = paired.sort_values("p_value", na_position="last").head(12)
        if paired.empty:
            lines.append("- No paired Wilcoxon contrasts were available.")
        else:
            lines.append("- Top exploratory paired contrasts by raw p-value. Treat q-values and leave-one-pair sensitivity as the stronger evidence filters:")
            for _, row in paired.iterrows():
                lines.append(f"  - {row['contrast']} / `{row['feature']}` ({row['window_size_s']}s): mean diff={row['mean_difference']:.4g}, dz={row['cohen_dz']:.3f}, p={row['p_value']:.4f}, q={row['q_value']:.4f}")
    lines.append("")

    lines.append("## Behavioral Vocabulary")
    cluster_quality = pd.read_csv(dirs["clustering"] / "cluster_quality_all_windows.csv") if (dirs["clustering"] / "cluster_quality_all_windows.csv").exists() else pd.DataFrame()
    if not cluster_quality.empty:
        ok = cluster_quality[cluster_quality["status"] == "ok"].sort_values("silhouette", ascending=False).head(5)
        lines.append("- Best clustering candidates by silhouette:")
        for _, row in ok.iterrows():
            lines.append(f"  - {row['window_size_s']}s `{row['algorithm']}` {row['k_or_param']}: silhouette={row['silhouette']:.3f}, DB={row['davies_bouldin']:.3f}, CH={row['calinski_harabasz']:.1f}, stability={row.get('stability_ari_mean', np.nan):.3f}")
    if not token_summary.empty:
        for _, row in token_summary.iterrows():
            lines.append(f"- Token transition JS divergence baseline vs competitive at {row['window_size_s']}s: {row['transition_js_divergence']:.4f}")
    lines.append("")

    lines.append("## Classification and Ablation")
    if classification_df.empty:
        lines.append("- No classification results were produced.")
    else:
        ok = classification_df[(classification_df["status"] == "ok") & (classification_df["target"] == "condition")].sort_values("macro_f1", ascending=False).head(10)
        lines.append("- Best baseline-vs-competitive classifiers under Leave-One-Pair-Out CV:")
        for _, row in ok.iterrows():
            lines.append(f"  - {row['window_size_s']}s `{row['feature_set']}` `{row['model']}`: accuracy={row['accuracy']:.3f}, macro F1={row['macro_f1']:.3f}, ROC-AUC={row['roc_auc']:.3f}")
        pair_note = classification_df[classification_df["target"] == "pair_identity"]
        if not pair_note.empty:
            lines.append("- Pair identity under Leave-One-Pair-Out is reported as not applicable because the held-out pair is an unseen class; a separate session-grouped control is saved for pair-signature diagnostics.")
    lines.append("")

    lines.append("## Explainability")
    if importance_df.empty or "feature" not in importance_df:
        lines.append("- Permutation importance was not available.")
    else:
        lines.append("- Top permutation-importance features for the best condition classifier:")
        for _, row in importance_df.head(10).iterrows():
            lines.append(f"  - `{row['feature']}`: {row['importance_mean']:.4f} +/- {row['importance_std']:.4f}")
    lines.append("")

    lines.append("## Limitations and Warnings")
    lines.append("- The final parquet files are fused interaction tables, not raw camera streams; per-camera validation is limited to calibration fields and gaze camera-id metadata.")
    lines.append("- Session 1 differs from Sessions 2-4 in both competitive pressure and task familiarity, so S1-vs-competitive contrasts cannot isolate competition alone.")
    lines.append("- Frame/window rows are not independent; statistical claims should prioritize within-pair tests, leave-one-pair-out validation, and robustness checks.")
    lines.append("- Unsupervised tokens are behavioral descriptors learned from this dataset, not a universal nonverbal vocabulary.")
    lines.append("- Findings with raw p-values but weak q-values or high leave-one-pair sensitivity should be described as exploratory.")

    out_path = config.output_dir / "FINAL_ANALYSIS_SUMMARY.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (dirs["tables"] / "FINAL_ANALYSIS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(config: PipelineConfig) -> None:
    dirs = make_result_dirs(config.output_dir)
    meta, warnings_out = build_metadata(config.input_dir)
    safe_to_csv(meta, dirs["validation"] / "dataset_metadata.csv")
    (config.output_dir / "pipeline_config.json").write_text(json.dumps({
        "input_dir": str(config.input_dir),
        "output_dir": str(config.output_dir),
        "window_sizes_s": list(config.window_sizes_s),
        "stride_ratio": config.stride_ratio,
        "max_model_windows": config.max_model_windows,
        "random_seed": config.random_seed,
        "warnings": warnings_out,
    }, indent=2), encoding="utf-8")

    validation_outputs = run_validation(meta, dirs)
    window_tables = build_window_tables(meta, config, dirs)
    stats_df = run_session_statistics(window_tables, dirs, config)
    token_tables = run_clustering(window_tables, dirs, config)
    token_summary = run_token_sequence_analysis(token_tables, dirs)
    classification_df, best_info = run_classification(window_tables, dirs, config)
    importance_df = run_explainability(best_info, dirs, config)
    write_final_summary(config, meta, validation_outputs, stats_df, classification_df, token_summary, importance_df, dirs)
    print(f"Wrote CENG488 final-project results to {config.output_dir}")
    print(f"Sessions={len(meta)} pairs={meta['dyad_id'].nunique()} windows={sum(len(v) for v in window_tables.values())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CENG488 dyadic interaction final analysis pipeline.")
    parser.add_argument("--input-dir", type=Path, default=Path("usta_pose/testing/final_dataset_parquets"))
    parser.add_argument("--output-dir", type=Path, default=Path("usta_pose/analiz/efe/results"))
    parser.add_argument("--window-sizes", type=str, default="1,2,5", help="Comma-separated window sizes in seconds.")
    parser.add_argument("--stride-ratio", type=float, default=1.0, help="Stride/window ratio. 1.0 gives non-overlapping windows.")
    parser.add_argument("--max-model-windows", type=int, default=8000)
    parser.add_argument("--max-plot-points", type=int, default=3000)
    parser.add_argument("--cluster-k-min", type=int, default=2)
    parser.add_argument("--cluster-k-max", type=int, default=10)
    parser.add_argument("--min-valid-interaction-ratio", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        window_sizes_s=parse_window_sizes(args.window_sizes),
        stride_ratio=args.stride_ratio,
        max_model_windows=args.max_model_windows,
        max_plot_points=args.max_plot_points,
        cluster_k_min=args.cluster_k_min,
        cluster_k_max=args.cluster_k_max,
        min_valid_interaction_ratio=args.min_valid_interaction_ratio,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        run_pipeline(config)


if __name__ == "__main__":
    main()
