#!/usr/bin/env python3
"""Build an anonymized individual-word nonverbal conversation layer.

The earlier HRI pipeline creates dyadic states. This script creates individual
nonverbal words from person-local and partner-relative features, then builds a
role-anonymized conversation stream such as:

    person_A:IW04 -> person_B:IW06

`person_A` and `person_B` are arbitrary within-dyad slots for readability only.
The learned vocabulary and response statistics never use p1/p2 as semantic
identity features.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_hri_analysis import (
    RunConfig,
    build_metadata,
    ensure_dir,
    finite_mean,
    finite_ratio,
    finite_std,
    point_array,
    prepare_frame_features,
    read_relevant_parquet,
    speed_from_points,
)

RANDOM_SEED = 42
EPS = 1e-9
LEFT_WRIST = 9
RIGHT_WRIST = 10

WORD_FEATURES = [
    "motion_energy_mean",
    "motion_energy_max",
    "motion_active_ratio",
    "motion_burst_count",
    "left_wrist_speed_mean",
    "right_wrist_speed_mean",
    "dominant_hand_speed_mean",
    "left_hand_to_other_head_mean",
    "right_hand_to_other_head_mean",
    "closest_hand_to_other_head_mean",
    "closest_hand_to_other_head_min",
    "left_hand_reach_delta",
    "right_hand_reach_delta",
    "closest_hand_reach_delta",
    "partner_gaze_ratio",
    "task_gaze_ratio",
    "gaze_switch_rate",
    "root_distance_mean",
    "partner_active_ratio",
]

MODEL_CONTEXT_COLUMNS = [
    "actor_word",
    "other_word_same_window",
    "prev_actor_word",
]


def bool_array(values) -> np.ndarray:
    return np.asarray(values, dtype=bool)


def finite_max(values) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return np.nan
    return float(np.nanmax(arr))


def finite_min(values) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return np.nan
    return float(np.nanmin(arr))


def count_boolean_bursts(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return 0
    starts = mask & np.r_[True, ~mask[:-1]]
    return int(starts.sum())


def switch_rate(mask: np.ndarray, timestamps_ms: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) < 2:
        return np.nan
    duration_s = max(float((timestamps_ms[-1] - timestamps_ms[0]) / 1000.0), EPS)
    return float(np.sum(mask[1:] != mask[:-1]) / duration_s)


def add_hand_speeds(frame: pd.DataFrame, raw_df: pd.DataFrame, timestamps_ms: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    for side, kpt in (("left", LEFT_WRIST), ("right", RIGHT_WRIST)):
        for person in ("p1", "p2"):
            prefix = f"{person}_kpt{kpt}_world"
            if all(f"{prefix}_{axis}" in raw_df.columns for axis in ("x", "y", "z")):
                out[f"{person}_{side}_wrist_speed"] = speed_from_points(point_array(raw_df, prefix), timestamps_ms)
            else:
                out[f"{person}_{side}_wrist_speed"] = np.nan
    return out


def prepare_session_frames(path: Path, base_config: RunConfig) -> pd.DataFrame:
    raw = read_relevant_parquet(path)
    frame = prepare_frame_features(raw, base_config)
    timestamps = frame["timestamp_ms"].to_numpy(float)
    frame = add_hand_speeds(frame, raw, timestamps)
    return frame


def summarize_person_window(w: pd.DataFrame, session_meta: pd.Series, window_index: int, person: str, slot: str) -> dict:
    other = "p2" if person == "p1" else "p1"
    timestamps = w["timestamp_ms"].to_numpy(float)
    motion = w[f"{person}_motion_energy"].to_numpy(float)
    active = bool_array(w[f"{person}_active"])
    partner_active = bool_array(w[f"{other}_active"])
    partner_gaze = bool_array(w[f"{person}_partner_gaze"])
    task_gaze = bool_array(w[f"{person}_task_gaze"])
    left_speed = w[f"{person}_left_wrist_speed"].to_numpy(float)
    right_speed = w[f"{person}_right_wrist_speed"].to_numpy(float)
    left_hand = w[f"{person}_closest_hand_to_other_head"].to_numpy(float)
    right_hand = left_hand.copy()

    # Use side-specific distance columns when available; otherwise closest distance
    # remains a useful partner-relative action proxy.
    if person == "p1":
        left_col = "p1_left_wrist_to_p2_head_distance"
        right_col = "p1_right_wrist_to_p2_head_distance"
    else:
        left_col = "p2_left_wrist_to_p1_head_distance"
        right_col = "p2_right_wrist_to_p1_head_distance"
    # These columns are not carried by prepared frames, so they are injected in
    # build_person_windows when present. If absent, fall back to closest hand.
    if left_col in w.columns:
        left_hand = w[left_col].to_numpy(float)
    if right_col in w.columns:
        right_hand = w[right_col].to_numpy(float)
    closest_hand = np.fmin(left_hand, right_hand)

    def delta(arr: np.ndarray) -> float:
        if len(arr) < 2 or not np.isfinite([arr[0], arr[-1]]).all():
            return np.nan
        return float(arr[-1] - arr[0])

    return {
        "session_label": session_meta["session_label"],
        "file_name": session_meta["file_name"],
        "dyad_id": session_meta["dyad_id"],
        "order": int(session_meta["order"]) if pd.notna(session_meta["order"]) else np.nan,
        "window_index": window_index,
        "window_start_ms": float(w["timestamp_ms"].iloc[0]),
        "window_end_ms": float(w["timestamp_ms"].iloc[-1]),
        "window_mid_ms": float((w["timestamp_ms"].iloc[0] + w["timestamp_ms"].iloc[-1]) / 2.0),
        "anon_person_slot": slot,
        "anon_person_id": f"{session_meta['dyad_id']}_{slot}",
        "source_side_audit": person,
        "n_frames": int(len(w)),
        "valid_interaction_ratio": finite_ratio(w["frame_interaction_valid"]),
        "motion_energy_mean": finite_mean(motion),
        "motion_energy_max": finite_max(motion),
        "motion_active_ratio": finite_ratio(active),
        "motion_burst_count": count_boolean_bursts(active),
        "left_wrist_speed_mean": finite_mean(left_speed),
        "right_wrist_speed_mean": finite_mean(right_speed),
        "dominant_hand_speed_mean": finite_mean(np.fmax(left_speed, right_speed)),
        "left_hand_to_other_head_mean": finite_mean(left_hand),
        "right_hand_to_other_head_mean": finite_mean(right_hand),
        "closest_hand_to_other_head_mean": finite_mean(closest_hand),
        "closest_hand_to_other_head_min": finite_min(closest_hand),
        "left_hand_reach_delta": delta(left_hand),
        "right_hand_reach_delta": delta(right_hand),
        "closest_hand_reach_delta": delta(closest_hand),
        "partner_gaze_ratio": finite_ratio(partner_gaze),
        "task_gaze_ratio": finite_ratio(task_gaze),
        "gaze_switch_rate": switch_rate(partner_gaze | task_gaze, timestamps),
        "root_distance_mean": finite_mean(w["root_distance"].to_numpy(float)),
        "partner_active_ratio": finite_ratio(partner_active),
    }


def inject_side_distance_columns(frame: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in [
        "p1_left_wrist_to_p2_head_distance",
        "p1_right_wrist_to_p2_head_distance",
        "p2_left_wrist_to_p1_head_distance",
        "p2_right_wrist_to_p1_head_distance",
    ]:
        if col in raw_df.columns:
            out[col] = pd.to_numeric(raw_df[col], errors="coerce").to_numpy(float)
    return out


def build_person_windows_for_session(path: Path, session_meta: pd.Series, window_s: float, stride_s: float) -> pd.DataFrame:
    base_config = RunConfig(input_dir=path.parent, output_dir=Path("."), window_s=window_s, stride_s=stride_s)
    raw = read_relevant_parquet(path)
    frame = prepare_frame_features(raw, base_config)
    frame = add_hand_speeds(frame, raw, frame["timestamp_ms"].to_numpy(float))
    frame = inject_side_distance_columns(frame, raw)
    if frame.empty:
        return pd.DataFrame()
    start_t = float(np.nanmin(frame["timestamp_ms"]))
    end_t = float(np.nanmax(frame["timestamp_ms"]))
    win_ms = window_s * 1000.0
    stride_ms = stride_s * 1000.0
    if not np.isfinite(start_t) or not np.isfinite(end_t):
        return pd.DataFrame()
    starts = [start_t] if end_t - start_t < win_ms else list(np.arange(start_t, end_t - win_ms + stride_ms, stride_ms))
    rows = []
    for idx, ws in enumerate(starts):
        w = frame[(frame["timestamp_ms"] >= ws) & (frame["timestamp_ms"] < ws + win_ms)]
        if len(w) < 3:
            continue
        rows.append(summarize_person_window(w, session_meta, idx, "p1", "person_A"))
        rows.append(summarize_person_window(w, session_meta, idx, "p2", "person_B"))
    return pd.DataFrame(rows)


def build_individual_windows(meta: pd.DataFrame, window_s: float, stride_s: float) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    warnings = []
    for _, session_meta in meta.iterrows():
        path = Path(session_meta["path"])
        try:
            person_windows = build_person_windows_for_session(path, session_meta, window_s, stride_s)
            if not person_windows.empty:
                rows.append(person_windows)
        except Exception as exc:
            warnings.append(f"Failed individual-word extraction for {path.name}: {exc}")
    if not rows:
        return pd.DataFrame(), warnings
    return pd.concat(rows, ignore_index=True), warnings


def discover_individual_words(person_df: pd.DataFrame, k: int, min_valid_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if person_df.empty:
        return person_df, pd.DataFrame(), {"status": "skipped", "reason": "no person windows"}
    use = person_df[person_df["valid_interaction_ratio"] >= min_valid_ratio].copy()
    if len(use) < max(100, 0.2 * len(person_df)):
        use = person_df[person_df["valid_interaction_ratio"] >= min(0.3, min_valid_ratio)].copy()
    if use.empty:
        use = person_df.copy()
    features = [c for c in WORD_FEATURES if c in use.columns and use[c].notna().sum() >= max(10, int(0.02 * len(use)))]
    if len(features) < 4 or len(use) < 20:
        return person_df, pd.DataFrame(), {"status": "skipped", "reason": "not enough person windows/features"}
    Xraw = use[features].apply(pd.to_numeric, errors="coerce")
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])
    X = pipe.fit_transform(Xraw)
    k = max(2, min(int(k), len(use) - 1))
    model = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=30, batch_size=1024)
    labels = model.fit_predict(X)
    use["word_id"] = labels
    use["word"] = [f"IW{label:02d}" for label in labels]

    centers = pipe.named_steps["scaler"].inverse_transform(model.cluster_centers_)
    centers_df = pd.DataFrame(centers, columns=features)
    med = Xraw.median(numeric_only=True)
    summary_rows = []
    for word_id in sorted(np.unique(labels)):
        group = use[use["word_id"] == word_id]
        center = centers_df.iloc[word_id]
        row = {
            "word_id": int(word_id),
            "word": f"IW{word_id:02d}",
            "description": describe_individual_word(center, med),
            "window_count": int(len(group)),
            "prevalence": float(len(group) / len(use)),
            "mean_order": finite_mean(group["order"].to_numpy(float)),
        }
        for order in sorted(use["order"].dropna().unique()):
            denom = max(1, int((use["order"] == order).sum()))
            row[f"prevalence_order{int(order)}"] = float(((use["order"] == order) & (use["word_id"] == word_id)).sum() / denom)
        for feat in features:
            row[f"centroid_{feat}"] = float(center[feat]) if np.isfinite(center[feat]) else np.nan
        summary_rows.append(row)

    person_out = person_df.merge(
        use[["session_label", "window_index", "anon_person_slot", "word_id", "word"]],
        on=["session_label", "window_index", "anon_person_slot"],
        how="left",
    )
    info = {"status": "ok", "n_person_windows": int(len(use)), "k": int(k), "features": features}
    return person_out, pd.DataFrame(summary_rows), info


def describe_individual_word(center: pd.Series, med: pd.Series) -> str:
    parts = []
    motion = center.get("motion_energy_mean", np.nan)
    motion_med = med.get("motion_energy_mean", np.nan)
    active = center.get("motion_active_ratio", np.nan)
    left_speed = center.get("left_wrist_speed_mean", np.nan)
    right_speed = center.get("right_wrist_speed_mean", np.nan)
    left_delta = center.get("left_hand_reach_delta", np.nan)
    right_delta = center.get("right_hand_reach_delta", np.nan)
    partner_gaze = center.get("partner_gaze_ratio", np.nan)
    task_gaze = center.get("task_gaze_ratio", np.nan)
    partner_active = center.get("partner_active_ratio", np.nan)

    if np.isfinite(motion) and np.isfinite(motion_med):
        if motion > motion_med * 1.35:
            parts.append("high body motion")
        elif motion < motion_med * 0.70:
            parts.append("low body motion")
        else:
            parts.append("moderate body motion")
    if np.isfinite(active):
        if active > 0.65:
            parts.append("sustained active")
        elif active > 0.25:
            parts.append("brief/intermittent active")
        else:
            parts.append("mostly still")

    if np.isfinite(left_speed) and np.isfinite(right_speed):
        if left_speed > right_speed * 1.25:
            parts.append("left-hand dominant")
        elif right_speed > left_speed * 1.25:
            parts.append("right-hand dominant")
        else:
            parts.append("balanced hands")
    if np.isfinite(left_delta) or np.isfinite(right_delta):
        deltas = {"left": left_delta, "right": right_delta}
        hand, value = min(((h, v) for h, v in deltas.items() if np.isfinite(v)), key=lambda hv: hv[1], default=(None, np.nan))
        if hand and value < -0.015:
            parts.append(f"{hand} hand moves toward partner")
        elif hand and value > 0.015:
            parts.append(f"{hand} hand withdraws")
    if np.isfinite(partner_gaze) and partner_gaze > 0.35:
        parts.append("partner-directed gaze")
    elif np.isfinite(task_gaze) and task_gaze > 0.45:
        parts.append("task-directed gaze")
    else:
        parts.append("low/uncertain gaze target")
    if np.isfinite(partner_active) and partner_active > 0.5:
        parts.append("partner also active")
    elif np.isfinite(partner_active):
        parts.append("partner mostly passive")
    return "; ".join(parts)


def build_conversation_turns(person_words: pd.DataFrame) -> pd.DataFrame:
    if person_words.empty or "word" not in person_words:
        return pd.DataFrame()
    rows = []
    for (session_label, window_index), group in person_words.dropna(subset=["word"]).groupby(["session_label", "window_index"]):
        if len(group) < 2:
            continue
        g = group.sort_values("anon_person_slot")
        a = g.iloc[0]
        b = g.iloc[1]
        a_score = float(a.get("motion_energy_mean", 0) or 0) + 0.15 * float(a.get("motion_active_ratio", 0) or 0)
        b_score = float(b.get("motion_energy_mean", 0) or 0) + 0.15 * float(b.get("motion_active_ratio", 0) or 0)
        if abs(a_score - b_score) < 0.005:
            actor, other = (a, b) if str(a["anon_person_slot"]) <= str(b["anon_person_slot"]) else (b, a)
            actor_status = "balanced_tie"
        elif a_score > b_score:
            actor, other = a, b
            actor_status = "primary_actor"
        else:
            actor, other = b, a
            actor_status = "primary_actor"
        rows.append({
            "session_label": session_label,
            "dyad_id": actor["dyad_id"],
            "order": int(actor["order"]),
            "window_index": int(window_index),
            "window_mid_ms": float(actor["window_mid_ms"]),
            "actor_slot": actor["anon_person_slot"],
            "other_slot": other["anon_person_slot"],
            "actor_word": actor["word"],
            "other_word_same_window": other["word"],
            "actor_status": actor_status,
            "actor_motion_energy": float(actor.get("motion_energy_mean", np.nan)),
            "other_motion_energy": float(other.get("motion_energy_mean", np.nan)),
            "activity_margin": float(abs(a_score - b_score)),
        })
    turns = pd.DataFrame(rows).sort_values(["session_label", "window_index"]).reset_index(drop=True)
    if turns.empty:
        return turns
    turns["prev_actor_slot"] = turns.groupby("session_label")["actor_slot"].shift(1)
    turns["prev_actor_word"] = turns.groupby("session_label")["actor_word"].shift(1)
    turns["next_actor_slot"] = turns.groupby("session_label")["actor_slot"].shift(-1)
    turns["next_actor_word"] = turns.groupby("session_label")["actor_word"].shift(-1)
    turns["next_window_index"] = turns.groupby("session_label")["window_index"].shift(-1)
    turns["actor_switch_next"] = (turns["actor_slot"] != turns["next_actor_slot"]) & turns["next_actor_slot"].notna()
    turns["next_is_adjacent"] = turns["next_window_index"] == turns["window_index"] + 1
    return turns


def build_response_pairs(turns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if turns.empty:
        return pd.DataFrame(), pd.DataFrame()
    pair_rows = []
    for _, row in turns[turns["actor_switch_next"] & turns["next_is_adjacent"]].iterrows():
        pair_rows.append({
            "session_label": row["session_label"],
            "dyad_id": row["dyad_id"],
            "order": int(row["order"]),
            "from_actor_slot": row["actor_slot"],
            "to_actor_slot": row["next_actor_slot"],
            "actor_word": row["actor_word"],
            "other_word_same_window": row["other_word_same_window"],
            "response_word": row["next_actor_word"],
            "prev_actor_word": row["prev_actor_word"] if pd.notna(row["prev_actor_word"]) else "START",
            "window_index": int(row["window_index"]),
            "response_window_index": int(row["next_window_index"]),
        })
    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        return pairs, pd.DataFrame()
    summary = pairs.groupby(["actor_word", "response_word"], dropna=False).size().reset_index(name="count")
    totals = summary.groupby("actor_word")["count"].transform("sum")
    summary["p_response_given_actor_word"] = summary["count"] / totals
    summary = summary.sort_values(["p_response_given_actor_word", "count"], ascending=False).reset_index(drop=True)
    for order in sorted(pairs["order"].dropna().unique()):
        order_counts = pairs[pairs["order"] == order].groupby(["actor_word", "response_word"]).size().rename(f"count_order{int(order)}")
        summary = summary.merge(order_counts.reset_index(), on=["actor_word", "response_word"], how="left")
        summary[f"count_order{int(order)}"] = summary[f"count_order{int(order)}"].fillna(0).astype(int)
    return pairs, summary


def evaluate_response_models(response_pairs: pd.DataFrame) -> pd.DataFrame:
    if response_pairs.empty or response_pairs["dyad_id"].nunique() < 3:
        return pd.DataFrame([{"model": "all", "target": "response_word", "status": "skipped", "reason": "not enough response pairs/dyads"}])
    df = response_pairs.dropna(subset=["actor_word", "response_word", "dyad_id"]).copy()
    if df["response_word"].nunique() < 2:
        return pd.DataFrame([{"model": "all", "target": "response_word", "status": "skipped", "reason": "single response class"}])
    groups = df["dyad_id"].astype(str).to_numpy()
    y = df["response_word"].astype(str).to_numpy()
    logo = LeaveOneGroupOut()
    rows = []

    unigram_pred = np.empty(len(df), dtype=object)
    markov_pred = np.empty(len(df), dtype=object)
    logistic_pred = np.empty(len(df), dtype=object)
    logistic_ok = True
    for train_idx, test_idx in logo.split(df, y, groups):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        default = train["response_word"].value_counts().index[0]
        unigram_pred[test_idx] = default
        table = train.groupby(["actor_word", "response_word"]).size().reset_index(name="count")
        best = table.sort_values(["actor_word", "count"], ascending=[True, False]).drop_duplicates("actor_word")
        best_map = dict(zip(best["actor_word"], best["response_word"]))
        markov_pred[test_idx] = [best_map.get(w, default) for w in test["actor_word"]]
        if train["response_word"].nunique() < 2:
            logistic_ok = False
            logistic_pred[test_idx] = default
            continue
        try:
            pre = ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore"), MODEL_CONTEXT_COLUMNS),
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), ["order"]),
            ])
            clf = Pipeline([
                ("pre", pre),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)),
            ])
            clf.fit(train[MODEL_CONTEXT_COLUMNS + ["order"]].fillna("START"), train["response_word"])
            logistic_pred[test_idx] = clf.predict(test[MODEL_CONTEXT_COLUMNS + ["order"]].fillna("START"))
        except Exception:
            logistic_ok = False
            logistic_pred[test_idx] = default

    labels = sorted(set(y) | set(unigram_pred) | set(markov_pred) | set(logistic_pred))
    rows.append(score_response_model("unigram_response", y, unigram_pred, labels))
    rows.append(score_response_model("actor_word_markov_response", y, markov_pred, labels))
    row = score_response_model("logistic_context_response", y, logistic_pred, labels)
    if not logistic_ok:
        row["note"] = "some folds fell back to unigram due training limitations"
    rows.append(row)
    return pd.DataFrame(rows)


def score_response_model(name: str, y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    return {
        "model": name,
        "target": "response_word",
        "status": "ok",
        "validation": "leave-one-dyad-out",
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "role_handling": "anonymous actor/responder; no p1/p2 semantic feature",
    }


def build_simulation_examples(response_pairs: pd.DataFrame, response_summary: pd.DataFrame, max_sessions: int = 6, max_rows: int = 18) -> list[str]:
    lines = []
    if response_pairs.empty or response_summary.empty:
        return ["No response examples available."]
    best_map = response_summary.sort_values(["actor_word", "p_response_given_actor_word", "count"], ascending=[True, False, False]).drop_duplicates("actor_word")
    best_map = dict(zip(best_map["actor_word"], best_map["response_word"]))
    for session_label, group in response_pairs.groupby("session_label"):
        if len(lines) >= max_sessions * (max_rows + 2):
            break
        lines.append(f"### {session_label}")
        shown = 0
        for _, row in group.head(max_rows).iterrows():
            pred = best_map.get(row["actor_word"], "UNK")
            hit = "OK" if pred == row["response_word"] else "MISS"
            lines.append(
                f"- t{int(row['window_index'])}: {row['from_actor_slot']}:{row['actor_word']} -> "
                f"model {row['to_actor_slot']}:{pred}; actual {row['to_actor_slot']}:{row['response_word']} ({hit})"
            )
            shown += 1
        if shown == 0:
            lines.append("- No actor-switch response pairs in first windows.")
        lines.append("")
    return lines


def write_conversation_sequences(turns: pd.DataFrame, out_path: Path, max_tokens_per_session: int = 240) -> None:
    lines = ["# Nonverbal Conversation Sequences", ""]
    if turns.empty:
        lines.append("No turns available.")
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    for session_label, group in turns.groupby("session_label"):
        lines.append(f"## {session_label}")
        parts = []
        last = None
        run_count = 0
        for _, row in group.sort_values("window_index").iterrows():
            item = f"{row['actor_slot']}:{row['actor_word']}"
            if item == last:
                run_count += 1
                continue
            if last is not None:
                parts.append(f"{last}x{run_count}" if run_count > 1 else last)
            last = item
            run_count = 1
            if len(parts) >= max_tokens_per_session:
                break
        if last is not None and len(parts) < max_tokens_per_session:
            parts.append(f"{last}x{run_count}" if run_count > 1 else last)
        lines.append(" -> ".join(parts) if parts else "No tokens.")
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_conversation_figures(word_summary: pd.DataFrame, response_summary: pd.DataFrame, output_dir: Path) -> list[str]:
    fig_dir = output_dir / "figures"
    ensure_dir(fig_dir)
    paths = []
    if not word_summary.empty:
        prev_cols = [c for c in word_summary.columns if c.startswith("prevalence_order")]
        if prev_cols:
            heat = word_summary.set_index("word")[prev_cols]
            plt.figure(figsize=(max(7, len(prev_cols) * 1.2), max(4, len(heat) * 0.4)))
            plt.imshow(heat.to_numpy(float), aspect="auto", cmap="magma")
            plt.colorbar(label="Prevalence")
            plt.xticks(range(len(prev_cols)), [c.replace("prevalence_", "") for c in prev_cols], rotation=45, ha="right")
            plt.yticks(range(len(heat.index)), heat.index.tolist())
            plt.title("Individual word prevalence by order")
            plt.tight_layout()
            path = fig_dir / "individual_word_prevalence_by_order.png"
            plt.savefig(path, dpi=150)
            plt.close()
            paths.append(str(path))
    if not response_summary.empty:
        top_words = response_summary.groupby("actor_word")["count"].sum().sort_values(ascending=False).head(12).index.tolist()
        top_responses = response_summary.groupby("response_word")["count"].sum().sort_values(ascending=False).head(12).index.tolist()
        mat = response_summary.pivot_table(index="actor_word", columns="response_word", values="p_response_given_actor_word", aggfunc="max").reindex(index=top_words, columns=top_responses).fillna(0)
        plt.figure(figsize=(max(7, len(top_responses) * 0.55), max(5, len(top_words) * 0.45)))
        plt.imshow(mat.to_numpy(float), aspect="auto", cmap="viridis")
        plt.colorbar(label="P(response | actor word)")
        plt.xticks(range(len(mat.columns)), mat.columns.tolist(), rotation=45, ha="right")
        plt.yticks(range(len(mat.index)), mat.index.tolist())
        plt.title("Anonymized response mapping")
        plt.tight_layout()
        path = fig_dir / "response_mapping_heatmap.png"
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(str(path))
    return paths


def write_report(output_dir: Path, meta: pd.DataFrame, person_words: pd.DataFrame, word_summary: pd.DataFrame, turns: pd.DataFrame, response_pairs: pd.DataFrame, response_summary: pd.DataFrame, model_results: pd.DataFrame, info: dict, warnings: list[str], figure_paths: list[str]) -> None:
    lines = ["# Individual Nonverbal Conversation Report", ""]
    lines.append("## Scope")
    lines.append("")
    lines.append("This layer creates individual nonverbal words and maps anonymous actor-response transitions. `person_A` and `person_B` are arbitrary within-dyad slots for readability; p1/p2 are not used as semantic identities.")
    lines.append("")
    lines.append("## Inventory")
    lines.append("")
    lines.append(f"- Parquet files: {len(meta)}")
    lines.append(f"- Dyads: {meta['dyad_id'].nunique() if not meta.empty else 0}")
    lines.append(f"- Person-windows: {len(person_words)}")
    lines.append(f"- Actor turns: {len(turns)}")
    lines.append(f"- Actor-switch response pairs: {len(response_pairs)}")
    if warnings:
        lines.append("")
        lines.append("## Warnings")
        lines.extend([f"- {w}" for w in warnings])
    lines.append("")
    lines.append("## Individual Vocabulary")
    lines.append("")
    if info.get("status") == "ok":
        lines.append(f"- Vocabulary size: {info.get('k')}")
        lines.append(f"- Windows used for vocabulary: {info.get('n_person_windows')}")
        for _, row in word_summary.iterrows():
            lines.append(f"- `{row['word']}`: {row['description']} (prevalence={row['prevalence']:.3f}, n={int(row['window_count'])})")
    else:
        lines.append(f"Vocabulary skipped: {info.get('reason', 'unknown')}")
    lines.append("")
    lines.append("## Response Mapping")
    lines.append("")
    if response_summary.empty:
        lines.append("No response mapping available.")
    else:
        lines.append("Most probable anonymized responses:")
        for _, row in response_summary.head(20).iterrows():
            lines.append(f"- `{row['actor_word']}` -> `{row['response_word']}`: P={row['p_response_given_actor_word']:.3f}, count={int(row['count'])}")
    lines.append("")
    lines.append("## Response Modeling")
    lines.append("")
    if model_results.empty:
        lines.append("No response model results.")
    else:
        for _, row in model_results.iterrows():
            if row.get("status") != "ok":
                lines.append(f"- {row.get('model', 'model')}: skipped ({row.get('reason', 'unknown')}).")
            else:
                lines.append(f"- `{row['model']}`: balanced accuracy={row['balanced_accuracy']:.3f}, macro F1={row['macro_f1']:.3f}, n={int(row['n_samples'])}, validation={row['validation']}.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The current mapping should be read as a symbolic interaction grammar, not as individual identity behavior. A transition `IW04 -> IW06` means that an anonymous primary actor produced an action state classified as `IW04`, and the other anonymous participant's next primary action state was classified as `IW06`.")
    lines.append("High response-model performance relative to a unigram baseline would suggest real sequential structure in the nonverbal language. Weak performance would mean the vocabulary is descriptive but not yet predictive enough for generative interaction.")
    lines.append("")
    lines.append("## Example Model Conversations")
    lines.append("")
    lines.extend(build_simulation_examples(response_pairs, response_summary, max_sessions=4, max_rows=10))
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    if figure_paths:
        for p in figure_paths:
            try:
                rel = Path(p).relative_to(output_dir)
            except ValueError:
                rel = Path(p)
            lines.append(f"- `{rel}`")
    else:
        lines.append("No figures generated.")
    (output_dir / "conversation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(input_dir: Path, output_dir: Path, window_s: float, stride_s: float, min_valid_ratio: float, word_k: int) -> None:
    ensure_dir(output_dir)
    meta, meta_warnings = build_metadata(input_dir)
    person_df, extraction_warnings = build_individual_windows(meta, window_s, stride_s)
    warnings = meta_warnings + extraction_warnings
    person_words, word_summary, vocab_info = discover_individual_words(person_df, word_k, min_valid_ratio, RANDOM_SEED)
    turns = build_conversation_turns(person_words)
    response_pairs, response_summary = build_response_pairs(turns)
    model_results = evaluate_response_models(response_pairs)
    figure_paths = make_conversation_figures(word_summary, response_summary, output_dir)

    meta.to_csv(output_dir / "conversation_metadata.csv", index=False)
    person_words.to_parquet(output_dir / "individual_words.parquet", index=False)
    word_summary.to_csv(output_dir / "individual_vocabulary_summary.csv", index=False)
    turns.to_parquet(output_dir / "conversation_turns.parquet", index=False)
    response_pairs.to_csv(output_dir / "response_pairs.csv", index=False)
    response_summary.to_csv(output_dir / "response_mapping.csv", index=False)
    model_results.to_csv(output_dir / "response_model_results.csv", index=False)
    write_conversation_sequences(turns, output_dir / "conversation_sequences.md")
    (output_dir / "conversation_config.json").write_text(json.dumps({
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "window_s": window_s,
        "stride_s": stride_s,
        "min_valid_ratio": min_valid_ratio,
        "word_k": word_k,
        "warnings": warnings,
        "vocabulary_info": vocab_info,
    }, indent=2), encoding="utf-8")
    write_report(output_dir, meta, person_words, word_summary, turns, response_pairs, response_summary, model_results, vocab_info, warnings, figure_paths)
    print(f"Wrote conversation outputs to {output_dir}")
    print(f"person_windows={len(person_words)} turns={len(turns)} response_pairs={len(response_pairs)}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build anonymized individual nonverbal conversation vocabulary and response mapping.")
    parser.add_argument("--input-dir", type=Path, default=Path("final_dataset_parquets"))
    parser.add_argument("--output-dir", type=Path, default=Path("usta_pose/analiz/efe/outputs_conversation"))
    parser.add_argument("--window-s", type=float, default=1.0)
    parser.add_argument("--stride-s", type=float, default=0.5)
    parser.add_argument("--min-valid-ratio", type=float, default=0.5)
    parser.add_argument("--word-k", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.input_dir.resolve(), args.output_dir.resolve(), args.window_s, args.stride_s, args.min_valid_ratio, args.word_k)


if __name__ == "__main__":
    main()
