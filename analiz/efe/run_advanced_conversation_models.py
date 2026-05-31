#!/usr/bin/env python3
"""Advanced models for anonymized nonverbal conversation.

This script fixes the immediate-listening-response issue by creating a second
response target: the partner's next active/non-listening movement word within a
future horizon. It also trains models that map a person's movement features into
one or several words from the learned individual vocabulary, using both current
movement and past movement context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

from run_nonverbal_conversation import WORD_FEATURES
from run_hri_analysis import ensure_dir

RANDOM_SEED = 42


def finite_numeric_features(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    cols = []
    for col in columns:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().sum() >= max(10, int(0.02 * len(df))):
            cols.append(col)
    return cols


def detect_wait_words(word_summary: pd.DataFrame, response_pairs: pd.DataFrame) -> tuple[list[str], str, pd.DataFrame]:
    dominant = response_pairs["response_word"].value_counts().index[0] if not response_pairs.empty else None
    rows = []
    wait_words = set()
    for _, row in word_summary.iterrows():
        word = row["word"]
        desc = str(row.get("description", ""))
        prevalence = float(row.get("prevalence", 0.0))
        no_explicit_hand_event = "moves toward" not in desc and "withdraws" not in desc
        gaze_monitoring = "partner-directed gaze" in desc
        mostly_passive_partner = "partner mostly passive" in desc
        is_wait = bool(
            word == dominant
            or (gaze_monitoring and mostly_passive_partner and no_explicit_hand_event and prevalence >= 0.03)
        )
        if is_wait:
            wait_words.add(word)
        rows.append({
            "word": word,
            "description": desc,
            "prevalence": prevalence,
            "is_wait_or_monitoring_word": is_wait,
            "reason": "dominant response" if word == dominant else "partner-gaze monitoring without explicit hand event" if is_wait else "active/non-default candidate",
        })
    return sorted(wait_words), dominant, pd.DataFrame(rows)


def build_active_response_pairs(turns: pd.DataFrame, wait_words: set[str], horizon_windows: int) -> pd.DataFrame:
    if turns.empty:
        return pd.DataFrame()
    rows = []
    turns = turns.sort_values(["session_label", "window_index"]).reset_index(drop=True)
    for session_label, group in turns.groupby("session_label", sort=False):
        group = group.sort_values("window_index").reset_index(drop=True)
        records = group.to_dict("records")
        for i, rec in enumerate(records):
            actor_slot = rec.get("actor_slot")
            other_slot = rec.get("other_slot")
            if pd.isna(actor_slot) or pd.isna(other_slot):
                continue
            immediate = None
            active = None
            for j in range(i + 1, len(records)):
                nxt = records[j]
                delta = int(nxt["window_index"]) - int(rec["window_index"])
                if delta <= 0:
                    continue
                if delta > horizon_windows:
                    break
                if nxt.get("actor_slot") != other_slot:
                    continue
                if immediate is None:
                    immediate = nxt
                if str(nxt.get("actor_word")) not in wait_words:
                    active = nxt
                    break
            if immediate is None:
                continue
            rows.append({
                "session_label": rec["session_label"],
                "dyad_id": rec["dyad_id"],
                "order": int(rec["order"]),
                "window_index": int(rec["window_index"]),
                "actor_slot": actor_slot,
                "responder_slot": other_slot,
                "actor_word": rec["actor_word"],
                "other_word_same_window": rec.get("other_word_same_window"),
                "prev_actor_word": rec.get("prev_actor_word") if pd.notna(rec.get("prev_actor_word")) else "START",
                "immediate_response_word": immediate.get("actor_word"),
                "immediate_response_lag_windows": int(immediate["window_index"]) - int(rec["window_index"]),
                "active_response_word": active.get("actor_word") if active else np.nan,
                "active_response_lag_windows": int(active["window_index"]) - int(rec["window_index"]) if active else np.nan,
                "found_active_response": bool(active is not None),
            })
    return pd.DataFrame(rows)


def add_lag_features(person_words: pd.DataFrame, n_lags: int = 3) -> pd.DataFrame:
    df = person_words.sort_values(["session_label", "anon_person_slot", "window_index"]).copy()
    base_feats = [c for c in WORD_FEATURES if c in df.columns]
    for lag in range(1, n_lags + 1):
        for feat in base_feats:
            df[f"lag{lag}_{feat}"] = df.groupby(["session_label", "anon_person_slot"])[feat].shift(lag)
        df[f"lag{lag}_word"] = df.groupby(["session_label", "anon_person_slot"])["word"].shift(lag)
    for feat in base_feats:
        df[f"past{n_lags}_mean_{feat}"] = df.groupby(["session_label", "anon_person_slot"])[feat].transform(
            lambda s: s.shift(1).rolling(n_lags, min_periods=1).mean()
        )
    return df


def label_encode(y_train: np.ndarray, y_test: np.ndarray) -> tuple[LabelEncoder, np.ndarray, np.ndarray]:
    enc = LabelEncoder()
    enc.fit(np.concatenate([y_train, y_test]))
    return enc, enc.transform(y_train), enc.transform(y_test)


def topk_from_proba(proba: np.ndarray, classes: np.ndarray, k: int = 3) -> list[list[str]]:
    order = np.argsort(proba, axis=1)[:, ::-1][:, :k]
    return [[str(classes[idx]) for idx in row] for row in order]


def score_multiclass(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None, classes: np.ndarray | None, model: str, target: str, feature_set: str, validation: str) -> dict:
    labels = sorted(set(y_true) | set(y_pred))
    row = {
        "model": model,
        "target": target,
        "feature_set": feature_set,
        "validation": validation,
        "n_samples": int(len(y_true)),
        "n_classes": int(len(set(y_true))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }
    if y_proba is not None and classes is not None and len(classes) > 1:
        try:
            row["top3_accuracy"] = float(top_k_accuracy_score(y_true, y_proba, k=min(3, len(classes)), labels=classes))
        except Exception:
            row["top3_accuracy"] = np.nan
    else:
        row["top3_accuracy"] = np.nan
    return row


def fit_predict_sklearn(df: pd.DataFrame, y_col: str, numeric_cols: list[str], categorical_cols: list[str], groups_col: str, model_name: str, estimator) -> tuple[dict, pd.DataFrame]:
    data = df.dropna(subset=[y_col]).copy()
    data[y_col] = data[y_col].astype(str)
    groups = data[groups_col].astype(str).to_numpy()
    y = data[y_col].to_numpy()
    preds = np.empty(len(data), dtype=object)
    proba_all = None
    all_classes = np.array(sorted(np.unique(y)), dtype=object)
    if len(all_classes) > 1:
        proba_all = np.zeros((len(data), len(all_classes)), dtype=float)
    logo = LeaveOneGroupOut()
    X = data[numeric_cols + categorical_cols].copy()
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            default = pd.Series(y[train_idx]).mode().iloc[0]
            preds[test_idx] = default
            continue
        pre = ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ])
        clf = Pipeline([("pre", pre), ("model", estimator)])
        clf.fit(X.iloc[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(X.iloc[test_idx])
        if proba_all is not None and hasattr(clf, "predict_proba"):
            fold_proba = clf.predict_proba(X.iloc[test_idx])
            fold_classes = clf.named_steps["model"].classes_
            for c_idx, cls in enumerate(fold_classes):
                global_idx = np.where(all_classes == cls)[0][0]
                proba_all[test_idx, global_idx] = fold_proba[:, c_idx]
    result = score_multiclass(y, preds, proba_all, all_classes, model_name, y_col, "mixed", "leave-one-dyad-out")
    pred_df = data[["session_label", "dyad_id", "order", "window_index"]].copy()
    if "anon_person_slot" in data.columns:
        pred_df["anon_person_slot"] = data["anon_person_slot"].values
    pred_df[f"true_{y_col}"] = y
    pred_df[f"pred_{y_col}"] = preds
    if proba_all is not None:
        top3 = topk_from_proba(proba_all, all_classes, 3)
        pred_df[f"top3_{y_col}"] = ["|".join(x) for x in top3]
    return result, pred_df


def fit_predict_xgb_numeric(df: pd.DataFrame, y_col: str, numeric_cols: list[str], groups_col: str, feature_set: str, n_estimators: int = 180) -> tuple[dict, pd.DataFrame]:
    if XGBClassifier is None:
        return {"model": "xgboost", "target": y_col, "feature_set": feature_set, "status": "skipped", "reason": "xgboost not installed"}, pd.DataFrame()
    data = df.dropna(subset=[y_col]).copy()
    data[y_col] = data[y_col].astype(str)
    groups = data[groups_col].astype(str).to_numpy()
    y = data[y_col].to_numpy()
    all_classes = np.array(sorted(np.unique(y)), dtype=object)
    preds = np.empty(len(data), dtype=object)
    proba_all = np.zeros((len(data), len(all_classes)), dtype=float)
    X = data[numeric_cols].apply(pd.to_numeric, errors="coerce")
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            default = pd.Series(y[train_idx]).mode().iloc[0]
            preds[test_idx] = default
            continue
        enc, y_train_enc, _ = label_encode(y[train_idx], y[test_idx])
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X.iloc[train_idx])
        X_test = imputer.transform(X.iloc[test_idx])
        clf = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_SEED,
            n_jobs=2,
            tree_method="hist",
        )
        clf.fit(X_train, y_train_enc)
        fold_proba = clf.predict_proba(X_test)
        fold_pred_enc = np.argmax(fold_proba, axis=1)
        fold_preds = enc.inverse_transform(fold_pred_enc)
        preds[test_idx] = fold_preds
        for c_idx, cls in enumerate(enc.classes_):
            global_idx = np.where(all_classes == cls)[0][0]
            proba_all[test_idx, global_idx] = fold_proba[:, c_idx]
    result = score_multiclass(y, preds, proba_all, all_classes, "xgboost_hist", y_col, feature_set, "leave-one-dyad-out")
    result["status"] = "ok"
    pred_df = data[["session_label", "dyad_id", "order", "window_index"]].copy()
    if "anon_person_slot" in data.columns:
        pred_df["anon_person_slot"] = data["anon_person_slot"].values
    pred_df[f"true_{y_col}"] = y
    pred_df[f"pred_{y_col}"] = preds
    pred_df[f"top3_{y_col}"] = ["|".join(x) for x in topk_from_proba(proba_all, all_classes, 3)]
    return result, pred_df


def train_word_generation_models(person_words: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = person_words.dropna(subset=["word", "dyad_id"]).copy()
    df = add_lag_features(df, n_lags=3)
    current_features = finite_numeric_features(df, WORD_FEATURES)
    past_features = finite_numeric_features(df, [c for c in df.columns if c.startswith("lag") and c not in {"lag1_word", "lag2_word", "lag3_word"}] + [c for c in df.columns if c.startswith("past3_mean_")])
    rows = []
    pred_frames = []

    sklearn_specs = [
        ("logistic_l2", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)),
        ("hist_gradient_boosting", HistGradientBoostingClassifier(max_iter=140, learning_rate=0.06, max_leaf_nodes=20, random_state=RANDOM_SEED)),
        ("random_forest_shallow", RandomForestClassifier(n_estimators=180, max_depth=6, min_samples_leaf=12, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)),
    ]
    for name, estimator in sklearn_specs:
        result, pred = fit_predict_sklearn(df, "word", current_features, [], "dyad_id", f"{name}_current", estimator)
        result["feature_set"] = "current_movement"
        result["status"] = "ok"
        rows.append(result)
        if name == "hist_gradient_boosting":
            pred["model"] = f"{name}_current"
            pred_frames.append(pred)
    if current_features:
        result, pred = fit_predict_xgb_numeric(df, "word", current_features, "dyad_id", "current_movement")
        rows.append(result)
        if not pred.empty:
            pred["model"] = "xgboost_hist_current"
            pred_frames.append(pred)
    if past_features:
        for name, estimator in sklearn_specs[:2]:
            result, pred = fit_predict_sklearn(df, "word", past_features, ["lag1_word", "lag2_word", "lag3_word"], "dyad_id", f"{name}_past3", estimator)
            result["feature_set"] = "past3_movement_plus_past_words"
            result["status"] = "ok"
            rows.append(result)
            if name == "hist_gradient_boosting":
                pred["model"] = f"{name}_past3"
                pred_frames.append(pred)
        result, pred = fit_predict_xgb_numeric(df, "word", past_features, "dyad_id", "past3_movement_numeric_only")
        rows.append(result)
        if not pred.empty:
            pred["model"] = "xgboost_hist_past3"
            pred_frames.append(pred)
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    return pd.DataFrame(rows), predictions


def merge_actor_features(active_pairs: pd.DataFrame, person_words: pd.DataFrame) -> pd.DataFrame:
    df = add_lag_features(person_words.dropna(subset=["word"]).copy(), n_lags=3)
    actor_cols = [
        "session_label", "dyad_id", "order", "window_index", "anon_person_slot", "word",
        *[c for c in WORD_FEATURES if c in df.columns],
        *[c for c in df.columns if c.startswith("lag") or c.startswith("past3_mean_")],
    ]
    actor = df[actor_cols].copy()
    actor = actor.rename(columns={"anon_person_slot": "actor_slot", "word": "actor_current_word"})
    merged = active_pairs.merge(actor, on=["session_label", "dyad_id", "order", "window_index", "actor_slot"], how="left")
    return merged


def train_active_response_models(active_pairs: pd.DataFrame, person_words: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = merge_actor_features(active_pairs[active_pairs["found_active_response"]].copy(), person_words)
    df = df.dropna(subset=["active_response_word", "dyad_id"])
    if df.empty or df["dyad_id"].nunique() < 3:
        return pd.DataFrame([{"model": "all", "target": "active_response_word", "status": "skipped", "reason": "not enough active response pairs"}]), pd.DataFrame()
    rows = []
    pred_frames = []
    current_features = finite_numeric_features(df, WORD_FEATURES)
    past_features = finite_numeric_features(df, [c for c in df.columns if c.startswith("lag") and c not in {"lag1_word", "lag2_word", "lag3_word"}] + [c for c in df.columns if c.startswith("past3_mean_")])

    # Simple categorical baselines.
    for cat_col, model_name in [("actor_word", "actor_word_markov_active"), ("immediate_response_word", "immediate_response_markov_active")]:
        result, pred = fit_predict_sklearn(
            df,
            "active_response_word",
            [],
            [cat_col],
            "dyad_id",
            model_name,
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
        )
        result["feature_set"] = cat_col
        result["status"] = "ok"
        rows.append(result)
    # Movement models.
    specs = [
        ("logistic_l2_active_current", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)),
        ("hist_gradient_boosting_active_current", HistGradientBoostingClassifier(max_iter=160, learning_rate=0.05, max_leaf_nodes=16, random_state=RANDOM_SEED)),
        ("random_forest_active_current", RandomForestClassifier(n_estimators=180, max_depth=5, min_samples_leaf=8, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)),
    ]
    for name, estimator in specs:
        result, pred = fit_predict_sklearn(df, "active_response_word", current_features, ["actor_word", "other_word_same_window", "prev_actor_word"], "dyad_id", name, estimator)
        result["feature_set"] = "actor_current_movement_plus_context_words"
        result["status"] = "ok"
        rows.append(result)
        if "hist_gradient" in name:
            pred["model"] = name
            pred_frames.append(pred)
    result, pred = fit_predict_xgb_numeric(df, "active_response_word", current_features, "dyad_id", "actor_current_movement_numeric_only", n_estimators=220)
    rows.append(result)
    if not pred.empty:
        pred["model"] = "xgboost_hist_active_current"
        pred_frames.append(pred)

    if past_features:
        result, pred = fit_predict_sklearn(
            df,
            "active_response_word",
            current_features + past_features,
            ["actor_word", "other_word_same_window", "prev_actor_word", "lag1_word", "lag2_word", "lag3_word"],
            "dyad_id",
            "hist_gradient_boosting_active_current_past3",
            HistGradientBoostingClassifier(max_iter=180, learning_rate=0.05, max_leaf_nodes=18, random_state=RANDOM_SEED),
        )
        result["feature_set"] = "actor_current_and_past3_movement_plus_words"
        result["status"] = "ok"
        rows.append(result)
        pred["model"] = "hist_gradient_boosting_active_current_past3"
        pred_frames.append(pred)
        result, pred = fit_predict_xgb_numeric(df, "active_response_word", current_features + past_features, "dyad_id", "actor_current_and_past3_numeric_only", n_estimators=240)
        rows.append(result)
        if not pred.empty:
            pred["model"] = "xgboost_hist_active_current_past3"
            pred_frames.append(pred)
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    return pd.DataFrame(rows), predictions


def active_response_mapping(active_pairs: pd.DataFrame) -> pd.DataFrame:
    df = active_pairs[active_pairs["found_active_response"]].copy()
    if df.empty:
        return pd.DataFrame()
    summary = df.groupby(["actor_word", "active_response_word"]).size().reset_index(name="count")
    totals = summary.groupby("actor_word")["count"].transform("sum")
    summary["p_active_response_given_actor_word"] = summary["count"] / totals
    summary = summary.sort_values(["p_active_response_given_actor_word", "count"], ascending=False).reset_index(drop=True)
    return summary


def write_talk_examples(active_pairs: pd.DataFrame, pred_df: pd.DataFrame, output_path: Path, max_sessions: int = 6, max_rows: int = 16) -> None:
    lines = ["# Model Talking With Real Recordings", ""]
    if active_pairs.empty or pred_df.empty:
        lines.append("No examples available.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    pred_col = "pred_active_response_word"
    top_col = "top3_active_response_word"
    pred = pred_df[pred_df.get("model", "") == "hist_gradient_boosting_active_current_past3"].copy()
    if pred.empty:
        pred = pred_df.copy()
    merged = active_pairs.merge(
        pred[["session_label", "dyad_id", "order", "window_index", pred_col, top_col]].drop_duplicates(["session_label", "window_index"]),
        on=["session_label", "dyad_id", "order", "window_index"],
        how="inner",
    )
    for session_label, group in merged.groupby("session_label"):
        if max_sessions <= 0:
            break
        lines.append(f"## {session_label}")
        for _, row in group.head(max_rows).iterrows():
            actual = row.get("active_response_word")
            hit = "OK" if row.get(pred_col) == actual else "MISS"
            lines.append(
                f"- t{int(row['window_index'])}: observed actor `{row['actor_word']}`; "
                f"immediate partner `{row['immediate_response_word']}`; model active reply `{row.get(pred_col)}` "
                f"top3 `{row.get(top_col)}`; actual next active `{actual}` at +{row.get('active_response_lag_windows')} windows ({hit})"
            )
        lines.append("")
        max_sessions -= 1
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(output_dir: Path, wait_words: list[str], dominant_word: str, active_pairs: pd.DataFrame, active_map: pd.DataFrame, word_results: pd.DataFrame, response_results: pd.DataFrame) -> None:
    lines = ["# Advanced Nonverbal Conversation Modeling", ""]
    lines.append("## Problem Fix")
    lines.append("")
    lines.append("The immediate response vocabulary was dominated by a waiting/listening word. This run keeps that state as meaningful, but creates a second target: the partner's next active/non-monitoring movement within a future horizon. This separates `attention as response` from `next movement as response`.")
    lines.append("")
    lines.append(f"- Dominant immediate response word: `{dominant_word}`")
    lines.append(f"- Wait/monitoring words excluded for active response target: {', '.join(f'`{w}`' for w in wait_words)}")
    lines.append(f"- Candidate response rows: {len(active_pairs)}")
    lines.append(f"- Rows with a future active response: {int(active_pairs['found_active_response'].sum()) if not active_pairs.empty else 0}")
    if not active_pairs.empty:
        lines.append(f"- Mean active response lag in windows: {active_pairs.loc[active_pairs['found_active_response'], 'active_response_lag_windows'].mean():.2f}")
    lines.append("")
    lines.append("## Word Generation Models")
    lines.append("")
    if word_results.empty:
        lines.append("No word generation results.")
    else:
        for _, row in word_results.sort_values("balanced_accuracy", ascending=False, na_position="last").iterrows():
            if row.get("status") == "skipped":
                lines.append(f"- `{row['model']}` skipped: {row.get('reason')}")
            else:
                lines.append(f"- `{row['model']}` ({row['feature_set']}): balanced accuracy={row['balanced_accuracy']:.3f}, macro F1={row['macro_f1']:.3f}, top3={row['top3_accuracy']:.3f}.")
    lines.append("")
    lines.append("## Active Response Models")
    lines.append("")
    if response_results.empty:
        lines.append("No active response model results.")
    else:
        for _, row in response_results.sort_values("balanced_accuracy", ascending=False, na_position="last").iterrows():
            if row.get("status") == "skipped":
                lines.append(f"- `{row['model']}` skipped: {row.get('reason')}")
            else:
                lines.append(f"- `{row['model']}` ({row['feature_set']}): balanced accuracy={row['balanced_accuracy']:.3f}, macro F1={row['macro_f1']:.3f}, top3={row['top3_accuracy']:.3f}.")
    lines.append("")
    lines.append("## Active Response Mapping")
    lines.append("")
    if active_map.empty:
        lines.append("No active response mapping available.")
    else:
        for _, row in active_map.head(20).iterrows():
            lines.append(f"- `{row['actor_word']}` -> `{row['active_response_word']}`: P={row['p_active_response_given_actor_word']:.3f}, count={int(row['count'])}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("If immediate responses are dominated by monitoring, active-response modeling asks a different question: after the other participant attends/waits, what is their next movement word? This is closer to a turn-taking grammar. The model outputs should therefore be read as delayed action replies, not instant reactions.")
    lines.append("Top-3 accuracy is especially relevant here, because a nonverbal context may license several plausible next movements rather than one deterministic reply.")
    (output_dir / "advanced_model_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(conversation_dir: Path, output_dir: Path, horizon_windows: int) -> None:
    ensure_dir(output_dir)
    person_words = pd.read_parquet(conversation_dir / "individual_words.parquet")
    turns = pd.read_parquet(conversation_dir / "conversation_turns.parquet")
    word_summary = pd.read_csv(conversation_dir / "individual_vocabulary_summary.csv")
    response_pairs = pd.read_csv(conversation_dir / "response_pairs.csv")

    wait_words, dominant, wait_table = detect_wait_words(word_summary, response_pairs)
    active_pairs = build_active_response_pairs(turns, set(wait_words), horizon_windows)
    active_map = active_response_mapping(active_pairs)
    word_results, word_predictions = train_word_generation_models(person_words, output_dir)
    response_results, response_predictions = train_active_response_models(active_pairs, person_words)

    wait_table.to_csv(output_dir / "wait_word_classification.csv", index=False)
    active_pairs.to_csv(output_dir / "active_response_pairs.csv", index=False)
    active_map.to_csv(output_dir / "active_response_mapping.csv", index=False)
    word_results.to_csv(output_dir / "word_generation_model_results.csv", index=False)
    word_predictions.to_parquet(output_dir / "word_generation_predictions.parquet", index=False)
    response_results.to_csv(output_dir / "active_response_model_results.csv", index=False)
    response_predictions.to_parquet(output_dir / "active_response_predictions.parquet", index=False)
    write_talk_examples(active_pairs, response_predictions, output_dir / "model_talk_examples.md")
    (output_dir / "advanced_model_config.json").write_text(json.dumps({
        "conversation_dir": str(conversation_dir),
        "output_dir": str(output_dir),
        "horizon_windows": horizon_windows,
        "dominant_immediate_response_word": dominant,
        "wait_words": wait_words,
        "xgboost_available": XGBClassifier is not None,
    }, indent=2), encoding="utf-8")
    write_report(output_dir, wait_words, dominant, active_pairs, active_map, word_results, response_results)
    print(f"Wrote advanced model outputs to {output_dir}")
    print(f"active_pairs={len(active_pairs)} found_active={int(active_pairs['found_active_response'].sum()) if not active_pairs.empty else 0}")
    print(f"wait_words={wait_words}; xgboost_available={XGBClassifier is not None}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train advanced word-generation and active-response models.")
    parser.add_argument("--conversation-dir", type=Path, default=Path("usta_pose/analiz/efe/outputs_conversation_full"))
    parser.add_argument("--output-dir", type=Path, default=Path("usta_pose/analiz/efe/outputs_advanced_models"))
    parser.add_argument("--horizon-windows", type=int, default=12, help="How many future windows to search for responder's active movement.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.conversation_dir.resolve(), args.output_dir.resolve(), args.horizon_windows)


if __name__ == "__main__":
    main()
