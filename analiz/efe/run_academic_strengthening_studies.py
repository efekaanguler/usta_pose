#!/usr/bin/env python3
"""Event-based nonverbal grammar and behavior-cloning studies.

This script strengthens academic claims by moving beyond fixed-window immediate
responses. It collapses individual words into events, builds delayed active
response pairs, trains leave-one-dyad-out behavior-cloning models, and evaluates
whether generated symbolic responses match human response distributions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
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
EPS = 1e-9


def detect_wait_words(conversation_dir: Path) -> tuple[set[str], str]:
    wait_path = conversation_dir.parent / "outputs_advanced_models" / "wait_word_classification.csv"
    if wait_path.exists():
        wait_df = pd.read_csv(wait_path)
        wait_words = set(wait_df.loc[wait_df["is_wait_or_monitoring_word"].astype(bool), "word"].astype(str))
    else:
        summary = pd.read_csv(conversation_dir / "individual_vocabulary_summary.csv")
        response = pd.read_csv(conversation_dir / "response_pairs.csv")
        dominant = response["response_word"].value_counts().index[0]
        wait_words = {dominant}
        for _, row in summary.iterrows():
            desc = str(row.get("description", ""))
            if "partner-directed gaze" in desc and "moves toward" not in desc and "withdraws" not in desc and row.get("prevalence", 0) >= 0.03:
                wait_words.add(str(row["word"]))
    response = pd.read_csv(conversation_dir / "response_pairs.csv")
    dominant = str(response["response_word"].value_counts().index[0])
    return wait_words, dominant


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return np.nan
    return float(np.nanmean(arr))


def collapse_person_events(person_words: pd.DataFrame, wait_words: set[str]) -> pd.DataFrame:
    df = person_words.dropna(subset=["word"]).copy()
    df = df.sort_values(["session_label", "anon_person_slot", "window_index"])
    rows = []
    numeric_cols = [c for c in WORD_FEATURES if c in df.columns]
    for (session, slot), group in df.groupby(["session_label", "anon_person_slot"], sort=False):
        group = group.sort_values("window_index").reset_index(drop=True)
        run_start = 0
        event_idx = 0
        for i in range(1, len(group) + 1):
            boundary = i == len(group) or group.loc[i, "word"] != group.loc[run_start, "word"] or int(group.loc[i, "window_index"]) != int(group.loc[i - 1, "window_index"]) + 1
            if not boundary:
                continue
            run = group.iloc[run_start:i]
            first = run.iloc[0]
            word = str(first["word"])
            out = {
                "session_label": session,
                "dyad_id": first["dyad_id"],
                "order": int(first["order"]),
                "anon_person_slot": slot,
                "event_index_person": event_idx,
                "word": word,
                "is_wait_word": word in wait_words,
                "is_active_word": word not in wait_words,
                "start_window": int(run["window_index"].min()),
                "end_window": int(run["window_index"].max()),
                "duration_windows": int(len(run)),
                "start_ms": float(run["window_start_ms"].iloc[0]),
                "end_ms": float(run["window_end_ms"].iloc[-1]),
                "mid_ms": float((run["window_start_ms"].iloc[0] + run["window_end_ms"].iloc[-1]) / 2.0),
            }
            for col in numeric_cols:
                out[f"event_mean_{col}"] = finite_mean(pd.to_numeric(run[col], errors="coerce").dropna().tolist())
            rows.append(out)
            event_idx += 1
            run_start = i
    events = pd.DataFrame(rows).sort_values(["session_label", "start_window", "anon_person_slot"]).reset_index(drop=True)
    return events


def add_event_history(events: pd.DataFrame, n_history: int = 3) -> pd.DataFrame:
    df = events.sort_values(["session_label", "anon_person_slot", "start_window"]).copy()
    for lag in range(1, n_history + 1):
        df[f"prev{lag}_own_word"] = df.groupby(["session_label", "anon_person_slot"])["word"].shift(lag).fillna("START")
        df[f"prev{lag}_own_active"] = df.groupby(["session_label", "anon_person_slot"])["is_active_word"].shift(lag).fillna(False).astype(bool)
        df[f"prev{lag}_own_duration"] = df.groupby(["session_label", "anon_person_slot"])["duration_windows"].shift(lag)
    return df


def build_event_response_pairs(events: pd.DataFrame, horizon_windows: int) -> pd.DataFrame:
    rows = []
    events = events.sort_values(["session_label", "start_window", "anon_person_slot"]).reset_index(drop=True)
    for session, group in events.groupby("session_label", sort=False):
        group = group.sort_values(["start_window", "anon_person_slot"]).reset_index(drop=True)
        recs = group.to_dict("records")
        for i, actor in enumerate(recs):
            if not actor["is_active_word"]:
                continue
            actor_slot = actor["anon_person_slot"]
            other_slot = "person_B" if actor_slot == "person_A" else "person_A"
            immediate_other = None
            active_response = None
            for j in range(i + 1, len(recs)):
                cand = recs[j]
                delta = int(cand["start_window"]) - int(actor["start_window"])
                if delta <= 0:
                    continue
                if delta > horizon_windows:
                    break
                if cand["anon_person_slot"] != other_slot:
                    continue
                if immediate_other is None:
                    immediate_other = cand
                if cand["is_active_word"]:
                    active_response = cand
                    break
            if immediate_other is None:
                continue
            out = {
                "session_label": actor["session_label"],
                "dyad_id": actor["dyad_id"],
                "order": int(actor["order"]),
                "actor_slot": actor_slot,
                "responder_slot": other_slot,
                "actor_start_window": int(actor["start_window"]),
                "actor_end_window": int(actor["end_window"]),
                "actor_word": actor["word"],
                "actor_duration_windows": int(actor["duration_windows"]),
                "immediate_other_word": immediate_other["word"],
                "immediate_other_is_wait": bool(immediate_other["is_wait_word"]),
                "immediate_lag_windows": int(immediate_other["start_window"]) - int(actor["start_window"]),
                "found_active_response": bool(active_response is not None),
                "active_response_word": active_response["word"] if active_response else np.nan,
                "active_response_lag_windows": int(active_response["start_window"]) - int(actor["start_window"]) if active_response else np.nan,
                "active_response_duration_windows": int(active_response["duration_windows"]) if active_response else np.nan,
            }
            for key, value in actor.items():
                if key.startswith("event_mean_"):
                    out[f"actor_{key}"] = value
            rows.append(out)
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    pairs = add_pair_history(pairs)
    return pairs


def add_pair_history(pairs: pd.DataFrame) -> pd.DataFrame:
    df = pairs.sort_values(["session_label", "actor_start_window"]).copy()
    for lag in range(1, 4):
        df[f"prev{lag}_actor_word_global"] = df.groupby("session_label")["actor_word"].shift(lag).fillna("START")
        df[f"prev{lag}_immediate_other_word_global"] = df.groupby("session_label")["immediate_other_word"].shift(lag).fillna("START")
        df[f"prev{lag}_response_word_global"] = df.groupby("session_label")["active_response_word"].shift(lag).fillna("START")
    return df


def mapping_table(pairs: pd.DataFrame) -> pd.DataFrame:
    df = pairs[pairs["found_active_response"]].dropna(subset=["active_response_word"])
    if df.empty:
        return pd.DataFrame()
    out = df.groupby(["actor_word", "immediate_other_word", "active_response_word"]).size().reset_index(name="count")
    totals = out.groupby(["actor_word", "immediate_other_word"])["count"].transform("sum")
    out["p_response_given_actor_and_immediate"] = out["count"] / totals
    out = out.sort_values(["p_response_given_actor_and_immediate", "count"], ascending=False).reset_index(drop=True)
    return out


def actor_response_mapping_table(pairs: pd.DataFrame) -> pd.DataFrame:
    df = pairs[pairs["found_active_response"]].dropna(subset=["active_response_word"])
    if df.empty:
        return pd.DataFrame()
    out = df.groupby(["actor_word", "active_response_word"]).size().reset_index(name="count")
    totals = out.groupby("actor_word")["count"].transform("sum")
    out["p_response_given_actor"] = out["count"] / totals
    return out.sort_values(["actor_word", "p_response_given_actor", "count"], ascending=[True, False, False]).reset_index(drop=True)


def wait_continuation_mapping_table(pairs: pd.DataFrame) -> pd.DataFrame:
    df = pairs[pairs["found_active_response"] & pairs["immediate_other_is_wait"]].dropna(subset=["active_response_word"])
    if df.empty:
        return pd.DataFrame()
    out = df.groupby(["actor_word", "immediate_other_word", "active_response_word"]).size().reset_index(name="count")
    totals = out.groupby(["actor_word", "immediate_other_word"])["count"].transform("sum")
    out["p_active_response_after_wait"] = out["count"] / totals
    return out.sort_values(["actor_word", "immediate_other_word", "p_active_response_after_wait", "count"], ascending=[True, True, False, False]).reset_index(drop=True)


def model_feature_columns(pairs: pd.DataFrame, mode: str) -> tuple[list[str], list[str]]:
    numeric = [c for c in pairs.columns if c.startswith("actor_event_mean_")]
    numeric += ["actor_duration_windows"]
    if mode == "observed_immediate":
        numeric += ["immediate_lag_windows"]
    numeric = [c for c in numeric if c in pairs.columns and pd.to_numeric(pairs[c], errors="coerce").notna().sum() > 20]
    categorical = ["actor_word", "order"]
    if mode in {"actor_history", "observed_immediate"}:
        categorical += [
            "prev1_actor_word_global",
            "prev2_actor_word_global",
            "prev3_actor_word_global",
            "prev1_response_word_global",
            "prev2_response_word_global",
            "prev3_response_word_global",
        ]
    if mode == "observed_immediate":
        categorical += [
            "immediate_other_word",
            "prev1_immediate_other_word_global",
            "prev2_immediate_other_word_global",
            "prev3_immediate_other_word_global",
        ]
    categorical = [c for c in categorical if c in pairs.columns]
    return numeric, categorical


def topk_strings(proba: np.ndarray, classes: np.ndarray, k: int = 3) -> list[str]:
    idx = np.argsort(proba, axis=1)[:, ::-1][:, :k]
    return ["|".join(str(classes[i]) for i in row) for row in idx]


def topk_accuracy_from_strings(y_true: np.ndarray, topk: list[str]) -> float:
    return float(np.mean([str(y) in str(preds).split("|") for y, preds in zip(y_true, topk)])) if len(y_true) else np.nan


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, top3: list[str] | None, name: str, feature_set: str) -> dict:
    labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    return {
        "model": name,
        "feature_set": feature_set,
        "n_samples": int(len(y_true)),
        "n_classes": int(len(set(map(str, y_true)))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "top3_accuracy": topk_accuracy_from_strings(y_true, top3) if top3 is not None else np.nan,
    }


def make_sklearn_estimator(family: str):
    if family == "logistic":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
    if family == "hist_gbdt":
        return HistGradientBoostingClassifier(max_iter=120, learning_rate=0.06, max_leaf_nodes=18, random_state=RANDOM_SEED)
    if family == "random_forest":
        return RandomForestClassifier(n_estimators=180, max_depth=8, min_samples_leaf=8, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
    raise ValueError(f"Unknown sklearn estimator family: {family}")


def model_specs() -> list[dict]:
    specs = [
        {"model": "majority", "kind": "majority", "feature_set": "baseline"},
        {"model": "actor_markov", "kind": "markov", "feature_set": "actor_word_only", "key_cols": ["actor_word"]},
        {"model": "actor_history_markov", "kind": "markov", "feature_set": "actor_word+previous_response", "key_cols": ["actor_word", "prev1_response_word_global"]},
        {"model": "actor_immediate_markov", "kind": "markov", "feature_set": "actor_word+immediate_other_word", "key_cols": ["actor_word", "immediate_other_word"]},
    ]
    for mode in ["actor_only", "actor_history", "observed_immediate"]:
        specs.extend([
            {"model": f"logistic_{mode}", "kind": "sklearn", "family": "logistic", "feature_set": mode},
            {"model": f"hist_gbdt_{mode}", "kind": "sklearn", "family": "hist_gbdt", "feature_set": mode},
            {"model": f"random_forest_{mode}", "kind": "sklearn", "family": "random_forest", "feature_set": mode},
        ])
        if XGBClassifier is not None:
            specs.append({"model": f"xgboost_{mode}", "kind": "xgb", "feature_set": mode})
    return specs


def fit_markov_fold(train: pd.DataFrame, test: pd.DataFrame, key_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    default = train["active_response_word"].value_counts().index[0]
    fallback = train["active_response_word"].value_counts().head(3).index.astype(str).tolist()
    tbl = train.groupby(key_cols + ["active_response_word"]).size().reset_index(name="count")
    best = tbl.sort_values(key_cols + ["count"], ascending=[True] * len(key_cols) + [False])
    lookup = {}
    top_lookup = {}
    for key, group in best.groupby(key_cols):
        norm_key = key if isinstance(key, tuple) else (key,)
        vals = group.sort_values("count", ascending=False)["active_response_word"].astype(str).tolist()
        lookup[norm_key] = vals[0]
        top_lookup[norm_key] = vals[:3]
    preds = []
    top3 = []
    for _, row in test.iterrows():
        key = tuple(row[col] for col in key_cols)
        preds.append(lookup.get(key, default))
        top3.append("|".join(top_lookup.get(key, fallback)))
    return np.asarray(preds, dtype=object), top3


def train_event_bc_models(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pairs[pairs["found_active_response"]].dropna(subset=["active_response_word", "dyad_id"]).copy()
    if data.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    y = data["active_response_word"].astype(str).to_numpy()
    groups = data["dyad_id"].astype(str).to_numpy()
    logo = LeaveOneGroupOut()
    rows = []
    pred_rows = []
    dist_rows = []

    for spec in model_specs():
        model_name = spec["model"]
        kind = spec["kind"]
        feature_set = spec["feature_set"]
        numeric, categorical = model_feature_columns(data, feature_set) if feature_set in {"actor_only", "actor_history", "observed_immediate"} else ([], [])
        preds = np.empty(len(data), dtype=object)
        top3 = [""] * len(data)
        for train_idx, test_idx in logo.split(data, y, groups):
            train = data.iloc[train_idx]
            test = data.iloc[test_idx]
            y_train = y[train_idx]
            if kind == "majority":
                default = pd.Series(y_train).mode().iloc[0]
                preds[test_idx] = default
                top3_choices = train["active_response_word"].value_counts().head(3).index.astype(str).tolist()
                for idx in test_idx:
                    top3[idx] = "|".join(top3_choices)
                continue
            if kind == "markov":
                pred, top = fit_markov_fold(train, test, spec["key_cols"])
                preds[test_idx] = pred
                for local, idx in enumerate(test_idx):
                    top3[idx] = top[local]
                continue
            if kind == "xgb":
                pred, top = fit_xgb_fold(train, test, y_train, numeric, categorical)
                preds[test_idx] = pred
                for local, idx in enumerate(test_idx):
                    top3[idx] = top[local]
                continue
            estimator = make_sklearn_estimator(spec["family"])
            pred, top = fit_sklearn_fold(train, test, y_train, estimator, numeric, categorical)
            preds[test_idx] = pred
            for local, idx in enumerate(test_idx):
                top3[idx] = top[local]
        rows.append(evaluate_predictions(y, preds, top3, model_name, feature_set))
        keep_cols = ["session_label", "dyad_id", "order", "actor_start_window", "actor_word", "immediate_other_word", "immediate_other_is_wait", "active_response_word"]
        tmp = data[keep_cols].copy()
        tmp["model"] = model_name
        tmp["feature_set"] = feature_set
        tmp["pred_active_response_word"] = preds
        tmp["top3_active_response_word"] = top3
        pred_rows.append(tmp)
        dist_rows.extend(distribution_eval(tmp, model_name))
    return pd.DataFrame(rows), pd.concat(pred_rows, ignore_index=True), pd.DataFrame(dist_rows)

def fit_sklearn_fold(train: pd.DataFrame, test: pd.DataFrame, y_train: np.ndarray, estimator, numeric: list[str], categorical: list[str]) -> tuple[np.ndarray, list[str]]:
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    clf = Pipeline([("pre", pre), ("model", estimator)])
    X_train = train[numeric + categorical].copy()
    X_test = test[numeric + categorical].copy()
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    if hasattr(clf.named_steps["model"], "predict_proba"):
        proba = clf.predict_proba(X_test)
        classes = clf.named_steps["model"].classes_
        top = topk_strings(proba, classes, 3)
    else:
        top = [str(p) for p in pred]
    return pred, top


def fit_xgb_fold(train: pd.DataFrame, test: pd.DataFrame, y_train: np.ndarray, numeric: list[str], categorical: list[str]) -> tuple[np.ndarray, list[str]]:
    enc_y = LabelEncoder()
    y_enc = enc_y.fit_transform(y_train)
    X_train = encode_xgb_features(train, train, numeric, categorical)
    X_test = encode_xgb_features(test, train, numeric, categorical)
    clf = XGBClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.07,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        n_jobs=2,
        tree_method="hist",
    )
    clf.fit(X_train, y_enc)
    proba = clf.predict_proba(X_test)
    pred = enc_y.inverse_transform(np.argmax(proba, axis=1))
    top = topk_strings(proba, enc_y.classes_, 3)
    return pred, top


def encode_xgb_features(df: pd.DataFrame, train_ref: pd.DataFrame, numeric: list[str], categorical: list[str]) -> np.ndarray:
    num = df[numeric].apply(pd.to_numeric, errors="coerce") if numeric else pd.DataFrame(index=df.index)
    med = train_ref[numeric].apply(pd.to_numeric, errors="coerce").median(numeric_only=True) if numeric else pd.Series(dtype=float)
    num = num.fillna(med).fillna(0.0)
    cats = []
    for col in categorical:
        levels = sorted(train_ref[col].astype(str).fillna("NA").unique().tolist())
        values = df[col].astype(str).fillna("NA")
        for level in levels:
            cats.append((values == level).astype(float).to_numpy()[:, None])
    arrays = [num.to_numpy(float)] if numeric else []
    arrays.extend(cats)
    return np.hstack(arrays) if arrays else np.zeros((len(df), 1), dtype=float)


def distribution_eval(pred_df: pd.DataFrame, model_name: str) -> list[dict]:
    rows = []
    for scope, group_cols in [("overall", []), ("by_order", ["order"]), ("by_dyad", ["dyad_id"] )]:
        iterator = [("all", pred_df)] if not group_cols else pred_df.groupby(group_cols, dropna=False)
        for key, group in iterator:
            true = group["active_response_word"].astype(str)
            pred = group["pred_active_response_word"].astype(str)
            labels = sorted(set(true) | set(pred))
            p = true.value_counts(normalize=True).reindex(labels, fill_value=0).to_numpy(float)
            q = pred.value_counts(normalize=True).reindex(labels, fill_value=0).to_numpy(float)
            js = float(jensenshannon(p, q, base=2.0) ** 2)
            rows.append({"model": model_name, "scope": scope, "group": str(key), "n": int(len(group)), "js_divergence": js})
    return rows


def write_report(output_dir: Path, events: pd.DataFrame, pairs: pd.DataFrame, model_results: pd.DataFrame, dist_eval: pd.DataFrame, wait_words: set[str], dominant_word: str) -> None:
    lines = ["# Academic Strengthening Studies", ""]
    lines.append("## Study Design")
    lines.append("")
    lines.append("This study strengthens the nonverbal-language claims by replacing fixed-window immediate response analysis with event-based delayed active response modeling. Consecutive identical individual words are collapsed into events. Monitoring/listening words are retained as meaningful turn states but are not treated as final active responses.")
    lines.append("")
    lines.append(f"- Events: {len(events)}")
    lines.append(f"- Active actor response opportunities: {len(pairs)}")
    found_active = int(pairs["found_active_response"].sum()) if not pairs.empty else 0
    wait_first = int((pairs["found_active_response"] & pairs["immediate_other_is_wait"]).sum()) if not pairs.empty else 0
    lines.append(f"- Found delayed active responses: {found_active}")
    lines.append(f"- Delayed responses where the first other-person event was wait/monitoring: {wait_first}")
    lines.append(f"- Wait/monitoring words: {', '.join(sorted(wait_words))}")
    lines.append(f"- Dominant immediate response word: {dominant_word}")
    if not pairs.empty and pairs["found_active_response"].any():
        lag = pairs.loc[pairs["found_active_response"], "active_response_lag_windows"]
        lines.append(f"- Mean delayed active response lag: {lag.mean():.2f} windows")
        lines.append(f"- Median delayed active response lag: {lag.median():.2f} windows")
    lines.append("")
    lines.append("## Behavior-Cloning Results")
    lines.append("")
    lines.append("Models are grouped by how much information they are allowed to observe. `actor_only` is the most autonomous setting: it predicts the other person's eventual active response from the actor's current event and movement features only. `actor_history` adds previous symbolic turns. `observed_immediate` additionally observes the other person's first event after the actor, so it is best interpreted as offline conversation interpretation rather than fully autonomous generation.")
    lines.append("")
    for feature_set in ["actor_only", "actor_history", "observed_immediate", "actor_word_only", "actor_word+previous_response", "actor_word+immediate_other_word", "baseline"]:
        subset = model_results[model_results["feature_set"] == feature_set]
        if subset.empty:
            continue
        lines.append(f"### {feature_set}")
        for _, row in subset.sort_values("balanced_accuracy", ascending=False, na_position="last").iterrows():
            lines.append(f"- `{row['model']}`: balanced accuracy={row['balanced_accuracy']:.3f}, macro F1={row['macro_f1']:.3f}, top3={row['top3_accuracy']:.3f}, accuracy={row['accuracy']:.3f}.")
        lines.append("")
    lines.append("## Distributional Human-Likeness")
    lines.append("")
    if not dist_eval.empty:
        overall = dist_eval[dist_eval["scope"] == "overall"].sort_values("js_divergence")
        for _, row in overall.iterrows():
            lines.append(f"- `{row['model']}` response distribution JS divergence: {row['js_divergence']:.4f} lower is more human-like.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The strongest claim is that dyadic recordings contain a reproducible symbolic nonverbal grammar at the event level. The wait/monitoring state is not just a failed response: in many cases it is an intermediate turn-holding state before a later active movement.")
    lines.append("Autonomous response generation should be claimed only from the `actor_only` and `actor_history` ablations. The `observed_immediate` models are still useful, but their higher scores partly come from observing the responder's first post-actor state; they support interpretation of real recordings more than fully independent generation.")
    lines.append("Top-3 accuracy should be interpreted as candidate-set quality, not exact prediction accuracy. For generative use, sampling among top candidates is more appropriate than forcing a single deterministic response because the same actor event can receive multiple plausible human responses.")
    lines.append("PPO should remain secondary until an environment with meaningful transition dynamics and rewards is defined. The current leave-one-dyad-out behavior cloning and distribution-matching evaluation are the more credible academic core.")
    (output_dir / "academic_strengthening_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def run(conversation_dir: Path, output_dir: Path, horizon_windows: int) -> None:
    ensure_dir(output_dir)
    person_words = pd.read_parquet(conversation_dir / "individual_words.parquet")
    wait_words, dominant = detect_wait_words(conversation_dir)
    events = collapse_person_events(person_words, wait_words)
    events = add_event_history(events)
    pairs = build_event_response_pairs(events, horizon_windows)
    mapping = mapping_table(pairs)
    actor_mapping = actor_response_mapping_table(pairs)
    wait_mapping = wait_continuation_mapping_table(pairs)
    model_results, predictions, dist_eval = train_event_bc_models(pairs)

    events.to_parquet(output_dir / "event_words.parquet", index=False)
    pairs.to_csv(output_dir / "event_response_pairs.csv", index=False)
    mapping.to_csv(output_dir / "event_response_mapping.csv", index=False)
    actor_mapping.to_csv(output_dir / "event_actor_response_mapping.csv", index=False)
    wait_mapping.to_csv(output_dir / "event_wait_continuation_mapping.csv", index=False)
    model_results.to_csv(output_dir / "event_bc_model_results.csv", index=False)
    predictions.to_parquet(output_dir / "event_bc_predictions.parquet", index=False)
    dist_eval.to_csv(output_dir / "event_bc_distribution_eval.csv", index=False)
    (output_dir / "academic_strengthening_config.json").write_text(json.dumps({
        "conversation_dir": str(conversation_dir),
        "output_dir": str(output_dir),
        "horizon_windows": horizon_windows,
        "wait_words": sorted(wait_words),
        "dominant_immediate_response_word": dominant,
        "xgboost_available": XGBClassifier is not None,
    }, indent=2), encoding="utf-8")
    write_report(output_dir, events, pairs, model_results, dist_eval, wait_words, dominant)
    print(f"Wrote academic strengthening outputs to {output_dir}")
    print(f"events={len(events)} pairs={len(pairs)} found_active={int(pairs['found_active_response'].sum()) if not pairs.empty else 0}")
    print(model_results.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run event-based academic strengthening studies for nonverbal grammar.")
    parser.add_argument("--conversation-dir", type=Path, default=Path("usta_pose/analiz/efe/outputs_conversation_full"))
    parser.add_argument("--output-dir", type=Path, default=Path("usta_pose/analiz/efe/outputs_academic_strengthening"))
    parser.add_argument("--horizon-windows", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.conversation_dir.resolve(), args.output_dir.resolve(), args.horizon_windows)


if __name__ == "__main__":
    main()
