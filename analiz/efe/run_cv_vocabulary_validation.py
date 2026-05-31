#!/usr/bin/env python3
"""Cross-dyad validation for learned individual nonverbal vocabulary.

This addresses the optimistic movement->word result where clustering was learned
on the full dataset. For each leave-one-dyad-out fold, this script:

1. learns the vocabulary only on train dyads,
2. assigns held-out dyad windows to the train vocabulary centroids,
3. maps train-fold cluster ids to the existing global vocabulary by train-majority
   matching for interpretability/stability analysis,
4. trains simple encoders on train labels and evaluates on held-out dyads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

from run_nonverbal_conversation import WORD_FEATURES
from run_hri_analysis import ensure_dir

RANDOM_SEED = 42


def usable_features(df: pd.DataFrame) -> list[str]:
    out = []
    for col in WORD_FEATURES:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().sum() >= max(10, int(0.02 * len(df))):
            out.append(col)
    return out


def topk_labels_from_proba(proba: np.ndarray, classes: np.ndarray, k: int = 3) -> list[list[str]]:
    idx = np.argsort(proba, axis=1)[:, ::-1][:, :k]
    return [[str(classes[i]) for i in row] for row in idx]


def topk_accuracy_strings(y_true: np.ndarray, topk: list[list[str]]) -> float:
    return float(np.mean([str(y) in preds for y, preds in zip(y_true, topk)])) if len(y_true) else np.nan


def metrics_row(name: str, target: str, y_true: np.ndarray, y_pred: np.ndarray, topk: list[list[str]] | None = None) -> dict:
    labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))
    return {
        "method": name,
        "target": target,
        "n_samples": int(len(y_true)),
        "n_classes": int(len(set(map(str, y_true)))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "top3_accuracy": topk_accuracy_strings(y_true, topk) if topk is not None else np.nan,
    }


def majority_cluster_to_word(cluster_labels: np.ndarray, global_words: np.ndarray) -> dict[int, str]:
    mapping = {}
    for cluster_id in sorted(np.unique(cluster_labels)):
        words = pd.Series(global_words[cluster_labels == cluster_id]).dropna()
        mapping[int(cluster_id)] = str(words.mode().iloc[0]) if not words.empty else f"CL{cluster_id:02d}"
    return mapping


def fit_predict_model(model_name: str, estimator, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if len(np.unique(y_train)) < 2:
        pred = np.full(len(X_test), y_train[0])
        return pred, None, None
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    if hasattr(clf.named_steps["model"], "predict_proba"):
        proba = clf.predict_proba(X_test)
        classes = clf.named_steps["model"].classes_
        return pred, proba, classes
    return pred, None, None


def fit_predict_xgb(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if XGBClassifier is None:
        return np.array([]), None, None
    enc = LabelEncoder()
    y_enc = enc.fit_transform(y_train)
    clf = XGBClassifier(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.08,
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
    pred_enc = np.argmax(proba, axis=1)
    pred = enc.inverse_transform(pred_enc)
    return pred, proba, enc.classes_


def run(input_path: Path, output_dir: Path, k: int) -> None:
    ensure_dir(output_dir)
    df = pd.read_parquet(input_path).dropna(subset=["word", "dyad_id"]).copy()
    features = usable_features(df)
    if len(features) < 4:
        raise RuntimeError("Not enough usable features for CV vocabulary validation.")
    Xraw = df[features].apply(pd.to_numeric, errors="coerce")
    groups = df["dyad_id"].astype(str).to_numpy()
    global_words = df["word"].astype(str).to_numpy()

    logo = LeaveOneGroupOut()
    all_rows = []
    fold_rows = []
    prediction_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(Xraw, global_words, groups), start=1):
        train_dyads = sorted(set(groups[train_idx]))
        test_dyad = sorted(set(groups[test_idx]))[0]
        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler()
        X_train_imp = imputer.fit_transform(Xraw.iloc[train_idx])
        X_test_imp = imputer.transform(Xraw.iloc[test_idx])
        X_train = scaler.fit_transform(X_train_imp)
        X_test = scaler.transform(X_test_imp)

        clusterer = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_SEED + fold_idx, n_init=30, batch_size=1024)
        train_cluster = clusterer.fit_predict(X_train)
        test_cluster = clusterer.predict(X_test)
        cluster_to_word = majority_cluster_to_word(train_cluster, global_words[train_idx])
        test_vocab_word = np.array([cluster_to_word[int(c)] for c in test_cluster], dtype=object)
        train_vocab_word = np.array([cluster_to_word[int(c)] for c in train_cluster], dtype=object)

        stability = metrics_row("train_only_kmeans_assignment", "global_word_stability", global_words[test_idx], test_vocab_word)
        stability.update({"fold": fold_idx, "test_dyad": test_dyad, "xgboost_available": XGBClassifier is not None})
        fold_rows.append(stability)
        fold_rows.append({
            "fold": fold_idx,
            "test_dyad": test_dyad,
            "method": "train_only_kmeans_assignment",
            "target": "global_word_cluster_similarity",
            "n_samples": int(len(test_idx)),
            "n_classes": int(len(set(global_words[test_idx]))),
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "macro_f1": np.nan,
            "top3_accuracy": np.nan,
            "adjusted_rand": float(adjusted_rand_score(global_words[test_idx], test_vocab_word)),
            "normalized_mutual_info": float(normalized_mutual_info_score(global_words[test_idx], test_vocab_word)),
            "xgboost_available": XGBClassifier is not None,
        })

        models = [
            ("logistic_l2", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)),
            ("hist_gradient_boosting", HistGradientBoostingClassifier(max_iter=100, learning_rate=0.07, max_leaf_nodes=20, random_state=RANDOM_SEED)),
        ]
        for model_name, estimator in models:
            pred_cluster, proba, classes = fit_predict_model(model_name, estimator, X_train, train_cluster, X_test)
            pred_word = np.array([cluster_to_word[int(c)] for c in pred_cluster], dtype=object)
            topk_cluster = topk_labels_from_proba(proba, classes, 3) if proba is not None and classes is not None else None
            topk_word = None
            if topk_cluster is not None:
                topk_word = [[cluster_to_word[int(c)] for c in row] for row in topk_cluster]
            r1 = metrics_row(model_name, "train_vocab_cluster", test_cluster.astype(str), pred_cluster.astype(str), [[str(x) for x in row] for row in topk_cluster] if topk_cluster else None)
            r1.update({"fold": fold_idx, "test_dyad": test_dyad, "xgboost_available": XGBClassifier is not None})
            fold_rows.append(r1)
            r2 = metrics_row(model_name, "global_word_via_train_vocab", global_words[test_idx], pred_word, topk_word)
            r2.update({"fold": fold_idx, "test_dyad": test_dyad, "xgboost_available": XGBClassifier is not None})
            fold_rows.append(r2)
            for local_i, row_idx in enumerate(test_idx[:200]):
                prediction_rows.append({
                    "fold": fold_idx,
                    "test_dyad": test_dyad,
                    "method": model_name,
                    "session_label": df.iloc[row_idx]["session_label"],
                    "window_index": int(df.iloc[row_idx]["window_index"]),
                    "anon_person_slot": df.iloc[row_idx]["anon_person_slot"],
                    "global_word": global_words[row_idx],
                    "train_vocab_word": test_vocab_word[local_i],
                    "pred_word": pred_word[local_i],
                    "pred_top3_words": "|".join(topk_word[local_i]) if topk_word else "",
                })

        if XGBClassifier is not None:
            pred_cluster, proba, classes = fit_predict_xgb(X_train, train_cluster, X_test)
            pred_word = np.array([cluster_to_word[int(c)] for c in pred_cluster], dtype=object)
            topk_cluster = topk_labels_from_proba(proba, classes, 3) if proba is not None and classes is not None else None
            topk_word = [[cluster_to_word[int(c)] for c in row] for row in topk_cluster] if topk_cluster else None
            r1 = metrics_row("xgboost_hist", "train_vocab_cluster", test_cluster.astype(str), pred_cluster.astype(str), [[str(x) for x in row] for row in topk_cluster] if topk_cluster else None)
            r1.update({"fold": fold_idx, "test_dyad": test_dyad, "xgboost_available": True})
            fold_rows.append(r1)
            r2 = metrics_row("xgboost_hist", "global_word_via_train_vocab", global_words[test_idx], pred_word, topk_word)
            r2.update({"fold": fold_idx, "test_dyad": test_dyad, "xgboost_available": True})
            fold_rows.append(r2)

    fold_df = pd.DataFrame(fold_rows)
    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "top3_accuracy", "adjusted_rand", "normalized_mutual_info"]
    summary = fold_df.groupby(["method", "target"], dropna=False)[metric_cols].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.reset_index().sort_values(["target", "balanced_accuracy_mean"], ascending=[True, False], na_position="last")

    fold_df.to_csv(output_dir / "cv_vocab_validation_by_fold.csv", index=False)
    summary.to_csv(output_dir / "cv_vocab_validation_summary.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "cv_vocab_validation_prediction_examples.csv", index=False)
    (output_dir / "cv_vocab_validation_config.json").write_text(json.dumps({
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "k": k,
        "features": features,
        "xgboost_available": XGBClassifier is not None,
        "interpretation": "Clusters are learned only on train dyads in each fold; held-out dyads are assigned to train centroids.",
    }, indent=2), encoding="utf-8")
    write_report(output_dir, summary, fold_df, k, features)
    print(f"Wrote CV vocabulary validation to {output_dir}")
    print(summary.to_string(index=False))


def write_report(output_dir: Path, summary: pd.DataFrame, fold_df: pd.DataFrame, k: int, features: list[str]) -> None:
    lines = ["# Cross-Dyad Vocabulary Validation", ""]
    lines.append("## Purpose")
    lines.append("")
    lines.append("This validation avoids the optimistic full-data clustering issue. In every leave-one-dyad-out fold, the individual vocabulary is learned only from train dyads. Test dyads are then assigned to those train centroids.")
    lines.append("")
    lines.append(f"- Vocabulary size per fold: {k}")
    lines.append(f"- Feature count: {len(features)}")
    lines.append("")
    lines.append("## How To Read The Metrics")
    lines.append("")
    lines.append("- `global_word_stability`: whether a train-only vocabulary assigns held-out samples to words that match the original full-data vocabulary after majority mapping. This is the key reliability metric for vocabulary stability.")
    lines.append("- `train_vocab_cluster`: whether a supervised model can reproduce the train-only centroid assignment on held-out dyads. This is an encoder approximation metric, not evidence that the words are behaviorally true.")
    lines.append("- `global_word_via_train_vocab`: model prediction mapped back to the original full-data word names; this combines encoder error and vocabulary stability error.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for _, row in summary.iterrows():
        lines.append(
            f"- `{row['method']}` / `{row['target']}`: "
            f"balanced accuracy mean={row.get('balanced_accuracy_mean', float('nan')):.3f}, "
            f"macro F1 mean={row.get('macro_f1_mean', float('nan')):.3f}, "
            f"top3 mean={row.get('top3_accuracy_mean', float('nan')):.3f}."
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("High encoder performance against `train_vocab_cluster` means the train-only symbolic encoder can be approximated by a supervised model. It does not by itself prove external behavioral validity because the target is still derived from movement features.")
    lines.append("The stronger academic question is `global_word_stability`: whether train-only clustering recovers the same broad word semantics on unseen dyads. Use this as the reliability estimate for the vocabulary itself.")
    (output_dir / "cv_vocab_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate individual vocabulary with train-only clustering and held-out dyad assignment.")
    parser.add_argument("--input", type=Path, default=Path("usta_pose/analiz/efe/outputs_conversation_full/individual_words.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("usta_pose/analiz/efe/outputs_cv_vocab_validation"))
    parser.add_argument("--k", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.input.resolve(), args.output_dir.resolve(), args.k)


if __name__ == "__main__":
    main()
