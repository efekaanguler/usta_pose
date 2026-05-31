# CENG488 Final Project Pipeline

This document records the reproducible final-project analysis pipeline requested for the dyadic pose/gaze parquet dataset.

## Script

`run_ceng488_final_pipeline.py` implements the full pipeline in modular stages:

1. Dataset validation and outlier checks.
2. Interpretable dyadic feature engineering.
3. 1s, 2s, and 5s non-overlapping window tables.
4. Within-pair session statistics with Wilcoxon, Friedman, sign-permutation tests, and effect sizes.
5. Unsupervised behavioral vocabulary discovery with PCA, t-SNE, KMeans, GMM, Agglomerative, DBSCAN, optional HDBSCAN, and cluster stability checks.
6. Token sequence analysis, n-grams, entropy, transition matrices, session similarity, and baseline-vs-competitive transition divergence.
7. Leave-One-Pair-Out classification for condition and session number.
8. Gaze-only, pose-only, and combined-feature ablation studies.
9. Permutation feature importance for the best condition classifier, with optional SHAP when available.
10. Robustness outputs across window sizes, leave-one-pair influence tables, and shuffled-label sanity checks.

## Command Used

```bash
/home/efekaan/Desktop/torch/torch_env/bin/python usta_pose/analiz/efe/run_ceng488_final_pipeline.py   --input-dir usta_pose/testing/final_dataset_parquets   --output-dir usta_pose/analiz/efe/results   --window-sizes 1,2,5   --stride-ratio 1.0   --max-model-windows 8000   --max-plot-points 2500
```

After the first full run, validation and the markdown summary were refreshed after tightening outlier detection so boolean/interpolation flags are not treated as continuous pose/gaze outliers.

## Main Outputs

Root summary:

- `results/FINAL_ANALYSIS_SUMMARY.md`
- `results/pipeline_config.json`

Validation:

- `results/validation/missing_by_feature.csv`
- `results/validation/missing_by_session_feature.csv`
- `results/validation/missing_by_session.csv`
- `results/validation/missing_by_participant.csv`
- `results/validation/missing_by_camera.csv`
- `results/validation/missing_by_pair_feature.csv`
- `results/validation/timestamp_frame_consistency.csv`
- `results/validation/outliers_by_feature.csv`
- `results/validation/outliers_by_session_feature.csv`

Features:

- `results/features/window_features_1s.csv` and `.parquet`
- `results/features/window_features_2s.csv` and `.parquet`
- `results/features/window_features_5s.csv` and `.parquet`
- `results/features/session_features_1s.csv`
- `results/features/session_features_2s.csv`
- `results/features/session_features_5s.csv`

Statistics:

- `results/statistics/session_statistics_all_windows.csv`
- `results/statistics/session_statistics_1s.csv`
- `results/statistics/session_statistics_2s.csv`
- `results/statistics/session_statistics_5s.csv`
- `results/statistics/leave_one_pair_influence_1s.csv`
- `results/statistics/leave_one_pair_influence_2s.csv`
- `results/statistics/leave_one_pair_influence_5s.csv`

Clustering and tokens:

- `results/clustering/cluster_quality_all_windows.csv`
- `results/clustering/cluster_profiles_1s.csv`
- `results/clustering/cluster_profiles_2s.csv`
- `results/clustering/cluster_profiles_5s.csv`
- `results/tokens/token_assignments_1s.csv`
- `results/tokens/token_assignments_2s.csv`
- `results/tokens/token_assignments_5s.csv`
- `results/tokens/token_sequence_summary_all_windows.csv`

Classification and explainability:

- `results/classification/classification_results_all_windows.csv`
- `results/classification/permutation_feature_importance.csv`
- `results/classification/shap_skipped.txt`
- confusion matrices for each condition/session-number classifier.

Figures:

- PNG figures are under `results/figures/validation`, `results/figures/features`, `results/figures/statistics`, `results/figures/clustering`, `results/figures/tokens`, and `results/figures/classification`.

## Important Methodological Notes

- The final parquet files are fused interaction tables. They do not preserve raw per-camera streams, so camera-level validation is limited to calibration fields and gaze camera-id metadata.
- Session 1 is both baseline/noncompetitive and first exposure. Session 1 versus Sessions 2-4 should therefore be written as baseline/first-exposure versus competitive/practiced, not as pure competition alone.
- Pair identity classification under Leave-One-Pair-Out is mathematically undefined because the held-out pair is an unseen class. The pipeline records this as not applicable and separately provides a session-grouped pair-signature control.
- The strongest findings are exploratory but reproducible in the sense that every table and figure is generated from the script. Statistical claims should prioritize q-values, leave-one-pair influence, and cross-window consistency.
