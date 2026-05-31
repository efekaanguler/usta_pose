# CENG488 Final Project Analysis Summary

## Dataset
- Input directory: `/home/efekaan/Desktop/romer/22kekec/usta_pose/testing/final_dataset_parquets`
- Sessions/parquets: 44
- Inferred pairs: 11
- Window sizes: 1.0s, 2.0s, 5.0s
- Total frame rows from parquet metadata: 183195

## Validation Findings
- Mean overall missing ratio across sessions: 0.1967
- Worst session missing ratio: 0.4559
- Total detected temporal gaps: 0
- Largest temporal gap: 0.00 ms
- Highest z-score outlier features:
  - `p2_motion_speed`: 0.0158
  - `p2_kpt50_world_y`: 0.0156
  - `p1_motion_speed`: 0.0154
  - `p2_kpt40_world_y`: 0.0153
  - `p2_kpt45_world_y`: 0.0147

## Main Session Statistics
- Top exploratory paired contrasts by raw p-value. Treat q-values and leave-one-pair sensitivity as the stronger evidence filters:
  - S1_vs_S2S3S4_mean / `any_partner_gaze_count_session_mean` (1.0s): mean diff=2.608, dz=1.092, p=0.0010, q=0.2592
  - S1_vs_S2S3S4_mean / `any_partner_gaze_count_session_mean` (2.0s): mean diff=5.214, dz=1.098, p=0.0010, q=0.2592
  - S1_vs_S2S3S4_mean / `p2_gaze_category_away_unknown_ratio_session_mean` (2.0s): mean diff=-0.06707, dz=-0.974, p=0.0010, q=0.2592
  - S1_vs_S2S3S4_mean / `p1_task_gaze_transition_count_session_mean` (1.0s): mean diff=0.25, dz=1.360, p=0.0020, q=0.2592
  - S1_vs_S2S3S4_mean / `p2_gaze_category_away_unknown_ratio_session_mean` (1.0s): mean diff=-0.06679, dz=-0.970, p=0.0020, q=0.2592
  - S1_vs_S2S3S4_mean / `any_partner_gaze_count_session_mean` (5.0s): mean diff=13.23, dz=1.058, p=0.0020, q=0.2592
  - S1_vs_S2 / `frame_interaction_valid_ratio_session_mean` (5.0s): mean diff=0.1123, dz=0.838, p=0.0020, q=0.2592
  - S1_vs_S2 / `frame_pose_valid_ratio_session_mean` (5.0s): mean diff=0.1123, dz=0.838, p=0.0020, q=0.2592
  - S1_vs_S2S3S4_mean / `p1_task_gaze_transition_count_session_mean` (2.0s): mean diff=0.5243, dz=1.366, p=0.0020, q=0.2592
  - S1_vs_S4 / `p1_task_gaze_transition_count_session_mean` (2.0s): mean diff=0.669, dz=1.370, p=0.0020, q=0.2592
  - S1_vs_S2 / `dyadic_motion_synchrony_zero_lag_session_mean` (5.0s): mean diff=0.05524, dz=1.432, p=0.0029, q=0.2592
  - S1_vs_S2S3S4_mean / `p2_gaze_category_away_unknown_ratio_session_mean` (5.0s): mean diff=-0.06748, dz=-0.974, p=0.0029, q=0.2592

## Behavioral Vocabulary
- Best clustering candidates by silhouette:
  - 5.0s `gmm` 2: silhouette=0.108, DB=2.798, CH=147.6, stability=nan
  - 5.0s `kmeans` 2: silhouette=0.101, DB=2.877, CH=156.5, stability=0.982
  - 2.0s `kmeans` 2: silhouette=0.089, DB=3.024, CH=351.5, stability=1.000
  - 5.0s `kmeans` 3: silhouette=0.086, DB=2.872, CH=130.9, stability=0.984
  - 2.0s `kmeans` 3: silhouette=0.084, DB=2.977, CH=303.2, stability=0.993
- Token transition JS divergence baseline vs competitive at 1.0s: 0.0774
- Token transition JS divergence baseline vs competitive at 2.0s: 0.0903
- Token transition JS divergence baseline vs competitive at 5.0s: 0.1081

## Classification and Ablation
- Best baseline-vs-competitive classifiers under Leave-One-Pair-Out CV:
  - 1.0s `pose` `random_forest`: accuracy=0.707, macro F1=0.621, ROC-AUC=0.613
  - 5.0s `gaze` `random_forest`: accuracy=0.689, macro F1=0.615, ROC-AUC=0.619
  - 5.0s `combined` `random_forest`: accuracy=0.700, macro F1=0.614, ROC-AUC=0.619
  - 2.0s `pose` `random_forest`: accuracy=0.712, macro F1=0.610, ROC-AUC=0.607
  - 5.0s `pose` `random_forest`: accuracy=0.707, macro F1=0.607, ROC-AUC=0.617
  - 2.0s `combined` `random_forest`: accuracy=0.681, macro F1=0.601, ROC-AUC=0.603
  - 1.0s `combined` `random_forest`: accuracy=0.660, macro F1=0.589, ROC-AUC=0.611
  - 2.0s `gaze` `random_forest`: accuracy=0.661, macro F1=0.588, ROC-AUC=0.608
  - 1.0s `gaze` `random_forest`: accuracy=0.647, macro F1=0.579, ROC-AUC=0.600
  - 5.0s `gaze` `logistic_regression`: accuracy=0.562, macro F1=0.531, ROC-AUC=0.564
- Pair identity under Leave-One-Pair-Out is reported as not applicable because the held-out pair is an unseen class; a separate session-grouped control is saved for pair-signature diagnostics.

## Explainability
- Top permutation-importance features for the best condition classifier:
  - `p1_gaze_pitch_max`: 0.0103 +/- 0.0016
  - `frame_gaze_valid_transition_count`: 0.0102 +/- 0.0018
  - `frame_pose_valid_ratio`: 0.0094 +/- 0.0031
  - `frame_gaze_valid_count`: 0.0093 +/- 0.0030
  - `p1_task_gaze_count`: 0.0079 +/- 0.0014
  - `p2_gaze_yaw_min`: 0.0078 +/- 0.0034
  - `root_distance_max`: 0.0077 +/- 0.0031
  - `p1_task_gaze_angle_max`: 0.0076 +/- 0.0023
  - `frame_interaction_valid_count`: 0.0075 +/- 0.0028
  - `frame_pose_valid_count`: 0.0074 +/- 0.0025

## Limitations and Warnings
- The final parquet files are fused interaction tables, not raw camera streams; per-camera validation is limited to calibration fields and gaze camera-id metadata.
- Session 1 differs from Sessions 2-4 in both competitive pressure and task familiarity, so S1-vs-competitive contrasts cannot isolate competition alone.
- Frame/window rows are not independent; statistical claims should prioritize within-pair tests, leave-one-pair-out validation, and robustness checks.
- Unsupervised tokens are behavioral descriptors learned from this dataset, not a universal nonverbal vocabulary.
- Findings with raw p-values but weak q-values or high leave-one-pair sensitivity should be described as exploratory.
