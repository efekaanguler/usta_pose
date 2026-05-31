# HRI Parquet Analysis Report

## Scope

This report was generated from tabular pose/gaze parquets only. It uses person-label-invariant dyadic features for primary analysis and models; `p1`/`p2` side diagnostics are retained only for auditing.

## Input and Inventory

- Input directory: `/home/efekaan/Desktop/romer/22kekec/usta_pose/testing/final_dataset_parquets`
- Parquet files found: 44
- Total parquet rows: 183195
- Inferred dyads: 11
- Complete four-order dyad blocks: 11

## Data Quality

- Total extracted windows: 14726
- Quality-passing windows: 10739
- Mean frame_pose_valid_ratio: 0.744
- Mean frame_gaze_valid_ratio: 0.742
- Mean frame_interaction_valid_ratio: 0.744

## Statistical Findings

- activity_state_entropy_mean increased for order4_vs_order2 (mean diff=0.05079, 95% CI [-0.02214, 0.1341], q=0.6423, n=11 dyads; confidence=weak/exploratory).
- activity_state_entropy_mean increased for competitive_order2_4_linear_slope (mean diff=0.0254, 95% CI [-0.01107, 0.06705], q=0.6423, n=11 dyads; confidence=weak/exploratory).
- closest_hand_to_other_head_mean_mean decreased for order1_vs_orders2_4 (mean diff=-0.01836, 95% CI [-0.04798, 0.01561], q=0.6423, n=11 dyads; confidence=weak/exploratory).
- closest_hand_to_other_head_mean_mean decreased for order4_vs_order2 (mean diff=-0.01946, 95% CI [-0.04944, 0.01042], q=0.6423, n=11 dyads; confidence=weak/exploratory).
- closest_hand_to_other_head_mean_mean decreased for competitive_order2_4_linear_slope (mean diff=-0.009732, 95% CI [-0.02472, 0.005208], q=0.6423, n=11 dyads; confidence=weak/exploratory).
- closest_hand_to_other_head_min_mean decreased for order1_vs_orders2_4 (mean diff=-0.02292, 95% CI [-0.05346, 0.01125], q=0.6423, n=11 dyads; confidence=weak/exploratory).
- closest_hand_to_other_head_min_mean decreased for order4_vs_order2 (mean diff=-0.02083, 95% CI [-0.05295, 0.01114], q=0.6423, n=11 dyads; confidence=weak/exploratory).
- closest_hand_to_other_head_min_mean decreased for competitive_order2_4_linear_slope (mean diff=-0.01042, 95% CI [-0.02647, 0.005571], q=0.6423, n=11 dyads; confidence=weak/exploratory).
- hand_to_hand_min_mean_mean decreased for order1_vs_orders2_4 (mean diff=-0.02834, 95% CI [-0.07017, 0.0158], q=0.6423, n=11 dyads; confidence=weak/exploratory).
- hand_to_hand_min_mean_mean decreased for order4_vs_order2 (mean diff=-0.03737, 95% CI [-0.07982, 0.0008159], q=0.6423, n=11 dyads; confidence=weak/exploratory).
- hand_to_hand_min_mean_mean decreased for competitive_order2_4_linear_slope (mean diff=-0.01869, 95% CI [-0.03991, 0.0004079], q=0.6423, n=11 dyads; confidence=weak/exploratory).
- motion_energy_absdiff_mean_mean increased for order4_vs_order2 (mean diff=0.01149, 95% CI [-0.003438, 0.02552], q=0.6423, n=11 dyads; confidence=weak/exploratory).

Summary: 0 strong and 0 moderate evidence rows after FDR correction. Treat all other rows as exploratory.

## Nonverbal Vocabulary

- Vocabulary size: 8
- Tokenized windows: 10739
- Clustering silhouette: 0.113
- Token descriptions:
  - `NV00`: high motion; alternating/one-sided activity; partner gaze; typical distance; motion-asymmetric (prevalence=0.022, windows=240)
  - `NV01`: moderate motion; alternating/one-sided activity; task-focused gaze; typical distance; motion-asymmetric (prevalence=0.190, windows=2037)
  - `NV02`: low motion; limited activity; partner gaze; typical distance (prevalence=0.189, windows=2030)
  - `NV03`: moderate motion; alternating/one-sided activity; task-focused gaze; separated posture (prevalence=0.233, windows=2499)
  - `NV04`: moderate motion; limited activity; partner gaze; typical distance (prevalence=0.049, windows=529)
  - `NV05`: high motion; alternating/one-sided activity; partner gaze; close posture; motion-asymmetric (prevalence=0.171, windows=1836)
  - `NV06`: moderate motion; alternating/one-sided activity; partner gaze; typical distance (prevalence=0.121, windows=1296)
  - `NV07`: high motion; alternating/one-sided activity; partner gaze; close posture; motion-asymmetric (prevalence=0.025, windows=272)

## Modeling

- majority_by_train_fold on `competitive_practiced_vs_order1`: balanced accuracy=0.500, macro F1=0.437, validation=grouped baseline.
- hist_gradient_boosting on `competitive_practiced_vs_order1`: balanced accuracy=0.499, macro F1=0.467, validation=leave-one-dyad-out.
- random_forest_shallow on `competitive_practiced_vs_order1`: balanced accuracy=0.443, macro F1=0.425, validation=leave-one-dyad-out.
- logistic_l2 on `competitive_practiced_vs_order1`: balanced accuracy=0.411, macro F1=0.364, validation=leave-one-dyad-out.
- markov_next_token on `next_vocabulary_token`: balanced accuracy=0.387, macro F1=0.384, validation=leave-one-dyad-out.
- random_forest_shallow on `order_1_2_3_4`: balanced accuracy=0.239, macro F1=0.228, validation=leave-one-dyad-out.
- hist_gradient_boosting on `order_1_2_3_4`: balanced accuracy=0.229, macro F1=0.226, validation=leave-one-dyad-out.
- logistic_l2 on `order_1_2_3_4`: balanced accuracy=0.227, macro F1=0.219, validation=leave-one-dyad-out.
- majority_by_train_fold on `order_1_2_3_4`: balanced accuracy=0.192, macro F1=0.130, validation=grouped baseline.
- unigram_next_token on `next_vocabulary_token`: balanced accuracy=0.089, macro F1=0.042, validation=leave-one-dyad-out.
- Model features exclude session labels, file names, timestamps as identifiers, and direct order metadata.
- Swap consistency is guaranteed by using symmetric dyad features only for models and vocabulary.

## Figures

- `figures/box_pair_motion_energy_mean_mean.png`
- `figures/box_both_active_ratio_mean.png`
- `figures/box_partner_gaze_ratio_mean_mean.png`
- `figures/box_root_distance_mean_mean.png`
- `figures/vocabulary_prevalence_by_order.png`

## Output Files

- `metadata.csv`
- `quality_summary.csv`
- `window_features.parquet`
- `session_features.parquet`
- `statistical_results.csv`
- `vocabulary_tokens.parquet`
- `vocabulary_summary.csv`
- `token_transitions.csv`
- `model_results.csv`
- `run_config.json`

## Interpretation Guardrails

- `order1` versus later orders is a baseline/noncompetitive first-exposure versus competitive/practiced contrast; competition is confounded with experience.
- Frames and windows are not treated as independent evidence for final claims; inferential statistics are paired at dyad/session level.
- `p1` and `p2` are table-side labels, not stable identities.
- Gaze findings should be weighted by gaze validity coverage.
