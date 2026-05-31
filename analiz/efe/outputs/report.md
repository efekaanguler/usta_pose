# HRI Parquet Analysis Report

## Scope

This report was generated from tabular pose/gaze parquets only. It uses person-label-invariant dyadic features for primary analysis and models; `p1`/`p2` side diagnostics are retained only for auditing.

## Input and Inventory

- Input directory: `/home/efekaan/Desktop/romer/22kekec/final_dataset_parquets`
- Parquet files found: 1
- Total parquet rows: 801
- Inferred dyads: 1
- Complete four-order dyad blocks: 0

## Warnings and Limitations

- Input directory has only 1 parquet file(s); dyad-level order statistics and group-validated models will be limited or skipped.
- Incomplete or ambiguous dyad blocks detected: dyad_001

## Data Quality

- Total extracted windows: 66
- Quality-passing windows: 60
- Mean frame_pose_valid_ratio: 0.874
- Mean frame_gaze_valid_ratio: 0.707
- Mean frame_interaction_valid_ratio: 0.874

## Statistical Findings

Dyad-level statistics were skipped or produced no valid contrasts. This usually means the input directory does not contain enough complete dyad/order blocks.

## Nonverbal Vocabulary

- Vocabulary size: 3
- Tokenized windows: 60
- Clustering silhouette: 0.149
- Token descriptions:
  - `NV00`: moderate motion; alternating/one-sided activity; partner gaze; separated posture; motion-asymmetric (prevalence=0.217, windows=13)
  - `NV01`: moderate motion; limited activity; partner gaze; close posture (prevalence=0.500, windows=30)
  - `NV02`: high motion; alternating/one-sided activity; partner gaze; typical distance (prevalence=0.283, windows=17)

## Modeling

- all / all: skipped (fewer than 3 dyads/windows).
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
