# What Was Implemented

This directory contains a parquet-only HRI analysis implementation for the goal described in `usta_pose/testing/goal.md`.

## Main Script

- `run_hri_analysis.py` is the reproducible entry point.
- It accepts `--input-dir`, `--output-dir`, window size/stride, quality threshold, lag window, vocabulary size, and model-window limit.
- The default input is repository-root `final_dataset_parquets`, matching the latest user request.

## Pipeline Stages

1. Metadata and validation: parses filenames, counts rows/columns, infers chronological dyad blocks, and flags incomplete order groups.
2. Feature extraction: reads only required parquet columns, computes frame-level motion/gaze/activity proxies, and aggregates them into sliding windows.
3. Person-label invariance: primary features are symmetric across the two people, using pair means, absolute differences, min/max proximity, activity-state ratios, and non-directional lag magnitudes.
4. Session aggregation: creates session-level summaries from quality-passing windows.
5. Statistics: runs paired dyad-level contrasts with bootstrap confidence intervals, Wilcoxon p-values, and Benjamini-Hochberg FDR correction.
6. Vocabulary discovery: clusters quality windows into interpretable nonverbal tokens and exports token timelines plus transition counts.
7. Modeling: when enough dyads exist, trains simple group-validated models and a Markov next-token baseline using leave-one-dyad-out validation.
8. Reporting: writes `report.md` with findings, quality limits, vocabulary descriptions, model results, and interpretation guardrails.

## Output Files

Each output directory contains:

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

## Runs Completed

The requested path was run exactly:

```bash
/home/efekaan/Desktop/torch/torch_env/bin/python usta_pose/analiz/efe/run_hri_analysis.py --input-dir final_dataset_parquets --output-dir usta_pose/analiz/efe/outputs
```

In this workspace, repository-root `final_dataset_parquets` currently contains only one parquet file, so dyad-level statistics and group-validated models are skipped there by design.

I also ran the same implementation on the complete 44-file parquet set described in the goal file:

```bash
/home/efekaan/Desktop/torch/torch_env/bin/python usta_pose/analiz/efe/run_hri_analysis.py --input-dir usta_pose/testing/final_dataset_parquets --output-dir usta_pose/analiz/efe/outputs_full
```

That full run produced 44 session summaries, 14,726 windows, a nonverbal vocabulary, dyad-level statistical contrasts, simple grouped models, and figures.

## Methodological Guardrails Implemented

- Primary analysis/modeling features do not treat `p1` and `p2` as identities.
- Models do not use filename, session ID, timestamp identity, or order metadata as input features.
- Model validation uses leave-one-dyad-out splitting when enough dyads are available.
- Statistical claims are based on session/dyad-level contrasts, not independent frame counts.
- Reports explicitly warn that `order1` versus later orders confounds competition and experience.

## Individual Nonverbal Conversation Extension

I added `run_nonverbal_conversation.py` to create anonymized individual words and response mappings. This script treats each participant separately inside each window, clusters person-level action/gaze/hand features into words (`IW00`, `IW01`, ...), then creates actor-switch response pairs such as:

```text
person_A:IW04 -> person_B:IW06
```

`person_A` and `person_B` are arbitrary within-dyad slots. The vocabulary and response models do not use p1/p2 as semantic identities.

The full conversation run is stored in `outputs_conversation_full/`. It produced 29,452 person-window rows, 10,739 primary actor turns, and 1,964 adjacent actor-switch response pairs. The academic interpretation is in `outputs_conversation_full/academic_conversation_interpretation.md`.

## Advanced Active-Response and Word-Generation Models

I added `run_advanced_conversation_models.py` to address the dominant waiting/listening response. It keeps monitoring words as meaningful, but creates a delayed active-response target by searching forward for the partner's next non-monitoring movement word.

It also trains movement-to-word encoders. These models take person-window movement/gaze features and predict the learned individual vocabulary word, including top-3 word proposals. XGBoost histogram GBDT is used when available; sklearn histogram gradient boosting is also run.

The full run is stored in `outputs_advanced_models/`. The academic interpretation is in `outputs_advanced_models/academic_advanced_model_interpretation.md`.

## Cross-Dyad Vocabulary Reliability Validation

I added `run_cv_vocabulary_validation.py` after noting that full-data clustering makes movement-to-word accuracy optimistic. The new validation learns clusters only on train dyads, assigns held-out dyads to those train centroids, and reports both train-vocabulary encoder performance and stability against the original full-data vocabulary.

This is documented in `outputs_cv_vocab_validation/cv_vocab_validation_report.md`.

## Academic Claim Strengthening Study

I added `run_academic_strengthening_studies.py` to make the nonverbal-communication claims more defensible. It collapses individual words into events, treats `IW01` and `IW11` as wait/monitoring turn states, searches for delayed active responses, and trains leave-one-dyad-out behavior-cloning models under three information regimes:

- `actor_only`: current actor event and movement features only.
- `actor_history`: current actor event plus previous symbolic turns.
- `observed_immediate`: actor event plus the responder's first post-actor event, useful for interpreting real recordings.

The study documents that current movement alone is weak, symbolic history gives moderate candidate-generation ability, and observed-immediate context strongly predicts delayed active responses. The results and academic wording are summarized in `ACADEMIC_CLAIM_STRENGTHENING.md` and `outputs_academic_strengthening/academic_strengthening_report.md`.
