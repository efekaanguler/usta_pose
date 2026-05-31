# Efe HRI Parquet Analysis

This folder implements the parquet-only HRI analysis requested in `usta_pose/testing/goal.md`.

## Entry Point

```bash
/home/efekaan/Desktop/torch/torch_env/bin/python usta_pose/analiz/efe/run_hri_analysis.py --input-dir final_dataset_parquets --output-dir usta_pose/analiz/efe/outputs
```

The script reads interaction parquets, infers dyads from chronological order blocks, extracts person-label-invariant dyadic features, runs dyad-level statistics when enough orders are present, discovers nonverbal vocabulary tokens, trains simple leave-one-dyad-out models when possible, and writes a markdown report.

## Generated Runs

- `outputs/`: run on repository-root `final_dataset_parquets`. In this workspace that directory contains one parquet, so statistics/models are intentionally skipped and the report explains the limitation.
- `outputs_full/`: run on the full 44-file set under `usta_pose/testing/final_dataset_parquets`, producing 44 session summaries, 14,726 windows, dyad-level statistics, vocabulary tokens, simple models, figures, and a full report.

## Main Documentation

- `WHAT_I_IMPLEMENTED.md`: implementation explanation.
- `outputs/report.md`: report for the user-specified one-file input directory.
- `outputs_full/report.md`: report for the complete 44-file dataset.

## Individual Conversation Layer

A second script builds anonymized person-level words and response mappings:

```bash
/home/efekaan/Desktop/torch/torch_env/bin/python usta_pose/analiz/efe/run_nonverbal_conversation.py --input-dir usta_pose/testing/final_dataset_parquets --output-dir usta_pose/analiz/efe/outputs_conversation_full
```

Key outputs:

- `outputs_conversation_full/individual_words.parquet`
- `outputs_conversation_full/individual_vocabulary_summary.csv`
- `outputs_conversation_full/conversation_turns.parquet`
- `outputs_conversation_full/response_pairs.csv`
- `outputs_conversation_full/response_mapping.csv`
- `outputs_conversation_full/salient_nondefault_response_mapping.csv`
- `outputs_conversation_full/response_model_results.csv`
- `outputs_conversation_full/conversation_sequences.md`
- `outputs_conversation_full/conversation_report.md`
- `outputs_conversation_full/academic_conversation_interpretation.md`

## Advanced Active-Response Models

A third script addresses the dominant waiting/listening response by searching for the partner's next active movement word:

```bash
/home/efekaan/Desktop/torch/torch_env/bin/python usta_pose/analiz/efe/run_advanced_conversation_models.py --conversation-dir usta_pose/analiz/efe/outputs_conversation_full --output-dir usta_pose/analiz/efe/outputs_advanced_models --horizon-windows 12
```

Key outputs:

- `outputs_advanced_models/word_generation_model_results.csv`
- `outputs_advanced_models/word_generation_predictions.parquet`
- `outputs_advanced_models/active_response_pairs.csv`
- `outputs_advanced_models/active_response_mapping.csv`
- `outputs_advanced_models/active_response_model_results.csv`
- `outputs_advanced_models/model_talk_examples.md`
- `outputs_advanced_models/advanced_model_report.md`
- `outputs_advanced_models/academic_advanced_model_interpretation.md`

## Cross-Dyad Vocabulary Validation

To avoid overclaiming the movement-to-word encoder, a train-only clustering validation was added:

```bash
/home/efekaan/Desktop/torch/torch_env/bin/python usta_pose/analiz/efe/run_cv_vocabulary_validation.py --input usta_pose/analiz/efe/outputs_conversation_full/individual_words.parquet --output-dir usta_pose/analiz/efe/outputs_cv_vocab_validation --k 12
```

Key outputs:

- `outputs_cv_vocab_validation/cv_vocab_validation_summary.csv`
- `outputs_cv_vocab_validation/cv_vocab_validation_by_fold.csv`
- `outputs_cv_vocab_validation/cv_vocab_validation_report.md`

## Academic Claim Strengthening

A fourth script adds event-level delayed-response ablations and distributional human-likeness evaluation:

```bash
/home/efekaan/Desktop/torch/torch_env/bin/python usta_pose/analiz/efe/run_academic_strengthening_studies.py --conversation-dir usta_pose/analiz/efe/outputs_conversation_full --output-dir usta_pose/analiz/efe/outputs_academic_strengthening --horizon-windows 12
```

Key outputs:

- `outputs_academic_strengthening/academic_strengthening_report.md`
- `outputs_academic_strengthening/event_bc_model_results.csv`
- `outputs_academic_strengthening/event_bc_distribution_eval.csv`
- `outputs_academic_strengthening/event_actor_response_mapping.csv`
- `outputs_academic_strengthening/event_wait_continuation_mapping.csv`
- `ACADEMIC_CLAIM_STRENGTHENING.md`

## CENG488 Final Project Pipeline

The final requested all-in-one analysis pipeline is implemented in `run_ceng488_final_pipeline.py` and documented in `CENG488_FINAL_PIPELINE.md`.

Full run command:

```bash
/home/efekaan/Desktop/torch/torch_env/bin/python usta_pose/analiz/efe/run_ceng488_final_pipeline.py --input-dir usta_pose/testing/final_dataset_parquets --output-dir usta_pose/analiz/efe/results --window-sizes 1,2,5 --stride-ratio 1.0 --max-model-windows 8000 --max-plot-points 2500
```

Main summary: `results/FINAL_ANALYSIS_SUMMARY.md`.
