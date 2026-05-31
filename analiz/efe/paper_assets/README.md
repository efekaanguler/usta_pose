# Paper Asset Pack

Title: Learning a Symbolic Nonverbal Vocabulary from Dyadic Pose and Gaze Interactions

Generated: 2026-05-30

Combined PDF: `paper_assets/pdf/paper_figures_tables_appendix_pack.pdf`

## Recommended Figure/Table Placement

| Asset | Recommended section | Files | Source files | Recommended caption |
|---|---|---|---|---|
| Figure 1 | Dataset section | `paper_assets/main_figures/fig1_experimental_setup.png`<br>`paper_assets/main_figures/fig1_experimental_setup.pdf` | `AGENTS.md`<br>`final_analysis.md` | Experimental setup and task structure. Each dyad completed four tabletop disk-ordering sessions while being recorded by four RealSense cameras. Session 1 served as the noncompetitive first-exposure baseline, whereas Sessions 2-4 introduced scoring and competitive/practiced interaction. |
| Table 1 | Dataset section | `paper_assets/main_tables/table1_dataset_summary.csv`<br>`paper_assets/main_tables/table1_dataset_summary.png`<br>`paper_assets/main_tables/table1_dataset_summary.pdf` | `results/validation/dataset_metadata.csv`<br>`Experimental task description from project prompt` | Summary of the processed dyadic tabletop interaction dataset. |
| Figure 2 | Methods section | `paper_assets/main_figures/fig2_pipeline.png`<br>`paper_assets/main_figures/fig2_pipeline.pdf` | `AGENTS.md`<br>`final_analysis.md` | Overview of the analysis pipeline. Multi-camera recordings were converted into synchronized pose/gaze tables, validated, segmented into temporal windows, discretized into symbolic nonverbal words, and evaluated through event-level response modeling. |
| Table 2 | Dataset or Results section | `paper_assets/main_tables/table2_validation_summary.csv`<br>`paper_assets/main_tables/table2_validation_summary.png`<br>`paper_assets/main_tables/table2_validation_summary.pdf` | `results/validation/missing_by_session.csv`<br>`results/validation/timestamp_frame_consistency.csv`<br>`results/validation/outliers_by_feature.csv` | Validation summary for missingness, temporal consistency, and extreme outliers. |
| Figure 3 | Results section, symbolic vocabulary subsection | `paper_assets/main_figures/fig3_iw_prevalence.png`<br>`paper_assets/main_figures/fig3_iw_prevalence.pdf` | `outputs_conversation_full/individual_vocabulary_summary.csv` | Prevalence of the learned individual nonverbal vocabulary. The vocabulary is dominated by IW01, interpreted as a monitoring/listening state, while the remaining words mostly correspond to rarer active manual movements differentiated by hand dominance, approach/withdrawal, and gaze direction. |
| Table 3 | Results section | `paper_assets/main_tables/table3_compact_iw_vocabulary.csv`<br>`paper_assets/main_tables/table3_compact_iw_vocabulary.png`<br>`paper_assets/main_tables/table3_compact_iw_vocabulary.pdf` | `outputs_conversation_full/individual_vocabulary_summary.csv` | Most important individual nonverbal words used in the main paper. |
| Figure 4 | Results section, delayed response analysis | `paper_assets/main_figures/fig4_action_monitoring_action.png`<br>`paper_assets/main_figures/fig4_action_monitoring_action.pdf` | `outputs_academic_strengthening/academic_strengthening_report.md` | Event-level symbolic analysis revealed a recurring action-monitoring-action structure. Participants often did not respond with an immediate active movement; instead, the first response was frequently a monitoring/listening state followed by a delayed active movement. |
| Figure 5 | Results section, response modeling | `paper_assets/main_figures/fig5_response_model_comparison.png`<br>`paper_assets/main_figures/fig5_response_model_comparison.pdf` | `outputs_academic_strengthening/event_bc_model_results.csv`<br>`outputs_academic_strengthening/event_bc_distribution_eval.csv` | Event-level response modeling results under leave-one-dyad-out evaluation. Actor-only models were weak, while symbolic interaction history improved plausible response prediction. Observed-immediate models performed best because they also observe the responder's first post-actor state, making them most appropriate for offline interaction interpretation rather than autonomous generation. |
| Table 5 | Results section | `paper_assets/main_tables/table5_response_model_summary.csv`<br>`paper_assets/main_tables/table5_response_model_summary.png`<br>`paper_assets/main_tables/table5_response_model_summary.pdf` | `outputs_academic_strengthening/event_bc_model_results.csv`<br>`outputs_academic_strengthening/event_bc_distribution_eval.csv` | Comparison of event-level response models for symbolic nonverbal interaction. |
| Figure 6 | Results or Discussion section | `paper_assets/main_figures/fig6_pair_identity_leakage.png`<br>`paper_assets/main_figures/fig6_pair_identity_leakage.pdf` | `results/classification/classification_results_all_windows.csv` | Pair-identity control experiment. High pair-recognition performance shows that dyads have recognizable signatures, justifying leave-one-dyad-out evaluation for condition and response modeling. |
| Table 4 | Results section | `paper_assets/main_tables/table4_stability_condition_signal.csv`<br>`paper_assets/main_tables/table4_stability_condition_signal.png`<br>`paper_assets/main_tables/table4_stability_condition_signal.pdf` | `outputs_cv_vocab_validation/cv_vocab_validation_summary.csv`<br>`results/classification/classification_results_all_windows.csv` | Cross-dyad vocabulary stability and best baseline-vs-competitive/practiced condition classifiers. |
| Appendix Figure A1 | Appendix | `paper_assets/appendix_figures/figA1_missingness_heatmap.png`<br>`paper_assets/appendix_figures/figA1_missingness_heatmap.pdf` | `results/validation/missing_by_session.csv` | Session-level missingness aggregated by dyad and session order. |
| Appendix Figure A2 | Appendix | `paper_assets/appendix_figures/figA2_token_transition_matrices.png`<br>`paper_assets/appendix_figures/figA2_token_transition_matrices.pdf` | `results/tokens/token_transitions_1s.csv`<br>`results/tokens/token_transitions_2s.csv`<br>`results/tokens/token_transitions_5s.csv`<br>`results/tokens/token_sequence_summary_all_windows.csv` | Baseline and competitive/practiced token transition matrices. JS divergences: 1s=0.0774, 2s=0.0903, 5s=0.1081. |
| Appendix Figure A3 | Appendix | `paper_assets/appendix_figures/figA3_condition_ablation.png`<br>`paper_assets/appendix_figures/figA3_condition_ablation.pdf` | `results/classification/classification_results_all_windows.csv` | Best non-dummy baseline-vs-competitive/practiced classifier per window size and feature set. |
| Appendix Table A1 | Appendix | `paper_assets/appendix_tables/tableA1_full_iw_vocabulary.csv`<br>`paper_assets/appendix_tables/tableA1_full_iw_vocabulary.png`<br>`paper_assets/appendix_tables/tableA1_full_iw_vocabulary.pdf` | `outputs_conversation_full/individual_vocabulary_summary.csv` | Full individual vocabulary with centroid-derived semantic interpretations. |
| Appendix Table A2 | Appendix | `paper_assets/appendix_tables/tableA2_top_statistical_contrasts.csv`<br>`paper_assets/appendix_tables/tableA2_top_statistical_contrasts.png`<br>`paper_assets/appendix_tables/tableA2_top_statistical_contrasts.pdf` | `results/statistics/session_statistics_all_windows.csv` | Top 10 exploratory paired contrasts sorted by raw p-value. |
| Appendix Table A3 | Appendix | `paper_assets/appendix_tables/tableA3_full_event_model_results.csv`<br>`paper_assets/appendix_tables/tableA3_full_event_model_results.png`<br>`paper_assets/appendix_tables/tableA3_full_event_model_results.pdf` | `outputs_academic_strengthening/event_bc_model_results.csv`<br>`outputs_academic_strengthening/event_bc_distribution_eval.csv` | Full actor-only, actor-history, observed-immediate, Markov, and majority model results. |
| Appendix Table A4 | Appendix | `paper_assets/appendix_tables/tableA4_feature_importance.csv`<br>`paper_assets/appendix_tables/tableA4_feature_importance.png`<br>`paper_assets/appendix_tables/tableA4_feature_importance.pdf` | `results/classification/permutation_feature_importance.csv` | Top 15 permutation feature importances for the best condition classifier analysis. |
| Appendix Table A5 | Appendix | `paper_assets/appendix_tables/tableA5_outlier_summary.csv`<br>`paper_assets/appendix_tables/tableA5_outlier_summary.png`<br>`paper_assets/appendix_tables/tableA5_outlier_summary.pdf` | `results/validation/outliers_by_feature.csv` | Top 10 features by z-score and IQR outlier ratios. |

## Missing Inputs or Assumptions

- No requested source file was missing during generation.
- Figure 1 and Figure 2 are clean schematics based on the project method description and existing analysis documentation.
- Table 1 includes task-design constants from the project description and processed dataset counts from `results/validation/dataset_metadata.csv`.
- Event grammar counts and lag values are parsed from `outputs_academic_strengthening/academic_strengthening_report.md`.

## Source Data Used

A copy or manifest entry for each source file used is stored in `source_data_used/`.

## Notes for Authors

- Use Figure 3 and Appendix Table A1 when discussing the learned IW vocabulary.
- Use Figure 4 and Table 5 when making claims about action-monitoring-action grammar and response prediction.
- Use Figure 6 to justify leave-one-dyad-out or pair-aware validation.
- Report condition-classification results as modest signal, not strong separability.
- Report observed-immediate response models as offline interpretation models because they observe the responder's first post-actor state.
- Report actor-history models as the closest autonomous/generative candidate, with moderate exact accuracy but useful top-3 and distributional behavior.