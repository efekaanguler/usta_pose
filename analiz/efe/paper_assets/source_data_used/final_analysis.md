# Final Analysis Report: Dyadic Pose/Gaze HRI Dataset

## 1. Purpose of This Document

This document synthesizes all analyses executed under `usta_pose/analiz/efe` for the dyadic interaction dataset. It is intended as a detailed academic working report from which the final CENG488 paper can selectively draw claims, methods, tables, and interpretations.

The dataset consists of processed dyadic interaction videos. Raw videos were already converted by the project pipeline into synchronized tabular `.parquet` files containing pose, gaze, dyad geometry, keypoint coordinates, validity flags, and derived interaction features. Each pair of participants completed four task sessions:

- Session/order 1: task-completion baseline, no competitive scoring.
- Sessions/orders 2, 3, and 4: competitive/practiced sessions with scoring based on numbered disks.

The central scientific question is whether the pose/gaze-derived tabular data contains reproducible evidence of behavioral changes across task context, increasing familiarity, dyadic coordination structure, and nonverbal communication patterns.

This report explicitly separates:

- strong or defensible claims;
- moderate exploratory claims;
- weak or unsupported claims;
- methodological limitations that must be reported.

A recurring caution applies throughout: Session 1 is both noncompetitive and first exposure. Therefore, Session 1 versus Sessions 2-4 should be interpreted as `baseline/noncompetitive first exposure versus competitive/practiced sessions`, not as a pure isolated competition effect.

## 2. Dataset and Processing Context

### 2.1 Source Data Used

The primary dataset used for the full analyses is:

```text
usta_pose/testing/final_dataset_parquets/
```

Observed dataset inventory:

- 44 parquet files.
- 11 dyads/pairs.
- 4 sessions/orders per dyad.
- 183,195 total frame rows.
- Approximately 3,436 columns per parquet.

The final parquet files are fused interaction tables. They contain pose/gaze information and calibration metadata, but they are not raw per-camera streams. This matters for camera-level validation: the pipeline can report camera-related metadata and gaze camera-id fields, but cannot fully validate each raw camera stream from these final tables alone.

### 2.2 Core Raw-to-Table Pipeline Context

The upstream processing pipeline, described in `AGENTS.md`, extracted pose and gaze from video and exported synchronized interaction parquets. The relevant upstream processing stages were:

- pose extraction from pose cameras, using RTMPose-style whole-body keypoints;
- gaze estimation from face/gaze cameras, using gaze pitch/yaw converted into 3D vectors;
- resampling and transformation into a shared coordinate frame;
- interaction parquet creation with absolute, person-relative, and dyad-relative features.

The analyses in this report did not re-run raw video inference. They use only the existing parquet tables.

## 3. Implemented Analysis Scripts and Output Folders

The following scripts were implemented or used under `usta_pose/analiz/efe`:

| Script | Purpose | Main output |
|---|---|---|
| `run_hri_analysis.py` | Initial dyadic HRI analysis using person-label-invariant window features, paired statistics, dyadic vocabulary, and simple grouped models. | `outputs_full/` |
| `run_nonverbal_conversation.py` | Individual anonymized word vocabulary and immediate actor-response conversation mapping. | `outputs_conversation_full/` |
| `run_advanced_conversation_models.py` | Delayed active-response analysis, wait-state handling, movement-to-word and past-context models. | `outputs_advanced_models/` |
| `run_cv_vocabulary_validation.py` | Train-only clustering validation to test vocabulary stability across held-out dyads. | `outputs_cv_vocab_validation/` |
| `run_academic_strengthening_studies.py` | Event-level response modeling and behavior-cloning ablations with leave-one-dyad-out evaluation. | `outputs_academic_strengthening/` |
| `run_ceng488_final_pipeline.py` | Full requested CENG488 pipeline: validation, feature engineering, 1s/2s/5s windows, statistics, clustering, token sequences, classification, ablations, explainability. | `results/` |

Additional documentation files produced:

- `WHAT_I_IMPLEMENTED.md`
- `README.md`
- `CENG488_FINAL_PIPELINE.md`
- `ACADEMIC_CLAIM_STRENGTHENING.md`
- `final_analysis.md` (this file)

## 4. Methodological Safeguards

Several safeguards were built into the analysis because the data are dyadic, repeated within pair, and heavily autocorrelated across frames/windows.

### 4.1 Pair/Person Identity Handling

`p1` and `p2` are table-side labels, not stable psychological identities. The analysis therefore avoids claims such as "person 1 is more dominant." For primary dyadic models and statistics, features were constructed symmetrically where possible:

- mean across participants;
- absolute difference between participants;
- min/max distances;
- dyadic synchrony and lag magnitudes;
- activity balance or asymmetry scores.

For individual vocabulary analyses, participant labels were anonymized as `person_A` and `person_B` within each dyad. These labels only make sequences readable; they are not interpreted as stable roles across dyads.

### 4.2 Leakage Prevention

All predictive claims that matter were evaluated using group-aware validation, especially leave-one-dyad/pair-out validation. This prevents windows from the same pair appearing in both train and test folds.

The pipelines avoid using filename, direct session identity, or raw timestamps as model features. Order/session labels are used as targets or grouping variables, not input identifiers.

### 4.3 Correct Statistical Unit

Frames and windows are not independent samples. Inferential statistics were therefore performed at the session or dyad level. The final pipeline ran within-pair comparisons using:

- Wilcoxon signed-rank tests;
- Friedman tests across S1-S4 where applicable;
- sign-permutation tests for paired robustness;
- effect sizes such as Cohen's dz and matched-pairs rank-biserial correlation;
- Benjamini-Hochberg FDR correction across many tested features.

This conservative approach matters because many raw window-level differences can look large simply due to autocorrelation and repeated measures.

## 5. Dataset Validation Results

The most complete validation was performed by `run_ceng488_final_pipeline.py`, with outputs under:

```text
results/validation/
```

### 5.1 Missingness

The final validation pass reported:

- Mean overall missing ratio across sessions: 0.1967.
- Worst session missing ratio: 0.4559.
- Minimum observed session missing ratio: 0.0672.

Sessions with the highest overall missingness included:

- `dyad_009 / 20260510_142206_order3`: 0.4559.
- `dyad_009 / 20260510_142353_order4`: 0.4103.
- `dyad_008 / 20260510_121708_order1`: 0.3891.
- `dyad_009 / 20260510_140907_order1`: 0.3885.
- `dyad_004 / 20260507_141202_order1`: 0.3499.

Interpretation: the dataset is usable, but missingness and tracking validity are non-trivial. Any gaze or pose claim should be interpreted alongside validity coverage. In particular, when model explainability points to validity/count features, this should be treated as a possible data-quality confound rather than a purely behavioral effect.

### 5.2 Timestamp and Frame Consistency

The final validation found:

- Total detected temporal gaps: 0.
- Largest temporal gap: 0.00 ms.
- Mean median frame interval: approximately 40.5 ms.

Interpretation: temporally, the dataset is stable enough for windowing and sequence analysis. This is one of the strongest technical claims: the synchronized parquet tables do not show timestamp discontinuities under the implemented gap detector.

### 5.3 Outlier Detection

Outliers were detected with z-score and IQR rules, after excluding boolean/flag columns from continuous outlier detection. The largest z-score outlier ratios were:

| Feature | z outlier ratio | IQR outlier ratio |
|---|---:|---:|
| `p2_motion_speed` | 0.0158 | 0.0373 |
| `p2_kpt50_world_y` | 0.0156 | 0.0273 |
| `p1_motion_speed` | 0.0154 | 0.0416 |
| `p2_kpt40_world_y` | 0.0153 | 0.0100 |
| `p2_kpt45_world_y` | 0.0147 | 0.0075 |

Interpretation: outlier ratios are not catastrophic but are concentrated in motion speed and some keypoint coordinates, especially for `p2` keypoints. This may reflect tracking noise, occlusion, or real rapid movements. The report should avoid interpreting extreme movement spikes without robustness filtering.

### 5.4 Camera-Level Limitation

The final parquets are fused interaction datasets. Per-camera validation is limited to available calibration fields and gaze camera-id metadata. Therefore, we can report missingness per camera-related feature family, but not a full raw camera stream health analysis from these parquets alone.

## 6. Feature Engineering and Windowing

The final pipeline created interpretable dyadic features across 1s, 2s, and 5s non-overlapping windows. Each window row includes metadata:

- `pair_id`
- `session_id`
- `order`
- `condition_label`
- `window_size_s`
- `start_time_s`
- `end_time_s`
- `participant_id` set to `dyad` for dyadic window features
- `camera_id` set to `fused`

### 6.1 Engineered Feature Families

The engineered features include:

- gaze category ratios: partner, table/task area, away/unknown;
- gaze switching rate;
- mutual gaze event ratio;
- joint attention to task/table area;
- body orientation difference proxy from shoulders/hips where available;
- gaze yaw/pitch summaries;
- movement intensity from pose/motion-energy features;
- lean-in / lean-out proxy from root distance delta;
- dyadic motion synchrony using zero-lag and lagged cross-correlation;
- asymmetric activity score between participants;
- validity and interaction-quality ratios.

Window tables were written to:

```text
results/features/window_features_1s.csv and .parquet
results/features/window_features_2s.csv and .parquet
results/features/window_features_5s.csv and .parquet
```

Session aggregates were written to:

```text
results/features/session_features_1s.csv
results/features/session_features_2s.csv
results/features/session_features_5s.csv
```

### 6.2 Window Counts

Across all window sizes, the final pipeline produced 12,477 windows:

- 1s windows: 7,354 model/evaluation samples after filtering/sample cap in classification outputs.
- 2s windows: 3,668 samples.
- 5s windows: 1,455 samples.

The earlier HRI-specific pipeline used a 1s/0.5s-style denser extraction and produced:

- 14,726 total windows.
- 10,739 quality-passing windows.
- Mean frame pose valid ratio: 0.744.
- Mean frame gaze valid ratio: 0.742.
- Mean frame interaction valid ratio: 0.744.

## 7. Dyadic HRI Analysis: Session-Level Statistics

Two related statistical analyses were performed.

### 7.1 Initial Dyadic HRI Pipeline Results

The initial `outputs_full/` run used person-label-invariant dyadic features and paired dyad-level contrasts. It reported:

- 44 sessions.
- 11 complete dyads.
- 14,726 total windows.
- 10,739 quality-passing windows.
- 0 strong and 0 moderate evidence rows after FDR correction.

Exploratory patterns included:

- activity state entropy increasing from order2 to order4;
- closest-hand-to-other-head distance decreasing in later sessions;
- hand-to-hand minimum distance decreasing in later sessions;
- motion asymmetry increasing in some competitive-later contrasts.

However, the q-values were high, and the confidence labels remained weak/exploratory.

Academic interpretation: the initial dyadic pipeline did not produce strong inferential evidence for session effects after multiple-comparison correction. It is most useful as a feature-extraction and hypothesis-generation layer.

### 7.2 Final Pipeline Statistical Results

The final CENG488 pipeline tested 2,415 statistical rows across 1s, 2s, and 5s windows. Results:

- Raw p < 0.05 rows: 274.
- FDR q < 0.05 rows: 0.
- Minimum q-value: 0.2592.

This is the most important statistical conclusion: many raw effects look suggestive, but none survive broad FDR correction across the full feature family.

Top exploratory paired contrasts by raw p-value included:

| Contrast | Feature | Window | Mean diff | dz | Raw p | q |
|---|---|---:|---:|---:|---:|---:|
| S1 vs S2-S4 | `any_partner_gaze_count_session_mean` | 1s | 2.608 | 1.092 | 0.0010 | 0.2592 |
| S1 vs S2-S4 | `any_partner_gaze_count_session_mean` | 2s | 5.214 | 1.098 | 0.0010 | 0.2592 |
| S1 vs S2-S4 | `p2_gaze_category_away_unknown_ratio_session_mean` | 2s | -0.067 | -0.974 | 0.0010 | 0.2592 |
| S1 vs S2-S4 | `p1_task_gaze_transition_count_session_mean` | 1s | 0.250 | 1.360 | 0.0020 | 0.2592 |
| S1 vs S4 | `p1_task_gaze_transition_count_session_mean` | 2s | 0.669 | 1.370 | 0.0020 | 0.2592 |
| S1 vs S2 | `dyadic_motion_synchrony_zero_lag_session_mean` | 5s | 0.055 | 1.432 | 0.0029 | 0.2592 |

Interpretation: the most consistent exploratory family is gaze/attention. Competitive/practiced sessions tend to show more partner-gaze counts and more task-gaze switching, while away/unknown gaze ratios decrease. However, because broad FDR correction removes conventional significance, this should be presented as an exploratory pattern rather than a confirmed effect.

Defensible wording:

> Across pairs, Sessions 2-4 showed consistent exploratory increases in partner-oriented gaze and task-gaze switching relative to Session 1. These effects had large paired effect sizes in several window scales but did not survive broad FDR correction across the complete feature set; therefore, they should be treated as hypothesis-generating rather than confirmatory.

## 8. Dyadic Behavioral Vocabulary From Window-Level Features

### 8.1 Initial Dyadic Vocabulary (`NV` Tokens)

The initial HRI pipeline clustered 10,739 quality-passing windows into 8 dyadic nonverbal vocabulary tokens (`NV00`-`NV07`). The silhouette score was 0.113.

The token descriptions were:

| Token | Description | Prevalence |
|---|---|---:|
| `NV00` | high motion; alternating/one-sided activity; partner gaze; typical distance; motion-asymmetric | 0.022 |
| `NV01` | moderate motion; alternating/one-sided activity; task-focused gaze; typical distance; motion-asymmetric | 0.190 |
| `NV02` | low motion; limited activity; partner gaze; typical distance | 0.189 |
| `NV03` | moderate motion; alternating/one-sided activity; task-focused gaze; separated posture | 0.233 |
| `NV04` | moderate motion; limited activity; partner gaze; typical distance | 0.049 |
| `NV05` | high motion; alternating/one-sided activity; partner gaze; close posture; motion-asymmetric | 0.171 |
| `NV06` | moderate motion; alternating/one-sided activity; partner gaze; typical distance | 0.121 |
| `NV07` | high motion; alternating/one-sided activity; partner gaze; close posture; motion-asymmetric | 0.025 |

Academic interpretation: the dyadic vocabulary is interpretable, but cluster separation is modest. It should be treated as an exploratory behavioral discretization rather than a final nonverbal lexicon.

### 8.2 Final Pipeline Clustering Across Window Sizes

The final pipeline evaluated KMeans, Gaussian Mixture Models, Agglomerative clustering, DBSCAN, and optional HDBSCAN. PCA and t-SNE visualizations were also generated. UMAP was skipped because `umap-learn` was not installed.

Best clustering candidates by silhouette:

| Window | Algorithm | k/param | Silhouette | Davies-Bouldin | Calinski-Harabasz | Stability ARI |
|---:|---|---|---:|---:|---:|---:|
| 5s | GMM | 2 | 0.108 | 2.798 | 147.6 | NA |
| 5s | KMeans | 2 | 0.101 | 2.877 | 156.5 | 0.982 |
| 2s | KMeans | 2 | 0.089 | 3.024 | 351.5 | 1.000 |
| 5s | KMeans | 3 | 0.086 | 2.872 | 130.9 | 0.984 |
| 2s | KMeans | 3 | 0.084 | 2.977 | 303.2 | 0.993 |
| 1s | KMeans | 2 | 0.084 | 3.165 | 648.2 | 1.000 |

Interpretation: KMeans solutions were highly stable across random seeds but had low silhouette scores. This means the algorithm repeatedly finds similar partitions, but the partitions are not strongly separated in feature space.

The final two-state cluster profiles were generally:

1. `moderate-motion + joint-table-attention`: higher validity, more partner/table attention, lower away/unknown gaze.
2. `moderate-motion + low-gaze + synchronous/asymmetric`: lower validity/interaction coverage and higher away/unknown gaze.

This is informative but also reveals a limitation: some cluster structure is influenced by validity and observability features. Therefore, the final behavioral tokens should not be overinterpreted as pure behavior states.

Defensible wording:

> Unsupervised clustering produced stable but weakly separated behavioral states. These states appear to capture a mixture of attentional behavior and data-validity/visibility structure. They are useful as exploratory descriptors but should not be presented as a universal nonverbal vocabulary.

## 9. Token Sequence Analysis

The final pipeline converted each session into behavioral token sequences and computed:

- token frequencies per session and condition;
- transition matrices;
- sequence entropy;
- most frequent 2-grams and 3-grams;
- session similarity from token-frequency vectors;
- Jensen-Shannon divergence between baseline and competitive transition distributions.

Baseline-versus-competitive transition divergence:

| Window size | JS divergence |
|---:|---:|
| 1s | 0.0774 |
| 2s | 0.0903 |
| 5s | 0.1081 |

Interpretation: symbolic transition distributions differ modestly between baseline and competitive/practiced sessions, with the strongest divergence at 5s. The effect is not large, but it supports the idea that task context affects the temporal organization of behavioral states.

Defensible wording:

> Baseline and competitive/practiced sessions showed modest differences in token transition structure, especially at the 5s temporal scale. This supports further investigation of nonverbal sequence organization but remains exploratory due to weak cluster separation.

## 10. Classification Experiments

Classification experiments were run under leave-one-pair-out cross-validation to prevent pair leakage. Targets were:

A. baseline vs competitive/practiced condition;
B. session number S1/S2/S3/S4;
C. pair identity as a control task.

Models included:

- Dummy baseline;
- Logistic Regression;
- Random Forest;
- Linear SVM.

Feature sets included:

- gaze-only;
- pose-only;
- combined gaze + pose.

### 10.1 Baseline vs Competitive/Practiced Classification

Best condition classifiers:

| Window | Feature set | Model | Accuracy | Macro F1 | ROC-AUC |
|---:|---|---|---:|---:|---:|
| 1s | pose | Random Forest | 0.707 | 0.621 | 0.613 |
| 5s | gaze | Random Forest | 0.689 | 0.615 | 0.619 |
| 5s | combined | Random Forest | 0.700 | 0.614 | 0.619 |
| 2s | pose | Random Forest | 0.712 | 0.610 | 0.607 |
| 5s | pose | Random Forest | 0.707 | 0.607 | 0.617 |
| 2s | combined | Random Forest | 0.681 | 0.601 | 0.603 |
| 1s | combined | Random Forest | 0.660 | 0.589 | 0.611 |

The dummy most-frequent baseline had accuracy around 0.707 because the dataset is imbalanced: each pair has one baseline session and three competitive/practiced sessions. However, dummy macro F1 was only about 0.414. Therefore, macro F1 is the more meaningful metric.

Interpretation: condition classification is modest but above dummy in macro F1. The model does not "accurately solve" the task, but it detects generalizable pose/gaze signal above a leakage-safe baseline.

Defensible wording:

> Pose/gaze-derived window features contained a modest but reproducible condition signal under leave-one-pair-out validation. Random Forest classifiers improved macro F1 over the majority baseline, although ROC-AUC remained modest and accuracy alone was inflated by class imbalance.

### 10.2 Session Number Classification

Session number classification was weak to moderate. Example results:

- 1s combined Random Forest: accuracy 0.348, macro F1 0.319, ROC-AUC 0.576.
- 1s pose Logistic Regression: accuracy 0.351, macro F1 0.330, ROC-AUC 0.590.
- 5s pose Random Forest: accuracy 0.355, macro F1 0.330, ROC-AUC 0.568.

Interpretation: models can detect some order-related structure, but not strongly. This is consistent with the idea that behavioral changes across orders are subtle, heterogeneous across dyads, and confounded by both experience and competition.

### 10.3 Pair Identity Control

Pair identity classification under leave-one-pair-out is mathematically undefined because the held-out pair is an unseen class. The pipeline correctly records this as not applicable.

A separate session-grouped pair-signature control was run. Results:

| Window | Model | Accuracy | Macro F1 | ROC-AUC |
|---:|---|---:|---:|---:|
| 1s | Random Forest | 0.660 | 0.607 | 0.924 |
| 2s | Random Forest | 0.704 | 0.659 | 0.940 |
| 5s | Random Forest | 0.768 | 0.728 | 0.961 |

Interpretation: the dataset contains strong pair-specific signatures. This is scientifically interesting but also a major leakage warning. Any model evaluated without pair-level grouping could appear to perform well by recognizing dyads rather than learning task-condition behavior.

Strong methodological claim:

> Pair-level grouping is essential for valid evaluation because dyads have highly recognizable motion/gaze signatures.

### 10.4 Shuffled-Label Sanity Checks

Shuffled-label condition classification produced near-random ROC-AUC:

- Logistic Regression ROC-AUC: approximately 0.48-0.50 across window sizes.
- Random Forest ROC-AUC: approximately 0.50-0.51 across window sizes.

Random Forest shuffled-label accuracy remained near the majority baseline because of class imbalance, but macro F1 and ROC-AUC showed that it was not learning true signal.

Interpretation: the real condition classifiers are not simply artifacts of the evaluation code; they detect some real signal, though the effect is modest.

## 11. Ablation Results: Gaze vs Pose vs Combined

The final pipeline compared condition classification using gaze-only, pose-only, and combined features.

Main observations:

- Best 1s model was pose-only Random Forest: macro F1 0.621.
- Best 5s gaze-only Random Forest: macro F1 0.615.
- Best 5s combined Random Forest: macro F1 0.614.
- Combined features did not consistently outperform gaze-only or pose-only models.

Interpretation: both pose and gaze independently carry condition-related information. Combining them may add noise, validity confounds, or redundant features in this small dataset.

Defensible wording:

> Both pose-only and gaze-only feature sets carried condition-related signal. Combined pose+gaze features did not reliably outperform the best single-modality models, suggesting either redundancy or noise introduced by the larger feature set.

## 12. Explainability Results

Permutation feature importance was computed for the best condition classifier. The top features were:

| Feature | Importance mean | Importance std |
|---|---:|---:|
| `p1_gaze_pitch_max` | 0.0103 | 0.0016 |
| `frame_gaze_valid_transition_count` | 0.0102 | 0.0018 |
| `frame_pose_valid_ratio` | 0.0094 | 0.0031 |
| `frame_gaze_valid_count` | 0.0093 | 0.0030 |
| `p1_task_gaze_count` | 0.0079 | 0.0014 |
| `p2_gaze_yaw_min` | 0.0078 | 0.0034 |
| `root_distance_max` | 0.0077 | 0.0031 |
| `p1_task_gaze_angle_max` | 0.0076 | 0.0023 |
| `frame_interaction_valid_count` | 0.0075 | 0.0028 |
| `frame_pose_valid_count` | 0.0074 | 0.0025 |

Interpretation: the condition classifier relied on a mixture of behavioral gaze/pose features and validity/visibility features. This weakens purely behavioral interpretation. It suggests the need for a confirmatory version of the classifier that excludes validity/count features and tests whether behavioral signal remains.

Defensible wording:

> Explainability analysis indicated that condition classification used both behavioral gaze/pose measures and data-validity features. Therefore, predictive performance should be interpreted as evidence of condition-related structure in the processed data, not as purely behavioral evidence without further validity-controlled analysis.

## 13. Individual Nonverbal Vocabulary (`IW` Words)

The dyadic vocabulary was extended into an individual-level nonverbal language using `run_nonverbal_conversation.py`.

### 13.1 How It Was Obtained

For each 1-second person-window, each participant was represented with person-local and partner-relative features, including:

- body motion;
- activity ratio;
- left/right wrist speed;
- hand movement toward or away from the partner;
- partner-directed gaze;
- task-directed gaze;
- partner activity;
- interpersonal distance.

The representation was anonymized. `person_A` and `person_B` are arbitrary within-dyad slots. The vocabulary and response models do not treat `p1` or `p2` as stable identities.

The full individual conversation run used:

- 44 sessions.
- 11 dyads.
- 29,452 person-window rows.
- 10,739 primary actor turns.
- 1,964 adjacent actor-switch response pairs.

### 13.2 Learned Individual Vocabulary

A 12-word vocabulary was learned: `IW00` to `IW11`.

The dominant word was `IW01`:

- Description: moderate body motion, brief/intermittent activity, balanced hands, partner-directed gaze, partner mostly passive.
- Window count: 12,842.
- Prevalence: 0.598.
- Prevalence by order: order1 0.656, order2 0.577, order3 0.538, order4 0.609.

Interpretation: `IW01` is best understood as an attentive monitoring/listening state, not as an explicit manual action.

Other notable words:

| Word | Description | Prevalence |
|---|---|---:|
| `IW08` | high body motion; left-hand dominant; left hand moves toward partner; partner-directed gaze | 0.047 |
| `IW06` | moderate body motion; balanced hands; left hand withdraws; task-directed gaze | 0.067 |
| `IW07` | moderate body motion; right hand moves toward partner; task-directed gaze | 0.054 |
| `IW04` | high body motion; right hand moves toward partner; task-directed gaze | 0.031 |
| `IW02` | high body motion; left hand withdraws; task-directed gaze | 0.041 |
| `IW11` | moderate body motion; balanced hands; partner-directed gaze; partner mostly passive | 0.064 |

Academic interpretation: the individual vocabulary separates a dominant monitoring/listening state from rarer active manual movement states. This supports the idea that nonverbal coordination is not only a sequence of movements, but also contains attention/backchannel-like states.

## 14. Immediate Nonverbal Response Mapping

The first response mapping examined adjacent actor-switch windows.

The dominant pattern was that many actor words were followed by `IW01` from the partner:

- `IW10 -> IW01`: P = 0.627, count = 42.
- `IW06 -> IW01`: P = 0.601, count = 89.
- `IW00 -> IW01`: P = 0.595, count = 44.
- `IW07 -> IW01`: P = 0.587, count = 61.
- `IW01 -> IW01`: P = 0.527, count = 564.

Initial response models had weak balanced performance:

- Unigram response baseline: balanced accuracy 0.083, macro F1 0.059.
- Actor-word Markov response: balanced accuracy 0.083, macro F1 0.059.
- Logistic context response: balanced accuracy 0.122, macro F1 0.096.

Interpretation: immediate responses are dominated by the monitoring/listening state. Actor word alone does not strongly predict diverse immediate response types. This led to the next analysis: delayed active responses.

## 15. Delayed Active Responses and the Waiting Problem

### 15.1 Motivation

The initial response mapping looked like:

```text
person acts -> other person waits/monitors
```

This is not wrong behaviorally, but it hides delayed active responses. The advanced analysis therefore treated `IW01` and `IW11` as wait/monitoring words and searched forward for the partner's next active non-monitoring word.

### 15.2 Advanced Window-Level Delayed Response Dataset

`run_advanced_conversation_models.py` found:

- 8,869 candidate response opportunities.
- 6,063 future active/non-monitoring responses within a 12-window horizon.
- Mean active response lag: 5.21 windows, approximately 2.6 seconds.

This suggests that responses are often delayed rather than immediate.

### 15.3 Delayed Active Response Patterns

Most common delayed active responses included:

- `IW00 -> IW08`: P = 0.266, count = 55.
- `IW09 -> IW08`: P = 0.263, count = 26.
- `IW07 -> IW08`: P = 0.257, count = 84.
- `IW04 -> IW08`: P = 0.249, count = 51.
- `IW02 -> IW08`: P = 0.237, count = 71.
- `IW06 -> IW08`: P = 0.225, count = 103.
- `IW07 -> IW06`: P = 0.211, count = 69.

Interpretation: after a monitoring state, the next active response often becomes a manual movement word, especially `IW08` or `IW06`. This points toward a two-step interaction structure.

### 15.4 Active Response Prediction Models

Poor/weak predictors:

| Model | Feature set | Balanced accuracy | Macro F1 | Top-3 |
|---|---|---:|---:|---:|
| actor-word Markov | actor word | 0.106 | 0.074 | 0.287 |
| logistic current | actor current movement + context words | 0.135 | 0.094 | 0.340 |
| XGBoost current | actor current numeric movement | 0.094 | 0.078 | 0.397 |
| XGBoost current+past3 | actor current and past numeric | 0.086 | 0.075 | 0.396 |

Strongest predictor:

| Model | Feature set | Balanced accuracy | Macro F1 | Top-3 |
|---|---|---:|---:|---:|
| immediate-response Markov | immediate response word | 0.568 | 0.582 | 0.659 |

Interpretation: the original actor movement is not enough to predict the partner's eventual active response. The partner's immediate monitoring/listening state carries much more information. This supports a grammar of the form:

```text
actor action -> responder monitoring/listening -> responder active movement
```

This is one of the most useful conceptual findings for HRI.

## 16. Movement-to-Word Encoders

The advanced models also trained encoders from continuous movement/gaze features to individual vocabulary words.

Current-window movement-to-word models:

| Model | Balanced accuracy | Macro F1 | Top-3 accuracy |
|---|---:|---:|---:|
| XGBoost histogram GBDT | 0.964 | 0.964 | 1.000 |
| HistGradientBoosting | 0.963 | 0.962 | 1.000 |
| Logistic Regression | 0.945 | 0.910 | 1.000 |
| Random Forest | 0.896 | 0.845 | 0.970 |

Past-only/past-context word prediction:

| Model | Balanced accuracy | Macro F1 | Top-3 accuracy |
|---|---:|---:|---:|
| Logistic past-3 | 0.260 | 0.175 | 0.575 |
| HistGradientBoosting past-3 | 0.123 | 0.130 | 0.791 |
| XGBoost past-3 | 0.121 | 0.123 | 0.798 |

Interpretation: once a vocabulary is defined, current movement/gaze features can reproduce the assigned word almost perfectly. However, this is expected because the vocabulary was clustered from similar features. Therefore, high encoder accuracy should not be used as independent proof that the vocabulary is scientifically valid.

Past-context prediction is much weaker, but top-3 performance suggests some sequential constraints. This supports the idea of plausible candidate generation rather than deterministic next-word prediction.

## 17. Cross-Dyad Vocabulary Reliability Validation

To address the optimistic nature of clustering and then encoding the same labels, `run_cv_vocabulary_validation.py` performed train-only clustering in leave-one-dyad-out folds.

### 17.1 How It Was Obtained

For each held-out dyad:

1. KMeans vocabulary was learned only on the training dyads.
2. Held-out dyad windows were assigned to train-learned centroids.
3. Train-only cluster labels were mapped to the original full-data word names via majority mapping.
4. Stability against the original full-data vocabulary was measured.

### 17.2 Results

Key reliability metrics:

- Train-only vocabulary stability against original full vocabulary:
  - balanced accuracy 0.627;
  - macro F1 0.586;
  - adjusted Rand index 0.791;
  - normalized mutual information 0.749.

Supervised recoverability of train-vocabulary assignments:

| Model | Target | Balanced accuracy | Macro F1 | Top-3 |
|---|---|---:|---:|---:|
| XGBoost | train vocabulary cluster | 0.955 | 0.955 | 1.000 |
| HistGradientBoosting | train vocabulary cluster | 0.936 | 0.929 | 0.996 |
| Logistic Regression | train vocabulary cluster | 0.931 | 0.891 | 0.997 |

Global-word-via-train-vocabulary:

| Model | Balanced accuracy | Macro F1 | Top-3 |
|---|---:|---:|---:|
| Logistic Regression | 0.639 | 0.585 | 0.923 |
| XGBoost | 0.627 | 0.585 | 0.886 |
| HistGradientBoosting | 0.609 | 0.567 | 0.880 |

Interpretation: the individual vocabulary has moderate cross-dyad stability. It is not arbitrary, but it is also not a definitive universal vocabulary. The strongest vocabulary claim is moderate stability and practical symbolic usefulness.

Defensible wording:

> A train-only clustering validation indicated moderate cross-dyad stability of the individual vocabulary. Supervised encoders reproduced train-vocabulary assignments with high accuracy, but this reflects recoverability of learned cluster labels rather than independent behavioral validity.

## 18. Event-Level Nonverbal Conversation Modeling

The final academic-strengthening study moved from fixed windows to event-level symbolic conversations.

### 18.1 How Events Were Constructed

Consecutive identical individual words were collapsed into events. For every active actor event, the pipeline searched within a 12-window horizon for:

1. the first event from the other participant, which may be wait/monitoring;
2. the next active non-wait response from the other participant.

`IW01` and `IW11` were treated as wait/monitoring words.

### 18.2 Event Dataset

The event-level run produced:

- 12,069 events.
- 5,894 active actor response opportunities.
- 5,102 found delayed active responses.
- 1,966 cases where the first other-person event was wait/monitoring before a later active response.
- Mean delayed active response lag: 3.23 windows.
- Median delayed active response lag: 2.00 windows.

Interpretation: waiting/monitoring is not merely absence of response. It frequently functions as an intermediate turn state before later active movement.

### 18.3 Event-Level Response Models

All event-level models used leave-one-dyad-out validation and predicted 10 active response classes over 5,102 active-response examples.

#### Actor-only models

These use only the actor's current event and movement features. Results:

| Model | Accuracy | Balanced accuracy | Macro F1 | Top-3 |
|---|---:|---:|---:|---:|
| Logistic actor-only | 0.093 | 0.117 | 0.091 | 0.315 |
| Random Forest actor-only | 0.097 | 0.108 | 0.091 | 0.338 |
| HistGBDT actor-only | 0.166 | 0.104 | 0.096 | 0.441 |
| XGBoost actor-only | 0.181 | 0.102 | 0.076 | 0.493 |
| Actor Markov | 0.185 | 0.097 | 0.061 | 0.521 |

Interpretation: current actor movement alone does not determine the partner's active response. This prevents overclaiming autonomous exact-response prediction.

#### Actor-history models

These use the current actor event plus previous symbolic turns. Results:

| Model | Accuracy | Balanced accuracy | Macro F1 | Top-3 |
|---|---:|---:|---:|---:|
| XGBoost actor-history | 0.470 | 0.403 | 0.429 | 0.677 |
| Logistic actor-history | 0.427 | 0.425 | 0.390 | 0.605 |
| Random Forest actor-history | 0.440 | 0.408 | 0.398 | 0.629 |
| HistGBDT actor-history | 0.442 | 0.376 | 0.401 | 0.656 |
| Actor-history Markov | 0.448 | 0.382 | 0.412 | 0.641 |

Interpretation: symbolic history gives a moderate ability to generate plausible response candidates. Exact prediction is not strong, but top-3 candidate quality is meaningful.

#### Observed-immediate models

These models see the actor event and the responder's first post-actor event. This is not fully autonomous generation; it is best interpreted as offline conversation interpretation.

| Model | Accuracy | Balanced accuracy | Macro F1 | Top-3 |
|---|---:|---:|---:|---:|
| XGBoost observed-immediate | 0.771 | 0.739 | 0.779 | 0.861 |
| Random Forest observed-immediate | 0.759 | 0.751 | 0.731 | 0.823 |
| Logistic observed-immediate | 0.748 | 0.747 | 0.702 | 0.826 |
| HistGBDT observed-immediate | 0.748 | 0.718 | 0.757 | 0.839 |
| Actor-immediate Markov | 0.709 | 0.663 | 0.740 | 0.810 |

Interpretation: once the partner's first post-actor state is observed, the eventual active response is highly structured. This supports the existence of a reproducible event-level nonverbal grammar in the recordings.

### 18.4 Distributional Human-Likeness

Response distribution matching was measured using Jensen-Shannon divergence. Lower is better.

Best overall JS divergence:

| Model | JS divergence |
|---|---:|
| Random Forest observed-immediate | 0.0022 |
| Random Forest actor-history | 0.0025 |
| HistGBDT observed-immediate | 0.0058 |
| XGBoost observed-immediate | 0.0068 |
| XGBoost actor-history | 0.0163 |
| Majority baseline | 0.6055 |

Interpretation: even when exact next-word prediction is moderate, actor-history models can reproduce the overall human response distribution very closely. This supports a distributional imitation claim, not a deterministic response claim.

Defensible wording:

> Models using symbolic interaction history produced plausible response candidates and matched the aggregate human response distribution closely, although exact autonomous response prediction remained limited.

## 19. Nonverbal Grammar Interpretation

Across the individual, delayed-response, and event-level analyses, the clearest grammar pattern is:

```text
actor active word -> responder wait/monitoring word -> responder active word
```

This structure appeared repeatedly:

- Immediate response mappings were dominated by `IW01` monitoring/listening.
- Delayed-response analysis showed that active movement often comes after a monitoring period.
- Event-level models showed that observing the responder's first post-actor event strongly improves prediction of the later active response.

Interpretation for HRI:

A human-like agent should not always respond to a partner's action with an immediate action. A more human-like policy may need to first produce an attentive monitoring/backchannel state, then select a later active response after additional context. This is a strong conceptual result, although the learned vocabulary itself remains exploratory.

## 20. Reinforcement Learning / PPO Assessment

The user asked whether PPO or another RL agent would make the work more credible. Based on the dataset and results, RL is not currently the strongest academic route.

Reasons:

- The dataset contains approximately two hours of recordings, not an interactive environment.
- PPO would require a hand-designed symbolic simulator and reward function.
- A high PPO return would mostly show that the agent optimized our reward, not that it learned human nonverbal communication.
- Behavior cloning and held-out distributional evaluation are more credible with the available data.

Recommended position:

1. Use behavior cloning, Markov baselines, and leave-one-dyad-out evaluation as the main evidence.
2. Evaluate generated symbolic responses with top-k accuracy, JS divergence, transition matrices, lag distributions, and held-out dyads.
3. If RL is added later, initialize from behavior cloning and use RL only for constrained symbolic fine-tuning.
4. Report PPO/RL as exploratory unless it improves held-out human-likeness metrics beyond behavior cloning.

## 21. Claim Strength Assessment

### 21.1 Strongest Defensible Claims

1. The processed dataset is temporally consistent and suitable for windowed dyadic analysis.

Evidence: 44 sessions, 183,195 frame rows, no detected timestamp gaps, largest temporal gap 0 ms.

2. Pair-level grouped validation is necessary.

Evidence: session-grouped pair identity control reached high pair-recognition performance (5s Random Forest accuracy 0.768, macro F1 0.728, ROC-AUC 0.961), showing strong pair-specific signatures.

3. Pose/gaze-derived features contain a modest generalizable condition signal.

Evidence: leave-one-pair-out condition classification improved macro F1 over dummy baselines. Best result: 1s pose Random Forest macro F1 0.621, ROC-AUC 0.613. Shuffled-label ROC-AUC was near 0.5.

4. The individual symbolic vocabulary has moderate cross-dyad stability.

Evidence: train-only clustering validation produced global-word stability balanced accuracy 0.627, macro F1 0.586, ARI 0.791, NMI 0.749.

5. Nonverbal interaction often follows an action-monitoring-action structure.

Evidence: immediate responses were dominated by `IW01`; 1,966 event-level cases had wait/monitoring as first response before later active response; observed-immediate models predicted delayed active responses with balanced accuracy up to 0.751 and macro F1 up to 0.779.

6. Symbolic interaction history can generate plausible response candidates and match response distributions.

Evidence: actor-history top-3 accuracy up to 0.677 and JS divergence as low as 0.0025 for Random Forest actor-history distribution matching.

### 21.2 Moderate / Exploratory Claims

1. Competitive/practiced sessions show increased partner/task-oriented gaze behavior relative to Session 1.

Evidence: large raw paired effects for partner gaze and task-gaze switching, but no FDR-significant results across all tested features. This should be written as exploratory.

2. Token transition structure differs between baseline and competitive/practiced sessions.

Evidence: JS divergence between baseline and competitive transition distributions increased from 0.077 at 1s to 0.108 at 5s. Cluster separation was weak, so this is exploratory.

3. Both pose and gaze independently encode condition-related information.

Evidence: pose-only and gaze-only Random Forest classifiers each achieved macro F1 around 0.61 in their best settings, but performance remains modest.

4. Dyadic vocabulary clusters are stable but weakly separated.

Evidence: KMeans stability ARI near 1.0 for k=2, but silhouette only around 0.08-0.10.

### 21.3 Weak or Unsupported Claims

The following should not be stated as firm conclusions:

- "Competition significantly changes behavior" as a broad claim. No broad FDR-corrected statistical result reached q < 0.05.
- "Session 1 versus later sessions isolates competition." It is confounded with practice/familiarity.
- "We discovered a universal nonverbal vocabulary." The vocabulary is exploratory and dataset-specific.
- "Current movement alone can predict the partner's exact response." Actor-only models were near chance in balanced accuracy.
- "High movement-to-word accuracy proves vocabulary validity." High encoder accuracy is expected because clustering labels were derived from the same feature space.
- "PPO/RL is more academically credible than behavior cloning." Without an environment and validated reward, RL would mainly optimize hand-designed objectives.

## 22. Recommended Academic Framing

A strong and defensible abstract-level framing would be:

> We analyze dyadic human interaction recordings converted into synchronized pose/gaze parquet tables. Using pair-aware validation and person-label-invariant features, we find that the processed dataset is temporally consistent and contains modest but reproducible task-condition signal under leave-one-pair-out evaluation. Exploratory statistics suggest increased partner/task-oriented gaze and gaze switching in competitive/practiced sessions relative to the baseline first exposure, although these effects do not survive broad FDR correction. We further derive an anonymized individual-level symbolic vocabulary and show moderate cross-dyad vocabulary stability. Event-level sequence modeling reveals a recurring action-monitoring-action structure: participants often respond to another's action first with an attentive monitoring state, followed by a delayed active movement. Symbolic history models generate plausible response candidates and closely match aggregate human response distributions, while exact autonomous response prediction from a single current movement remains weak.

A concise results statement for the paper could be:

> The strongest empirical result is not a single statistically significant behavioral contrast, but a reproducible analysis pipeline showing that dyadic pose/gaze data carry modest task-condition information and support an interpretable, moderately stable symbolic representation of nonverbal interaction. The most meaningful interaction pattern is delayed response through monitoring: actions are often followed by partner attention before later active movement.

## 23. Suggested Paper Structure From These Results

### 23.1 Methods

Include:

- dataset description: 11 dyads, 44 sessions, S1 baseline, S2-S4 competitive/practiced;
- upstream pose/gaze parquet generation;
- validation metrics: missingness, timestamp consistency, outliers;
- feature families: gaze category, task gaze, mutual gaze, motion intensity, synchrony, activity asymmetry, lean proxy;
- windowing: 1s, 2s, 5s;
- statistical tests: Wilcoxon, Friedman, sign-permutation, FDR;
- classification: Leave-One-Pair-Out, dummy/logistic/RF/SVM, ablations;
- vocabulary: clustering, PCA/t-SNE, token transitions;
- individual conversation: anonymized individual words, event-level delayed-response analysis.

### 23.2 Results

Recommended order:

1. Dataset quality and temporal consistency.
2. Exploratory session differences in gaze/task attention.
3. Condition classification and ablations.
4. Pair identity control and leakage warning.
5. Dyadic tokenization and transition divergence.
6. Individual vocabulary and monitoring-state dominance.
7. Event-level delayed response grammar.
8. Behavior-cloning and distributional imitation.

### 23.3 Discussion

Main discussion points:

- HRI relevance of monitoring/listening states.
- Why immediate action-reaction is the wrong model for this task.
- Importance of pair-level validation.
- Difference between predictive signal and statistically significant behavioral claim.
- Vocabulary as a useful symbolic representation, not a final lexicon.
- Future work: validity-controlled classifiers, event segmentation by motion onset, larger dataset, manual annotation, robot policy evaluation.

## 24. Files to Cite in the Academic Report

Primary summary files:

- `results/FINAL_ANALYSIS_SUMMARY.md`
- `outputs_full/report.md`
- `outputs_conversation_full/academic_conversation_interpretation.md`
- `outputs_advanced_models/academic_advanced_model_interpretation.md`
- `outputs_cv_vocab_validation/cv_vocab_validation_report.md`
- `ACADEMIC_CLAIM_STRENGTHENING.md`

Primary result tables:

- `results/validation/missing_by_session.csv`
- `results/validation/timestamp_frame_consistency.csv`
- `results/validation/outliers_by_feature.csv`
- `results/features/window_features_1s.csv`
- `results/features/window_features_2s.csv`
- `results/features/window_features_5s.csv`
- `results/statistics/session_statistics_all_windows.csv`
- `results/clustering/cluster_quality_all_windows.csv`
- `results/clustering/cluster_profiles_1s.csv`
- `results/clustering/cluster_profiles_2s.csv`
- `results/clustering/cluster_profiles_5s.csv`
- `results/tokens/token_sequence_summary_all_windows.csv`
- `results/classification/classification_results_all_windows.csv`
- `results/classification/permutation_feature_importance.csv`
- `outputs_conversation_full/individual_vocabulary_summary.csv`
- `outputs_advanced_models/word_generation_model_results.csv`
- `outputs_advanced_models/active_response_model_results.csv`
- `outputs_cv_vocab_validation/cv_vocab_validation_summary.csv`
- `outputs_academic_strengthening/event_bc_model_results.csv`
- `outputs_academic_strengthening/event_bc_distribution_eval.csv`

## 25. Final Takeaway

The overall evidence supports a careful but valuable academic contribution. The project should not be framed as having discovered definitive behavioral laws or a universal nonverbal language. Instead, the defensible contribution is a reproducible HRI analysis pipeline that:

1. validates a dyadic pose/gaze parquet dataset;
2. extracts interpretable dyadic and individual features;
3. demonstrates modest task-condition signal under leakage-safe validation;
4. shows that pair identity is a serious confound if grouping is ignored;
5. constructs a moderately stable symbolic vocabulary;
6. identifies monitoring/listening as a central nonverbal response state;
7. models delayed active responses better than immediate action-reaction models;
8. provides a foundation for future human-like nonverbal response generation.

The most scientifically interesting behavioral insight is the action-monitoring-action structure. In this task, nonverbal communication often appears less like immediate turn exchange and more like attentional buffering: one participant acts, the other monitors, and only later produces an active movement. For HRI, this suggests that human-like agents may need to model waiting, attention, and delayed response as meaningful communicative actions rather than as inactivity.

## 26. Selected Best Vocabulary and Best Models

This section explicitly selects the most useful vocabulary and model family from all experiments. The goal is to make the final academic report easier to write: instead of treating every output as equally important, this section identifies which vocabulary and which models should be foregrounded, what they do, how they were built, and what their scores mean.

### 26.1 Best Vocabulary: Individual 12-Word `IW` Vocabulary

The best vocabulary for the academic report is the individual-level 12-word vocabulary `IW00`-`IW11`, created by `run_nonverbal_conversation.py` and used by the later delayed-response and event-level modeling scripts.

This vocabulary is better than the dyadic `NV` vocabulary for the final nonverbal-language claim because it represents each person separately. The dyadic `NV` vocabulary describes whole-dyad window states, which is useful for session-level behavioral analysis, but it cannot naturally express a conversation such as:

```text
person_A performs word IW04 -> person_B responds with word IW06
```

The `IW` vocabulary can express this kind of anonymized individual turn structure. It is therefore the best candidate for a nonverbal communication vocabulary.

#### How the `IW` Vocabulary Was Created

Each 1-second window was represented twice: once for each participant. The participant identity was anonymized into within-dyad slots, so the model does not treat `p1` or `p2` as stable identities. For each person-window, the script computed person-local and partner-relative features, including:

- body motion energy and maximum motion;
- active ratio and motion burst count;
- left and right wrist speed;
- dominant-hand speed;
- left/right/closest hand distance to the partner's head;
- hand approach or withdrawal over the window;
- partner-directed gaze ratio;
- task-directed gaze ratio;
- gaze switching rate;
- root/interpersonal distance;
- partner active ratio.

The vocabulary was learned with clustering over these 19 movement/gaze/context features. The full individual-word dataset contained:

- 29,452 person-window rows;
- 12 learned words;
- 44 sessions;
- 11 dyads.

The full-data vocabulary is useful for symbolic analysis, but its reliability was separately checked using train-only clustering. In leave-one-dyad-out train-only vocabulary validation, held-out dyads mapped back to the original full-data vocabulary with:

- balanced accuracy: 0.627;
- macro F1: 0.586;
- adjusted Rand index: 0.791;
- normalized mutual information: 0.749.

Interpretation: the `IW` vocabulary is not arbitrary, because train-only clusterings recover similar broad word structure on held-out dyads. However, it is not strong enough to be called a universal nonverbal lexicon. The correct academic phrasing is that it is a moderately stable, dataset-derived symbolic representation of individual nonverbal behavior.

### 26.2 Detailed Explanation of Every `IW` Word

The following word interpretations are based on the cluster centroids in `outputs_conversation_full/individual_vocabulary_summary.csv`. These names are not manually annotated ground truth labels; they are tentative semantic descriptions derived from movement, hand, gaze, and partner-context features.

| Word | Prevalence | Main centroid description | Interpretation |
|---|---:|---|---|
| `IW00` | 3.02% | Moderate body motion; brief/intermittent activity; balanced hands; right hand moves toward partner; task-directed gaze; partner mostly passive. | A task-oriented right-hand approach or placement-like action. Because gaze is task-directed and the partner is mostly passive, this may represent a participant manipulating or preparing an object while the other monitors. |
| `IW01` | 59.79% | Moderate body motion; brief/intermittent activity; balanced hands; partner-directed gaze; partner mostly passive. | The dominant monitoring/listening word. This is the core nonverbal attention state. It should not be treated as empty waiting; it often functions like an attentive backchannel or readiness state. |
| `IW02` | 4.06% | High body motion; brief/intermittent activity; balanced hands; left hand withdraws; task-directed gaze; partner mostly passive. | A task-focused left-hand withdrawal or completion movement. This may occur after reaching, placing, or adjusting an object, when attention remains on the task area. |
| `IW03` | 0.86% | High body motion; brief/intermittent activity; right-hand dominant; right hand moves toward partner; partner-directed gaze; partner mostly passive. | A rare, salient right-hand approach with partner monitoring. Because it combines partner gaze and right-hand dominance, it may correspond to a communicative reach, offer, or assertive manual action, but counts are low. |
| `IW04` | 3.12% | High body motion; brief/intermittent activity; balanced hands; right hand moves toward partner; task-directed gaze; partner mostly passive. | A stronger task-focused right-hand action than `IW00`. It likely captures active manipulation or reaching toward the shared workspace while visual attention remains on the task rather than the partner. |
| `IW05` | 1.61% | High body motion; brief/intermittent activity; left-hand dominant; partner-directed gaze; partner mostly passive. | A left-hand-dominant active state with partner monitoring. This may reflect action performed while checking the partner, but it is relatively rare and should be interpreted cautiously. |
| `IW06` | 6.67% | Moderate body motion; brief/intermittent activity; balanced hands; left hand withdraws; task-directed gaze; partner mostly passive. | A common task-focused left-hand withdrawal/action word. It appears frequently as an active response candidate and may represent completing or retracting a manual move after task interaction. |
| `IW07` | 5.38% | Moderate body motion; brief/intermittent activity; balanced hands; right hand moves toward partner; task-directed gaze; partner mostly passive. | A common task-focused right-hand approach word. It is similar to `IW00`/`IW04` but occurs more often and with moderate rather than high motion. |
| `IW08` | 4.72% | High body motion; brief/intermittent activity; left-hand dominant; left hand moves toward partner; partner-directed gaze; partner mostly passive. | The most important active response word in the delayed-response grammar. It often appears as the later active answer after a monitoring phase. It likely represents an active left-hand approach/reach while visually monitoring the partner. |
| `IW09` | 1.01% | High body motion; brief/intermittent activity; right-hand dominant; left hand withdraws; partner-directed gaze; partner mostly passive. | A rare mixed-action word: right-hand dominant body movement with left-hand withdrawal and partner gaze. It may represent a transition between manual action and partner monitoring, but low prevalence limits confidence. |
| `IW10` | 3.34% | Moderate body motion; brief/intermittent activity; balanced hands; right hand moves toward partner; partner-directed gaze; partner mostly passive. | A partner-monitored right-hand approach word. Compared with `IW07`, gaze is partner-directed rather than task-directed, so it may be more socially coordinated or checking-oriented. |
| `IW11` | 6.40% | Moderate body motion; brief/intermittent activity; balanced hands; partner-directed gaze; partner mostly passive. | A secondary monitoring/waiting word, similar in function to `IW01` but less dominant. In delayed-response analysis it was treated as a wait/monitoring state, and it often preceded later active responses such as `IW08`. |

#### Vocabulary-Level Interpretation

The vocabulary has one dominant monitoring state, `IW01`, and a secondary monitoring state, `IW11`. The other words mostly describe active manual movements differentiated by:

- left versus right hand dominance;
- approach versus withdrawal;
- task-directed versus partner-directed gaze;
- moderate versus high body motion.

The most important scientific point is that monitoring/listening is not noise. The dominant nonverbal word is an attention state, and later event-level modeling showed that such states often occur between another person's action and one's own later active response.

### 26.3 Why This Vocabulary Is the Best Choice

The `IW` vocabulary is selected as the best vocabulary because it satisfies four requirements better than the alternatives:

1. It is individual-level, so it can represent turn-like nonverbal exchanges.
2. It is anonymized, so it does not rely on `p1` or `p2` as stable identities.
3. It has moderate cross-dyad stability under train-only validation.
4. It supports downstream response modeling, delayed-response analysis, and behavior-cloning experiments.

The dyadic `NV` vocabulary remains useful for describing global interaction states, but it is less appropriate for the project's long-term goal of representing sessions as nonverbal conversations.

### 26.4 Best Model for Offline Conversation Interpretation

The best model for offline interpretation of real recordings is the event-level observed-immediate response model, especially the `xgboost_observed_immediate` and `random_forest_observed_immediate` variants from `run_academic_strengthening_studies.py`.

This model predicts the partner's eventual active response word after observing:

- the actor's current event word;
- the actor event's movement/gaze summary features;
- the other participant's first post-actor event word;
- previous symbolic context;
- event duration and lag information.

The target is the responder's next active non-wait word. `IW01` and `IW11` are treated as wait/monitoring states and are not used as final active-response labels.

The data for this model consisted of:

- 12,069 collapsed symbolic events;
- 5,894 active actor response opportunities;
- 5,102 found active delayed responses;
- 10 active response classes;
- leave-one-dyad-out evaluation.

Best observed-immediate scores:

| Model | Accuracy | Balanced accuracy | Macro F1 | Top-3 accuracy | JS divergence |
|---|---:|---:|---:|---:|---:|
| `xgboost_observed_immediate` | 0.771 | 0.739 | 0.779 | 0.861 | 0.0068 |
| `random_forest_observed_immediate` | 0.759 | 0.751 | 0.731 | 0.823 | 0.0022 |
| `logistic_observed_immediate` | 0.748 | 0.747 | 0.702 | 0.826 | 0.0084 |
| `hist_gbdt_observed_immediate` | 0.748 | 0.718 | 0.757 | 0.839 | 0.0058 |
| Markov actor+immediate baseline | 0.709 | 0.663 | 0.740 | 0.810 | 0.0394 |
| Majority baseline | 0.203 | 0.100 | 0.034 | 0.523 | 0.6055 |

Interpretation:

- `xgboost_observed_immediate` is the best exact classifier by accuracy and macro F1.
- `random_forest_observed_immediate` is the best distribution-matching model by JS divergence and has the best balanced accuracy among the observed-immediate models.
- Both models strongly outperform majority and actor-only baselines.

This is the strongest model for explaining real recordings because it shows that once the partner's first response state is observed, the later active response is highly structured.

However, this is not a fully autonomous generative model. It observes the responder's first post-actor state, which would not be available to an agent that must decide a response before the responder acts.

Defensible academic claim:

> The event-level observed-immediate model shows that delayed active responses are highly structured once the responder's first monitoring/action state is known. This supports the existence of a reproducible symbolic event grammar in the recordings.

### 26.5 Best Model for Autonomous or Generative Use

For autonomous response generation, the observed-immediate model is too informed because it sees the responder's first post-actor event. The best autonomous candidate is therefore the actor-history event model.

This model predicts the partner's active response from:

- the actor's current event;
- actor event movement/gaze summaries;
- previous symbolic turns;
- previous response words;
- session/order context.

It does not observe the responder's immediate future state.

Best actor-history scores:

| Model | Accuracy | Balanced accuracy | Macro F1 | Top-3 accuracy | JS divergence |
|---|---:|---:|---:|---:|---:|
| `xgboost_actor_history` | 0.470 | 0.403 | 0.429 | 0.677 | 0.0163 |
| `logistic_actor_history` | 0.427 | 0.425 | 0.390 | 0.605 | 0.0275 |
| `random_forest_actor_history` | 0.440 | 0.408 | 0.398 | 0.629 | 0.0025 |
| `hist_gbdt_actor_history` | 0.442 | 0.376 | 0.401 | 0.656 | 0.0166 |
| Actor-history Markov baseline | 0.448 | 0.382 | 0.412 | 0.641 | 0.0295 |

Interpretation:

- `xgboost_actor_history` is the best exact/top-3 autonomous candidate by accuracy, macro F1, and top-3 accuracy.
- `random_forest_actor_history` is the best distributional generator by JS divergence, meaning its aggregate response distribution is closest to the human response distribution.
- Exact prediction remains moderate, so the model should not be described as deterministic human-response prediction.

Best academic wording:

> For autonomous response generation, symbolic history supports plausible candidate generation rather than exact deterministic prediction. The best actor-history models reached top-3 accuracy up to 0.677 and closely matched the aggregate human response distribution, but exact balanced accuracy remained moderate.

### 26.6 Best Movement-to-Word Encoder

If the task is to translate continuous parquet movement/gaze features into symbolic words, the best encoder is the current-window XGBoost histogram model from `run_advanced_conversation_models.py`.

Scores:

| Model | Target | Feature set | Accuracy | Balanced accuracy | Macro F1 | Top-3 accuracy |
|---|---|---|---:|---:|---:|---:|
| `xgboost_hist` | `word` | current movement | 0.985 | 0.964 | 0.964 | 1.000 |
| `hist_gradient_boosting_current` | `word` | current movement | 0.984 | 0.963 | 0.962 | 1.000 |
| `logistic_l2_current` | `word` | current movement | 0.932 | 0.945 | 0.910 | 1.000 |

This model is practically useful as a symbolic encoder: it can convert a participant's current movement/gaze features into an `IW` word. But its high score is not independent evidence for vocabulary validity because the vocabulary itself was clustered from similar features.

Correct interpretation:

> Once a vocabulary is defined, a supervised model can recover the assigned symbolic word from current movement/gaze features with very high accuracy. This supports practical encoding of recordings into symbolic language, but vocabulary validity must be judged from cross-dyad stability and downstream grammar analyses, not from encoder accuracy alone.

### 26.7 Best Task-Condition Classifier

For the separate task of classifying Session 1 baseline versus Sessions 2-4 competitive/practiced condition, the best model from the final CENG488 pipeline was the 1s pose-only Random Forest.

Scores:

| Model | Window | Feature set | Accuracy | Macro F1 | ROC-AUC |
|---|---:|---|---:|---:|---:|
| Random Forest | 1s | pose-only | 0.707 | 0.621 | 0.613 |
| Random Forest | 5s | gaze-only | 0.689 | 0.615 | 0.619 |
| Random Forest | 5s | combined | 0.700 | 0.614 | 0.619 |

This classifier is not the best model for nonverbal conversation. It is the best model for detecting task-condition signal. Its performance is modest but above the dummy macro-F1 baseline. Because accuracy is inflated by class imbalance, macro F1 and ROC-AUC should be emphasized.

Correct interpretation:

> The best task-condition classifier indicates modest generalizable signal in pose/gaze features, not strong deterministic separability between baseline and competitive sessions.

### 26.8 Final Selection for the Academic Report

For the academic report, the recommended selections are:

1. Best vocabulary: the individual 12-word `IW` vocabulary.
2. Best exact offline conversation model: `xgboost_observed_immediate` event-level response model.
3. Best balanced/distributional offline conversation model: `random_forest_observed_immediate`.
4. Best autonomous/generative candidate: actor-history event models, especially `xgboost_actor_history` for top-3 candidate quality and `random_forest_actor_history` for distribution matching.
5. Best movement-to-word encoder: current-window `xgboost_hist`, with the explicit caveat that this is an encoder of learned cluster labels, not proof of vocabulary validity.
6. Best condition classifier: 1s pose-only Random Forest, useful for showing modest task-condition signal.

The main report should foreground the `IW` vocabulary and the event-level response models, because these are most aligned with the project's long-term aim: representing human nonverbal communication as an anonymized symbolic conversation and eventually generating plausible human-like response patterns.

