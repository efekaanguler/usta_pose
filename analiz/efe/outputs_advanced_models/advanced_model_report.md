# Advanced Nonverbal Conversation Modeling

## Problem Fix

The immediate response vocabulary was dominated by a waiting/listening word. This run keeps that state as meaningful, but creates a second target: the partner's next active/non-monitoring movement within a future horizon. This separates `attention as response` from `next movement as response`.

- Dominant immediate response word: `IW01`
- Wait/monitoring words excluded for active response target: `IW01`, `IW11`
- Candidate response rows: 8869
- Rows with a future active response: 6063
- Mean active response lag in windows: 5.21

## Word Generation Models

- `xgboost_hist` (current_movement): balanced accuracy=0.964, macro F1=0.964, top3=1.000.
- `hist_gradient_boosting_current` (current_movement): balanced accuracy=0.963, macro F1=0.962, top3=1.000.
- `logistic_l2_current` (current_movement): balanced accuracy=0.945, macro F1=0.910, top3=1.000.
- `random_forest_shallow_current` (current_movement): balanced accuracy=0.896, macro F1=0.845, top3=0.970.
- `logistic_l2_past3` (past3_movement_plus_past_words): balanced accuracy=0.260, macro F1=0.175, top3=0.575.
- `hist_gradient_boosting_past3` (past3_movement_plus_past_words): balanced accuracy=0.123, macro F1=0.130, top3=0.791.
- `xgboost_hist` (past3_movement_numeric_only): balanced accuracy=0.121, macro F1=0.123, top3=0.798.

## Active Response Models

- `immediate_response_markov_active` (immediate_response_word): balanced accuracy=0.568, macro F1=0.582, top3=0.659.
- `logistic_l2_active_current` (actor_current_movement_plus_context_words): balanced accuracy=0.135, macro F1=0.094, top3=0.340.
- `random_forest_active_current` (actor_current_movement_plus_context_words): balanced accuracy=0.116, macro F1=0.067, top3=0.289.
- `actor_word_markov_active` (actor_word): balanced accuracy=0.106, macro F1=0.074, top3=0.287.
- `hist_gradient_boosting_active_current` (actor_current_movement_plus_context_words): balanced accuracy=0.099, macro F1=0.089, top3=0.409.
- `hist_gradient_boosting_active_current_past3` (actor_current_and_past3_movement_plus_words): balanced accuracy=0.096, macro F1=0.086, top3=0.405.
- `xgboost_hist` (actor_current_movement_numeric_only): balanced accuracy=0.094, macro F1=0.078, top3=0.397.
- `xgboost_hist` (actor_current_and_past3_numeric_only): balanced accuracy=0.086, macro F1=0.075, top3=0.396.

## Active Response Mapping

- `IW00` -> `IW08`: P=0.266, count=55
- `IW09` -> `IW08`: P=0.263, count=26
- `IW07` -> `IW08`: P=0.257, count=84
- `IW04` -> `IW08`: P=0.249, count=51
- `IW02` -> `IW08`: P=0.237, count=71
- `IW06` -> `IW08`: P=0.225, count=103
- `IW07` -> `IW06`: P=0.211, count=69
- `IW05` -> `IW08`: P=0.200, count=26
- `IW10` -> `IW06`: P=0.186, count=41
- `IW03` -> `IW08`: P=0.184, count=16
- `IW08` -> `IW08`: P=0.179, count=66
- `IW01` -> `IW08`: P=0.175, count=567
- `IW10` -> `IW08`: P=0.172, count=38
- `IW11` -> `IW08`: P=0.169, count=71
- `IW08` -> `IW06`: P=0.168, count=62
- `IW01` -> `IW06`: P=0.162, count=525
- `IW06` -> `IW06`: P=0.162, count=74
- `IW05` -> `IW07`: P=0.162, count=21
- `IW11` -> `IW06`: P=0.157, count=66
- `IW00` -> `IW10`: P=0.155, count=32

## Interpretation

If immediate responses are dominated by monitoring, active-response modeling asks a different question: after the other participant attends/waits, what is their next movement word? This is closer to a turn-taking grammar. The model outputs should therefore be read as delayed action replies, not instant reactions.
Top-3 accuracy is especially relevant here, because a nonverbal context may license several plausible next movements rather than one deterministic reply.
