# Academic Strengthening Studies

## Study Design

This study strengthens the nonverbal-language claims by replacing fixed-window immediate response analysis with event-based delayed active response modeling. Consecutive identical individual words are collapsed into events. Monitoring/listening words are retained as meaningful turn states but are not treated as final active responses.

- Events: 12069
- Active actor response opportunities: 5894
- Found delayed active responses: 5102
- Delayed responses where the first other-person event was wait/monitoring: 1966
- Wait/monitoring words: IW01, IW11
- Dominant immediate response word: IW01
- Mean delayed active response lag: 3.23 windows
- Median delayed active response lag: 2.00 windows

## Behavior-Cloning Results

Models are grouped by how much information they are allowed to observe. `actor_only` is the most autonomous setting: it predicts the other person's eventual active response from the actor's current event and movement features only. `actor_history` adds previous symbolic turns. `observed_immediate` additionally observes the other person's first event after the actor, so it is best interpreted as offline conversation interpretation rather than fully autonomous generation.

### actor_only
- `logistic_actor_only`: balanced accuracy=0.117, macro F1=0.091, top3=0.315, accuracy=0.093.
- `random_forest_actor_only`: balanced accuracy=0.108, macro F1=0.091, top3=0.338, accuracy=0.097.
- `hist_gbdt_actor_only`: balanced accuracy=0.104, macro F1=0.096, top3=0.441, accuracy=0.166.
- `xgboost_actor_only`: balanced accuracy=0.102, macro F1=0.076, top3=0.493, accuracy=0.181.

### actor_history
- `logistic_actor_history`: balanced accuracy=0.425, macro F1=0.390, top3=0.605, accuracy=0.427.
- `random_forest_actor_history`: balanced accuracy=0.408, macro F1=0.398, top3=0.629, accuracy=0.440.
- `xgboost_actor_history`: balanced accuracy=0.403, macro F1=0.429, top3=0.677, accuracy=0.470.
- `hist_gbdt_actor_history`: balanced accuracy=0.376, macro F1=0.401, top3=0.656, accuracy=0.442.

### observed_immediate
- `random_forest_observed_immediate`: balanced accuracy=0.751, macro F1=0.731, top3=0.823, accuracy=0.759.
- `logistic_observed_immediate`: balanced accuracy=0.747, macro F1=0.702, top3=0.826, accuracy=0.748.
- `xgboost_observed_immediate`: balanced accuracy=0.739, macro F1=0.779, top3=0.861, accuracy=0.771.
- `hist_gbdt_observed_immediate`: balanced accuracy=0.718, macro F1=0.757, top3=0.839, accuracy=0.748.

### actor_word_only
- `actor_markov`: balanced accuracy=0.097, macro F1=0.061, top3=0.521, accuracy=0.185.

### actor_word+previous_response
- `actor_history_markov`: balanced accuracy=0.382, macro F1=0.412, top3=0.641, accuracy=0.448.

### actor_word+immediate_other_word
- `actor_immediate_markov`: balanced accuracy=0.663, macro F1=0.740, top3=0.810, accuracy=0.709.

### baseline
- `majority`: balanced accuracy=0.100, macro F1=0.034, top3=0.523, accuracy=0.203.

## Distributional Human-Likeness

- `random_forest_observed_immediate` response distribution JS divergence: 0.0022 lower is more human-like.
- `random_forest_actor_history` response distribution JS divergence: 0.0025 lower is more human-like.
- `hist_gbdt_observed_immediate` response distribution JS divergence: 0.0058 lower is more human-like.
- `xgboost_observed_immediate` response distribution JS divergence: 0.0068 lower is more human-like.
- `logistic_observed_immediate` response distribution JS divergence: 0.0084 lower is more human-like.
- `xgboost_actor_history` response distribution JS divergence: 0.0163 lower is more human-like.
- `hist_gbdt_actor_history` response distribution JS divergence: 0.0166 lower is more human-like.
- `logistic_actor_history` response distribution JS divergence: 0.0275 lower is more human-like.
- `actor_history_markov` response distribution JS divergence: 0.0295 lower is more human-like.
- `actor_immediate_markov` response distribution JS divergence: 0.0394 lower is more human-like.
- `hist_gbdt_actor_only` response distribution JS divergence: 0.0731 lower is more human-like.
- `random_forest_actor_only` response distribution JS divergence: 0.1108 lower is more human-like.
- `logistic_actor_only` response distribution JS divergence: 0.1567 lower is more human-like.
- `xgboost_actor_only` response distribution JS divergence: 0.2150 lower is more human-like.
- `actor_markov` response distribution JS divergence: 0.3349 lower is more human-like.
- `majority` response distribution JS divergence: 0.6055 lower is more human-like.

## Interpretation

The strongest claim is that dyadic recordings contain a reproducible symbolic nonverbal grammar at the event level. The wait/monitoring state is not just a failed response: in many cases it is an intermediate turn-holding state before a later active movement.
Autonomous response generation should be claimed only from the `actor_only` and `actor_history` ablations. The `observed_immediate` models are still useful, but their higher scores partly come from observing the responder's first post-actor state; they support interpretation of real recordings more than fully independent generation.
Top-3 accuracy should be interpreted as candidate-set quality, not exact prediction accuracy. For generative use, sampling among top candidates is more appropriate than forcing a single deterministic response because the same actor event can receive multiple plausible human responses.
PPO should remain secondary until an environment with meaningful transition dynamics and rewards is defined. The current leave-one-dyad-out behavior cloning and distribution-matching evaluation are the more credible academic core.
