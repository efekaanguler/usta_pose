# Academic Claim Strengthening Summary

This note summarizes the latest event-level nonverbal-conversation analysis. The purpose is to make the academic claims more defensible by separating fully autonomous generation claims from offline conversation-interpretation claims.

## Input and Protocol

- Input conversation layer: `outputs_conversation_full/individual_words.parquet`.
- Original parquet source: `usta_pose/testing/final_dataset_parquets/`.
- Evaluation unit: anonymized dyads, using leave-one-dyad-out validation.
- Person labels are anonymized as `person_A` and `person_B`; no claim treats `p1` or `p2` as identity.
- Vocabulary is individual-level (`IW00` to `IW11`), not combined dyad words.
- Monitoring/waiting words are retained as real turn states but excluded as final active-response targets.

## Event-Based Conversation Construction

The analysis collapses consecutive identical individual words into symbolic events. For each active actor event, it searches the other person's later events within a 12-window horizon and records:

- the first other-person event (`immediate_other_word`), which can be wait/monitoring;
- the next non-wait active response (`active_response_word`), if found;
- response lag and duration.

Counts from the full run:

- Events: 12,069.
- Active actor response opportunities: 5,894.
- Found delayed active responses: 5,102.
- Cases where the first other-person event was wait/monitoring before a later active response: 1,966.
- Wait/monitoring words: `IW01`, `IW11`.
- Mean delayed active response lag: 3.23 windows.
- Median delayed active response lag: 2.00 windows.

This directly addresses the earlier problem where many responses looked like only waiting. Waiting is now modeled as an intermediate conversational state, not as absence of communication.

## Model Results by Claim Strength

All metrics below are leave-one-dyad-out results on 5,102 active-response examples with 10 active response classes.

### Strongest Offline Interpretation Claim

`observed_immediate` models are allowed to observe the actor event and the other person's first event after it. This is appropriate for interpreting real recordings and predicting the eventual active response after a wait/monitoring state. It is not a fully autonomous response generator because it sees the responder's first post-actor state.

Best results:

- `xgboost_observed_immediate`: accuracy 0.771, balanced accuracy 0.739, macro F1 0.779, top-3 accuracy 0.861.
- `random_forest_observed_immediate`: accuracy 0.759, balanced accuracy 0.751, macro F1 0.731, top-3 accuracy 0.823.
- `logistic_observed_immediate`: accuracy 0.748, balanced accuracy 0.747, macro F1 0.702, top-3 accuracy 0.826.
- Markov baseline with actor word + immediate other word: accuracy 0.709, balanced accuracy 0.663, macro F1 0.740, top-3 accuracy 0.810.

Interpretation: once the partner's first post-actor state is known, the eventual active response is highly structured. This supports a reproducible event-level nonverbal grammar in the recordings.

### Moderate Autonomous/History-Based Generation Claim

`actor_history` models do not observe the responder's immediate future state. They use the current actor event plus previous symbolic turns. This is the most credible setting for a simple agent that tries to continue a conversation using past context.

Best results:

- `xgboost_actor_history`: accuracy 0.470, balanced accuracy 0.403, macro F1 0.429, top-3 accuracy 0.677.
- `logistic_actor_history`: accuracy 0.427, balanced accuracy 0.425, macro F1 0.390, top-3 accuracy 0.605.
- `random_forest_actor_history`: accuracy 0.440, balanced accuracy 0.408, macro F1 0.398, top-3 accuracy 0.629.
- Markov baseline with actor word + previous response: accuracy 0.448, balanced accuracy 0.382, macro F1 0.412, top-3 accuracy 0.641.

Interpretation: exact next-response prediction is only moderate, but top-3 candidate quality is meaningful. This supports generating a small set of plausible human-like responses rather than claiming deterministic imitation.

### Weak Fully Current-Movement Generation Claim

`actor_only` models see only the actor's current event and movement features. They do not use previous turns or the responder's first event.

Best results are near chance in balanced accuracy:

- `logistic_actor_only`: balanced accuracy 0.117.
- `random_forest_actor_only`: balanced accuracy 0.108.
- `hist_gbdt_actor_only`: balanced accuracy 0.104.
- `xgboost_actor_only`: balanced accuracy 0.102.
- Actor-word-only Markov baseline: balanced accuracy 0.097.

Interpretation: the current actor movement alone is not enough to determine the other person's exact active response. A paper should not claim strong autonomous response prediction from single-event movement alone.

## Distributional Human-Likeness

Exact next-word accuracy is not the only relevant generative criterion. Human nonverbal response is stochastic, so distribution matching is also important. Jensen-Shannon divergence compares the predicted response-word distribution to the true human response-word distribution; lower is better.

Best overall JS divergence:

- `random_forest_observed_immediate`: 0.0022.
- `random_forest_actor_history`: 0.0025.
- `hist_gbdt_observed_immediate`: 0.0058.
- `xgboost_observed_immediate`: 0.0068.
- `xgboost_actor_history`: 0.0163.
- Majority baseline: 0.6055.

Interpretation: even when exact prediction is moderate, the actor-history generator can reproduce the overall human response distribution very closely. This is a defensible claim for human-like distributional imitation, not exact one-to-one response prediction.

## Nonverbal Grammar Pattern

The dominant structure is:

```text
actor active word -> responder wait/monitoring word -> responder active word
```

For actor-only response probabilities, many actor words most often lead to `IW08`, followed by `IW06` and `IW07`. Examples:

- `IW00 -> IW08`: probability 0.235.
- `IW02 -> IW08`: probability 0.212.
- `IW04 -> IW08`: probability 0.229.
- `IW06 -> IW08`: probability 0.194.
- `IW07 -> IW06`: probability 0.189, very close to `IW07 -> IW08`: 0.188.

When the first responder event is wait/monitoring, the later active response is still structured. Examples:

- `IW00 -> IW01(wait) -> IW08`: probability 0.230.
- `IW00 -> IW11(wait) -> IW08`: probability 0.375.
- `IW02 -> IW01(wait) -> IW08`: probability 0.202.
- `IW02 -> IW11(wait) -> IW08`: probability 0.329.
- `IW04 -> IW01(wait) -> IW08`: probability 0.250.
- `IW04 -> IW11(wait) -> IW08`: probability 0.433.

Interpretation: `IW01` and `IW11` are not just idle noise. They function like monitoring/listening states that often precede a later active manual response, especially `IW08`.

## Vocabulary Reliability Caveat

The movement-to-word encoder accuracy from the earlier full-data clustering run is not the strongest evidence because the vocabulary was learned on the same data. The more appropriate validation is the train-only clustering study:

- Leave-one-dyad-out train-only vocabulary stability against the original vocabulary: balanced accuracy 0.627, macro F1 0.586, ARI 0.791, NMI 0.749.
- Train-vocabulary recoverability with supervised models remains high, but this should be interpreted as recoverability of the learned train vocabulary, not proof that the vocabulary is an objective ground truth.

Academic wording should say that the vocabulary is moderately stable across held-out dyads and useful as a symbolic representation, not that it is a final universal lexicon.

## RL/PPO Position

PPO or another RL method is not currently the strongest academic route. With only about two hours of recordings and no real interaction environment, PPO would require hand-designed rewards and simulated transitions. A high PPO return would mostly prove that the agent learned our reward function, not that it learned human nonverbal communication.

A more credible sequence is:

1. Use the current behavior-cloning and Markov baselines as the main evidence.
2. Evaluate generated sequences with held-out dyads, top-k response accuracy, JS divergence, transition matrices, lag distributions, and order-specific distribution matching.
3. If RL is added later, initialize from behavior cloning and use RL only for constrained fine-tuning in a clearly defined symbolic environment.
4. Report RL as an exploratory extension unless human or held-out distributional evaluation shows improvement over behavior cloning.

Installed packages such as `gymnasium`, `stable-baselines3`, `sb3_contrib`, `imitation`, and `xgboost` are sufficient for future RL/imitation experiments. They are not required for the current strongest claim.

## Files Produced

- `outputs_academic_strengthening/event_words.parquet`
- `outputs_academic_strengthening/event_response_pairs.csv`
- `outputs_academic_strengthening/event_response_mapping.csv`
- `outputs_academic_strengthening/event_actor_response_mapping.csv`
- `outputs_academic_strengthening/event_wait_continuation_mapping.csv`
- `outputs_academic_strengthening/event_bc_model_results.csv`
- `outputs_academic_strengthening/event_bc_predictions.parquet`
- `outputs_academic_strengthening/event_bc_distribution_eval.csv`
- `outputs_academic_strengthening/academic_strengthening_report.md`

## Recommended Paper Claim Wording

A defensible wording is:

> We derive an anonymized individual-level symbolic vocabulary from dyadic pose/gaze features and show that event-level nonverbal response patterns generalize across held-out dyads. Monitoring/waiting states frequently act as intermediate turn states before later active responses. Models using symbolic interaction history produce plausible response candidates and closely match human response distributions, while exact autonomous prediction from a single current movement event remains weak.

Avoid claiming:

- that the learned words are a universal human nonverbal vocabulary;
- that current movement alone is enough to generate the exact human response;
- that PPO/RL would be more academically credible without a validated environment and reward.
