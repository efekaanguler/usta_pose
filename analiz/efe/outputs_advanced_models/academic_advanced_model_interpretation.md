# Academic Interpretation: Delayed Active Responses and Movement-to-Word Models

## Why This Extension Was Needed

The first individual conversation model showed that many immediate responses were `IW01`, a broad attentive monitoring/listening state. This is a real behavioral state, but it makes the language look too much like:

```text
person acts -> other waits
```

The extension therefore keeps waiting/listening as meaningful, but adds a delayed-response target: after the partner enters a monitoring state, what is their next active movement word within the next 12 windows? With 0.5 s stride, this searches roughly the next 6 seconds.

Two words were treated as wait/monitoring words for this active-response analysis:

- `IW01`: dominant immediate response; partner-directed gaze, balanced hands, mostly passive partner context.
- `IW11`: similar partner-gaze monitoring state without explicit hand movement.

## Active Response Dataset

The advanced run found:

- 8,869 candidate response opportunities.
- 6,063 cases where a future active/non-monitoring response appeared within the horizon.
- Mean active response lag: 5.21 windows, approximately 2.6 seconds.

This changes the interpretation from instant reaction to delayed nonverbal turn-taking. The partner often does not immediately answer with a manual action; they first attend/monitor, then act later.

## Movement-to-Word Translation

Models were trained to convert a person's movement features into one of the learned individual words. This is the requested translation from parquet-style movement features into symbolic nonverbal language.

Best current-window models:

- XGBoost histogram GBDT: balanced accuracy 0.964, macro F1 0.964, top-3 accuracy approximately 1.000.
- sklearn histogram gradient boosting: balanced accuracy 0.963, macro F1 0.962, top-3 accuracy approximately 1.000.
- Logistic regression: balanced accuracy 0.945, macro F1 0.910, top-3 accuracy approximately 1.000.

Interpretation: the learned vocabulary is technically recoverable from movement/gaze features. In other words, a model can reliably translate a real participant's current parquet features into one or several nonverbal word candidates.

This high performance is expected because the words were originally clustered from these same feature families. It is still useful: it gives us a practical encoder from continuous body/gaze data into symbolic words.

## Past-Only Word Prediction

Past-context models attempted to predict the current word from previous movement windows. These were much weaker:

- Logistic past-3 model: balanced accuracy 0.260, top-3 accuracy 0.575.
- Histogram gradient boosting past-3: balanced accuracy 0.123, top-3 accuracy 0.791.
- XGBoost past-3 numeric-only: balanced accuracy 0.121, top-3 accuracy 0.798.

Interpretation: exact next/current word is difficult to infer from past movement alone, but top-3 prediction is often informative. This suggests the nonverbal language is partially constrained by recent motion history but not deterministic.

## Fixing the Waiting-Response Problem

When waiting/monitoring words are skipped and we search for the partner's next active movement, the response grammar becomes more interpretable.

Most common delayed active responses:

- `IW00 -> IW08`: P=0.266, count=55.
- `IW09 -> IW08`: P=0.263, count=26.
- `IW07 -> IW08`: P=0.257, count=84.
- `IW04 -> IW08`: P=0.249, count=51.
- `IW02 -> IW08`: P=0.237, count=71.
- `IW06 -> IW08`: P=0.225, count=103.
- `IW07 -> IW06`: P=0.211, count=69.

`IW08` is a high-motion, left-hand-dominant, left-hand-toward-partner, partner-gaze word. `IW06` is a moderate-motion, left-hand-withdrawal, task-gaze word. This suggests that after the monitoring phase, many active replies are hand-action states, often left-hand dominant in the current coordinate/pose convention.

## Response Model Results

The key result is that actor movement alone does not predict the partner's active response very well.

Poor/weak predictors:

- Actor word only: balanced accuracy 0.106, macro F1 0.074.
- Actor current movement + context words with logistic model: balanced accuracy 0.135.
- Actor current movement GBDT/XGBoost: balanced accuracy around 0.09-0.10.
- Actor current + past movement context: did not improve meaningfully.

Strongest predictor:

- Immediate response word -> delayed active response: balanced accuracy 0.568, macro F1 0.582, top-3 accuracy 0.659.

Interpretation: the partner's initial monitoring/listening state is not empty. It carries information about that partner's later active movement. This means the nonverbal conversation has a two-step structure:

```text
actor action -> partner monitoring/listening state -> partner active movement
```

This is a better model of the data than a simple immediate action-reaction grammar.

## Academic Meaning

The main theoretical implication is that in this cooperative/competitive disk task, nonverbal interaction appears to include an attentional buffer. Participants often do not immediately answer another person's movement with a movement. Instead, they enter a monitoring state, and that monitoring state predicts the next active response better than the original actor's movement does.

For HRI, this matters because a robot or agent imitating human-like coordination should not always respond immediately with action. A human-like policy may need to produce an attentive waiting/backchannel state first, then select an action after additional context.

## Practical Conclusion

The current system can now do three things:

1. Translate real person movement windows into symbolic individual words with high reliability.
2. Represent interactions as anonymous actor-response sequences.
3. Predict a delayed active partner response better when the immediate monitoring word is included.

The next best improvement is event-based segmentation: detect true movement onsets and offsets instead of fixed 1-second windows. That should reduce repeated waiting tokens and make active response prediction cleaner.
