# Academic Interpretation: Individual Nonverbal Conversation Layer

## Method Summary

The dyadic vocabulary was extended into an individual nonverbal conversation representation. For every 1-second window, both participants were represented separately with person-local and partner-relative features: body motion, activity ratio, left/right wrist speed, hand movement toward or away from the partner, partner-directed gaze, task-directed gaze, partner activity, and interpersonal distance.

The representation is anonymized. `person_A` and `person_B` are arbitrary within-dyad labels used only to make sequences readable. The clustering and response models do not use `p1` or `p2` as semantic identity features. A transition such as `IW04 -> IW06` means: an anonymous current actor produced individual word `IW04`, and the other anonymous participant's next primary action was classified as `IW06`.

The full run used 44 sessions, 11 dyads, 29,452 person-window rows, 10,739 primary-actor turns, and 1,964 adjacent actor-switch response pairs.

## Vocabulary Findings

The learned individual vocabulary has 12 words (`IW00`-`IW11`). The dominant word is `IW01`, representing moderate/intermittent movement, balanced hands, partner-directed gaze, and a mostly passive partner context. It accounts for 59.8% of person-windows. This should be interpreted as an attentive monitoring or nonverbal listening state rather than an action command.

Several rarer words encode more active manual behavior:

- `IW03`: high body motion, right-hand dominant, right hand moving toward the partner, partner-directed gaze.
- `IW08`: high body motion, left-hand dominant, left hand moving toward the partner, partner-directed gaze.
- `IW04` and `IW07`: right-hand movement toward the partner with task-directed gaze.
- `IW02` and `IW06`: left-hand withdrawal with task-directed gaze.

Across orders, the passive/monitoring word `IW01` is highest in order1 (65.6%), drops in order2/order3 (57.7% and 53.8%), and rises again in order4 (60.9%). Task/manual words such as `IW06`, `IW07`, and `IW04` are more common in order3 than order1. This suggests that the middle competitive trials may contain more explicit manual action variation, while order4 may return to a more routinized structure with more monitoring/listening states.

## Response Mapping Findings

The strongest response pattern is not an active counter-action. Most actor words are followed by `IW01` as the next primary action from the other participant:

- `IW10 -> IW01`: P=0.627, count=42
- `IW06 -> IW01`: P=0.601, count=89
- `IW00 -> IW01`: P=0.595, count=44
- `IW07 -> IW01`: P=0.587, count=61
- `IW01 -> IW01`: P=0.527, count=564

This means the first discovered nonverbal grammar is dominated by an action-monitoring pattern: one participant acts, and the other often responds by looking/monitoring rather than immediately producing a distinct manual action. In HRI terms, this resembles backchanneling, attention allocation, or readiness rather than turn-taking in spoken language.

After excluding the dominant `IW01` response, the non-default response map becomes more diverse but weaker. Examples include:

- `IW03 -> IW06`: P=0.286 among non-default responses, count=2
- `IW09 -> IW06`: P=0.267 among non-default responses, count=4
- `IW05 -> IW02`: P=0.250 among non-default responses, count=4
- `IW00 -> IW08`: P=0.233 among non-default responses, count=7
- `IW07 -> IW11`: P=0.233 among non-default responses, count=10

These mappings are not yet statistically strong because counts are small, but they are useful hypotheses for a refined event-based vocabulary.

## Model Interpretation

Three response predictors were evaluated with leave-one-dyad-out validation:

- Unigram response baseline: balanced accuracy 0.083, macro F1 0.059.
- Actor-word Markov response: balanced accuracy 0.083, macro F1 0.059.
- Logistic context response: balanced accuracy 0.122, macro F1 0.096.

Overall accuracy for the unigram and Markov models is high only because `IW01` dominates the response distribution. Balanced accuracy shows that actor word alone does not yet predict diverse responses. The logistic context model improves balanced accuracy slightly, which suggests weak context sensitivity, but the result is far from sufficient for a robust generative interaction model.

## Academic Conclusion

The individual nonverbal language currently supports a cautious conclusion: dyadic interaction in these sessions contains a strong attentive-monitoring component. Many manual or gaze-action states are followed by a broad listening/monitoring state from the partner. This is a meaningful HRI finding because it suggests that nonverbal coordination is not only action-action exchange; it also includes action-attention coupling.

However, the learned language is not yet a mature conversational grammar. The vocabulary is dominated by one broad state, and response prediction remains weak after accounting for class imbalance. A future version should separate listening/backchannel states from manual action states, segment events by motion onset rather than fixed windows, and train response models on event-level sequences rather than all adjacent windows.

The current implementation is still valuable because it creates the first anonymized, person-level symbolic representation:

```text
anonymous actor word -> anonymous partner response word
```

This is the correct foundation for later attempts to make a model “talk” with a real recording in the learned nonverbal language.
