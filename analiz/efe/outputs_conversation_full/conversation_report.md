# Individual Nonverbal Conversation Report

## Scope

This layer creates individual nonverbal words and maps anonymous actor-response transitions. `person_A` and `person_B` are arbitrary within-dyad slots for readability; p1/p2 are not used as semantic identities.

## Inventory

- Parquet files: 44
- Dyads: 11
- Person-windows: 29452
- Actor turns: 10739
- Actor-switch response pairs: 1964

## Individual Vocabulary

- Vocabulary size: 12
- Windows used for vocabulary: 21478
- `IW00`: moderate body motion; brief/intermittent active; balanced hands; right hand moves toward partner; task-directed gaze; partner mostly passive (prevalence=0.030, n=649)
- `IW01`: moderate body motion; brief/intermittent active; balanced hands; partner-directed gaze; partner mostly passive (prevalence=0.598, n=12842)
- `IW02`: high body motion; brief/intermittent active; balanced hands; left hand withdraws; task-directed gaze; partner mostly passive (prevalence=0.041, n=873)
- `IW03`: high body motion; brief/intermittent active; right-hand dominant; right hand moves toward partner; partner-directed gaze; partner mostly passive (prevalence=0.009, n=185)
- `IW04`: high body motion; brief/intermittent active; balanced hands; right hand moves toward partner; task-directed gaze; partner mostly passive (prevalence=0.031, n=671)
- `IW05`: high body motion; brief/intermittent active; left-hand dominant; partner-directed gaze; partner mostly passive (prevalence=0.016, n=346)
- `IW06`: moderate body motion; brief/intermittent active; balanced hands; left hand withdraws; task-directed gaze; partner mostly passive (prevalence=0.067, n=1433)
- `IW07`: moderate body motion; brief/intermittent active; balanced hands; right hand moves toward partner; task-directed gaze; partner mostly passive (prevalence=0.054, n=1156)
- `IW08`: high body motion; brief/intermittent active; left-hand dominant; left hand moves toward partner; partner-directed gaze; partner mostly passive (prevalence=0.047, n=1013)
- `IW09`: high body motion; brief/intermittent active; right-hand dominant; left hand withdraws; partner-directed gaze; partner mostly passive (prevalence=0.010, n=217)
- `IW10`: moderate body motion; brief/intermittent active; balanced hands; right hand moves toward partner; partner-directed gaze; partner mostly passive (prevalence=0.033, n=718)
- `IW11`: moderate body motion; brief/intermittent active; balanced hands; partner-directed gaze; partner mostly passive (prevalence=0.064, n=1375)

## Response Mapping

Most probable anonymized responses:
- `IW10` -> `IW01`: P=0.627, count=42
- `IW03` -> `IW01`: P=0.611, count=11
- `IW06` -> `IW01`: P=0.601, count=89
- `IW00` -> `IW01`: P=0.595, count=44
- `IW07` -> `IW01`: P=0.587, count=61
- `IW08` -> `IW01`: P=0.557, count=68
- `IW04` -> `IW01`: P=0.556, count=35
- `IW09` -> `IW01`: P=0.531, count=17
- `IW01` -> `IW01`: P=0.527, count=564
- `IW02` -> `IW01`: P=0.522, count=47
- `IW05` -> `IW01`: P=0.515, count=17
- `IW11` -> `IW01`: P=0.503, count=72
- `IW09` -> `IW06`: P=0.125, count=4
- `IW09` -> `IW11`: P=0.125, count=4
- `IW05` -> `IW02`: P=0.121, count=4
- `IW03` -> `IW06`: P=0.111, count=2
- `IW02` -> `IW06`: P=0.100, count=9
- `IW07` -> `IW11`: P=0.096, count=10
- `IW04` -> `IW11`: P=0.095, count=6
- `IW00` -> `IW08`: P=0.095, count=7

## Response Modeling

- `unigram_response`: balanced accuracy=0.083, macro F1=0.059, n=1964, validation=leave-one-dyad-out.
- `actor_word_markov_response`: balanced accuracy=0.083, macro F1=0.059, n=1964, validation=leave-one-dyad-out.
- `logistic_context_response`: balanced accuracy=0.122, macro F1=0.096, n=1964, validation=leave-one-dyad-out.

## Interpretation

The current mapping should be read as a symbolic interaction grammar, not as individual identity behavior. A transition `IW04 -> IW06` means that an anonymous primary actor produced an action state classified as `IW04`, and the other anonymous participant's next primary action state was classified as `IW06`.
High response-model performance relative to a unigram baseline would suggest real sequential structure in the nonverbal language. Weak performance would mean the vocabulary is descriptive but not yet predictive enough for generative interaction.

## Example Model Conversations

### 20260504_183351_order1
- t8: person_A:IW01 -> model person_B:IW01; actual person_B:IW08 (MISS)
- t29: person_B:IW01 -> model person_A:IW01; actual person_A:IW01 (OK)
- t31: person_A:IW01 -> model person_B:IW01; actual person_B:IW06 (MISS)
- t40: person_B:IW01 -> model person_A:IW01; actual person_A:IW01 (OK)
- t42: person_A:IW01 -> model person_B:IW01; actual person_B:IW08 (MISS)
- t62: person_B:IW07 -> model person_A:IW01; actual person_A:IW01 (OK)
- t68: person_A:IW08 -> model person_B:IW01; actual person_B:IW01 (OK)
- t78: person_A:IW01 -> model person_B:IW01; actual person_B:IW10 (MISS)
- t79: person_B:IW10 -> model person_A:IW01; actual person_A:IW01 (OK)
- t80: person_A:IW01 -> model person_B:IW01; actual person_B:IW07 (MISS)

### 20260504_183529_order2
- t4: person_A:IW01 -> model person_B:IW01; actual person_B:IW06 (MISS)
- t6: person_B:IW06 -> model person_A:IW01; actual person_A:IW01 (OK)
- t9: person_A:IW09 -> model person_B:IW01; actual person_B:IW08 (MISS)
- t11: person_B:IW06 -> model person_A:IW01; actual person_A:IW01 (OK)
- t12: person_A:IW01 -> model person_B:IW01; actual person_B:IW04 (MISS)
- t15: person_B:IW01 -> model person_A:IW01; actual person_A:IW01 (OK)
- t17: person_A:IW01 -> model person_B:IW01; actual person_B:IW11 (MISS)
- t21: person_B:IW01 -> model person_A:IW01; actual person_A:IW01 (OK)
- t25: person_A:IW11 -> model person_B:IW01; actual person_B:IW05 (MISS)
- t31: person_B:IW01 -> model person_A:IW01; actual person_A:IW01 (OK)

### 20260504_184034_order3
- t6: person_B:IW11 -> model person_A:IW01; actual person_A:IW01 (OK)
- t8: person_A:IW08 -> model person_B:IW01; actual person_B:IW07 (MISS)
- t9: person_B:IW07 -> model person_A:IW01; actual person_A:IW01 (OK)
- t17: person_A:IW01 -> model person_B:IW01; actual person_B:IW07 (MISS)
- t18: person_B:IW07 -> model person_A:IW01; actual person_A:IW06 (MISS)
- t22: person_A:IW01 -> model person_B:IW01; actual person_B:IW01 (OK)
- t26: person_B:IW11 -> model person_A:IW01; actual person_A:IW11 (MISS)
- t33: person_A:IW08 -> model person_B:IW01; actual person_B:IW06 (MISS)
- t35: person_B:IW11 -> model person_A:IW01; actual person_A:IW11 (MISS)
- t41: person_A:IW03 -> model person_B:IW01; actual person_B:IW06 (MISS)

### 20260504_184304_order4
- t2: person_B:IW06 -> model person_A:IW01; actual person_A:IW01 (OK)
- t4: person_A:IW10 -> model person_B:IW01; actual person_B:IW07 (MISS)
- t15: person_B:IW06 -> model person_A:IW01; actual person_A:IW01 (OK)
- t18: person_A:IW06 -> model person_B:IW01; actual person_B:IW10 (MISS)
- t21: person_B:IW01 -> model person_A:IW01; actual person_A:IW01 (OK)
- t22: person_A:IW01 -> model person_B:IW01; actual person_B:IW01 (OK)
- t30: person_B:IW02 -> model person_A:IW01; actual person_A:IW01 (OK)
- t31: person_A:IW01 -> model person_B:IW01; actual person_B:IW11 (MISS)
- t34: person_B:IW07 -> model person_A:IW01; actual person_A:IW08 (MISS)
- t35: person_A:IW08 -> model person_B:IW01; actual person_B:IW07 (MISS)


## Figures

- `figures/individual_word_prevalence_by_order.png`
- `figures/response_mapping_heatmap.png`
