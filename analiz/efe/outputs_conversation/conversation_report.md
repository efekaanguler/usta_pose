# Individual Nonverbal Conversation Report

## Scope

This layer creates individual nonverbal words and maps anonymous actor-response transitions. `person_A` and `person_B` are arbitrary within-dyad slots for readability; p1/p2 are not used as semantic identities.

## Inventory

- Parquet files: 1
- Dyads: 1
- Person-windows: 132
- Actor turns: 60
- Actor-switch response pairs: 12

## Warnings
- Input directory has only 1 parquet file(s); dyad-level order statistics and group-validated models will be limited or skipped.
- Incomplete or ambiguous dyad blocks detected: dyad_001

## Individual Vocabulary

- Vocabulary size: 12
- Windows used for vocabulary: 120
- `IW00`: high body motion; brief/intermittent active; right-hand dominant; partner-directed gaze; partner also active (prevalence=0.133, n=16)
- `IW01`: high body motion; brief/intermittent active; balanced hands; right hand moves toward partner; partner-directed gaze; partner mostly passive (prevalence=0.083, n=10)
- `IW02`: high body motion; brief/intermittent active; balanced hands; right hand moves toward partner; task-directed gaze; partner mostly passive (prevalence=0.050, n=6)
- `IW03`: moderate body motion; mostly still; right-hand dominant; right hand withdraws; partner-directed gaze; partner mostly passive (prevalence=0.292, n=35)
- `IW04`: moderate body motion; mostly still; right-hand dominant; partner-directed gaze; partner mostly passive (prevalence=0.008, n=1)
- `IW05`: moderate body motion; brief/intermittent active; balanced hands; left hand moves toward partner; partner-directed gaze; partner mostly passive (prevalence=0.175, n=21)
- `IW06`: moderate body motion; brief/intermittent active; right-hand dominant; left hand withdraws; task-directed gaze; partner mostly passive (prevalence=0.108, n=13)
- `IW07`: moderate body motion; brief/intermittent active; right-hand dominant; right hand moves toward partner; partner-directed gaze; partner mostly passive (prevalence=0.017, n=2)
- `IW08`: high body motion; sustained active; left-hand dominant; right hand moves toward partner; partner-directed gaze; partner also active (prevalence=0.008, n=1)
- `IW09`: moderate body motion; brief/intermittent active; right-hand dominant; right hand moves toward partner; partner-directed gaze; partner mostly passive (prevalence=0.008, n=1)
- `IW10`: high body motion; sustained active; left-hand dominant; left hand moves toward partner; partner-directed gaze; partner mostly passive (prevalence=0.033, n=4)
- `IW11`: moderate body motion; brief/intermittent active; right-hand dominant; right hand moves toward partner; partner-directed gaze; partner mostly passive (prevalence=0.083, n=10)

## Response Mapping

Most probable anonymized responses:
- `IW06` -> `IW05`: P=1.000, count=2
- `IW03` -> `IW06`: P=1.000, count=1
- `IW10` -> `IW00`: P=1.000, count=1
- `IW11` -> `IW00`: P=1.000, count=1
- `IW00` -> `IW00`: P=0.500, count=1
- `IW00` -> `IW11`: P=0.500, count=1
- `IW02` -> `IW06`: P=0.500, count=1
- `IW02` -> `IW10`: P=0.500, count=1
- `IW05` -> `IW03`: P=0.333, count=1
- `IW05` -> `IW05`: P=0.333, count=1
- `IW05` -> `IW11`: P=0.333, count=1

## Response Modeling

- all: skipped (not enough response pairs/dyads).

## Interpretation

The current mapping should be read as a symbolic interaction grammar, not as individual identity behavior. A transition `IW04 -> IW06` means that an anonymous primary actor produced an action state classified as `IW04`, and the other anonymous participant's next primary action state was classified as `IW06`.
High response-model performance relative to a unigram baseline would suggest real sequential structure in the nonverbal language. Weak performance would mean the vocabulary is descriptive but not yet predictive enough for generative interaction.

## Example Model Conversations

### 20260507_155542_order1
- t4: person_B:IW06 -> model person_A:IW05; actual person_A:IW05 (OK)
- t12: person_A:IW03 -> model person_B:IW06; actual person_B:IW06 (OK)
- t16: person_B:IW05 -> model person_A:IW03; actual person_A:IW05 (MISS)
- t17: person_A:IW05 -> model person_B:IW03; actual person_B:IW03 (OK)
- t20: person_B:IW02 -> model person_A:IW06; actual person_A:IW06 (OK)
- t22: person_A:IW06 -> model person_B:IW05; actual person_B:IW05 (OK)
- t23: person_B:IW05 -> model person_A:IW03; actual person_A:IW11 (MISS)
- t39: person_A:IW00 -> model person_B:IW00; actual person_B:IW11 (MISS)
- t40: person_B:IW11 -> model person_A:IW00; actual person_A:IW00 (OK)
- t51: person_B:IW00 -> model person_A:IW00; actual person_A:IW00 (OK)


## Figures

- `figures/individual_word_prevalence_by_order.png`
- `figures/response_mapping_heatmap.png`
