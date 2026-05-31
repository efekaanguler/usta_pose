# Cross-Dyad Vocabulary Validation

## Purpose

This validation avoids the optimistic full-data clustering issue. In every leave-one-dyad-out fold, the individual vocabulary is learned only from train dyads. Test dyads are then assigned to those train centroids.

- Vocabulary size per fold: 12
- Feature count: 19

## How To Read The Metrics

- `global_word_stability`: whether a train-only vocabulary assigns held-out samples to words that match the original full-data vocabulary after majority mapping. This is the key reliability metric for vocabulary stability.
- `train_vocab_cluster`: whether a supervised model can reproduce the train-only centroid assignment on held-out dyads. This is an encoder approximation metric, not evidence that the words are behaviorally true.
- `global_word_via_train_vocab`: model prediction mapped back to the original full-data word names; this combines encoder error and vocabulary stability error.

## Summary

- `train_only_kmeans_assignment` / `global_word_cluster_similarity`: balanced accuracy mean=nan, macro F1 mean=nan, top3 mean=nan.
- `train_only_kmeans_assignment` / `global_word_stability`: balanced accuracy mean=0.627, macro F1 mean=0.586, top3 mean=nan.
- `logistic_l2` / `global_word_via_train_vocab`: balanced accuracy mean=0.639, macro F1 mean=0.585, top3 mean=0.923.
- `xgboost_hist` / `global_word_via_train_vocab`: balanced accuracy mean=0.627, macro F1 mean=0.585, top3 mean=0.886.
- `hist_gradient_boosting` / `global_word_via_train_vocab`: balanced accuracy mean=0.609, macro F1 mean=0.567, top3 mean=0.880.
- `xgboost_hist` / `train_vocab_cluster`: balanced accuracy mean=0.955, macro F1 mean=0.955, top3 mean=1.000.
- `hist_gradient_boosting` / `train_vocab_cluster`: balanced accuracy mean=0.936, macro F1 mean=0.929, top3 mean=0.996.
- `logistic_l2` / `train_vocab_cluster`: balanced accuracy mean=0.931, macro F1 mean=0.891, top3 mean=0.997.

## Interpretation

High encoder performance against `train_vocab_cluster` means the train-only symbolic encoder can be approximated by a supervised model. It does not by itself prove external behavioral validity because the target is still derived from movement features.
The stronger academic question is `global_word_stability`: whether train-only clustering recovers the same broad word semantics on unseen dyads. Use this as the reliability estimate for the vocabulary itself.

## Practical Takeaway

The earlier high movement-to-word accuracy should not be cited as independent proof that the vocabulary is scientifically valid. It is best cited as evidence that, once a vocabulary is defined, a supervised encoder can reproduce that vocabulary from current movement/gaze features.

The more reliable vocabulary-validity number is the train-only stability result: held-out dyads assigned to train-learned centroids match the original full-data vocabulary with mean balanced accuracy 0.627, macro F1 0.586, adjusted Rand index 0.791, and normalized mutual information 0.749. This is a moderate stability result: the vocabulary is not arbitrary, but it is not fully locked down either.

For academic reporting, use this language:

> A train-only clustering validation indicated moderate cross-dyad stability of the individual vocabulary. Although supervised encoders reproduced train-vocabulary assignments with high accuracy, this reflects recoverability of cluster labels from the same feature space. The stronger validation is that train-learned vocabularies mapped onto held-out dyads with moderate agreement to the full-data vocabulary, supporting the vocabulary as a useful but still exploratory representation.
