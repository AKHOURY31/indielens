# Experiment 002 — Item-Based Collaborative Filtering

## Date
25/10/2026

## Dataset source link
https://www.kaggle.com/datasets/tamber/steam-video-games

## Command to run code
.\.venv\Scripts\python.exe -m indielens.cli --data data/raw/steam-200k.csv --top-k 10

## Objective
Evaluate whether personalized recommendations based on shared players improve validation Recall@10 compared with the popularity baseline.

## Dataset
- Source: Steam 200K
- Unique users: 12,393
- Unique games: 5,153
- Total user-game interactions: 128,801
- Training interactions: 116,891
- Validation interactions: 5,217
- Reserved test interactions: 6,693

## Model
Item-based collaborative filtering with a popularity fallback.

The model builds a sparse binary user-game matrix from training interactions. Each unique user-game pair receives a value of 1; playtime weights are not used.

Cosine similarity compares games by their player audiences. Self-similarities are set to zero. Each candidate game's score is the sum of its similarities to the games in the user's training history.

Already-seen games are excluded, and games with positive scores are ranked in descending order. Equal scores are resolved alphabetically. Popularity fills remaining recommendation slots when there are too few positive scores and provides recommendations for unknown users. Both similarities and popularity counts are learned from training data only.

## Evaluation
- Split: two successive random leave-one-out splits.
- First split: reserve one test interaction per user with at least two distinct games, using seed 42.
- Second split: reserve one validation interaction per user with at least two remaining games, using seed 43.
- Each user retains at least one training interaction. Users with only one original game remain training-only.
- Both models are fitted on the same training partition and evaluated on the same validation partition.
- Metric: Recall@10
- Popularity validation result: **0.2105 (21.05%)**
- Item similarity with popularity fallback validation result: **0.3862 (38.62%)**
- Improvement: approximately **17.57 percentage points**, based on the rounded scores.
- The reserved test set is not scored in this experiment.

## Observations
- Item similarity with popularity fallback achieves higher validation Recall@10 than popularity alone on this split.
- Recommendations depend on the user's training history rather than only on global game popularity.
- The reported score includes the popularity fallback; it does not isolate the contribution of similarity scoring alone.
- Every validation user has training history, so the unknown-user fallback is not exercised by this evaluation. Filling remaining slots with popularity can still occur.
- Experiment 001's 24.20% test score is not directly comparable with these validation results because the training and evaluation partitions differ.

## Limitations
- The split is random rather than chronological because the dataset has no timestamps.
- Validation covers users with at least three original games; it does not measure performance for new users or users with shorter histories.
- Binary interactions ignore differences in playtime and do not prove that a user liked a game.
- Missing interactions do not necessarily indicate dislikes.
- Games absent from training cannot be recommended.
- Recall@10 measures recovery of hidden interactions, not user satisfaction or indie discovery.
- No game metadata or indie-label policy is used.
- Results come from one fixed split; performance may differ with other splits.

## Next Steps
1. Inspect recommendations alongside several users' training histories to understand their relevance.
2. Compare validation results with and without popularity fallback to measure its contribution.
3. Implement implicit matrix factorization and compare it using the same training and validation partitions.
4. Keep the test set reserved until the final model and settings have been selected.
