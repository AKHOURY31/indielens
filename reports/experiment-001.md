# Experiment 001 — Popularity-Based Recommendation Baseline

## Date
15/10/2026

## Dataset source link
https://www.kaggle.com/datasets/tamber/steam-video-games

## Command to run code
.\.venv\Scripts\python.exe -m indielens.cli --data data/raw/steam-200k.csv --top-k 10

## Objective
Establish a simple recommendation baseline against which future personalized recommendation models can be evaluated.

## Dataset
- Source: Steam 200K
- Unique users: 12,393
- Unique games: 5,153
- Total user-game interactions: 128,801
- Training interactions: 122,108
- Test interactions: 6,693

## Model
Popularity-based recommender.

The model ranks games according to their popularity in the training dataset and recommends the most popular games that the user has not already interacted with.

## Evaluation
- Split: random leave-one-out, with seed 42.
- Eligibility: users with at least two distinct games.
- Users with one game remain in training only.
- Metric: Recall@10
- Recall@10: 1,620 of 6,693 held-out games appeared among the corresponding users' top ten recommendations.
- Result: **0.2420 (24.2%)**

## Observations
- The baseline recommends highly popular games such as Dota 2, Team Fortress 2, and Counter-Strike.
- The model does not learn personalized preferences.
- Popularity may be a useful baseline, but it can disadvantage less-popular indie games.

## Limitations
- Only 54.01% of users are evaluated.
- Seventeen test interactions involve games absent from training,
  which this model cannot recommend.
- The split is random rather than chronological.
- Missing interactions do not necessarily indicate dislikes.

## Next Steps
1. Implement a personalized recommendation model using matrix factorization.
2. Compare its Recall@10 against the popularity baseline using the same data split.
3. Investigate whether personalized recommendations improve discovery of less-popular games.
