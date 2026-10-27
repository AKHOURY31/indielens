# Experiment 003 — Implicit Matrix Factorization and Validation Tuning

## Date
26/10/2026

## Dataset source link
https://www.kaggle.com/datasets/tamber/steam-video-games

## Command to run code

Current CLI comparison of the selected ALS configuration (64 factors, alpha 10)
with popularity and item similarity:

```powershell
.\.venv\Scripts\python.exe -m indielens.cli --data data/raw/steam-200k.csv --top-k 10
```

Six-configuration ALS validation experiment:

```powershell
.\.venv\Scripts\python.exe -m indielens.tune_als --data data/raw/steam-200k.csv --output reports/experiment-003-results.csv
```

## Objective
Establish an implicit matrix-factorization baseline and select ALS settings from a small, predefined grid using validation Recall@10.

## Dataset

- Source: Steam 200K
- Unique users: 12,393
- Unique games: 5,153
- Total user-game interactions: 128,801
- Training interactions: 116,891
- Validation interactions: 5,217
- Reserved test interactions: 6,693

## Model
Implicit matrix factorization using CPU alternating least squares (ALS), implemented with `implicit==0.7.3`.

The model uses a binary sparse user-game matrix. Each unique training pair receives a value of 1; playtime weights are not used. ALS learns a vector for every training user and game. Recommendations rank unseen games by their dot product with the user vector, including games with negative scores. Equal scores are resolved alphabetically.

The library multiplies observed confidence by alpha; missing pairs have a target of zero with confidence 1. These missing pairs are not confirmed dislikes. Popularity fitted on the same training data provides a fallback for unknown users. Every validation user has training history, so that fallback is not exercised in this evaluation.

Initial settings:

- Factors: 32
- Alpha: 10
- Regularization: 0.1
- Iterations: 20
- Random state: 42
- ALS threads: 1; BLAS threads limited to 1 during fitting and evaluation

The CLI now explicitly uses 64 factors and alpha 10, selected from the grid below.
The initial 32-factor result is preserved in this report and can be reproduced by
the tuning command. The class default remains 32 factors.

## Evaluation

- Split: reserve test interactions using random leave-one-out with seed 42, then split the remaining data using seed 43 to obtain validation interactions.
- Validation eligibility: users with at least three original distinct games, preserving at least one training game.
- All configurations use exactly the same training and validation partitions.
- Metric: Recall@10 over the full training catalog, excluding each user's seen games; no sampled negatives.
- The test set is reserved and is not scored or used to choose settings.
- Fixed parameters: regularization 0.1, iterations 20, random state 42.
- Planned grid: factors in {16, 32, 64}, alpha in {1, 10}.

Measured validation results:

| Factors | Alpha | Hits / 5,217 | Recall@10 |
|---|---|---|---|
| 16 | 1 | 1,902 | 0.364577 (36.46%) |
| 16 | 10 | 2,034 | 0.389879 (38.99%) |
| 32 | 1 | 1,873 | 0.359019 (35.90%) |
| 32 | 10 | 2,079 | 0.398505 (39.85%) |
| 64 | 1 | 1,904 | 0.364961 (36.50%) |
| **64** | **10** | **2,113** | **0.405022 (40.50%)** |

Reference validation scores on the same split:

- Popularity: **0.2105 (21.05%)**
- Item similarity with popularity fallback: **0.3862 (38.62%)**
- Initial ALS configuration: **0.398505 (39.85%)**, reproduced in this grid.
- Best ALS configuration in this grid: **0.405022 (40.50%)**

Full-precision scores, hit counts, configuration values, library version, and elapsed times are saved in [experiment-003-results.csv](experiment-003-results.csv). Times cover fitting and validation scoring and depend on the machine and load.

## Observations

- The best configuration among these six uses 64 factors and alpha 10; this is the selected ALS setting for future comparisons, not the final model selection across all approaches.
- It recovers 34 more hidden games than the initial ALS configuration, improving validation recall by approximately 0.65 percentage points.
- Its improvement over item similarity is approximately 1.88 percentage points, using the rounded reference score.
- Alpha 10 outperforms alpha 1 at every tested factor count on this split. More factors alone do not consistently improve recall.
- The main CLI uses the selected 64-factor, alpha-10 configuration. Add `--inspect` to compare its recommendations with item similarity for the same three validation users, selected by training-history length.

Inspection on 27/10/2026 produced these examples. Users were selected by training-history length, not by recommendation success:

| User | Training games | Hidden validation game | Item similarity rank | Selected ALS rank |
|---|---|---|---|---|
| 33755449 | 2 | Day of Defeat Source | 3 | 2 |
| 161663718 | 6 | Toribash | Outside top 10 | Outside top 10 |
| 115671217 | 17 | Mark of the Ninja | 1 | 3 |

Both models retrieve the hidden game for two of these users and miss it for the third. These three examples illustrate rankings; they do not estimate overall performance. Item similarity still recommends `Psychonauts Demo`, and both lists for the medium-history user contain Fallout New Vegas add-ons. ALS does not by itself resolve the catalog filtering issue.

## Limitations

- These are validation results used for model selection, not an independent final test estimate.
- Only one split and initialization seed are evaluated; small score differences may not generalize.
- Regularization and iteration count are held fixed, and only six configurations are tested. The selected settings are best within this grid, not proven optimal.
- The split is random rather than chronological because timestamps are unavailable.
- New users and users with fewer than three original games are not evaluated.
- Games absent from training cannot be recommended.
- Binary interactions ignore playtime intensity, and missing interactions do not establish dislike.
- The catalog still includes demos, DLC, and soundtracks; metadata filtering and indie labels have not been added.
- Recall@10 measures recovery of hidden interactions, not user satisfaction or indie discovery.

## Next Steps

1. Preserve these inspection examples as a qualitative reference for future models.
2. Implement a two-tower model and compare it on the same validation split.
3. Add metadata validation and consistent catalog policies before evaluating indie discovery.
4. After selecting the final approach and settings, refit on training plus validation and evaluate on the reserved test set.
