# IndieLens 🎮

**Goal:** A reproducible, indie-aware video game recommendation system that learns from a user's play history. Built as an ML-engineering portfolio project.

## Milestone 1 — reproducible popularity baseline

The baseline counts **unique users per game** and recommends popular games not already seen. Its job is to give later collaborative filtering / PyTorch retrieval models a clear benchmark to beat. **It does not yet identify indie games.**

### Set up (PowerShell / Windows 11)

```powershell
cd path\to\indielens_starter
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
```

If PowerShell execution policy blocks activation, you can use ` .\.venv\Scripts\python.exe -m pip install -e ".[dev]" ` and ` .\.venv\Scripts\python.exe -m pytest -q ` without activating.

### Get the data

1. Download `steam-200k.csv` from https://www.kaggle.com/datasets/tamber/steam-video-games (Kaggle account may be required).
2. Put it at `data/raw/steam-200k.csv`.
3. Run:

```powershell
python -m indielens.cli --data data/raw/steam-200k.csv --top-k 10
```

The repository deliberately does not redistribute external data.

### Data rules

- Dataset columns: `user_id, game, behavior, value, unused` (no header).
- `purchase` always has numeric value `1`; **never** treat that as an hour played.
- `play` value means hours played; repeated user/game records are merged.
- Hold out one random interaction for each user with >=2 unique games. Users with only one game remain in training. This split is **not chronological** because the data has no timestamps.
- Report **Recall@10** for held-out interactions, not prediction accuracy. Missing user-game pairs are unknown, not proven negatives.

### Roadmap

- [x] Clean Python packaging and tests
- [x] Offline Steam 200K loader and popularity baseline
- [ ] Validate real dataset counts and baseline metrics
- [ ] Item-based similarity / matrix-factorization baseline
- [ ] PyTorch two-tower candidate retriever
- [ ] FAISS retrieval benchmark
- [ ] Game metadata validation and indie-label policy (indie / AAA / unknown)
- [ ] Ranking policy with configurable indie promotion; track relevance tradeoffs
- [ ] Cold-start evaluation and sparse-indie coverage
- [ ] FastAPI, Docker, CI, then optional Cloud Run deployment
- [ ] Optional Steam import (never required for core training)

### Project design decisions

- Train **offline** from a downloadable data snapshot. APIs are optional and cannot block progress.
- Keep user-game interactions in a generic schema so a music dataset could replace games later.
- Separate **predicted user preference** from **business/policy reranking** (indie preference).
- An indie tag is not reliable proof of independent development. Don't infer AAA from price or popularity alone.
- Use a second holdout or cross-validation for tuning before final reporting. Avoid evaluating many design choices against the same test set.

### First experiment notes

After running the CLI, record dataset dimensions, Recall@10, limitations, and at least three possible improvements in `reports/experiment-001.md`. Keep measured numbers here, not invented numbers.
