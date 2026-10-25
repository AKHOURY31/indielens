"""User-based holdout and Recall@K for implicit-feedback evaluation."""
import numpy as np
import pandas as pd


def split_leave_one_out(interactions: pd.DataFrame, seed: int = 42):
    """Hold out one interaction per eligible user (>=2 games).

    This is a RANDOM holdout because Steam 200K has no timestamps.
    Users with one interaction remain training-only. Do not interpret missing
    interactions as explicit dislikes; offline recall is only a proxy.
    """
    '''
    Creates a random number generator. Using the same seed with the same input and ordering makes the
    selections repeatable. The value 42 has no special meaning here.
    '''
    
    rng = np.random.default_rng(seed)
    eligible = interactions.groupby("user_id").filter(lambda g: len(g) >= 2)
    holdout_indices = []
    for _, group in eligible.groupby("user_id", sort=True):
        '''
        Selects one row from the current user’s group:
        1. group.index gets that group’s row labels.
        2. .to_numpy() converts those labels to a NumPy array.
        3. rng.choice(...) randomly chooses one label. 
        '''
        holdout_indices.append(rng.choice(group.index.to_numpy()))

    '''
    Builds the test table from the selected rows.
    .loc[...] selects rows by their index labels, and .copy() creates a separate table.
    '''
    test = interactions.loc[holdout_indices].copy()
    train = interactions.drop(index=holdout_indices).copy()
    return train, test


def recall_at_k(model, test: pd.DataFrame, k: int = 10) -> float:
    '''
    This function measures the model's success.

    Parameters:
        model (model): A recommender already fitted on train.
        test (Datafame) : Hidden interactions.
        k (int, default = 10): Number of recommendations to check.

    Returns:
        float: Ratio of successful recommendation/ total number of tests.
    '''
    if test.empty:
        return float("nan")
    hits = sum(row.game in model.recommend(row.user_id, k) for row in test.itertuples())
    '''
    This is an expanded version of the line above. It counts successful recommendations:
    hits = 0

    for row in test.itertuples():
        recommendations = model.recommend(row.user_id, k)

        if row.game in recommendations:
            hits += 1
    '''
    return hits / len(test)
