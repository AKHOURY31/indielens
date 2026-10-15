"""Read the public Steam 200K interactions dataset into a common schema."""
from pathlib import Path
import numpy as np
import pandas as pd


def load_steam200k(path: str | Path) -> pd.DataFrame:
    """Return unique user/game pairs with implicit weights from playtime.

    Steam 200K has no header: user_id, game_title, behavior, value, unused.
    'purchase' has value 1 regardless of playtime and must not be read as hours.
    A purchase with no play row receives a weak positive weight of 1.
    """
    frame = pd.read_csv(path, header=None, names=["user_id", "game", "behavior", "value", "unused"])
    frame = frame.drop(columns="unused")
    frame = frame[frame["behavior"].isin(["purchase", "play"])].copy()
    frame["user_id"] = frame["user_id"].astype(str)
    frame["game"] = frame["game"].astype(str).str.strip()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"])
    frame = frame[(frame["value"] >= 0) & (frame["game"] != "")]
    frame["hours"] = np.where(frame["behavior"].eq("play"), frame["value"], 0.0)
    # Aggregate duplicated rows, count purchases once, avoid treating purchases as playtime.
    group = frame.groupby(["user_id", "game"], as_index=False).agg(
        hours=("hours", "max"),
        purchased=("behavior", lambda s: (s == "purchase").any()),
    )
    group["weight"] = np.where(group["hours"] > 0, 1 + np.log1p(group["hours"]), 1.0)
    return group[["user_id", "game", "hours", "purchased", "weight"]]
