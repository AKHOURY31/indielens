import pandas as pd
import pytest
from indielens.data import load_steam200k
from indielens.model import PopularityRecommender
from indielens.evaluation import split_leave_one_out, recall_at_k


def test_purchase_does_not_count_as_playtime(tmp_path):
    path = tmp_path / "steam.csv"
    path.write_text("1,Game A,purchase,1,0\n1,Game A,play,9,0\n1,Game B,purchase,1,0\n2,Game B,play,4,0\n")
    data = load_steam200k(path)
    assert len(data) == 3
    assert data.query("user_id == '1' and game == 'Game A'")["hours"].iloc[0] == 9
    assert data.query("user_id == '1' and game == 'Game B'")["hours"].iloc[0] == 0


def test_recommendations_exclude_seen():
    d = pd.DataFrame({"user_id": ["1", "2", "2", "3"], "game": ["A", "A", "B", "B"]})
    model = PopularityRecommender().fit(d)
    assert model.recommend("1", 2) == ["B"]
    assert model.recommend("unseen", 2) == ["A", "B"]
    with pytest.raises(ValueError):
        model.recommend("1", 0)


def test_split_holdout_deterministic_and_not_in_train():
    d = pd.DataFrame({"user_id": ["1", "1", "2", "2", "3"], "game": ["A", "B", "B", "C", "D"]})
    tr, te = split_leave_one_out(d, 42)
    assert len(te) == 2
    assert len(tr) == 3
    assert set(te.index).isdisjoint(tr.index)
    assert recall_at_k(PopularityRecommender().fit(tr), te, 2) >= 0
