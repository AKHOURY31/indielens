import numpy as np
import pandas as pd
import pytest
from scipy.sparse import isspmatrix_csr
from indielens.data import load_steam200k
from indielens.model import PopularityRecommender, ItemSimilarityRecommender
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
    """
    Test checks:
    1- The split sizes
    2- Train/test row indices don’t overlap
    3- Checks that the two DataFrames have matching contents, indices, and types
    """

    d = pd.DataFrame({"user_id": ["1", "1", "2", "2", "3"], "game": ["A", "B", "B", "C", "D"]})
    tr, te = split_leave_one_out(d, 42)

    assert len(te) == 2
    assert len(tr) == 3

    assert set(te.index).isdisjoint(tr.index)

    train_a, test_a = split_leave_one_out(d, seed=42)
    train_b, test_b = split_leave_one_out(d, seed=42)
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)

    assert recall_at_k(PopularityRecommender().fit(tr), te, 2) >= 0


def test_split_holds_out_one_game_per_eligible_user():
    """
    Checks that each eligible user has exactly one hidden game and that a user with only one game 
    stays in training
    """

    d = pd.DataFrame({
        "user_id": ["1", "1", "1", "2", "2", "3"],
        "game": ["A", "B", "C", "A", "D", "E"],
    })

    train, test = split_leave_one_out(d, seed=42)

    # Users 1 and 2 have multiple games: each gets one test interaction.
    test_counts = test.groupby("user_id").size().to_dict()
    assert test_counts == {"1": 1, "2": 1}

    # Both eligible users still have games available for training.
    train_counts = train.groupby("user_id").size().to_dict()
    assert train_counts == {"1": 2, "2": 1, "3": 1}

    # User 3's only interaction remains unchanged in training
    # (User 3’s rows in training must be identical to their rows in the original dataset)
    pd.testing.assert_frame_equal(
        train[train["user_id"] == "3"],
        d[d["user_id"] == "3"],
    )

def test_recall_at_k_known_answer():
    """
    It checks Recall@K using fixed recommendations whose expected score we can calculate ourselves
    """
    class FakeRecommender:
        def recommend(self, user_id, k=10):
            recommendations = {
                "1": ["B", "D"],
                "2": ["A", "D"],
            }
            return recommendations[user_id][:k]

    # This is the "test" results that we actually fabricated
    test = pd.DataFrame({
        "user_id": ["1", "2"],
        "game": ["B", "D"],
    })

    model = FakeRecommender()

    assert recall_at_k(model, test, k=1) == pytest.approx(0.5)
    assert recall_at_k(model, test, k=2) == pytest.approx(1.0)


def test_train_validation_test_split():
    """Check that the three partitions preserve all rows without overlap."""
    d = pd.DataFrame({
        "user_id": ["1", "1", "1", "2", "2", "3"],
        "game": ["A", "B", "C", "A", "D", "E"],
    })

    train_validation, test = split_leave_one_out(d, seed=42)
    train, validation = split_leave_one_out(train_validation, seed=43)

    train_indices = set(train.index)
    validation_indices = set(validation.index)
    test_indices = set(test.index)

    # No row belongs to more than one partition.
    assert train_indices.isdisjoint(validation_indices)
    assert train_indices.isdisjoint(test_indices)
    assert validation_indices.isdisjoint(test_indices)

    # Every original row belongs to one partition.
    assert train_indices | validation_indices | test_indices == set(d.index)

    # User 1 appears in all three partitions; user 2 in training and test.
    # User 3's only game stays in training.
    assert train.groupby("user_id").size().to_dict() == {
        "1": 1, "2": 1, "3": 1,
    }
    assert validation.groupby("user_id").size().to_dict() == {"1": 1}
    assert test.groupby("user_id").size().to_dict() == {"1": 1, "2": 1}


def test_item_similarity_builds_binary_user_game_matrix():
    # Deliberately unordered, with a duplicate: user 2's Game B counts once.
    interactions = pd.DataFrame({
        "user_id": ["2", "1", "3", "1", "2", "2", "2"],
        "game": ["B", "A", "C", "B", "A", "C", "B"],
    })

    model = ItemSimilarityRecommender().fit(interactions)

    assert list(model.users) == ["1", "2", "3"]
    assert list(model.games) == ["A", "B", "C"]
    assert model.user_to_index == {"1": 0, "2": 1, "3": 2}
    assert model.game_to_index == {"A": 0, "B": 1, "C": 2}
    assert isspmatrix_csr(model.interaction_matrix)
    np.testing.assert_array_equal(
        model.interaction_matrix.toarray(),
        [[1, 1, 0], [1, 1, 1], [0, 0, 1]],
    )


def test_item_similarity_cosine_known_values():
    # A and B have identical audiences; C shares one of their two players.
    # D has no players in common with any other game.
    interactions = pd.DataFrame({
        "user_id": ["1", "1", "2", "2", "2", "3", "4"],
        "game": ["A", "B", "A", "B", "C", "C", "D"],
    })

    model = ItemSimilarityRecommender().fit(interactions)

    assert list(model.games) == ["A", "B", "C", "D"]
    assert isspmatrix_csr(model.similarity_matrix)
    np.testing.assert_allclose(
        model.similarity_matrix.toarray(),
        [
            [0, 1, 0.5, 0],
            [1, 0, 0.5, 0],
            [0.5, 0.5, 0, 0],
            [0, 0, 0, 0],
        ],
        rtol=1e-6,
        atol=1e-7,
    )


def test_item_recommendations_rank_unseen_games_and_use_fallback():
    interactions = pd.DataFrame({
        "user_id": ["target", "target", "u", "u", "v", "v", "w", "w", "loner"],
        "game": ["A", "B", "A", "C", "B", "C", "A", "D", "E"],
    })
    model = ItemSimilarityRecommender().fit(interactions)

    # C is related to both A and B (C = 1 iff (A = 1 OR B = 1)); D only to A. E has no shared players.
    assert model.recommend("target", k=1) == ["C"]
    assert model.recommend("target", k=10) == ["C", "D", "E"]
    # No similarity signal for the loner: use popularity, excluding their E.
    assert model.recommend("loner", k=10) == ["A", "B", "C", "D"]
    assert model.recommend("unknown", k=3) == ["A", "B", "C"]

    # To make sure that we correclty catch the error in recommend
    with pytest.raises(ValueError, match="k must be positive"):
        model.recommend("target", k=0)


def test_item_recommendations_when_every_game_is_seen():
    model = ItemSimilarityRecommender().fit(pd.DataFrame({
        "user_id": ["1", "1"],
        "game": ["A", "B"],
    }))
    assert model.recommend("1", k=10) == []
    assert model.recommend(1, k=10) == []
