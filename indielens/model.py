"""Popularity baseline and item-based collaborative filtering."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class PopularityRecommender:
    """
    This class defines a popularity recommender: it recommends popular games while
    excluding games the user has already interacted with
    """
    def fit(self, interactions: pd.DataFrame):
        # Popularity represents each game's popularity (nbr of purchases)
        self.popularity = (
            interactions.groupby("game")["user_id"].nunique()
            .sort_values(ascending=False, kind="stable") #kind="stable": games with equal counts keep their relative order from before this sorting step.
        )
        # Seen lets the recommender look up a user’s existing games stored in the form of a dictionnary.
        self.seen = interactions.groupby("user_id")["game"].agg(set).to_dict()
        return self

    def recommend(self, user_id: str, k: int = 10) -> list[str]:
        """
        Method that generates recommendations
        """
        if k < 1:
            raise ValueError("k must be positive")
        seen = self.seen.get(str(user_id), set())
        # .index returns the game names because they are the index, not because they occupy the first column.
        # In other tables, the index might simply be row numbers such as 0, 1, 2.
        return [game for game in self.popularity.index if game not in seen][:k]


class ItemSimilarityRecommender:
    """First step: build a binary user-game matrix from training interactions."""

    def fit(self, interactions: pd.DataFrame):
        # Each user/game pair contributes one 1, even if it appears repeatedly.

        # ["user_id", "game"]             # A list of column names
        # interactions[that_list]         # Select those columns
        pairs = interactions[["user_id", "game"]].copy()    #.copy() to create a separate copy
        pairs["user_id"] = pairs["user_id"].astype(str)
        pairs = pairs.drop_duplicates()

        # Replace labels with integer row/column positions.
        user_rows, self.users = pd.factorize(pairs["user_id"], sort=True)
        game_columns, self.games = pd.factorize(pairs["game"], sort=True)
        
        # Building a dico here
        self.user_to_index = {
            user_id: index for index, user_id in enumerate(self.users)
        }
        self.game_to_index = {
            game: index for index, game in enumerate(self.games)
        }

        # CSR stores observed entries efficiently; missing entries are zero.
        self.interaction_matrix = csr_matrix(
            (np.ones(len(pairs), dtype=np.float32), (user_rows, game_columns)),
            shape=(len(self.users), len(self.games)),
        )
        return self
