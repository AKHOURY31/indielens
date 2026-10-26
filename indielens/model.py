"""Popularity baseline and item-based collaborative filtering."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


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
    """Build binary training interactions and cosine similarities between games.
    Item similarity, with popularity fallback used for unknown users and to fill remaining recommendation slots."""

    def fit(self, interactions: pd.DataFrame):
        # Each user/game pair contributes one 1, even if it appears repeatedly.

        # ["user_id", "game"]             # A list of column names
        # interactions[that_list]         # Select those columns
        pairs = interactions[["user_id", "game"]].copy()    #.copy() to create a separate copy
        # Pairs is a table with to columns (user_id, then game) each row representing an interaction
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
        # in this csr_matrix function, we are placing ones at user_rows corresponding to game_columns
        self.interaction_matrix = csr_matrix(
            # len(pairs) returns the number of rows
            (np.ones(len(pairs), dtype=np.float32), (user_rows, game_columns)),
            shape=(len(self.users), len(self.games)),
        )

        # cosine_similarity compares rows, so transpose to games x users first.
        # We choose CSR because later we’ll access rows of game similarities, and CSR supports efficient row access.

        self.similarity_matrix = cosine_similarity(
            self.interaction_matrix.T, dense_output=False
        ).tocsr() 
        # Exclude each game's similarity to itself from recommendation scores.
        self.similarity_matrix.setdiag(0)
        self.similarity_matrix.eliminate_zeros()
        self.popularity_fallback = PopularityRecommender().fit(pairs)
        return self

    def recommend(self, user_id: str, k: int = 10) -> list[str]:
        """Rank unseen games by summed similarity to the user's training games."""
        if k < 1:
            raise ValueError("k must be positive")
        user_id = str(user_id)
        # Reminder: user_to_index is a dico
        user_index = self.user_to_index.get(user_id)

        # we recommend the game based on most popular games (first model) if no user id found
        if user_index is None:
            return self.popularity_fallback.recommend(user_id, k)

        # One row represents a user's interaction with every game (all columns)
        # Length of a user's history = nnumber of games he played
        history = self.interaction_matrix.getrow(user_index)

        # One score per game: sum similarities from every game in the history.
        #.toarray() changes the storage: sparse matrix -> regular np array
        #.ravel() changes the shape: a two-dimensional array ([[1.0, 1.0, 2.0]]) -> a one-dimensional array ([1.0, 1.0, 2.0])
        scores = (history @ self.similarity_matrix).toarray().ravel()
        '''
        history.indices contains the column positions of the observed interactions: (where val is different from 0)
        which means the game was played.
        -np.inf means negative infinity. Assigning it makes those games rank below every finite score so they never
        get recommended'''
        scores[history.indices] = -np.inf

        # Games are alphabetical, so stable sorting also resolves ties consistently.
        ranked_indices = np.argsort(-scores, kind="stable")
        recommendations = [
            self.games[index] for index in ranked_indices if scores[index] > 0
        ][:k]

        # Fill remaining places with popular unseen games without duplicates.
        if len(recommendations) < k:
            selected = set(recommendations)
            for game in self.popularity_fallback.recommend(user_id, len(self.games)):
                if game not in selected:
                    recommendations.append(game)
                    if len(recommendations) == k:
                        break
        return recommendations
