"""Popularity baseline and item-based collaborative filtering."""
import numpy as np
import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from threadpoolctl import threadpool_limits


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


class ImplicitALSRecommender:
    """Learn user/game vectors from binary training interactions using CPU ALS.

    alpha is the confidence for observed pairs; missing pairs have confidence 1.
    """

    def __init__(
        self, factors=32, regularization=0.1, iterations=20,
        random_state=42, alpha=10.0,
    ):
        if factors < 1 or iterations < 1:
            raise ValueError("factors and iterations must be positive")
        if regularization <= 0 or alpha <= 0:
            raise ValueError("regularization and alpha must be positive")
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.alpha = alpha

    def fit(self, interactions: pd.DataFrame):
        if interactions.empty:
            raise ValueError("training interactions must not be empty")

        # Use the same binary representation and ordering as item similarity.
        pairs = interactions[["user_id", "game"]].copy()
        pairs["user_id"] = pairs["user_id"].astype(str)
        pairs = pairs.drop_duplicates()
        user_rows, self.users = pd.factorize(pairs["user_id"], sort=True)
        game_columns, self.games = pd.factorize(pairs["game"], sort=True)
        self.user_to_index = {
            user_id: index for index, user_id in enumerate(self.users)
        }
        self.game_to_index = {
            game: index for index, game in enumerate(self.games)
        }
        self.interaction_matrix = csr_matrix(
            (np.ones(len(pairs), dtype=np.float32), (user_rows, game_columns)),
            shape=(len(self.users), len(self.games)),
        )

        # Create a fresh model on each fit so previous factors are not reused.
        # Limit BLAS threads during fitting, as recommended by implicit.
        with threadpool_limits(limits=1, user_api="blas"):
            self.als = AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                random_state=self.random_state,
                alpha=self.alpha,
                num_threads=1,
            )
            # Unlike cosine similarity, ALS expects users x games: no transpose.
            self.als.fit(self.interaction_matrix, show_progress=False)
        self.user_factors = self.als.user_factors
        self.item_factors = self.als.item_factors
        self.popularity_fallback = PopularityRecommender().fit(pairs)
        return self

    def recommend(self, user_id: str, k: int = 10) -> list[str]:
        """Rank unseen games by dot product with the learned user vector."""
        if k < 1:
            raise ValueError("k must be positive")
        user_id = str(user_id)
        user_index = self.user_to_index.get(user_id)
        if user_index is None:
            return self.popularity_fallback.recommend(user_id, k)

        # A NumPy array with shape (2,) like self.user_factors[user_index] is one-dimensional, 
        # so it isn’t explicitly a row or column vector. 
        # NumPy handles its role according to which side of @ it appears on.
        scores = self.item_factors @ self.user_factors[user_index]
        history = self.interaction_matrix.getrow(user_index)
        scores[history.indices] = -np.inf

        # Negative ALS scores are valid. Exclude non-finite scores (seen games).
        candidates = np.flatnonzero(np.isfinite(scores))
        # Stable sorting resolves score ties alphabetically, as in item similarity.
        ranked_indices = candidates[np.argsort(-scores[candidates], kind="stable")]
        return [self.games[index] for index in ranked_indices[:k]]
