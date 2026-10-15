"""A transparent popularity baseline. Later models must beat it."""
import pandas as pd


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
