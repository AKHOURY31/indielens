"""IndieLens first runnable milestone."""
import argparse
from pathlib import Path
from threadpoolctl import threadpool_limits
from .data import load_steam200k
from .model import PopularityRecommender, ItemSimilarityRecommender, ImplicitALSRecommender
from .evaluation import split_leave_one_out, recall_at_k


def inspect_recommendations(models, train, validation, k=10):
    """Inspect users at the 25th, 50th, and 75th history-length percentiles.
    This function displays three users' recommendations, selected around the 25th, 50th, and 75th 
    percentiles of training-history length.
    models maps display names to fitted recommenders. For each user, show the
    same training history and hidden game, then each model's recommendations."""
    
    if validation.empty:
        print("No validation users available for inspection.")
        return

    # Select by training history length only, without looking at model outcomes.
    counts = train.groupby("user_id").size()

    # Keeps only users who appear in validation.
    counts = counts[counts.index.isin(validation["user_id"])]

    counts = counts.sort_values(kind="stable")
    '''
    validation.set_index("user_id"): Changes the table's main row labels (the index) from generic numbers
    (0, 1, 2...) to the actual user_id values.
    ["game"]: keeps only the "game" column
    Now, instead of searching the whole table,.loc[] can be used to instantly see what game a user played
    '''
    validation_games = validation.set_index("user_id")["game"]
    selected_users = set()

    for label, fraction in [("short", 0.25), ("medium", 0.5), ("long", 0.75)]:
        position = round((len(counts) - 1) * fraction) # Calculates a position in the sorted user list
        user_id = counts.index[position]
        if user_id in selected_users:
            continue
        selected_users.add(user_id)
        # Keep rows belonging to this user, and return their game column. Sort them alphabetically
        history = sorted(train.loc[train["user_id"] == user_id, "game"]) 
        hidden_game = validation_games.loc[user_id]

        print(f"\nUser {user_id} ({label} history; {len(history)} training games)")
        print("Training history (up to 20 titles, alphabetical):")
        for game in history[:20]:
            print(f"  - {game}")
        if len(history) > 20:
            print(f"  ... {len(history) - 20} more training games")
        print(f"Hidden validation game: {hidden_game}")
        # Compare both models for this user before moving to the next user.
        for model_name, model in models.items():
            recommendations = model.recommend(user_id, k)
            print(f"{model_name} top {k} recommendations:")
            for rank, game in enumerate(recommendations, start=1):
                print(f"  {rank}. {game}")
            print(f"{model_name} Hit@{k}: {'yes' if hidden_game in recommendations else 'no'}")


def main():
    # Creates an argument parser. It reads command-line options and provides help text when you run
    parser = argparse.ArgumentParser(description="Train and inspect IndieLens's first baseline")
    # type=Path: converts the supplied text into a Path object.
    parser.add_argument("--data", type=Path, default=Path("data/raw/steam-200k.csv"))
    parser.add_argument("--top-k", type=int, default=10)
    '''
    If we add --inspect, the inspect_recommendations function will run, otherwise it won't'''
    parser.add_argument(
        "--inspect", action="store_true",
        help="Inspect validation users at three training-history lengths",
    )
    # Reads the supplied command-line options and stores them in args.
    args = parser.parse_args()
    if not args.data.exists():
        parser.error(f"Dataset missing: {args.data}. See README for download instructions.")
    interactions = load_steam200k(args.data)

    train_validation, test = split_leave_one_out(interactions, seed=42)
    train, validation = split_leave_one_out(train_validation, seed=43)
    model = PopularityRecommender().fit(train)
    #:, formats the number with thousands separators: 12345 becomes 12,345.
    print(f"Unique users: {interactions.user_id.nunique():,}")
    print(f"Unique games: {interactions.game.nunique():,}")
    print(f"User-game interactions: {len(interactions):,}")
    print(
        f"Train interactions: {len(train):,}; "
        f"validation interactions: {len(validation):,}; "
        f"test interactions: {len(test):,}"
    )
    print(
        f"Popularity Validation Recall@{args.top_k}: "
        f"{recall_at_k(model, validation, args.top_k):.4f}"
    )
    item_model = ItemSimilarityRecommender().fit(train)
    print(
        f"Item similarity Validation Recall@{args.top_k}: "
        f"{recall_at_k(item_model, validation, args.top_k):.4f}"
    )
    als_model = ImplicitALSRecommender(factors=64, alpha=10.0).fit(train)

    '''Small per-user vector products avoid the overhead of many BLAS (Basic Linear Algebra Subprograms)
    threads. This operation can use BLAS for ex: scores = self.item_factors @ self.user_factors[user_index]
    A BLAS library can distribute calculations across several CPU threads—workers that perform computation. 
    That helps with large operations, but coordinating workers adds overhead. For many small operations, 
    one thread can be faster.'''
    with threadpool_limits(limits=1, user_api="blas"): # Limit BLAS to one thread
        als_score = recall_at_k(als_model, validation, args.top_k)
    print(f"Implicit ALS Validation Recall@{args.top_k}: {als_score:.4f}")

    '''
    Selects a user for an example:
    1. train.user_id gets the user ID column.
    2. .iloc[0] gets its first value by position.
    Unlike .loc, which uses index labels, .iloc uses positions: 0 means first, 1 means second.
    '''
    example_user = train.user_id.iloc[0]
    print(f"Example item-similarity recommendations for anonymized user {example_user}:")
    for game in item_model.recommend(example_user, args.top_k):
        print(f"  - {game}")
    if args.inspect:
        with threadpool_limits(limits=1, user_api="blas"):
            inspect_recommendations(
                {"Item similarity": item_model, "Implicit ALS (64 factors)": als_model},
                train, validation, args.top_k,
            )


if __name__ == "__main__":
    main()
