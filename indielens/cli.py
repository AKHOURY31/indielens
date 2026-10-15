"""IndieLens first runnable milestone."""
import argparse
from pathlib import Path
from .data import load_steam200k
from .model import PopularityRecommender
from .evaluation import split_leave_one_out, recall_at_k


def main():
    # Creates an argument parser. It reads command-line options and provides help text when you run
    parser = argparse.ArgumentParser(description="Train and inspect IndieLens's first baseline")
    # type=Path: converts the supplied text into a Path object.
    parser.add_argument("--data", type=Path, default=Path("data/raw/steam-200k.csv"))
    parser.add_argument("--top-k", type=int, default=10)
    # Reads the supplied command-line options and stores them in args.
    args = parser.parse_args()
    if not args.data.exists():
        parser.error(f"Dataset missing: {args.data}. See README for download instructions.")
    interactions = load_steam200k(args.data)
    train, test = split_leave_one_out(interactions)
    model = PopularityRecommender().fit(train)
    #:, formats the number with thousands separators: 12345 becomes 12,345.
    print(f"Unique users: {interactions.user_id.nunique():,}")
    print(f"Unique games: {interactions.game.nunique():,}")
    print(f"User-game interactions: {len(interactions):,}")
    print(f"Train interactions: {len(train):,}; test interactions: {len(test):,}")
    print(f"Popularity Recall@{args.top_k}: {recall_at_k(model, test, args.top_k):.4f}")
    '''
    Selects a user for an example:
    1. train.user_id gets the user ID column.
    2. .iloc[0] gets its first value by position.
    Unlike .loc, which uses index labels, .iloc uses positions: 0 means first, 1 means second.
    '''
    example_user = train.user_id.iloc[0]
    print(f"Example recommendations for anonymized user {example_user}:")
    for game in model.recommend(example_user, args.top_k):
        print(f"  - {game}")


if __name__ == "__main__":
    main()
