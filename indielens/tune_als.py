"""Compare a small, predefined ALS grid on validation; never score the test set."""
import argparse
from importlib.metadata import version
from itertools import product
from pathlib import Path
from time import perf_counter

import pandas as pd
from threadpoolctl import threadpool_limits

from .data import load_steam200k
from .evaluation import recall_at_k, split_leave_one_out
from .model import ImplicitALSRecommender


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/steam-200k.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/experiment-003-results.csv"),
    )
    args = parser.parse_args()
    if not args.data.exists():
        parser.error(f"Dataset missing: {args.data}")

    interactions = load_steam200k(args.data)
    train_validation, reserved_test = split_leave_one_out(interactions, seed=42)
    train, validation = split_leave_one_out(train_validation, seed=43)
    if validation.empty:
        parser.error("No validation interactions available for tuning")
    print(
        f"Train: {len(train):,}; validation: {len(validation):,}; "
        f"reserved test (not scored): {len(reserved_test):,}", flush=True,
    )

    '''parents=True means "create any missing folders needed to reach this folder."
    Suppose your output file is: reports/als/run1/results.csv
    If only reports exists: With parents=True, Python creates both missing folders:
    als and then run1. without parents=True, Python tries to create only run1, but fails 
    because its containing folder, als, doesn't exist. 
    exist_ok=True: doesn't raise an error if the folder already exists. 
    '''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []

    # Six planned settings; data, split, metric, and other parameters stay fixed.
    for factors, alpha in product([16, 32, 64], [1.0, 10.0]):
        # flush=True forces the program to output the text to your terminal or log file immediately
        print(f"Training factors={factors}, alpha={alpha:g} ...", flush=True)
        start = perf_counter() # More performant than time.time()
        model = ImplicitALSRecommender(
            factors=factors, alpha=alpha, regularization=0.1,
            iterations=20, random_state=42,
        ).fit(train)
        with threadpool_limits(limits=1, user_api="blas"):
            recall = recall_at_k(model, validation, k=10)
        results.append({
            "factors": factors,
            "alpha": alpha,
            "regularization": 0.1,
            "iterations": 20,
            "random_state": 42,
            "test_split_seed": 42,
            "validation_split_seed": 43,
            "k": 10,
            "train_interactions": len(train),
            "validation_interactions": len(validation),
            "validation_hits": round(recall * len(validation)),
            "validation_recall_at_10": recall,
            "elapsed_seconds": round(perf_counter() - start, 3),
            "implicit_version": version("implicit"), # automatically extracts and logs the exact version number 
                                                     # of the installed Python package named implicit
        })
        # Save after every configuration so completed measurements are preserved.
        pd.DataFrame(results).to_csv(args.output, index=False)
        print(f"Validation Recall@10: {recall:.6f}", flush=True)

    ranked = pd.DataFrame(results).sort_values(
        "validation_recall_at_10", ascending=False, kind="stable",
    )
    print("\nResults (best validation score first):")
    print(ranked[["factors", "alpha", "validation_recall_at_10"]].to_string(index=False))
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
