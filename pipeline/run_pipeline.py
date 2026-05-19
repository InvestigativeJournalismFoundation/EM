from __future__ import annotations

import argparse
import time

from .build_gold import create_gold
from .build_train_valid_test import build_splits
from .build_predict_pairs import build_predict_pairs
from .train_model import train
from .test_and_analyze import run_test
from .predict_and_analyze import run_predict
from .fetch_data import fetch_data
from .fetch_model import fetch_model
from dotenv import load_dotenv

STAGES = [
    "build_gold",
    "build_splits",
    "build_predict",
    "train",
    "test",
    "predict",
    "fetch_data",
    "fetch_model",
]

FULL_RUN =  [
    "fetch_data",
    "build_gold",
    "build_splits",
    "build_predict",
    "train",
    "test",
    "predict",
]

INFERENCE_RUN = [
    "fetch_model",
    "fetch_data",
    "build_predict",
    "predict",
]


def run(dataset: str, stage: str, size: int = None, batch_size: int = None) -> None:
    load_dotenv()
    if stage == "all":
        seq = FULL_RUN
    elif stage == "inference":
        seq = INFERENCE_RUN
    else:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        seq = [stage]

    timings: list[tuple[str, float]] = []
    pipeline_start = time.time()

    for s in seq:
        print(f"\n=== Stage: {s} ===")
        stage_start = time.time()
        if s == "build_gold":
            create_gold(dataset)
        elif s == "build_splits":
            build_splits(dataset)
        elif s == "build_predict":
            build_predict_pairs(dataset)
        elif s == "train":
            train(dataset)
        elif s == "test":
            run_test(dataset)
        elif s == "predict":
            run_predict(dataset)
        elif s == "fetch_data":
            fetch_data(dataset, size=size, batch_size=batch_size)
        elif s == "fetch_model":
            fetch_model(dataset)
        else:
            raise ValueError(f"Unknown stage: {s}")
        elapsed = time.time() - stage_start
        timings.append((s, elapsed))
        print(f"    completed in {elapsed:.1f}s")

    if len(timings) > 1:
        total = time.time() - pipeline_start
        print("\n=== Timing Summary ===")
        for name, dur in timings:
            print(f"  {name:<20} {dur:>8.1f}s")
        print(f"  {'TOTAL':<20} {total:>8.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run end-to-end Ditto pipeline by stage.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--stage", default="all", help="all | fetch_data | fetch_model | build_gold | build_splits | build_predict | train | test | predict")
    ap.add_argument("--size", type=int, help="Number of rows to fetch")
    ap.add_argument("--batch_size", type=int, help="Batch size for training and inference")
    # match mode controls whether we check matches in the dataset, or between the dataset and the full dataset, or both
    ap.add_argument("--match_mode", default="self", help="self | full | both")
    args = ap.parse_args()
    run(args.dataset, args.stage, size=args.size, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
