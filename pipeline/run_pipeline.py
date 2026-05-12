from __future__ import annotations

import argparse

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
    "build_gold",
    "build_predict",
    "predict",
]


def run(dataset: str, stage: str, size: int = None) -> None:
    load_dotenv()
    if stage == "all":
        seq = FULL_RUN
    elif stage == "inference":
        seq = INFERENCE_RUN
    else:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        seq = [stage]

    for s in seq:
        print(f"\n=== Stage: {s} ===")
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
            fetch_data(dataset, size=size)
        elif s == "fetch_model":
            fetch_model(dataset)
        else:
            raise ValueError(f"Unknown stage: {s}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run end-to-end Ditto pipeline by stage.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--stage", default="all", help="all | fetch_data | fetch_model | build_gold | build_splits | build_predict | train | test | predict")
    ap.add_argument("--size", type=int, help="Number of rows to fetch")
    args = ap.parse_args()
    run(args.dataset, args.stage, size=args.size)


if __name__ == "__main__":
    main()
