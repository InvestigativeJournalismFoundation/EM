from __future__ import annotations

import argparse

from .build_gold import create_gold
from .build_train_valid_test import build_splits
from .build_predict_pairs import build_predict_pairs
from .train_model import train
from .test_and_analyze import run_test
from .predict_and_analyze import run_predict


STAGES = [
    "build_gold",
    "build_splits",
    "build_predict",
    "train",
    "test",
    "predict",
]


def run(dataset: str, stage: str) -> None:
    if stage == "all":
        seq = STAGES
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Run end-to-end Ditto pipeline by stage.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--stage", default="all", help="all | build_gold | build_splits | build_predict | train | test | predict")
    args = ap.parse_args()
    run(args.dataset, args.stage)


if __name__ == "__main__":
    main()
