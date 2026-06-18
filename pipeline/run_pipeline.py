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
from .build_graph import build_graph
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
    "build_graph",
]

TRAIN = [
    "fetch_data",
    "build_gold",
    "build_splits",
    "train",
    "test",
]

INFERENCE_RUN = [
    "fetch_model",
    "fetch_data",
    "build_predict",
    "predict",
    "build_graph",
]

FULL_RUN = [
    "fetch_data",        # no size limit — fetches all rows for training
    "build_gold",
    "build_splits",
    "train",
    "test",
    "fetch_data",        # SIZE limit — re-fetches for inference
    "build_predict",
    "predict",
    "build_graph",
]


def run(
    dataset: str,
    stage: str,
    size: int = None,
    batch_size: int = None,
    top_k_train: int = None,
    top_k_predict: int = None,
    target_total_pairs: int = None,
    resolution: float = None,
    offset: int = None,
    model: str = None,
    model_path: str = None,
) -> None:
    load_dotenv()
    if stage == "train":
        seq = TRAIN
    elif stage == "inference":
        seq = INFERENCE_RUN
    elif stage == "full":
        seq = FULL_RUN
    else:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        seq = [stage]

    timings: list[tuple[str, float]] = []
    pipeline_start = time.time()
    fetch_count = 0

    for s in seq:
        print(f"\n=== Stage: {s} ===")
        stage_start = time.time()
        if s == "build_gold":
            create_gold(dataset)
        elif s == "build_splits":
            build_splits(dataset, top_k_train=top_k_train, target_total_pairs=target_total_pairs)
        elif s == "build_predict":
            build_predict_pairs(dataset, top_k_predict=top_k_predict, size=size, offset=offset)
        elif s == "train":
            train(dataset, model=model, model_path=model_path)
        elif s == "test":
            run_test(dataset)
        elif s == "predict":
            run_predict(dataset, model=model)
        elif s == "fetch_data":
            fetch_count += 1
            # inference and full stages always fetch the complete dataset;
            # SIZE/OFFSET are only used by build_predict to slice the anchor set
            effective_size = None if stage in ("full", "inference") else size
            fetch_data(dataset, size=effective_size, batch_size=batch_size)
        elif s == "fetch_model":
            fetch_model(dataset, model_path=model_path)
        elif s == "build_graph":
            build_graph(dataset, resolution=resolution)
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
    ap.add_argument("--stage", default="all", help="fetch_data | fetch_model | build_gold | build_splits | build_predict | train | test | predict | full | inference | train")
    ap.add_argument("--size", type=int, help="Number of rows to fetch")
    ap.add_argument("--offset", type=int, help="Offset for fetching data (used for batching)")
    ap.add_argument("--batch_size", type=int, help="Batch size for training and inference")
    ap.add_argument("--top_k_train", type=int, help="Top-k candidates per record during train blocking (overrides blocking.yaml)")
    ap.add_argument("--top_k_predict", type=int, help="Top-k candidates per record during predict blocking (overrides blocking.yaml)")
    ap.add_argument("--target_total_pairs", type=int, help="Target total pairs for train blocking (overrides blocking.yaml)")
    ap.add_argument("--resolution", type=float, help="Leiden resolution parameter for graph clustering (overrides default)")
    ap.add_argument("--model", help="Language model to use for training: distilbert | roberta | bert (overrides training.yaml)")
    ap.add_argument("--model_path", help="S3 key suffix for model checkpoint (overrides config s3.model_path)")
    ap.add_argument("--match_mode", default="self", help="self | full | both")
    args = ap.parse_args()
    run(
        args.dataset,
        args.stage,
        size=args.size,
        offset=args.offset,
        batch_size=args.batch_size,
        top_k_train=args.top_k_train,
        top_k_predict=args.top_k_predict,
        target_total_pairs=args.target_total_pairs,
        resolution=args.resolution,
        model=args.model,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
