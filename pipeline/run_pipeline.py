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
from .leiden_ensemble_test import run_leiden_ensemble_test, run_leiden_all_models
from .graph_cluster_eval import run_graph_cluster_eval
from .entity_holdout_eval import run_entity_holdout_eval
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
    "leiden_test",
    "leiden_test_each",
    "graph_cluster",
    "entity_holdout",
]

TRAIN = [
    "fetch_data",
    "build_gold",
    "build_splits",
    "build_predict",
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
    "fetch_data",
    "build_gold",
    "build_splits",
    "build_predict",
    "train",
    "test",
    "predict",
    "build_graph",
]


def run(
    dataset: str,
    stage: str,
    lm: str = None,
    n_epochs: int = None,
    model_tag: str = None,
    size: int = None,
    batch_size: int = None,
    retrain: bool = False,
) -> None:
    load_dotenv()
    if stage == "all":
        seq = STAGES
    elif stage == "train_run":
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
            train(dataset, lm=lm, n_epochs=n_epochs, model_tag=model_tag)
        elif s == "test":
            run_test(dataset, lm=lm, model_tag=model_tag)
        elif s == "predict":
            run_predict(dataset, lm=lm, model_tag=model_tag)
        elif s == "fetch_data":
            fetch_data(dataset, size=size, batch_size=batch_size)
        elif s == "fetch_model":
            fetch_model(dataset)
        elif s == "build_graph":
            build_graph(dataset, model_tag=model_tag)
        elif s == "leiden_test":
            run_leiden_ensemble_test(dataset)
        elif s == "leiden_test_each":
            run_leiden_all_models(dataset)
        elif s == "graph_cluster":
            run_graph_cluster_eval(
                dataset,
                lm=lm or "bert",
                n_epochs=n_epochs or 10,
                retrain=retrain,
            )
        elif s == "entity_holdout":
            run_entity_holdout_eval(dataset, lm=lm or "bert")
        else:
            raise ValueError(f"Unknown stage: {s}")
        elapsed = time.time() - stage_start
        timings.append((s, elapsed))
        print(f"[TIMING] {s}: {elapsed:.2f}s")

    if len(timings) > 1:
        total = time.time() - pipeline_start
        print("\n=== Timing Summary ===")
        for name, dur in timings:
            print(f"  {name:<20} {dur:>8.1f}s")
        print(f"  {'TOTAL':<20} {total:>8.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run end-to-end Ditto pipeline by stage.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--stage", default="all", help="all | full | train_run | inference | fetch_data | fetch_model | build_gold | build_splits | build_predict | train | test | predict | build_graph | leiden_test | leiden_test_each")
    ap.add_argument("--lm", default=None, help="Language model override: distilbert | bert | roberta")
    ap.add_argument("--n_epochs", type=int, default=None, help="Number of training epochs override")
    ap.add_argument("--model_tag", default=None, help="Tag used for output folder isolation (default: same as --lm)")
    ap.add_argument("--size", type=int, help="Number of rows to fetch")
    ap.add_argument("--batch_size", type=int, help="Batch size for training and inference")
    ap.add_argument("--retrain", action="store_true", help="Force retrain from scratch (used by graph_cluster stage)")
    args = ap.parse_args()
    run(args.dataset, args.stage, lm=args.lm, n_epochs=args.n_epochs, model_tag=args.model_tag, size=args.size, batch_size=args.batch_size, retrain=args.retrain)


if __name__ == "__main__":
    main()
