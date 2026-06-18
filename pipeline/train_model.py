from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3

from .config import load_dataset_config, load_training_config, to_abs
from .modeling import TrainHyperParams, train_save_and_eval


def train(dataset: str, model: str | None = None, model_path: str | None = None) -> str:
    dcfg = load_dataset_config(dataset)
    tcfg = load_training_config()

    train_path = to_abs(dcfg["output"]["train_txt"])
    valid_path = to_abs(dcfg["output"]["valid_txt"])
    test_path = to_abs(dcfg["output"]["test_txt"])

    hp = TrainHyperParams(
        lm=model if model is not None else tcfg.get("lm", "distilbert"),
        max_len=int(tcfg.get("max_len", 256)),
        batch_size_train=int(tcfg.get("batch_size_train", 32)),
        batch_size_eval=int(tcfg.get("batch_size_eval", 128)),
        n_epochs=int(tcfg.get("n_epochs", 10)),
        lr=float(tcfg.get("lr", 2e-5)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
        seed=int(tcfg.get("seed", 42)),
        device=tcfg.get("device", None),
    )

    out_dir = Path(to_abs(f"models/{dataset}"))
    summary = train_save_and_eval(
        train_path,
        valid_path,
        test_path,
        str(out_dir),
        hp,
        threshold=float(tcfg.get("threshold", 0.5)),
    )

    ckpt = Path(summary["checkpoint"])
    bucket = os.environ.get("S3_BUCKET")
    if bucket:
        prefix = dcfg.get("s3", {}).get("prefix", dataset)
        resolved_model_path = model_path or dcfg.get("s3", {}).get("model_path", f"models/{dataset}/best_model.pt")
        s3_key = f"{prefix}/{resolved_model_path}"
        boto3.client("s3").upload_file(str(ckpt), bucket, s3_key)
        print(f"[train] Uploaded model to s3://{bucket}/{s3_key}")

    return str(ckpt)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Ditto model from config-defined train/valid/test files.")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    train(args.dataset)


if __name__ == "__main__":
    main()
