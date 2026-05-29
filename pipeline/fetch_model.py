from __future__ import annotations

import os
from pathlib import Path

import boto3

from .config import load_dataset_config, to_abs


def fetch_model(dataset: str) -> None:
    cfg = load_dataset_config(dataset)
    out_path = Path(to_abs(cfg["model"]["filename"]))

    if out_path.exists():
        print(f"[fetch_model] Model already exists at {out_path}, skipping download.")
        return

    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        raise RuntimeError("S3_BUCKET environment variable is not set")

    prefix = cfg.get("s3", {}).get("prefix", dataset)
    s3_key = f"{prefix}/models/{dataset}/best_model.pt"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[fetch_model] Downloading s3://{bucket}/{s3_key} → {out_path}")
    boto3.client("s3").download_file(bucket, s3_key, str(out_path))
    print(f"[fetch_model] Done.")
