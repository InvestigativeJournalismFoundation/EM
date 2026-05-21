from __future__ import annotations

import os
from pathlib import Path

import boto3
import pandas as pd
from dotenv import load_dotenv

from config import load_ground_truth_config, to_abs


def fetch_pairs(dataset: str) -> str:
    load_dotenv()

    cfg = load_ground_truth_config(dataset)
    s3cfg = cfg["s3"]
    prefix = s3cfg["pairs_path"].rstrip("/")
    filename = s3cfg.get("pairs_filename", f"{dataset}_predict.csv")

    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        raise RuntimeError("S3_BUCKET environment variable is not set")

    s3 = boto3.client("s3")

    # List run-timestamp subfolders under the prefix
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix + "/", Delimiter="/")

    run_folders = []
    for page in pages:
        for cp in page.get("CommonPrefixes", []):
            run_folders.append(cp["Prefix"])

    if not run_folders:
        raise FileNotFoundError(
            f"No run folders found under s3://{bucket}/{prefix}/"
        )

    # Timestamp folders sort lexicographically (YYYYMMDDTHHMMSSZ)
    latest_folder = sorted(run_folders)[-1]
    s3_key = f"{latest_folder}{filename}"

    print(f"[fetch_pairs] Latest run: s3://{bucket}/{s3_key}")

    out_path = Path(to_abs(cfg["paths"]["pairs_csv"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    s3.download_file(bucket, s3_key, str(out_path))

    df = pd.read_csv(out_path)
    n_match = int((df["pred"] == 1).sum()) if "pred" in df.columns else "?"
    print(f"[fetch_pairs] {len(df)} pairs ({n_match} predicted matches) → {out_path}")

    return str(out_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    fetch_pairs(args.dataset)
