from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
import boto3
import pandas as pd

from .config import load_dataset_config, load_training_config, to_abs
from .modeling import load_model_for_inference, predict_from_txt
from .record_format import build_record_text


def run_predict(dataset: str) -> str:
    dcfg = load_dataset_config(dataset)
    tcfg = load_training_config()

    predict_txt = to_abs(dcfg["output"]["predict_txt"])
    out_dir = Path(to_abs(dcfg["output"]["predict_result_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(to_abs(dcfg["model"]["filename"]))
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Run training first.")

    # Build text → (rid, name) lookup from raw CSV
    schema = dcfg["schema"]
    name_col = schema["text_fields"][0]
    text_fields = schema.get("text_fields", [])
    raw_df = pd.read_csv(to_abs(dcfg["paths"]["raw_csv"]), low_memory=False)
    raw_df["_text"] = raw_df.apply(lambda r: build_record_text(r, text_fields), axis=1)
    text_to_name = dict(zip(raw_df["_text"], raw_df[name_col]))

    model = load_model_for_inference(str(ckpt), lm=tcfg.get("lm", "distilbert"), device=tcfg.get("device", None))
    rows = predict_from_txt(
        model,
        predict_txt,
        lm=tcfg.get("lm", "distilbert"),
        max_len=int(tcfg.get("max_len", 256)),
        batch_size=int(tcfg.get("batch_size_eval", 128)),
        threshold=float(tcfg.get("threshold", 0.5)),
    )
    df = pd.DataFrame(rows).drop(columns=["true_label"], errors="ignore")
    df = pd.DataFrame({
        "name1": df["record1"].map(text_to_name),
        "name2": df["record2"].map(text_to_name),
        "score": df["prob_match"],
        "pred":  df["pred_label"],
    })

    pred_csv = out_dir / f"{dataset}_predict.csv"
    df.to_csv(pred_csv, index=False)

    n = len(df)
    n_match = int((df["pred"] == 1).sum()) if n else 0
    n_non = int((df["pred"] == 0).sum()) if n else 0

    analysis = out_dir / f"{dataset}_predict_analysis.txt"
    with analysis.open("w", encoding="utf-8") as f:
        f.write(f"dataset: {dataset}\n")
        f.write(f"rows: {n}\n")
        f.write(f"matches: {n_match} ({(n_match/n*100 if n else 0):.2f}%)\n")
        f.write(f"non_matches: {n_non} ({(n_non/n*100 if n else 0):.2f}%)\n")
        if n:
            f.write(f"score_mean: {df['score'].mean():.6f}\n")
            f.write(f"score_std: {df['score'].std():.6f}\n")
            f.write(f"score_q25: {df['score'].quantile(0.25):.6f}\n")
            f.write(f"score_q50: {df['score'].quantile(0.50):.6f}\n")
            f.write(f"score_q75: {df['score'].quantile(0.75):.6f}\n")

    print(f"[predict] Wrote {pred_csv}")
    print(f"[predict] Wrote {analysis}")

    bucket = os.environ.get("S3_BUCKET")
    if bucket:
        s3 = boto3.client("s3")
        prefix = dcfg.get("s3", {}).get("prefix", dataset)
        run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for local_path, s3_key in [
            (pred_csv,  f"{prefix}/{run_ts}/{dataset}_predict.csv"),
            (analysis,  f"{prefix}/{run_ts}/{dataset}_predict_analysis.txt"),
        ]:
            s3.upload_file(str(local_path), bucket, s3_key)
            print(f"[predict] Uploaded s3://{bucket}/{s3_key}")

    return str(pred_csv)


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict match/non-match on predict.txt and write CSV + analysis.")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    run_predict(args.dataset)


if __name__ == "__main__":
    main()
