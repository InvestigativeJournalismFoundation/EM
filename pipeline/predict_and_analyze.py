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


def run_predict(dataset: str, lm: str = None, model_tag: str = None) -> str:
    dcfg = load_dataset_config(dataset)
    tcfg = load_training_config()

    effective_lm = lm or tcfg.get("lm", "distilbert")
    tag = model_tag or effective_lm

    predict_txt = to_abs(dcfg["output"]["predict_txt"])
    out_dir = Path(to_abs(f"predict_output/{dataset}_{tag}_predict_result"))
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_ckpt = dcfg.get("model", {}).get("filename")
    ckpt = Path(to_abs(cfg_ckpt)) if cfg_ckpt else Path(to_abs(f"models/{dataset}_{tag}/best_model.pt"))
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Run training first.")

    # Build per-field text → value lookups from all source CSVs so every record
    # text that can appear in the pairs (from raw, predict, or gold) is covered.
    schema = dcfg.get("schema", {})
    text_fields = schema.get("text_fields", [])
    name_col = text_fields[0] if text_fields else None
    extra_cols = text_fields[1:] if len(text_fields) > 1 else []

    field_lookups: dict[str, dict] = {f: {} for f in text_fields}
    seen_paths: set = set()
    for csv_key in ("raw_csv", "predict_csv", "gold_csv"):
        csv_path = to_abs(dcfg.get("paths", {}).get(csv_key, ""))
        if not csv_path or not Path(csv_path).exists():
            continue
        resolved = str(Path(csv_path).resolve())
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        df_src = pd.read_csv(csv_path, low_memory=False)
        df_src["_text"] = df_src.apply(lambda r: build_record_text(r, text_fields), axis=1)
        for field in text_fields:
            if field in df_src.columns:
                field_lookups[field].update(zip(df_src["_text"], df_src[field]))

    model = load_model_for_inference(str(ckpt), lm=effective_lm, device=tcfg.get("device", None))
    rows = predict_from_txt(
        model,
        predict_txt,
        lm=effective_lm,
        max_len=int(tcfg.get("max_len", 256)),
        batch_size=int(tcfg.get("batch_size_eval", 128)),
        threshold=float(tcfg.get("threshold", 0.5)),
    )
    df = pd.DataFrame(rows).drop(columns=["true_label"], errors="ignore")
    out: dict = {}
    if name_col:
        out["name1"] = df["record1"].map(field_lookups[name_col])
        out["name2"] = df["record2"].map(field_lookups[name_col])
    for field in extra_cols:
        out[f"{field}1"] = df["record1"].map(field_lookups[field])
        out[f"{field}2"] = df["record2"].map(field_lookups[field])
    out["score"] = df["prob_match"]
    out["pred"] = df["pred_label"]
    df = pd.DataFrame(out)

    pred_csv = out_dir / f"{dataset}_{tag}_predict.csv"
    df.to_csv(pred_csv, index=False)

    n = len(df)
    n_match = int((df["pred"] == 1).sum()) if n else 0
    n_non = int((df["pred"] == 0).sum()) if n else 0

    analysis = out_dir / f"{dataset}_{tag}_predict_analysis.txt"
    with analysis.open("w", encoding="utf-8") as f:
        f.write(f"dataset: {dataset}\n")
        f.write(f"model: {tag}\n")
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
    ap.add_argument("--lm", default=None)
    ap.add_argument("--model_tag", default=None)
    args = ap.parse_args()
    run_predict(args.dataset, lm=args.lm, model_tag=args.model_tag)


if __name__ == "__main__":
    main()
