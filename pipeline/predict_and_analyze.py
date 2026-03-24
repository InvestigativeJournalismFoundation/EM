from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .config import load_dataset_config, load_training_config, to_abs
from .modeling import load_model_for_inference, predict_from_txt


def run_predict(dataset: str) -> str:
    dcfg = load_dataset_config(dataset)
    tcfg = load_training_config()

    predict_txt = to_abs(dcfg["output"]["predict_txt"])
    out_dir = Path(to_abs(dcfg["output"]["predict_result_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(to_abs(f"models/{dataset}/best_model.pt"))
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Run training first.")

    model = load_model_for_inference(str(ckpt), lm=tcfg.get("lm", "distilbert"), device=tcfg.get("device", None))
    rows = predict_from_txt(
        model,
        predict_txt,
        lm=tcfg.get("lm", "distilbert"),
        max_len=int(tcfg.get("max_len", 256)),
        batch_size=int(tcfg.get("batch_size_eval", 128)),
        threshold=float(tcfg.get("threshold", 0.5)),
    )
    df = pd.DataFrame(rows)

    pred_csv = out_dir / f"{dataset}_predict.csv"
    df.to_csv(pred_csv, index=False)

    n = len(df)
    n_match = int((df["pred_label"] == 1).sum()) if n else 0
    n_non = int((df["pred_label"] == 0).sum()) if n else 0

    analysis = out_dir / f"{dataset}_predict_analysis.txt"
    with analysis.open("w", encoding="utf-8") as f:
        f.write(f"dataset: {dataset}\n")
        f.write(f"rows: {n}\n")
        f.write(f"matches: {n_match} ({(n_match/n*100 if n else 0):.2f}%)\n")
        f.write(f"non_matches: {n_non} ({(n_non/n*100 if n else 0):.2f}%)\n")
        if n:
            f.write(f"prob_mean: {df['prob_match'].mean():.6f}\n")
            f.write(f"prob_std: {df['prob_match'].std():.6f}\n")
            f.write(f"prob_q25: {df['prob_match'].quantile(0.25):.6f}\n")
            f.write(f"prob_q50: {df['prob_match'].quantile(0.50):.6f}\n")
            f.write(f"prob_q75: {df['prob_match'].quantile(0.75):.6f}\n")

    print(f"[predict] Wrote {pred_csv}")
    print(f"[predict] Wrote {analysis}")
    return str(pred_csv)


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict match/non-match on predict.txt and write CSV + analysis.")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    run_predict(args.dataset)


if __name__ == "__main__":
    main()
