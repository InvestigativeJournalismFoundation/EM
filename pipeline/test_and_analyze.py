from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .config import load_dataset_config, load_training_config, to_abs
from .modeling import load_model_for_inference, predict_from_txt


def run_test(dataset: str) -> str:
    dcfg = load_dataset_config(dataset)
    tcfg = load_training_config()

    test_txt = to_abs(dcfg["output"]["test_txt"])
    out_dir = Path(to_abs(dcfg["output"]["test_result_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(to_abs(f"models/{dataset}/best_model.pt"))
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Run training first.")

    model = load_model_for_inference(str(ckpt), lm=tcfg.get("lm", "distilbert"), device=tcfg.get("device", None))
    rows = predict_from_txt(
        model,
        test_txt,
        lm=tcfg.get("lm", "distilbert"),
        max_len=int(tcfg.get("max_len", 256)),
        batch_size=int(tcfg.get("batch_size_eval", 128)),
        threshold=float(tcfg.get("threshold", 0.5)),
    )
    df = pd.DataFrame(rows)

    test_csv = out_dir / f"{dataset}_test.csv"
    df.to_csv(test_csv, index=False)

    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
    y_true = df["true_label"].fillna(0).astype(int)
    y_pred = df["pred_label"].astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    a = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    analysis = out_dir / f"{dataset}_test_analysis.txt"
    with analysis.open("w", encoding="utf-8") as f:
        f.write(f"dataset: {dataset}\n")
        f.write(f"rows: {len(df)}\n")
        f.write(f"f1: {f1:.6f}\n")
        f.write(f"precision: {p:.6f}\n")
        f.write(f"recall: {r:.6f}\n")
        f.write(f"accuracy: {a:.6f}\n")
        f.write(f"tp: {tp}\nfp: {fp}\ntn: {tn}\nfn: {fn}\n")

    print(f"[test] Wrote {test_csv}")
    print(f"[test] Wrote {analysis}")
    return str(test_csv)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Ditto on test file and produce CSV + analysis report.")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    run_test(args.dataset)


if __name__ == "__main__":
    main()
