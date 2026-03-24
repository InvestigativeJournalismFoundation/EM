from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DITTO_CORE = REPO_ROOT / 'FAIR-DA4ER' / 'ditto'
if str(DITTO_CORE) not in sys.path:
    sys.path.insert(0, str(DITTO_CORE))

import pandas as pd
import numpy as np

from .config import load_dataset_config, load_blocking_config, to_abs
from .record_format import build_record_text
from er_pipeline.sbert_blocking import encode_texts


def build_predict_pairs(dataset: str) -> str:
    dcfg = load_dataset_config(dataset)
    bcfg = load_blocking_config()

    gold_csv = Path(to_abs(dcfg["paths"]["gold_csv"]))
    predict_csv = Path(to_abs(dcfg["paths"]["predict_csv"]))
    out_txt = Path(to_abs(dcfg["output"]["predict_txt"]))
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    text_fields = dcfg["schema"].get("text_fields", [])
    k = int(bcfg.get("top_k_predict", 1000))
    scfg = bcfg.get("sbert", {})
    model_name = scfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    batch_size = int(scfg.get("batch_size_encode", 512))
    use_gpu = bool(scfg.get("use_gpu", True))

    gdf = pd.read_csv(gold_csv)
    pdf = pd.read_csv(predict_csv)

    if "record_text" not in gdf.columns:
        gdf["record_text"] = gdf.apply(lambda r: build_record_text(r, text_fields), axis=1)
    pdf["record_text"] = pdf.apply(lambda r: build_record_text(r, text_fields), axis=1)

    gold_texts = gdf["record_text"].astype(str).tolist()
    pred_texts = pdf["record_text"].astype(str).tolist()

    ge = encode_texts(gold_texts, model_name, batch_size=batch_size, device=("cuda" if use_gpu else "cpu"))
    pe = encode_texts(pred_texts, model_name, batch_size=batch_size, device=("cuda" if use_gpu else "cpu"))

    ge = ge / (np.linalg.norm(ge, axis=1, keepdims=True) + 1e-8)
    pe = pe / (np.linalg.norm(pe, axis=1, keepdims=True) + 1e-8)

    with out_txt.open("w", encoding="utf-8") as f:
        for i in range(pe.shape[0]):
            sims = pe[i] @ ge.T
            idx = np.argpartition(-sims, min(k, len(sims)-1))[:k]
            idx = idx[np.argsort(-sims[idx])]
            for j in idx:
                f.write(f"{pred_texts[i]}\t{gold_texts[int(j)]}\t0\n")

    print(f"[build_predict_pairs] Wrote {out_txt}")
    return str(out_txt)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create <dataset>_predict.txt via top-k matching against gold.")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    build_predict_pairs(args.dataset)


if __name__ == "__main__":
    main()
