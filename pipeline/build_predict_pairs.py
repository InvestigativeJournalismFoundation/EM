from __future__ import annotations

import argparse
import os
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
from er_pipeline.tfidf_blocking import TfidfBlockingConfig, write_predict_pairs_tfidf


def build_predict_pairs(dataset: str, top_k_predict: int | None = None, size: int | None = None, offset: int | None = None) -> str:
    dcfg = load_dataset_config(dataset)
    bcfg = load_blocking_config()

    raw_csv = Path(to_abs(dcfg["paths"]["raw_csv"]))
    out_txt = Path(to_abs(dcfg["output"]["predict_txt"]))
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    text_fields = dcfg["schema"].get("text_fields", [])
    k = top_k_predict if top_k_predict is not None else int(bcfg.get("top_k_predict", 20))
    strategy = bcfg.get("strategy", "sbert").lower()

    full_df = pd.read_csv(raw_csv, low_memory=False)

    name_col = text_fields[0] if text_fields else None
    if name_col:
        before = len(full_df)
        full_df = full_df[full_df[name_col].notna() & (full_df[name_col].astype(str).str.strip() != "")]
        print(f"[build_predict_pairs] Dropped {before - len(full_df)} rows with missing '{name_col}'")

    full_df["record_text"] = full_df.apply(lambda r: build_record_text(r, text_fields), axis=1)

    # Corpus (right side): always the full raw CSV
    gold_texts = full_df["record_text"].astype(str).drop_duplicates().tolist()

    # Anchors (left side): a size-limited slice if offset/size given, otherwise the full corpus
    if offset is not None or size is not None:
        start = offset or 0
        end = (start + size) if size is not None else len(full_df)
        pred_df = full_df.iloc[start:end]
        print(f"[build_predict_pairs] Predict subset: rows {start}–{end} ({len(pred_df)} rows)")
    else:
        pred_df = full_df

    pred_texts = pred_df["record_text"].astype(str).drop_duplicates().tolist()

    if strategy == "tfidf":
        tfcfg = bcfg.get("tfidf", {})
        cfg = TfidfBlockingConfig(
            max_features=int(tfcfg.get("max_features", 50_000)),
            min_df=int(tfcfg.get("min_df", 2)),
            max_df=float(tfcfg.get("max_df", 0.95)),
            sublinear_tf=bool(tfcfg.get("sublinear_tf", True)),
            top_k=k,
            anchor_batch_size=int(tfcfg.get("anchor_batch_size", 32)),
            block_rows=int(tfcfg.get("block_rows", 32)),
            seed=int(bcfg.get("seed", 42)),
        )
        write_predict_pairs_tfidf(gold_texts, pred_texts, str(out_txt), k, cfg, skip_self=True)
        print(f"[build_predict_pairs] Wrote {out_txt}")
        return str(out_txt)

    scfg = bcfg.get("sbert", {})
    model_name = scfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    batch_size = int(scfg.get("batch_size_encode", 512))
    use_gpu = bool(scfg.get("use_gpu", True))

    device = "cuda" if use_gpu else "cpu"

    # Gold embeddings cache: encoding 1.4M rows takes ~3 min; cache on a persistent volume
    # so subsequent runs (new predict.csv, same gold table) skip re-encoding entirely.
    gold_cache_path = os.environ.get("GOLD_EMBED_CACHE", "")
    if gold_cache_path and os.path.exists(gold_cache_path):
        print(f"[build_predict_pairs] Loading cached gold embeddings from {gold_cache_path}")
        ge = np.load(gold_cache_path)
    else:
        print(f"[build_predict_pairs] Encoding {len(gold_texts)} gold records with SBERT ...")
        ge = encode_texts(gold_texts, model_name, batch_size=batch_size, device=device)
        ge = ge / (np.linalg.norm(ge, axis=1, keepdims=True) + 1e-8)
        if gold_cache_path:
            os.makedirs(os.path.dirname(gold_cache_path), exist_ok=True)
            np.save(gold_cache_path, ge)
            print(f"[build_predict_pairs] Saved gold embeddings to {gold_cache_path}")

    pe = encode_texts(pred_texts, model_name, batch_size=batch_size, device=device)
    pe = pe / (np.linalg.norm(pe, axis=1, keepdims=True) + 1e-8)

    with out_txt.open("w", encoding="utf-8") as f:
        for i in range(pe.shape[0]):
            sims = pe[i] @ ge.T
            idx = np.argpartition(-sims, min(k, len(sims)-1))[:k]
            idx = idx[np.argsort(-sims[idx])]
            for j in idx:
                if pred_texts[i] == gold_texts[int(j)]:
                    continue
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
