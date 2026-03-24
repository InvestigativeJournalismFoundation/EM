from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DITTO_CORE = REPO_ROOT / 'FAIR-DA4ER' / 'ditto'
if str(DITTO_CORE) not in sys.path:
    sys.path.insert(0, str(DITTO_CORE))

from typing import List, Tuple
import numpy as np
import pandas as pd

from .config import load_dataset_config, load_blocking_config, to_abs

# reuse existing modular code
from er_pipeline.sbert_blocking import (
    SbertBlockingConfig,
    build_blocking_datasets_from_csv,
    train_valid_test_split_indices,
    write_ditto_files,
    sample_anchor_indices_for_target_pairs,
)
from er_pipeline.ann_blocking import AnnBlockingConfig, build_ann_blocking_datasets_from_csv


def _rename_split_files(tmp_dir: Path, out_train: Path, out_valid: Path, out_test: Path) -> None:
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_valid.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)
    (tmp_dir / "train.txt").replace(out_train)
    (tmp_dir / "valid.txt").replace(out_valid)
    (tmp_dir / "test.txt").replace(out_test)


def _char_ngrams(s: str, n: int = 3) -> set[str]:
    s = (s or "").lower().strip()
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _ngram_pairs(texts: List[str], top_k: int, target_total_pairs: int | None, ngram_n: int, seed: int) -> List[Tuple[int, int]]:
    gram_sets = [_char_ngrams(t, ngram_n) for t in texts]
    inv = {}
    for i, grams in enumerate(gram_sets):
        for g in grams:
            inv.setdefault(g, []).append(i)

    anchors = sample_anchor_indices_for_target_pairs(len(texts), top_k, target_total_pairs, seed=seed)
    pairs = []
    for i in anchors:
        cand = {}
        for g in gram_sets[i]:
            for j in inv.get(g, []):
                if j == i:
                    continue
                cand[j] = cand.get(j, 0) + 1
        # top by shared gram count
        js = sorted(cand.items(), key=lambda x: x[1], reverse=True)[:top_k]
        pairs.extend((int(i), int(j)) for j, _ in js)
    return pairs


def build_splits(dataset: str) -> tuple[str, str, str]:
    dcfg = load_dataset_config(dataset)
    bcfg = load_blocking_config()

    gold_csv = to_abs(dcfg["paths"]["gold_csv"])
    out_train = Path(to_abs(dcfg["output"]["train_txt"]))
    out_valid = Path(to_abs(dcfg["output"]["valid_txt"]))
    out_test = Path(to_abs(dcfg["output"]["test_txt"]))

    strategy = bcfg.get("strategy", "sbert").lower()
    tmp_dir = out_train.parent / f"._tmp_{dataset}_splits"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if strategy == "sbert":
        scfg = bcfg.get("sbert", {})
        cfg = SbertBlockingConfig(
            model_name=scfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            top_k=int(bcfg.get("top_k_train", 10000)),
            target_total_pairs=bcfg.get("target_total_pairs", 200000),
            batch_size_encode=int(scfg.get("batch_size_encode", 512)),
            block_rows=int(scfg.get("block_rows", 800)),
            anchor_batch_size=int(scfg.get("anchor_batch_size", 64)),
            use_gpu=bool(scfg.get("use_gpu", True)),
            seed=int(bcfg.get("seed", 42)),
        )
        build_blocking_datasets_from_csv(gold_csv, "record_text", dcfg["schema"]["canonical_col"], str(tmp_dir), cfg)

    elif strategy == "ann":
        acfg = bcfg.get("ann", {})
        scfg = bcfg.get("sbert", {})
        cfg = AnnBlockingConfig(
            model_name=scfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            top_k=int(bcfg.get("top_k_train", 10000)),
            target_total_pairs=bcfg.get("target_total_pairs", 200000),
            batch_size_encode=int(scfg.get("batch_size_encode", 512)),
            nlist=int(acfg.get("nlist", 4096)),
            nprobe=int(acfg.get("nprobe", 16)),
            seed=int(bcfg.get("seed", 42)),
        )
        build_ann_blocking_datasets_from_csv(gold_csv, "record_text", dcfg["schema"]["canonical_col"], str(tmp_dir), cfg)

    elif strategy == "ngram":
        ncfg = bcfg.get("ngram", {})
        top_k = int(bcfg.get("top_k_train", 10000))
        target_total_pairs = bcfg.get("target_total_pairs", 200000)
        seed = int(bcfg.get("seed", 42))
        ngram_n = int(ncfg.get("n", 3))

        df = pd.read_csv(gold_csv)
        texts = df["record_text"].astype(str).tolist()
        canon = df[dcfg["schema"]["canonical_col"]].tolist()
        pairs = _ngram_pairs(texts, top_k, target_total_pairs, ngram_n, seed)
        labeled = [(i, j, int(canon[i] == canon[j])) for i, j in pairs]

        tr, va, te = train_valid_test_split_indices(
            len(labeled),
            train_ratio=float(dcfg["split"]["train_ratio"]),
            valid_ratio=float(dcfg["split"]["valid_ratio"]),
            seed=int(dcfg["split"]["seed"]),
        )
        write_ditto_files(labeled, texts, str(tmp_dir), tr, va, te)
    else:
        raise ValueError(f"Unsupported blocking strategy: {strategy}")

    _rename_split_files(tmp_dir, out_train, out_valid, out_test)
    print(f"[build_splits] train={out_train} valid={out_valid} test={out_test}")
    return str(out_train), str(out_valid), str(out_test)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build dataset-prefixed Ditto train/valid/test files.")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    build_splits(args.dataset)


if __name__ == "__main__":
    main()
