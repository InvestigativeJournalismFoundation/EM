"""
FAISS-based approximate nearest neighbours (ANN) blocking utilities.

This module mirrors the Sentence-BERT blocking API but uses a FAISS
index to scale to larger datasets. It expects pre-computed embeddings
for a single table and produces candidate index pairs and Ditto-style
train/valid/test files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import os
import pandas as pd
import torch

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    faiss = None  # type: ignore

from .sbert_blocking import (
    SbertBlockingConfig,
    encode_texts,
    label_pairs_from_clusters,
    train_valid_test_split_indices,
    write_ditto_files,
)


@dataclass
class AnnBlockingConfig:
    """Configuration for ANN (FAISS) blocking on a single table."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 200
    target_total_pairs: Optional[int] = 200_000
    batch_size_encode: int = 512
    nlist: int = 4096
    nprobe: int = 16
    seed: int = 42


def build_faiss_index(
    embeddings: np.ndarray,
    nlist: int = 4096,
    nprobe: int = 16,
    seed: int = 42,
) -> "faiss.IndexIVFFlat":
    """
    Build a FAISS IndexIVFFlat index (inner product on L2-normalised vectors).
    """
    if faiss is None:
        raise ImportError(
            "faiss is not installed. Install faiss-cpu or faiss-gpu depending "
            "on your environment."
        )

    d = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = nprobe

    rng = np.random.default_rng(seed)
    # Train on a subset for speed if the dataset is very large.
    n_train = min(embeddings.shape[0], 200_000)
    train_idx = rng.choice(embeddings.shape[0], size=n_train, replace=False)
    index.train(embeddings[train_idx])

    index.add(embeddings)
    return index


def ann_topk_self_join(
    embeddings: np.ndarray,
    top_k: int,
    nlist: int = 4096,
    nprobe: int = 16,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    """
    Use FAISS to get approximate top‑K neighbours for each record.

    Returns
    -------
    pairs : list of (i, j)
        Index pairs in zero-based indexing.
    """
    # L2-normalise.
    V = embeddings.astype("float32")
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
    V = V / norms

    index = build_faiss_index(V, nlist=nlist, nprobe=nprobe, seed=seed)

    # Search all records.
    sims, idx = index.search(V, top_k)

    pairs: List[Tuple[int, int]] = []
    for i in range(idx.shape[0]):
        for j in idx[i]:
            if j < 0:
                continue
            if j == i:
                continue
            pairs.append((int(i), int(j)))
    return pairs


def build_ann_blocking_datasets_from_csv(
    csv_path: str,
    text_col: str,
    canonical_col: str,
    out_dir: str,
    config: Optional[AnnBlockingConfig] = None,
) -> Tuple[str, str, str]:
    """
    High-level helper to:
        - Read CSV with text and canonical_int columns.
        - Encode with Sentence-BERT.
        - Build a FAISS ANN index and perform top‑K search.
        - Label pairs and split into Ditto train/valid/test files.
    """
    if config is None:
        config = AnnBlockingConfig()

    df = pd.read_csv(csv_path)
    assert text_col in df.columns, f"text_col '{text_col}' not in {csv_path}"
    assert canonical_col in df.columns, f"canonical_col '{canonical_col}' not in {csv_path}"

    valid_rows = df[df[text_col].notna()].reset_index(drop=True)
    texts = valid_rows[text_col].astype(str).tolist()
    canonical_ids = valid_rows[canonical_col].tolist()

    print(f"[ann_blocking] Loaded {len(texts):,} records from {csv_path}")

    emb = encode_texts(
        texts,
        model_name=config.model_name,
        batch_size=config.batch_size_encode,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    all_pairs = ann_topk_self_join(
        emb,
        top_k=config.top_k,
        nlist=config.nlist,
        nprobe=config.nprobe,
        seed=config.seed,
    )
    print(f"[ann_blocking] Generated {len(all_pairs):,} ANN candidate pairs")

    labeled_pairs = label_pairs_from_clusters(all_pairs, canonical_ids)
    print(f"[ann_blocking] Labeled {len(labeled_pairs):,} pairs")

    train_idx, valid_idx, test_idx = train_valid_test_split_indices(
        n_pairs=len(labeled_pairs),
        train_ratio=0.8,
        valid_ratio=0.1,
        seed=config.seed,
    )

    os.makedirs(out_dir, exist_ok=True)
    write_ditto_files(labeled_pairs, texts, out_dir, train_idx, valid_idx, test_idx)

    train_path = os.path.join(out_dir, "train.txt")
    valid_path = os.path.join(out_dir, "valid.txt")
    test_path = os.path.join(out_dir, "test.txt")
    print(f"[ann_blocking] Wrote train/valid/test to {out_dir}")
    return train_path, valid_path, test_path

