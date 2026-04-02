"""
TF-IDF based blocking for a single entity table.

Uses L2-normalized sparse TF-IDF vectors so cosine similarity equals the
dot product. Mirrors the Sentence-BERT blocking flow (anchor sampling vs
full self-join) for train/valid/test generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import os
import pandas as pd

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
except Exception:  # pragma: no cover
    TfidfVectorizer = None  # type: ignore
    normalize = None  # type: ignore

from .sbert_blocking import (
    label_pairs_from_clusters,
    sample_anchor_indices_for_target_pairs,
    train_valid_test_split_indices,
    write_ditto_files,
)


@dataclass
class TfidfBlockingConfig:
    """Configuration for TF-IDF blocking on a single table."""

    max_features: int = 50_000
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True
    top_k: int = 500
    target_total_pairs: Optional[int] = None
    seed: int = 42
    anchor_batch_size: int = 32
    block_rows: int = 32


def _fit_tfidf_matrix(
    texts: Sequence[str],
    config: TfidfBlockingConfig,
):
    if TfidfVectorizer is None:
        raise ImportError("scikit-learn is required for TF-IDF blocking. pip install scikit-learn")

    vectorizer = TfidfVectorizer(
        max_features=config.max_features,
        min_df=config.min_df,
        max_df=config.max_df,
        dtype=np.float32,
        sublinear_tf=config.sublinear_tf,
    )
    X = vectorizer.fit_transform(list(texts))
    X = normalize(X, norm="l2", axis=1)
    return X.tocsr(), vectorizer


def tfidf_topk_for_anchors(
    X,
    anchor_indices: np.ndarray,
    top_k: int,
    anchor_batch_size: int = 32,
) -> List[Tuple[int, int]]:
    """Top-k neighbours per anchor row (cosine = dot on L2-normalized TF-IDF)."""
    n = X.shape[0]
    if n == 0 or len(anchor_indices) == 0:
        return []

    k_eff = min(top_k, max(0, n - 1))
    if k_eff <= 0:
        return []

    pairs: List[Tuple[int, int]] = []
    anchors = np.asarray(anchor_indices, dtype=np.int64)

    for s in range(0, len(anchors), anchor_batch_size):
        batch_idx = anchors[s : s + anchor_batch_size]
        block = X[batch_idx]
        sim = (block @ X.T).toarray()
        for r, global_i in enumerate(batch_idx.tolist()):
            row = sim[r].copy()
            row[int(global_i)] = -np.inf
            part = np.argpartition(-row, k_eff - 1)[:k_eff]
            order = part[np.argsort(-row[part])][:k_eff]
            for j in order:
                pairs.append((int(global_i), int(j)))

    return pairs


def tfidf_topk_self_join(
    X,
    top_k: int,
    block_rows: int = 32,
) -> List[Tuple[int, int]]:
    """Full-table blocking: each row as anchor, top-k over all rows."""
    n = X.shape[0]
    if n == 0:
        return []

    k_eff = min(top_k, max(0, n - 1))
    if k_eff <= 0:
        return []

    pairs: List[Tuple[int, int]] = []
    for start in range(0, n, block_rows):
        end = min(start + block_rows, n)
        block = X[start:end]
        sim = (block @ X.T).toarray()
        for local_i, global_i in enumerate(range(start, end)):
            row = sim[local_i].copy()
            row[global_i] = -np.inf
            part = np.argpartition(-row, k_eff - 1)[:k_eff]
            order = part[np.argsort(-row[part])][:k_eff]
            for j in order:
                pairs.append((int(global_i), int(j)))

    return pairs


def build_tfidf_blocking_datasets_from_csv(
    csv_path: str,
    text_col: str,
    canonical_col: str,
    out_dir: str,
    config: Optional[TfidfBlockingConfig] = None,
) -> Tuple[str, str, str]:
    """
    Read gold CSV, TF-IDF encode, top-k blocking, label, split, write Ditto files.
    """
    if config is None:
        config = TfidfBlockingConfig()

    df = pd.read_csv(csv_path, low_memory=False)
    assert text_col in df.columns, f"text_col '{text_col}' not in {csv_path}"
    assert canonical_col in df.columns, f"canonical_col '{canonical_col}' not in {csv_path}"

    valid_rows = df[df[text_col].notna()].reset_index(drop=True)
    texts = valid_rows[text_col].astype(str).tolist()
    canonical_ids = valid_rows[canonical_col].tolist()

    print(f"[tfidf_blocking] Loaded {len(texts):,} records from {csv_path}")
    print("[tfidf_blocking] Fitting TF-IDF ...")
    X, _vec = _fit_tfidf_matrix(texts, config)

    anchors = sample_anchor_indices_for_target_pairs(
        n_records=len(texts),
        top_k=config.top_k,
        target_total_pairs=config.target_total_pairs,
        seed=config.seed,
    )
    print(f"[tfidf_blocking] Using {len(anchors):,} anchor records out of {len(texts):,}")

    n_records = len(texts)
    if len(anchors) < n_records:
        filtered_pairs = tfidf_topk_for_anchors(
            X,
            anchors,
            top_k=config.top_k,
            anchor_batch_size=config.anchor_batch_size,
        )
    else:
        filtered_pairs = tfidf_topk_self_join(
            X,
            top_k=config.top_k,
            block_rows=config.block_rows,
        )

    print(f"[tfidf_blocking] Generated {len(filtered_pairs):,} raw candidate pairs")

    labeled_pairs = label_pairs_from_clusters(filtered_pairs, canonical_ids)
    print(f"[tfidf_blocking] Labeled {len(labeled_pairs):,} pairs (match / non-match)")

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

    print(f"[tfidf_blocking] Wrote train/valid/test to {out_dir}")
    return train_path, valid_path, test_path


def write_predict_pairs_tfidf(
    gold_texts: Sequence[str],
    pred_texts: Sequence[str],
    out_path: str,
    top_k: int,
    config: Optional[TfidfBlockingConfig] = None,
) -> None:
    """
    Fit TF-IDF on gold + predict (shared vocabulary), then for each predict row
    write top-k gold neighbours (label 0) in Ditto format.
    """
    if config is None:
        config = TfidfBlockingConfig()

    gold_list = list(gold_texts)
    pred_list = list(pred_texts)
    n_g = len(gold_list)
    if n_g == 0:
        raise ValueError("gold_texts is empty")

    print(f"[tfidf_blocking] predict: fitting TF-IDF on {n_g:,} gold + {len(pred_list):,} predict rows")
    X, _ = _fit_tfidf_matrix(gold_list + pred_list, config)
    X_g = X[:n_g]
    X_p = X[n_g:]

    k_eff = min(top_k, n_g)
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    batch = max(1, config.anchor_batch_size)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in range(0, X_p.shape[0], batch):
            e = min(s + batch, X_p.shape[0])
            block = X_p[s:e]
            sim = (block @ X_g.T).toarray()
            for local_i, global_pi in enumerate(range(s, e)):
                row = sim[local_i]
                part = np.argpartition(-row, k_eff - 1)[:k_eff]
                order = part[np.argsort(-row[part])][:k_eff]
                for j in order:
                    f.write(f"{pred_list[global_pi]}\t{gold_list[int(j)]}\t0\n")

    print(f"[tfidf_blocking] Wrote predict pairs to {out_path}")
