"""
Sentence-BERT based blocking utilities.

The goal of this module is to encapsulate the logic that was originally
implemented in the notebook for:
    - Encoding entity records with Sentence-BERT.
    - Performing dense top‑K similarity search (self-join style).
    - Building labeled Ditto-style datasets (train/valid/test) from a
      source table with a text column and a canonical_int (cluster id).

The functions are dataset-agnostic: you pass paths and column names,
and receive either in-memory pairs or written Ditto text files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import os
import pandas as pd
import torch

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


@dataclass
class SbertBlockingConfig:
    """Configuration for Sentence-BERT blocking on a single table."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 500
    target_total_pairs: Optional[int] = None  # e.g. 200_000
    batch_size_encode: int = 512
    block_rows: int = 800  # rows per block when running dense top‑K
    use_gpu: bool = True
    seed: int = 42


def _get_device(prefer_gpu: bool = True) -> str:
    if prefer_gpu and torch.cuda.is_available():  # pragma: no cover - depends on env
        return "cuda"
    return "cpu"


def encode_texts(
    texts: Sequence[str],
    model_name: str,
    batch_size: int = 512,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Encode a list of strings into a float32 numpy array using Sentence-BERT.

    Parameters
    ----------
    texts : list of str
        Input records to encode.
    model_name : str
        Hugging Face model id for Sentence-BERT.
    batch_size : int
        Batch size for encoding.
    device : str or None
        'cuda', 'cpu', or None to auto-detect.
    """
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Install it with `pip install sentence-transformers`."
        )

    if device is None:
        device = _get_device()

    model = SentenceTransformer(model_name, device=device)
    # SentenceTransformer already handles batching internally, but we keep
    # the explicit batch_size argument for clarity and future tuning.
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype("float32")

    # Free model and GPU memory eagerly if needed.
    if device == "cuda":  # pragma: no cover - env-dependent
        del model
        torch.cuda.empty_cache()

    return embeddings


def dense_topk_self_join(
    embeddings: np.ndarray,
    top_k: int,
    block_rows: int = 800,
    use_gpu: bool = True,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    """
    Perform a dense top‑K similarity search over a single set of embeddings.

    For each anchor index i, find the top_k most similar j (including i),
    using cosine similarity computed via batched matrix multiplication.

    Returns
    -------
    pairs : list of (i, j)
        Index pairs in zero-based indexing.
    """
    rng = np.random.default_rng(seed)
    n, d = embeddings.shape
    if n == 0:
        return []

    # L2-normalise for cosine similarity.
    V = embeddings.astype("float32")
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
    V = V / norms

    device = _get_device(prefer_gpu=use_gpu)
    use_gpu = use_gpu and device == "cuda"

    pairs: List[Tuple[int, int]] = []

    if use_gpu:  # pragma: no cover - depends on GPU availability
        V_torch = torch.from_numpy(V).to(device)

        for start in range(0, n, block_rows):
            end = min(start + block_rows, n)
            block = V_torch[start:end]  # (B, d)
            # Cosine similarity as dot product because vectors are normalised.
            sims = torch.matmul(block, V_torch.T)  # (B, n)
            # Optional: set self-similarity to very small to avoid trivial self-pair.
            idx = torch.arange(start, end, device=device)
            sims[torch.arange(end - start, device=device), idx] = -1e9

            topk_vals, topk_idx = torch.topk(sims, k=top_k, dim=1)
            # Build (i, j) pairs.
            i_indices = torch.arange(start, end, device=device).unsqueeze(1).repeat(1, top_k)
            block_pairs = torch.stack([i_indices.reshape(-1), topk_idx.reshape(-1)], dim=1)
            pairs.extend([(int(i), int(j)) for i, j in block_pairs.cpu().tolist()])

        # Free.
        del V_torch
        torch.cuda.empty_cache()
    else:
        # CPU fallback: use numpy dot-products in blocks.
        for start in range(0, n, block_rows):
            end = min(start + block_rows, n)
            block = V[start:end]  # (B, d)
            sims = np.dot(block, V.T)  # (B, n)
            np.fill_diagonal(sims[:, start:end], -1e9)
            topk_idx = np.argpartition(-sims, top_k, axis=1)[:, :top_k]
            for i_local, js in enumerate(topk_idx):
                i = start + i_local
                for j in js:
                    pairs.append((i, int(j)))

    return pairs


def sample_anchor_indices_for_target_pairs(
    n_records: int,
    top_k: int,
    target_total_pairs: Optional[int],
    seed: int = 42,
) -> np.ndarray:
    """
    Decide which anchor rows to use in blocking so that the total number
    of candidate pairs roughly matches target_total_pairs.

    If target_total_pairs is None, all records are used.
    """
    if target_total_pairs is None or target_total_pairs <= 0:
        return np.arange(n_records, dtype=int)

    approx_n_anchors = max(1, target_total_pairs // max(1, top_k))
    approx_n_anchors = min(approx_n_anchors, n_records)
    rng = np.random.default_rng(seed)
    anchors = rng.choice(n_records, size=approx_n_anchors, replace=False)
    anchors.sort()
    return anchors


def label_pairs_from_clusters(
    pairs: Iterable[Tuple[int, int]],
    canonical_ids: Sequence[int],
) -> List[Tuple[int, int, int]]:
    """
    Turn index pairs (i, j) into labeled triples (i, j, label) where
    label = 1 if canonical_ids[i] == canonical_ids[j] else 0.
    """
    canon = np.asarray(canonical_ids)
    labeled: List[Tuple[int, int, int]] = []
    for i, j in pairs:
        if i < 0 or j < 0 or i >= len(canon) or j >= len(canon):
            continue
        lab = int(canon[i] == canon[j])
        labeled.append((int(i), int(j), lab))
    return labeled


def train_valid_test_split_indices(
    n_pairs: int,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple pair-level random split into train/valid/test.
    """
    assert 0 < train_ratio < 1 and 0 < valid_ratio < 1
    rng = np.random.default_rng(seed)
    idx = np.arange(n_pairs, dtype=int)
    rng.shuffle(idx)

    n_train = int(train_ratio * n_pairs)
    n_valid = int(valid_ratio * n_pairs)
    train_idx = idx[:n_train]
    valid_idx = idx[n_train : n_train + n_valid]
    test_idx = idx[n_train + n_valid :]
    return train_idx, valid_idx, test_idx


def write_ditto_files(
    labeled_pairs: Sequence[Tuple[int, int, int]],
    texts: Sequence[str],
    out_dir: str,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """
    Write Ditto-format train/valid/test text files from labeled index pairs.
    """
    os.makedirs(out_dir, exist_ok=True)

    def _write(path: str, sel_idx: np.ndarray) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for k in sel_idx:
                i, j, lab = labeled_pairs[int(k)]
                rec1 = texts[i]
                rec2 = texts[j]
                f.write(f"{rec1}\t{rec2}\t{lab}\n")

    _write(os.path.join(out_dir, "train.txt"), train_idx)
    _write(os.path.join(out_dir, "valid.txt"), valid_idx)
    _write(os.path.join(out_dir, "test.txt"), test_idx)


def build_blocking_datasets_from_csv(
    csv_path: str,
    text_col: str,
    canonical_col: str,
    out_dir: str,
    config: Optional[SbertBlockingConfig] = None,
) -> Tuple[str, str, str]:
    """
    High-level helper to:
        - Read a CSV with a text column and canonical_int column.
        - Encode texts with Sentence-BERT.
        - Run dense top‑K blocking (optionally on a sampled subset of anchors).
        - Label pairs as matches / non-matches.
        - Randomly split into train/valid/test.
        - Write Ditto-format text files into out_dir.

    Returns
    -------
    train_path, valid_path, test_path : str
        Paths to the three generated Ditto files.
    """
    if config is None:
        config = SbertBlockingConfig()

    df = pd.read_csv(csv_path)
    assert text_col in df.columns, f"text_col '{text_col}' not in {csv_path}"
    assert canonical_col in df.columns, f"canonical_col '{canonical_col}' not in {csv_path}"

    valid_rows = df[df[text_col].notna()].reset_index(drop=True)
    texts = valid_rows[text_col].astype(str).tolist()
    canonical_ids = valid_rows[canonical_col].tolist()

    print(f"[sbert_blocking] Loaded {len(texts):,} records from {csv_path}")

    # Encode
    emb = encode_texts(
        texts,
        model_name=config.model_name,
        batch_size=config.batch_size_encode,
        device=_get_device(prefer_gpu=config.use_gpu),
    )

    # Optionally sample anchors to hit target_total_pairs.
    anchors = sample_anchor_indices_for_target_pairs(
        n_records=len(texts),
        top_k=config.top_k,
        target_total_pairs=config.target_total_pairs,
        seed=config.seed,
    )
    print(f"[sbert_blocking] Using {len(anchors):,} anchor records out of {len(texts):,}")

    # Run dense top‑K only for the selected anchors.
    all_pairs = dense_topk_self_join(
        emb,
        top_k=config.top_k,
        block_rows=config.block_rows,
        use_gpu=config.use_gpu,
        seed=config.seed,
    )
    # Filter to anchor-based pairs.
    anchor_set = set(int(a) for a in anchors)
    filtered_pairs = [(i, j) for (i, j) in all_pairs if i in anchor_set]

    print(f"[sbert_blocking] Generated {len(filtered_pairs):,} raw candidate pairs")

    labeled_pairs = label_pairs_from_clusters(filtered_pairs, canonical_ids)
    print(f"[sbert_blocking] Labeled {len(labeled_pairs):,} pairs (match / non-match)")

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

    print(f"[sbert_blocking] Wrote train/valid/test to {out_dir}")
    return train_path, valid_path, test_path

