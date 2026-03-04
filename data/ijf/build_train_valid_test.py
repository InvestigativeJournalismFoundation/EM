"""
Build train.txt, valid.txt, test.txt for the IJF pro_supplier_standardization dataset.

- Excludes id, created_at from record representation (not used in train/test).
- Uses canonical: same canonical = matching pair (label 1).
- Blocking: 3-gram and/or sentence-BERT. Mode: union (default), intersection (smaller, higher match density), or embedding_only (smallest, highest density).
- Optional --target_match_ratio (e.g. 0.18): resample to ~15-20% matches for robust train/valid/test (stratified split preserved).
- Output: train.txt, valid.txt, test.txt in the same folder as the CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

# Allow importing blocking from ditto/blocking
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DITTO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
if _DITTO_ROOT not in sys.path:
    sys.path.insert(0, _DITTO_ROOT)

from blocking.blocking import (
    load_csv,
    blocking_ngram,
    blocking_embedding,
    write_ditto_splits,
)

# Default paths
DEFAULT_CSV_PATH = os.path.join(_SCRIPT_DIR, "pro_supplier_standardization_v.csv")
EXCLUDE_COLUMNS = ["id", "created_at", "canonical"]  # id/created_at not used; canonical used only for labeling
DEFAULT_OUTPUT_DIR = _SCRIPT_DIR
DEFAULT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VALID_RATIO = 0.1
# test = 1 - train - valid
NGRAM_N = 3
DEFAULT_EMBEDDING_K = 5
# Target match ratio for robust training (e.g. 0.18 = 18% matches). None = no resampling.
DEFAULT_TARGET_MATCH_RATIO = None  # set to 0.18 for ~15-20% matches


def load_canonicals(path: str, encoding: str = "utf-8") -> list:
    """Return list of canonical values by row index."""
    canonicals = []
    with open(path, encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            canonicals.append(str(row.get("canonical", "")).strip())
    return canonicals


def main(
    csv_path: str = DEFAULT_CSV_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    seed: int = DEFAULT_SEED,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    valid_ratio: float = DEFAULT_VALID_RATIO,
    skip_embedding: bool = False,
    blocking_mode: str = "union",
    embedding_k: int = DEFAULT_EMBEDDING_K,
    target_match_ratio: float | None = DEFAULT_TARGET_MATCH_RATIO,
) -> None:
    # blocking_mode: union (3-gram ∪ embedding), intersection (3-gram ∩ embedding), embedding_only (only top-k by cosine)
    print("Step 1: Load CSV (excluding id, created_at, canonical from record).")
    records, columns = load_csv(
        csv_path,
        id_column=None,  # use row index as id
        exclude_columns=EXCLUDE_COLUMNS,
        encoding="utf-8",
    )
    canonicals = load_canonicals(csv_path, encoding="utf-8")
    assert len(records) == len(canonicals), "Record and canonical length mismatch"
    n = len(records)
    print(f"  Loaded {n} records. Record columns: {columns}")

    pairs_ngram: set = set()
    if blocking_mode != "embedding_only":
        print("Step 2: Apply 3-gram blocking (self-join).")
        pairs_ngram = blocking_ngram(
            records,
            records_right=None,
            n=NGRAM_N,
            attr=None,
            ngram_type="char",
        )
        print(f"  3-gram candidates: {len(pairs_ngram)}")
    else:
        print("Step 2: Skipping 3-gram (blocking_mode=embedding_only).")

    if blocking_mode == "embedding_only" or blocking_mode == "intersection" or not skip_embedding:
        print("Step 3: Apply sentence-BERT (cosine similarity) blocking.")
        pairs_embedding = blocking_embedding(
            records,
            records_right=None,
            model=None,
            k=embedding_k,
            threshold=None,
            batch_size=512,
        )
        print(f"  Sentence-BERT candidates: {len(pairs_embedding)}")
        if blocking_mode == "embedding_only":
            pairs_union = pairs_embedding
        elif blocking_mode == "intersection":
            pairs_union = pairs_ngram & pairs_embedding
            print(f"  Intersection (3-gram ∩ BERT): {len(pairs_union)}")
        else:
            pairs_union = pairs_ngram | pairs_embedding
    else:
        assert blocking_mode == "union" and skip_embedding
        print("Step 3: Skipping sentence-BERT (--skip_embedding).")
        pairs_union = pairs_ngram
    print(f"  Total candidates: {len(pairs_union)}")

    print("Step 4: Label by canonical (same canonical = 1).")
    labeled = []
    for (i, j) in pairs_union:
        label = 1 if canonicals[i] == canonicals[j] and canonicals[i] else 0
        labeled.append((i, j, label))

    if not labeled:
        print("  No candidate pairs; nothing to write.")
        return

    labeled = np.array(labeled, dtype=object)
    labels = np.array([x[2] for x in labeled])
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    print(f"  Raw labels: {n_pos} matches ({100*n_pos/len(labels):.1f}%), {n_neg} non-matches ({100*n_neg/len(labels):.1f}%)")

    # Optional: resample to achieve target match ratio (e.g. 15-20%) for robust train/valid/test
    if target_match_ratio is not None and 0 < target_match_ratio < 1:
        np.random.seed(seed)
        pos_idx_all = np.where(labels == 1)[0]
        neg_idx_all = np.where(labels == 0)[0]
        n_neg_want = int(round(n_pos * (1 - target_match_ratio) / target_match_ratio))
        if n_neg_want < len(neg_idx_all):
            neg_idx_sampled = np.random.choice(neg_idx_all, size=n_neg_want, replace=False)
            keep_idx = np.concatenate([pos_idx_all, neg_idx_sampled])
            np.random.shuffle(keep_idx)
            labeled = labeled[keep_idx]
            labels = np.array([x[2] for x in labeled])
            print(f"  Resampled to target match ratio {target_match_ratio:.0%}: {len(labeled)} pairs ({int(labels.sum())} matches, {len(labeled)-int(labels.sum())} non-matches, {100*labels.mean():.1f}% match)")
        else:
            print(f"  Target match ratio {target_match_ratio:.0%} would require {n_neg_want} negatives but only {len(neg_idx_all)} available; keeping all (match rate {100*n_pos/len(labels):.1f}%)")

    print("Step 5: Stratified split into train/valid/test.")
    np.random.seed(seed)
    # Stratified split: same train/valid/test ratio for positives and negatives
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    def split_indices(idxs):
        n = len(idxs)
        if n == 0:
            return [], [], []
        t = max(1, int(n * train_ratio))
        v = max(0, int(n * valid_ratio))
        te = n - t - v
        return idxs[:t], idxs[t : t + v], idxs[t + v :]

    pos_t, pos_v, pos_te = split_indices(pos_idx)
    neg_t, neg_v, neg_te = split_indices(neg_idx)

    train_idx = np.concatenate([pos_t, neg_t])
    valid_idx = np.concatenate([pos_v, neg_v])
    test_idx = np.concatenate([pos_te, neg_te])
    np.random.shuffle(train_idx)
    np.random.shuffle(valid_idx)
    np.random.shuffle(test_idx)

    def to_lines(indices):
        return [
            (records[labeled[i][0]][1], records[labeled[i][1]][1], labeled[i][2])
            for i in indices
        ]

    train_lines = to_lines(train_idx)
    valid_lines = to_lines(valid_idx)
    test_lines = to_lines(test_idx)

    print(f"  Train: {len(train_lines)}, Valid: {len(valid_lines)}, Test: {len(test_lines)}")

    write_ditto_splits(
        train_lines,
        test_lines,
        output_dir,
        valid_lines=valid_lines,
    )
    print(f"  Wrote train.txt, valid.txt, test.txt to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build train/valid/test.txt for IJF pro_supplier_standardization from CSV (3-gram + optional sentence-BERT blocking).",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="Path to pro_supplier_standardization CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write train.txt, valid.txt, test.txt (default: same as CSV).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Fraction of labeled pairs for training (default: 0.8).",
    )
    parser.add_argument(
        "--valid_ratio",
        type=float,
        default=DEFAULT_VALID_RATIO,
        help="Fraction of labeled pairs for validation (default: 0.1). Test = 1 - train - valid.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for train/valid/test split (default: 42).",
    )
    parser.add_argument(
        "--skip_embedding",
        action="store_true",
        help="Use only 3-gram blocking (no sentence-BERT). Ignored if --blocking_mode is embedding_only or intersection.",
    )
    parser.add_argument(
        "--blocking_mode",
        type=str,
        default="union",
        choices=["union", "intersection", "embedding_only"],
        help="union: 3-gram ∪ BERT (largest, lowest match density). intersection: 3-gram ∩ BERT (smaller, higher density). embedding_only: only BERT top-k (smallest, highest density).",
    )
    parser.add_argument(
        "--embedding_k",
        type=int,
        default=DEFAULT_EMBEDDING_K,
        help="For sentence-BERT blocking: top-k most similar pairs per record (default: 5). Lower k = smaller dataset, higher density.",
    )
    parser.add_argument(
        "--target_match_ratio",
        type=float,
        default=DEFAULT_TARGET_MATCH_RATIO,
        metavar="R",
        help="Resample so match rate is ~R (e.g. 0.18 for 18%% matches). Keeps all positives, subsamples negatives. Use for robust 15-20%% match splits (stratified train/valid/test).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        skip_embedding=args.skip_embedding,
        blocking_mode=args.blocking_mode,
        embedding_k=args.embedding_k,
        target_match_ratio=args.target_match_ratio,
    )
