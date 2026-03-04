"""
Analysis utilities for model trust, intuition, and simple ablation.

These helpers are thin wrappers around the logic prototyped in the
notebook (casebook export, counterfactuals, label-shuffle sanity check),
but written as reusable functions that operate on files and arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import os
import pandas as pd

import matplotlib.pyplot as plt  # type: ignore


@dataclass
class CasebookConfig:
    n_per_class: int = 20


def build_casebook(
    tp_csv: str,
    fp_csv: str,
    tn_csv: str,
    fn_csv: str,
    out_csv: str,
    config: Optional[CasebookConfig] = None,
) -> pd.DataFrame:
    """
    Build a compact casebook from TP/FP/TN/FN CSVs of the form produced
    by the notebook (index, true_label, pred_label, record1, record2).
    """
    if config is None:
        config = CasebookConfig()

    def _load(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame(columns=["index", "true_label", "pred_label", "record1", "record2"])
        return pd.read_csv(path)

    dfs: Dict[str, pd.DataFrame] = {
        "TP": _load(tp_csv),
        "FP": _load(fp_csv),
        "TN": _load(tn_csv),
        "FN": _load(fn_csv),
    }

    samples: List[pd.DataFrame] = []
    # Prioritise FP/FN first so they show up early for inspection.
    order = ["FP", "FN", "TP", "TN"]

    for src in order:
        df = dfs[src]
        if df.empty:
            continue
        n_take = min(config.n_per_class, len(df))
        df_sample = df.sample(n=n_take, random_state=42) if len(df) > n_take else df.copy()
        df_sample = df_sample.assign(source=src)
        samples.append(df_sample)

    if not samples:
        raise RuntimeError("No examples available in any of TP/FP/TN/FN CSVs.")

    casebook_df = pd.concat(samples, ignore_index=True)
    cols = ["source", "index", "true_label", "pred_label", "record1", "record2"]
    casebook_df = casebook_df[[c for c in cols if c in casebook_df.columns]]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    casebook_df.to_csv(out_csv, index=False)
    return casebook_df


def label_shuffle_sanity_check(
    true_labels: Sequence[int],
    preds: Sequence[int],
    seed: int = 123,
) -> Dict[str, float]:
    """
    Compare performance on real labels vs shuffled labels.
    """
    from sklearn.metrics import f1_score, accuracy_score

    true_arr = np.asarray(true_labels, dtype=int)
    preds_arr = np.asarray(preds, dtype=int)

    real_f1 = f1_score(true_arr, preds_arr)
    real_acc = accuracy_score(true_arr, preds_arr)

    rng = np.random.default_rng(seed)
    shuffled = true_arr.copy()
    rng.shuffle(shuffled)

    shuffled_f1 = f1_score(shuffled, preds_arr)
    shuffled_acc = accuracy_score(shuffled, preds_arr)

    return {
        "real_f1": float(real_f1),
        "real_accuracy": float(real_acc),
        "shuffled_f1": float(shuffled_f1),
        "shuffled_accuracy": float(shuffled_acc),
    }


def prob_histograms_by_label(
    probs: Sequence[float],
    labels: Sequence[int],
    title: str = "Probability distributions by true label",
) -> None:
    """
    Plot two overlaid histograms: predicted P(match) for true positives
    and true negatives.
    """
    probs_arr = np.asarray(probs, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)

    plt.figure(figsize=(8, 5))
    plt.hist(
        probs_arr[labels_arr == 1],
        bins=30,
        alpha=0.6,
        label="True matches (label=1)",
    )
    plt.hist(
        probs_arr[labels_arr == 0],
        bins=30,
        alpha=0.6,
        label="True non-matches (label=0)",
    )
    plt.xlabel("Predicted probability of match P(label=1)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def threshold_sweep_curves(
    probs: Sequence[float],
    labels: Sequence[int],
    title: str = "Threshold vs performance",
) -> None:
    """
    Plot F1 / precision / recall as a function of decision threshold.
    """
    from sklearn.metrics import precision_recall_fscore_support

    probs_arr = np.asarray(probs, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)

    thresholds = np.linspace(0.1, 0.9, 17)
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for th in thresholds:
        preds = (probs_arr >= th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            labels_arr, preds, average="binary", zero_division=0
        )
        precisions.append(float(p))
        recalls.append(float(r))
        f1s.append(float(f1))

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1s, label="F1")
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.xlabel("Decision threshold on P(label=1)")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

