"""Evaluate Leiden clustering on the test set using an ensemble of BERT, RoBERTa, and DistilBERT probabilities.

Steps:
1. Load cached per-pair test probabilities for each model (runs inference if missing).
2. Combine probabilities across models (default: mean).
3. Build a weighted graph: nodes = unique records, edges = test pairs weighted by ensemble score.
4. Run Leiden CPMVertexPartition clustering.
5. For each labeled test pair, predict match = same cluster; compute F1 / precision / recall.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import load_dataset_config, load_training_config, to_abs
from .modeling import load_model_for_inference, predict_from_txt


ENSEMBLE_MODELS = ["bert", "roberta", "distilbert"]


def _load_or_run_test(dataset: str, lm: str, tcfg: dict, dcfg: dict) -> pd.DataFrame:
    """Return test-set predictions for one model, loading a cached CSV when available."""
    tag = lm
    test_csv = Path(to_abs(f"Test_Output/{dataset}_{tag}_test_result/{dataset}_{tag}_test.csv"))
    if test_csv.exists():
        print(f"[leiden_test] Loading cached: {test_csv}")
        return pd.read_csv(test_csv)

    print(f"[leiden_test] Running inference for lm={lm} ...")
    test_txt = to_abs(dcfg["output"]["test_txt"])
    ckpt = Path(to_abs(f"models/{dataset}_{tag}/best_model.pt"))
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Run training first.")

    model = load_model_for_inference(str(ckpt), lm=lm, device=tcfg.get("device", None))
    rows = predict_from_txt(
        model,
        test_txt,
        lm=lm,
        max_len=int(tcfg.get("max_len", 256)),
        batch_size=int(tcfg.get("batch_size_eval", 128)),
        threshold=float(tcfg.get("threshold", 0.5)),
    )
    df = pd.DataFrame(rows)
    test_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(test_csv, index=False)
    print(f"[leiden_test] Saved inference results: {test_csv}")
    return df


def run_leiden_ensemble_test(
    dataset: str,
    models: Optional[List[str]] = None,
    combination: str = "mean",
    resolution: float = 1.0,
) -> Dict:
    """Run Leiden ensemble evaluation on the test set.

    Args:
        dataset: Dataset name matching a configs/datasets/<dataset>.yaml file.
        models: List of language models to ensemble (default: bert, roberta, distilbert).
        combination: How to combine per-model probabilities — "mean", "max", or "min".
        resolution: Leiden CPMVertexPartition resolution parameter.

    Returns:
        Dict with f1, precision, recall, accuracy, tp/fp/tn/fn, n_clusters, n_nodes.
    """
    models = models or ENSEMBLE_MODELS[:]
    dcfg = load_dataset_config(dataset)
    tcfg = load_training_config()

    # 1. Load / run inference for each model
    dfs: Dict[str, pd.DataFrame] = {}
    for lm in models:
        dfs[lm] = _load_or_run_test(dataset, lm, tcfg, dcfg)

    # 2. Verify all CSVs are row-aligned (same pairs in same order)
    base_lm = models[0]
    base = dfs[base_lm][["record1", "record2", "true_label"]].copy().reset_index(drop=True)
    for lm in models[1:]:
        other = dfs[lm].reset_index(drop=True)
        misaligned = (
            (other["record1"] != base["record1"]) | (other["record2"] != base["record2"])
        ).sum()
        if misaligned > 0:
            raise ValueError(
                f"Row misalignment between {base_lm} and {lm}: {misaligned} rows differ. "
                "Delete cached test CSVs and re-run."
            )

    # 3. Compute ensemble probability
    prob_matrix = np.stack([dfs[lm]["prob_match"].values for lm in models], axis=1)
    if combination == "mean":
        ensemble_prob = prob_matrix.mean(axis=1)
    elif combination == "max":
        ensemble_prob = prob_matrix.max(axis=1)
    elif combination == "min":
        ensemble_prob = prob_matrix.min(axis=1)
    else:
        raise ValueError(f"Unknown combination method: {combination!r}")

    base["ensemble_prob"] = ensemble_prob
    for lm in models:
        base[f"prob_{lm}"] = dfs[lm]["prob_match"].values

    # 4. Build weighted graph: nodes = unique records, edges = test pairs
    all_records = pd.concat([base["record1"], base["record2"]]).unique()
    name_to_idx = {name: i for i, name in enumerate(all_records)}

    sources = base["record1"].map(name_to_idx).tolist()
    targets = base["record2"].map(name_to_idx).tolist()
    weights = ensemble_prob.tolist()

    g = ig.Graph(n=len(all_records), edges=list(zip(sources, targets)), directed=False)
    g.vs["name"] = list(all_records)
    g.es["weight"] = weights

    # 5. Leiden clustering (mirrors build_graph.py settings)
    partition = la.find_partition(
        g,
        la.CPMVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=-1,
        seed=42,
    )

    record_to_cluster: Dict[str, int] = {}
    for cluster_id, vertex_indices in enumerate(partition):
        for idx in vertex_indices:
            record_to_cluster[g.vs[idx]["name"]] = cluster_id

    # 6. Evaluate: same cluster → predicted match
    base["cluster1"] = base["record1"].map(record_to_cluster)
    base["cluster2"] = base["record2"].map(record_to_cluster)
    base["leiden_pred"] = (base["cluster1"] == base["cluster2"]).astype(int)

    y_true = base["true_label"].fillna(0).astype(int).values
    y_pred = base["leiden_pred"].values

    f1 = f1_score(y_true, y_pred, zero_division=0)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    a = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # 7. Save outputs — single model gets its own dir; ensemble gets a shared dir
    tag = models[0] if len(models) == 1 else "ensemble"
    out_dir = Path(to_abs(f"Test_Output/{dataset}_{tag}_leiden_test_result"))
    out_dir.mkdir(parents=True, exist_ok=True)

    result_csv = out_dir / f"{dataset}_{tag}_leiden_test.csv"
    base.to_csv(result_csv, index=False)

    analysis_path = out_dir / f"{dataset}_{tag}_leiden_test_analysis.txt"
    with analysis_path.open("w", encoding="utf-8") as fh:
        fh.write(f"dataset: {dataset}\n")
        fh.write(f"models: {', '.join(models)}\n")
        fh.write(f"combination: {combination}\n")
        fh.write(f"leiden_resolution: {resolution}\n")
        fh.write(f"n_clusters: {len(partition)}\n")
        fh.write(f"n_nodes: {len(all_records)}\n")
        fh.write(f"rows: {len(base)}\n")
        fh.write(f"f1: {f1:.6f}\n")
        fh.write(f"precision: {p:.6f}\n")
        fh.write(f"recall: {r:.6f}\n")
        fh.write(f"accuracy: {a:.6f}\n")
        fh.write(f"tp: {tp}\nfp: {fp}\ntn: {tn}\nfn: {fn}\n")

    print(f"[leiden_test/{tag}] n_clusters={len(partition)}  n_nodes={len(all_records)}")
    print(f"[leiden_test/{tag}] F1={f1:.6f}  Precision={p:.6f}  Recall={r:.6f}  Accuracy={a:.6f}")
    print(f"[leiden_test/{tag}] TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"[leiden_test/{tag}] Wrote {result_csv}")
    print(f"[leiden_test/{tag}] Wrote {analysis_path}")

    return {
        "f1": f1,
        "precision": p,
        "recall": r,
        "accuracy": a,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "n_clusters": len(partition),
        "n_nodes": len(all_records),
    }


def run_leiden_all_models(
    dataset: str,
    combination: str = "mean",
    resolution: float = 1.0,
) -> Dict[str, Dict]:
    """Run Leiden evaluation separately for each individual classifier and print a comparison table."""
    results: Dict[str, Dict] = {}
    for lm in ENSEMBLE_MODELS:
        print(f"\n--- Leiden test: {lm} ---")
        results[lm] = run_leiden_ensemble_test(
            dataset,
            models=[lm],
            combination=combination,
            resolution=resolution,
        )

    # Comparison table
    cols = ["f1", "precision", "recall", "accuracy", "tp", "fp", "tn", "fn", "n_clusters"]
    header = f"{'model':<12}" + "".join(f"{c:>12}" for c in cols)
    print(f"\n{'='*len(header)}")
    print("Leiden Clustering — Per-Classifier Results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for lm, m in results.items():
        row = f"{lm:<12}" + "".join(
            f"{m[c]:>12.4f}" if isinstance(m[c], float) else f"{m[c]:>12}"
            for c in cols
        )
        print(row)
    print("=" * len(header))

    return results


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate Leiden clustering on the test set using individual or ensemble classifiers."
    )
    ap.add_argument("--dataset", required=True, help="Dataset name, e.g. pro_supplier")
    ap.add_argument(
        "--models",
        default="bert,roberta,distilbert",
        help="Comma-separated list of models to ensemble (default: bert,roberta,distilbert)",
    )
    ap.add_argument(
        "--combination",
        default="mean",
        choices=["mean", "max", "min"],
        help="How to combine per-model probabilities (default: mean)",
    )
    ap.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Leiden CPMVertexPartition resolution parameter (default: 1.0)",
    )
    ap.add_argument(
        "--each",
        action="store_true",
        help="Run Leiden separately for each classifier and print a comparison table",
    )
    args = ap.parse_args()
    if args.each:
        run_leiden_all_models(args.dataset, combination=args.combination, resolution=args.resolution)
    else:
        run_leiden_ensemble_test(
            args.dataset,
            models=args.models.split(","),
            combination=args.combination,
            resolution=args.resolution,
        )


if __name__ == "__main__":
    main()
