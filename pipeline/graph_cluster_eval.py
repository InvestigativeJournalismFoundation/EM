"""Graph-cluster evaluation pipeline.

Steps:
1. Train Ditto (BERT by default, configurable epochs).
2. Infer on all training pairs  → build training graph (n edges)  → Leiden clusters.
3. Infer on all test pairs      → build combined graph (n + m edges) → recluster.
4. Predict each test pair: same cluster in the combined graph → match.
5. Report Ditto classifier metrics (threshold) vs Leiden graph metrics side-by-side.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import igraph as ig
import leidenalg as la
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import load_dataset_config, load_training_config, to_abs
from .modeling import TrainHyperParams, load_model_for_inference, predict_from_txt, train_save_and_eval


# ── helpers ───────────────────────────────────────────────────────────────────

def _leiden(pairs_df: pd.DataFrame, resolution: float = 1.0):
    """Build a weighted undirected graph from (record1, record2, prob_match) rows and run Leiden.

    Returns (record_to_cluster, n_clusters, n_nodes).
    """
    # Drop self-loops; keep highest probability for any duplicate pair
    df = pairs_df[pairs_df["record1"] != pairs_df["record2"]].copy()
    df = (
        df.groupby(["record1", "record2"], sort=False)["prob_match"]
        .max()
        .reset_index()
    )

    all_records = pd.concat([df["record1"], df["record2"]]).unique()
    name_to_idx = {name: i for i, name in enumerate(all_records)}

    g = ig.Graph(
        n=len(all_records),
        edges=list(zip(df["record1"].map(name_to_idx), df["record2"].map(name_to_idx))),
        directed=False,
    )
    g.vs["name"] = list(all_records)
    g.es["weight"] = df["prob_match"].tolist()

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

    return record_to_cluster, len(partition), len(all_records)


def _metrics(df: pd.DataFrame, pred_col: str) -> Dict:
    y_true = df["true_label"].fillna(0).astype(int).values
    y_pred = df[pred_col].astype(int).values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def _print_metrics_table(ditto_m: Dict, leiden_m: Dict) -> None:
    print(f"\n  {'Metric':<12} {'Ditto (threshold)':>20} {'Leiden (graph)':>20}")
    print(f"  {'-'*52}")
    for k in ["f1", "precision", "recall", "accuracy"]:
        print(f"  {k:<12} {ditto_m[k]:>20.6f} {leiden_m[k]:>20.6f}")
    for k in ["tp", "fp", "tn", "fn"]:
        print(f"  {k:<12} {ditto_m[k]:>20} {leiden_m[k]:>20}")


# ── main function ──────────────────────────────────────────────────────────────

def run_graph_cluster_eval(
    dataset: str,
    lm: str = "bert",
    n_epochs: int = 10,
    resolution: float = 1.0,
    retrain: bool = False,
) -> Dict:
    """Train Ditto, build train graph, extend to combined (n+m) graph, evaluate on test set.

    Args:
        dataset:    Dataset name matching configs/datasets/<dataset>.yaml.
        lm:         Classifier backbone — "bert", "roberta", or "distilbert".
        n_epochs:   Training epochs (only used when training from scratch).
        resolution: Leiden CPMVertexPartition resolution parameter.
        retrain:    If True, always retrain even when a checkpoint exists.
    """
    dcfg = load_dataset_config(dataset)
    tcfg = load_training_config()

    tag = lm
    train_path   = to_abs(dcfg["output"]["train_txt"])
    valid_path   = to_abs(dcfg["output"]["valid_txt"])
    test_path    = to_abs(dcfg["output"]["test_txt"])
    threshold    = float(tcfg.get("threshold", 0.5))
    max_len      = int(tcfg.get("max_len", 256))
    batch_train  = int(tcfg.get("batch_size_train", 32))
    batch_eval   = int(tcfg.get("batch_size_eval", 128))

    out_dir = Path(to_abs(f"Test_Output/{dataset}_{tag}_graph_cluster_result"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(to_abs(f"models/{dataset}_{tag}/best_model.pt"))

    # ── Step 1: Train ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Step 1 — Train Ditto  [lm={lm}  epochs={n_epochs}]")
    print(f"{'='*60}")

    if retrain or not ckpt.exists():
        hp = TrainHyperParams(
            lm=lm,
            max_len=max_len,
            batch_size_train=batch_train,
            batch_size_eval=batch_eval,
            n_epochs=n_epochs,
            lr=float(tcfg.get("lr", 2e-5)),
            weight_decay=float(tcfg.get("weight_decay", 0.0)),
            seed=int(tcfg.get("seed", 42)),
            device=tcfg.get("device", None),
        )
        train_save_and_eval(
            train_path, valid_path, test_path,
            str(Path(to_abs(f"models/{dataset}_{tag}"))),
            hp, threshold,
            resume=(not retrain),
        )
        print(f"[graph_cluster] Saved checkpoint: {ckpt}")
    else:
        print(f"[graph_cluster] Checkpoint already exists — skipping training.")
        print(f"[graph_cluster] {ckpt}")
        print(f"[graph_cluster] Pass --retrain to force retraining from scratch.")

    # Load model once; reuse for both train-pair and test-pair inference
    model = load_model_for_inference(str(ckpt), lm=lm, device=tcfg.get("device", None))

    # ── Step 2: Training graph (n pairs) ───────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Step 2 — Infer on training pairs → build training graph (n)")
    print(f"{'='*60}")

    train_df = pd.DataFrame(
        predict_from_txt(model, train_path, lm=lm, max_len=max_len,
                         batch_size=batch_eval, threshold=threshold)
    )
    _, train_n_clusters, train_n_nodes = _leiden(train_df, resolution)
    print(f"[graph_cluster] Training graph : {len(train_df):,} pairs | "
          f"{train_n_nodes:,} nodes | {train_n_clusters:,} clusters")

    # ── Step 3: Combined graph (n + m pairs) and recluster ─────────────────────
    print(f"\n{'='*60}")
    print(f"  Step 3 — Infer on test pairs → build combined graph (n+m) → recluster")
    print(f"{'='*60}")

    test_df = pd.DataFrame(
        predict_from_txt(model, test_path, lm=lm, max_len=max_len,
                         batch_size=batch_eval, threshold=threshold)
    )

    combined_df = pd.concat(
        [train_df[["record1", "record2", "prob_match"]],
         test_df[["record1", "record2", "prob_match"]]],
        ignore_index=True,
    )
    combined_clusters, combined_n_clusters, combined_n_nodes = _leiden(combined_df, resolution)
    print(f"[graph_cluster] Combined graph  : {len(combined_df):,} pairs | "
          f"{combined_n_nodes:,} nodes | {combined_n_clusters:,} clusters")

    # ── Step 4: Predict on test pairs via combined cluster membership ───────────
    print(f"\n{'='*60}")
    print(f"  Step 4 — Predict test pairs from combined cluster membership")
    print(f"{'='*60}")

    test_df["leiden_cluster1"] = test_df["record1"].map(combined_clusters)
    test_df["leiden_cluster2"] = test_df["record2"].map(combined_clusters)
    test_df["leiden_pred"]     = (test_df["leiden_cluster1"] == test_df["leiden_cluster2"]).astype(int)

    missing = test_df["leiden_cluster1"].isna() | test_df["leiden_cluster2"].isna()
    if missing.any():
        print(f"[graph_cluster] Warning: {missing.sum()} test pairs had a record not found in combined graph. "
              f"Falling back to Ditto prediction for those rows.")
        test_df.loc[missing, "leiden_pred"] = test_df.loc[missing, "pred_label"]

    # ── Step 5: Metrics ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Step 5 — Metrics  [{dataset} | lm={lm} | epochs={n_epochs} | resolution={resolution}]")
    print(f"{'='*60}")

    ditto_m  = _metrics(test_df, "pred_label")
    leiden_m = _metrics(test_df, "leiden_pred")

    print(f"  Training graph : {len(train_df):,} pairs | {train_n_nodes:,} nodes | {train_n_clusters:,} clusters")
    print(f"  Combined graph : {len(combined_df):,} pairs | {combined_n_nodes:,} nodes | {combined_n_clusters:,} clusters")
    _print_metrics_table(ditto_m, leiden_m)

    # ── Save outputs ───────────────────────────────────────────────────────────
    result_csv = out_dir / f"{dataset}_{tag}_graph_cluster_test.csv"
    test_df.to_csv(result_csv, index=False)

    analysis_path = out_dir / f"{dataset}_{tag}_graph_cluster_analysis.txt"
    with analysis_path.open("w", encoding="utf-8") as fh:
        fh.write(f"dataset:           {dataset}\n")
        fh.write(f"lm:                {lm}\n")
        fh.write(f"n_epochs:          {n_epochs}\n")
        fh.write(f"leiden_resolution: {resolution}\n")
        fh.write(f"\n[training graph]\n")
        fh.write(f"n_pairs:    {len(train_df)}\n")
        fh.write(f"n_nodes:    {train_n_nodes}\n")
        fh.write(f"n_clusters: {train_n_clusters}\n")
        fh.write(f"\n[combined graph n+m]\n")
        fh.write(f"n_pairs:    {len(combined_df)}\n")
        fh.write(f"n_nodes:    {combined_n_nodes}\n")
        fh.write(f"n_clusters: {combined_n_clusters}\n")
        fh.write(f"\n[ditto classifier — threshold={threshold}]\n")
        for k, v in ditto_m.items():
            fh.write(f"{k}: {v}\n")
        fh.write(f"\n[leiden graph — combined n+m]\n")
        for k, v in leiden_m.items():
            fh.write(f"{k}: {v}\n")

    print(f"\n  Wrote: {result_csv}")
    print(f"  Wrote: {analysis_path}")

    return {
        "ditto":  ditto_m,
        "leiden": leiden_m,
        "train_graph":    {"n_pairs": len(train_df),    "n_nodes": train_n_nodes,    "n_clusters": train_n_clusters},
        "combined_graph": {"n_pairs": len(combined_df), "n_nodes": combined_n_nodes, "n_clusters": combined_n_clusters},
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train Ditto, build train + combined (n+m) Leiden graphs, evaluate on test set."
    )
    ap.add_argument("--dataset",    required=True,         help="Dataset name, e.g. pro_supplier")
    ap.add_argument("--lm",         default="bert",        help="bert | roberta | distilbert (default: bert)")
    ap.add_argument("--n_epochs",   type=int, default=10,  help="Training epochs (default: 10)")
    ap.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution (default: 1.0)")
    ap.add_argument("--retrain",    action="store_true",   help="Force retrain even if checkpoint exists")
    args = ap.parse_args()
    run_graph_cluster_eval(
        args.dataset,
        lm=args.lm,
        n_epochs=args.n_epochs,
        resolution=args.resolution,
        retrain=args.retrain,
    )


if __name__ == "__main__":
    main()
