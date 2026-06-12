"""Entity-disjoint holdout evaluation.

Entities in the holdout predict set are completely absent from the training data.
This tests true generalisation — the model must match entities it has never seen.

Steps:
1. Build holdout predict set: canonical groups whose records never appear in train.txt.
   - Positive pairs : within-group (same canonical_int → true match).
   - Negative pairs : cross-group (different canonical_int → true non-match).
2. Run BERT inference on holdout pairs → Ditto classifier metrics.
3. Build combined Leiden graph (train edges + holdout edges) → Leiden cluster metrics.
4. Report both side-by-side.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

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
from .modeling import load_model_for_inference, predict_from_txt


# ── shared helpers ─────────────────────────────────────────────────────────────

def _leiden(pairs_df: pd.DataFrame, resolution: float = 1.0) -> Tuple[Dict[str, int], int, int]:
    """Build weighted graph from (record1, record2, prob_match) and run Leiden.

    Returns (record_to_cluster, n_clusters, n_nodes).
    """
    df = pairs_df[pairs_df["record1"] != pairs_df["record2"]].copy()
    df = df.groupby(["record1", "record2"], sort=False)["prob_match"].max().reset_index()

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


# ── step 1: build holdout predict set ─────────────────────────────────────────

def build_entity_holdout_predict(
    dataset: str,
    target_pairs: int = 10_000,
    neg_pos_ratio: int = 4,
    seed: int = 42,
) -> str:
    """Build a labeled holdout predict set whose entities never appear in train.txt.

    Writes:
      data/<dataset>/<dataset>_entity_holdout_predict.txt  — Ditto tab-separated format
      dataset/<dataset>/<dataset>_entity_holdout_predict.csv — CSV with record1/record2/label columns

    Returns path to the .txt file.
    """
    dcfg = load_dataset_config(dataset)
    gold_csv  = to_abs(dcfg["paths"]["gold_csv"])
    train_txt = to_abs(dcfg["output"]["train_txt"])

    print(f"[entity_holdout] Loading gold CSV: {gold_csv}")
    df = pd.read_csv(gold_csv, low_memory=False, usecols=["record_text", "canonical_int"])

    # Records seen during training
    train_records: set = set()
    with open(train_txt) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                train_records.add(parts[0])
                train_records.add(parts[1])
    print(f"[entity_holdout] Training records: {len(train_records):,}")

    # Group gold records by canonical_int
    canon_to_records: Dict = defaultdict(list)
    for record, canonical in zip(df["record_text"], df["canonical_int"]):
        canon_to_records[canonical].append(record)

    # Unseen groups: 2+ records, none in training
    unseen_groups = {
        cid: recs
        for cid, recs in canon_to_records.items()
        if len(recs) >= 2 and not any(r in train_records for r in recs)
    }
    print(f"[entity_holdout] Unseen canonical groups (2+ records, none in training): {len(unseen_groups):,}")

    rng = random.Random(seed)

    # ── Positive pairs (within-group) ─────────────────────────────────────────
    all_pos: List[Tuple] = []
    for cid, recs in unseen_groups.items():
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                all_pos.append((recs[i], recs[j], 1))
    rng.shuffle(all_pos)

    n_pos = min(len(all_pos), target_pairs // (neg_pos_ratio + 1))
    pos_pairs = all_pos[:n_pos]

    # ── Negative pairs (cross-group) ──────────────────────────────────────────
    flat_records = [r for recs in unseen_groups.values() for r in recs]
    record_to_canon = {r: cid for cid, recs in unseen_groups.items() for r in recs}

    n_neg = n_pos * neg_pos_ratio
    neg_pairs: List[Tuple] = []
    max_attempts = n_neg * 20
    attempts = 0
    while len(neg_pairs) < n_neg and attempts < max_attempts:
        r1 = rng.choice(flat_records)
        r2 = rng.choice(flat_records)
        if r1 != r2 and record_to_canon[r1] != record_to_canon[r2]:
            neg_pairs.append((r1, r2, 0))
        attempts += 1

    all_pairs = pos_pairs + neg_pairs
    rng.shuffle(all_pairs)

    n_pos_final = sum(1 for _, _, l in all_pairs if l == 1)
    n_neg_final = sum(1 for _, _, l in all_pairs if l == 0)
    print(f"[entity_holdout] Pairs built: {n_pos_final:,} positive + {n_neg_final:,} negative = {len(all_pairs):,} total")

    # ── Save .txt (Ditto format) ───────────────────────────────────────────────
    out_txt = Path(to_abs(f"data/{dataset}/{dataset}_entity_holdout_predict.txt"))
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8") as fh:
        for r1, r2, label in all_pairs:
            fh.write(f"{r1}\t{r2}\t{label}\n")

    # ── Save .csv ─────────────────────────────────────────────────────────────
    out_csv = Path(to_abs(f"dataset/{dataset}/{dataset}_entity_holdout_predict.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_pairs, columns=["record1", "record2", "label"]).to_csv(out_csv, index=False)

    print(f"[entity_holdout] Wrote {out_txt}")
    print(f"[entity_holdout] Wrote {out_csv}")
    return str(out_txt)


# ── step 2–4: inference + graph + evaluate ────────────────────────────────────

def run_entity_holdout_eval(
    dataset: str,
    lm: str = "bert",
    resolution: float = 1.0,
    target_pairs: int = 10_000,
    neg_pos_ratio: int = 4,
    seed: int = 42,
    rebuild: bool = False,
) -> Dict:
    """Evaluate BERT + Leiden on the entity-disjoint holdout predict set.

    Args:
        dataset:       Dataset name.
        lm:            Classifier backbone used for inference.
        resolution:    Leiden CPMVertexPartition resolution.
        target_pairs:  Target holdout set size (built once, cached).
        neg_pos_ratio: Negative : positive pair ratio for holdout set.
        seed:          RNG seed for holdout sampling.
        rebuild:       Force rebuild the holdout file even if it exists.
    """
    dcfg = load_dataset_config(dataset)
    tcfg = load_training_config()

    tag        = lm
    train_txt  = to_abs(dcfg["output"]["train_txt"])
    threshold  = float(tcfg.get("threshold", 0.5))
    max_len    = int(tcfg.get("max_len", 256))
    batch_eval = int(tcfg.get("batch_size_eval", 128))

    out_dir = Path(to_abs(f"Test_Output/{dataset}_{tag}_entity_holdout_result"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: build holdout file if needed ──────────────────────────────────
    holdout_txt = Path(to_abs(f"data/{dataset}/{dataset}_entity_holdout_predict.txt"))
    if rebuild or not holdout_txt.exists():
        print(f"\n{'='*60}")
        print(f"  Step 1 — Building entity-disjoint holdout predict set")
        print(f"{'='*60}")
        build_entity_holdout_predict(dataset, target_pairs, neg_pos_ratio, seed)
    else:
        n_lines = sum(1 for _ in holdout_txt.open())
        print(f"\n[entity_holdout] Using existing holdout file ({n_lines:,} pairs): {holdout_txt}")
        print(f"[entity_holdout] Pass --rebuild to regenerate.")

    # ── Step 2: load model and infer on holdout pairs ─────────────────────────
    print(f"\n{'='*60}")
    print(f"  Step 2 — Inferring on holdout pairs  [lm={lm}]")
    print(f"{'='*60}")

    ckpt = Path(to_abs(f"models/{dataset}_{tag}/best_model.pt"))
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Run training first.")

    model = load_model_for_inference(str(ckpt), lm=lm, device=tcfg.get("device", None))

    holdout_df = pd.DataFrame(
        predict_from_txt(model, str(holdout_txt), lm=lm, max_len=max_len,
                         batch_size=batch_eval, threshold=threshold)
    )
    print(f"[entity_holdout] Holdout pairs inferred: {len(holdout_df):,}")

    # ── Step 3: build combined graph (train + holdout) and recluster ──────────
    print(f"\n{'='*60}")
    print(f"  Step 3 — Building combined graph (train edges + holdout edges)")
    print(f"{'='*60}")

    train_df = pd.DataFrame(
        predict_from_txt(model, train_txt, lm=lm, max_len=max_len,
                         batch_size=batch_eval, threshold=threshold)
    )
    print(f"[entity_holdout] Training pairs inferred: {len(train_df):,}")

    combined_df = pd.concat(
        [train_df[["record1", "record2", "prob_match"]],
         holdout_df[["record1", "record2", "prob_match"]]],
        ignore_index=True,
    )
    combined_clusters, n_clusters, n_nodes = _leiden(combined_df, resolution)
    print(f"[entity_holdout] Combined graph: {len(combined_df):,} pairs | {n_nodes:,} nodes | {n_clusters:,} clusters")

    # ── Step 4: predict holdout pairs from combined cluster membership ─────────
    holdout_df["leiden_cluster1"] = holdout_df["record1"].map(combined_clusters)
    holdout_df["leiden_cluster2"] = holdout_df["record2"].map(combined_clusters)
    holdout_df["leiden_pred"]     = (
        holdout_df["leiden_cluster1"] == holdout_df["leiden_cluster2"]
    ).astype(int)

    missing = holdout_df["leiden_cluster1"].isna() | holdout_df["leiden_cluster2"].isna()
    if missing.any():
        print(f"[entity_holdout] Warning: {missing.sum()} pairs had a record missing from graph. "
              "Falling back to Ditto prediction for those.")
        holdout_df.loc[missing, "leiden_pred"] = holdout_df.loc[missing, "pred_label"]

    # ── Step 5: metrics ────────────────────────────────────────────────────────
    ditto_m  = _metrics(holdout_df, "pred_label")
    leiden_m = _metrics(holdout_df, "leiden_pred")

    W = 60
    print(f"\n{'='*W}")
    print(f"  Results — {dataset} | lm={lm} | resolution={resolution}")
    print(f"  Holdout: entities completely absent from training")
    print(f"{'='*W}")
    print(f"  Holdout pairs  : {len(holdout_df):,}  "
          f"(+{int(holdout_df['true_label'].sum()):,} / -{int((holdout_df['true_label']==0).sum()):,})")
    print(f"  Combined graph : {len(combined_df):,} pairs | {n_nodes:,} nodes | {n_clusters:,} clusters")
    _print_metrics_table(ditto_m, leiden_m)
    print(f"{'='*W}")

    # ── Save outputs ───────────────────────────────────────────────────────────
    result_csv = out_dir / f"{dataset}_{tag}_entity_holdout_test.csv"
    holdout_df.to_csv(result_csv, index=False)

    analysis_path = out_dir / f"{dataset}_{tag}_entity_holdout_analysis.txt"
    with analysis_path.open("w", encoding="utf-8") as fh:
        fh.write(f"dataset:           {dataset}\n")
        fh.write(f"lm:                {lm}\n")
        fh.write(f"leiden_resolution: {resolution}\n")
        fh.write(f"holdout_pairs:     {len(holdout_df)}\n")
        fh.write(f"holdout_positive:  {int(holdout_df['true_label'].sum())}\n")
        fh.write(f"holdout_negative:  {int((holdout_df['true_label']==0).sum())}\n")
        fh.write(f"combined_pairs:    {len(combined_df)}\n")
        fh.write(f"combined_nodes:    {n_nodes}\n")
        fh.write(f"combined_clusters: {n_clusters}\n")
        fh.write(f"\n[ditto classifier — threshold={threshold}]\n")
        for k, v in ditto_m.items():
            fh.write(f"{k}: {v}\n")
        fh.write(f"\n[leiden graph — combined train+holdout]\n")
        for k, v in leiden_m.items():
            fh.write(f"{k}: {v}\n")

    print(f"\n  Wrote: {result_csv}")
    print(f"  Wrote: {analysis_path}")

    return {"ditto": ditto_m, "leiden": leiden_m}


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate BERT + Leiden on entity-disjoint holdout (unseen entities)."
    )
    ap.add_argument("--dataset",       required=True,          help="Dataset name, e.g. pro_supplier")
    ap.add_argument("--lm",            default="bert",         help="bert | roberta | distilbert")
    ap.add_argument("--resolution",    type=float, default=1.0, help="Leiden resolution (default: 1.0)")
    ap.add_argument("--target_pairs",  type=int, default=10_000, help="Target holdout set size (default: 10000)")
    ap.add_argument("--neg_pos_ratio", type=int, default=4,    help="Negatives per positive (default: 4)")
    ap.add_argument("--seed",          type=int, default=42,   help="RNG seed (default: 42)")
    ap.add_argument("--rebuild",       action="store_true",    help="Force rebuild holdout file")
    args = ap.parse_args()
    run_entity_holdout_eval(
        args.dataset,
        lm=args.lm,
        resolution=args.resolution,
        target_pairs=args.target_pairs,
        neg_pos_ratio=args.neg_pos_ratio,
        seed=args.seed,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
