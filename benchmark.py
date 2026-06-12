"""
Sequential benchmark: train / test / predict (Ditto) / build_graph (Leiden)
for bert, distilbert, roberta — one model at a time to avoid GPU contention.

Usage:
    python benchmark.py --dataset pro_supplier --n_epochs 10
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MODELS = ["bert", "distilbert", "roberta"]
STAGES = ["train", "test", "predict", "build_graph"]


def _run_stage(dataset: str, model_tag: str, stage: str, n_epochs: int, log_file) -> float:
    cmd = [
        sys.executable, "-m", "pipeline.run_pipeline",
        "--dataset", dataset,
        "--stage", stage,
        "--lm", model_tag,
        "--model_tag", model_tag,
    ]
    if stage == "train":
        cmd += ["--n_epochs", str(n_epochs)]
    t0 = time.time()
    result = subprocess.run(cmd, stdout=log_file, stderr=log_file, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0
    log_file.flush()
    if result.returncode != 0:
        print(f"  [FAILED] {model_tag}/{stage} (exit {result.returncode}) — check benchmark_logs/{model_tag}.log")
    return elapsed


def _load_metrics(dataset: str, model_tag: str) -> dict:
    path = REPO_ROOT / "models" / f"{dataset}_{model_tag}" / "train_metrics.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_test_metrics(dataset: str, model_tag: str) -> dict:
    path = REPO_ROOT / "Test_Output" / f"{dataset}_{model_tag}_test_result" / f"{dataset}_{model_tag}_test_analysis.txt"
    out = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                try:
                    out[k.strip()] = float(v.strip())
                except ValueError:
                    out[k.strip()] = v.strip()
    except Exception:
        pass
    return out


def _load_predict_metrics(dataset: str, model_tag: str) -> dict:
    path = REPO_ROOT / "predict_output" / f"{dataset}_{model_tag}_predict_result" / f"{dataset}_{model_tag}_predict_analysis.txt"
    out = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                try:
                    out[k.strip()] = float(v.strip())
                except ValueError:
                    out[k.strip()] = v.strip()
    except Exception:
        pass
    return out


def _load_cluster_info(dataset: str, model_tag: str) -> dict:
    import pandas as pd
    path = REPO_ROOT / "predict_output" / f"{dataset}_{model_tag}_cluster_result" / f"{dataset}_{model_tag}_clusters.csv"
    try:
        df = pd.read_csv(path)
        return {
            "total_records": len(df),
            "total_clusters": df["cluster_id"].nunique(),
            "avg_cluster_size": round(len(df) / df["cluster_id"].nunique(), 2),
        }
    except Exception:
        return {}


def _print_report(results: list[dict]) -> None:
    sep = "-" * 110

    print("\n" + "=" * 110)
    print("BENCHMARK RESULTS — Pure Ditto (pairwise classification)")
    print("=" * 110)
    print(f"{'Model':<14} {'Train(s)':>9} {'Test(s)':>8} {'Pred(s)':>8} {'Best Ep':>8} {'val_f1':>8} {'test_f1':>8} {'precision':>10} {'recall':>8} {'accuracy':>10}")
    print(sep)
    for r in results:
        m = r.get("train_metrics", {})
        print(
            f"{r['model']:<14}"
            f"{r.get('train_time', 0):>9.1f}"
            f"{r.get('test_time', 0):>8.1f}"
            f"{r.get('predict_time', 0):>8.1f}"
            f"{str(m.get('epoch', '-')):>8}"
            f"{m.get('val_f1', 0):>8.4f}"
            f"{m.get('test_f1', 0):>8.4f}"
            f"{m.get('test_precision', 0):>10.4f}"
            f"{m.get('test_recall', 0):>8.4f}"
            f"{m.get('test_accuracy', 0):>10.4f}"
        )
    print(sep)

    print("\n" + "=" * 110)
    print("BENCHMARK RESULTS — Graph (Leiden community detection on Ditto scores)")
    print("=" * 110)
    print(f"{'Model':<14} {'Graph(s)':>9} {'Records':>9} {'Clusters':>10} {'Avg Size':>10}")
    print(sep)
    for r in results:
        c = r.get("cluster_info", {})
        print(
            f"{r['model']:<14}"
            f"{r.get('graph_time', 0):>9.1f}"
            f"{c.get('total_records', '-'):>9}"
            f"{c.get('total_clusters', '-'):>10}"
            f"{c.get('avg_cluster_size', '-'):>10}"
        )
    print(sep)

    out = REPO_ROOT / "benchmark_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull results saved to: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pro_supplier")
    ap.add_argument("--n_epochs", type=int, default=10)
    args = ap.parse_args()

    dataset = args.dataset
    n_epochs = args.n_epochs
    logs_dir = REPO_ROOT / "benchmark_logs"
    logs_dir.mkdir(exist_ok=True)

    print(f"Sequential benchmark: {MODELS}  |  epochs={n_epochs}  |  dataset={dataset}")
    print("Logs → benchmark_logs/<model>.log\n")

    results = []

    for tag in MODELS:
        log_path = logs_dir / f"{tag}.log"
        print(f"\n{'='*60}")
        print(f"Model: {tag}")
        print(f"{'='*60}")

        times = {}
        with log_path.open("w", encoding="utf-8") as log_file:
            for stage in STAGES:
                print(f"  [{stage}] starting...", end=" ", flush=True)
                elapsed = _run_stage(dataset, tag, stage, n_epochs, log_file)
                times[stage] = elapsed
                print(f"done in {elapsed:.1f}s")

        results.append({
            "model": tag,
            "train_time": times.get("train", 0),
            "test_time": times.get("test", 0),
            "predict_time": times.get("predict", 0),
            "graph_time": times.get("build_graph", 0),
            "train_metrics": _load_metrics(dataset, tag),
            "test_metrics": _load_test_metrics(dataset, tag),
            "predict_metrics": _load_predict_metrics(dataset, tag),
            "cluster_info": _load_cluster_info(dataset, tag),
            "log": str(log_path),
        })

    _print_report(results)


if __name__ == "__main__":
    main()
