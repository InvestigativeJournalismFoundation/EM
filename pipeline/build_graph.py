import leidenalg as la
import igraph as ig
from .config import load_dataset_config, to_abs
import pandas as pd
import os
import boto3
import datetime
from datetime import timezone

def build_graph(dataset: str):
    cfg = load_dataset_config(dataset)
    pred_path = cfg["output"]["predict_result_dir"] + "/" + f"{dataset}_predict.csv"
    pred_df = pd.read_csv(to_abs(pred_path))

    # add nodes, which is every unique spelling
    all_records = pd.concat([pred_df["name1"], pred_df["name2"]]).unique()

    # Drop self-loops TODO remove this from the predict step instead
    df = pred_df[pred_df["name1"] != pred_df["name2"]]

    # all edges, which is every pair with weight equal to the predicted score
    edges = df[["name1", "name2", "score"]]

    # Map names to vertex indices for graph construction
    name_to_index = {name: idx for idx, name in enumerate(all_records)}

    sources = edges["name1"].map(name_to_index).tolist()
    targets = edges["name2"].map(name_to_index).tolist()
    weights = edges["score"].tolist()

    # build the graph and run Leiden clustering
    g = ig.Graph(n=len(all_records), edges=list(zip(sources, targets)), directed=False)
    g.vs["name"] = list(all_records)
    g.es["weight"] = weights

    # empirically tuned resolution param to 1
    partition = la.find_partition(
        g,
        la.CPMVertexPartition,
        weights="weight",
        resolution_parameter=1,
        n_iterations=-1,
        seed=42,
    )

    # extract the cluster assignments and save results
    clusters = []
    for cluster_id, vertex_indices in enumerate(partition):
        for idx in vertex_indices:
            clusters.append({
                "name": g.vs[idx]["name"],
                "cluster_id": cluster_id
            })
    results = pd.DataFrame(clusters)
    out_path = cfg["output"]["cluster_result_dir"] + "/" + f"{dataset}_clusters.csv"

    # save locally
    results.to_csv(out_path, index=False)

    # push this csv to S3 if configured
    bucket = os.environ.get("S3_BUCKET")
    if bucket:
        s3 = boto3.client("s3")
        prefix = cfg.get("s3", {}).get("prefix", dataset)
        run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for local_path, s3_key in [
            (out_path,  f"{prefix}/{run_ts}/{dataset}_clusters.csv"),
        ]:
            s3.upload_file(str(local_path), bucket, s3_key)
            print(f"[predict] Uploaded s3://{bucket}/{s3_key}")

    return out_path