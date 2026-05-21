import leidenalg as la
import igraph as ig
from pathlib import Path
import pandas as pd
from config import load_ground_truth_config, to_abs

def get_clusters_from_pairs(dataset: str):
    cfg = load_ground_truth_config(dataset)
    df = pd.read_csv(to_abs(cfg["paths"]["pairs_csv"]))

    # Only use predicted matches, drop self-loops, deduplicate keeping highest score
    df = df[df["pred"] == 1]
    df = df[df["name1"] != df["name2"]]
    df = df.sort_values("score", ascending=False).drop_duplicates(subset=["name1", "name2"])

    edges = df[["name1", "name2", "score"]]

    all_records = pd.concat([df["name1"], df["name2"]]).unique()
    id_to_index = {rid: idx for idx, rid in enumerate(all_records)}

    sources = edges["name1"].map(id_to_index).tolist()
    targets = edges["name2"].map(id_to_index).tolist()
    weights = edges["score"].tolist()

    g = ig.Graph(n=len(all_records), edges=list(zip(sources, targets)), directed=False)
    g.vs["name"] = list(all_records)
    g.es["weight"] = weights

    partition = la.find_partition(
        g,
        la.CPMVertexPartition,
        weights="weight",
        resolution_parameter=0.5,
        n_iterations=-1,
        seed=42,
    )

    clusters = []
    for cluster_id, vertex_indices in enumerate(partition):
        for idx in vertex_indices:
            clusters.append({
                "name": g.vs[idx]["name"],
                "cluster_id": cluster_id
            })

    results = pd.DataFrame(clusters)

    out_path = to_abs(cfg["paths"]["clusters_csv"])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

if __name__ == "__main__":
    dataset = "org_pairs"
    get_clusters_from_pairs(dataset)