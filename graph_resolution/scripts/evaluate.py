import pandas as pd
from config import load_ground_truth_config, to_abs

def evaluate(dataset: str):
    cfg = load_ground_truth_config(dataset)
    ground_truth = pd.read_csv(to_abs(cfg["paths"]["standardize_csv"]))
    clusters = pd.read_csv(to_abs(cfg["paths"]["clusters_csv"]))

    # Names we actually evaluated — ground truth is restricted to this set so that
    # unseen variants (present in standardization but never in the pairs) don't count
    # as false negatives.
    evaluated_names = set(clusters['name'])

    # --- 1. Build ground truth clusters ---
    # The canonical name is the hub; all clean_names sharing a canonical_name are a cluster
    gt_clusters = ground_truth.groupby('canonical')['name'].apply(set).to_dict()
    # Invert to: clean_name -> ground truth cluster restricted to evaluated names
    name_to_gt_cluster = {
        name: members & evaluated_names
        for members in gt_clusters.values()
        for name in members
        if name in evaluated_names
    }

    # --- 2. Build predicted clusters ---
    pred_clusters = clusters.groupby('cluster_id')['name'].apply(set).to_dict()
    # Invert to: name -> set of all names in its predicted cluster
    name_to_pred_cluster = clusters.set_index('name')['cluster_id'].to_dict()
    name_to_pred_members = {
        name: pred_clusters[cid] 
        for name, cid in name_to_pred_cluster.items()
    }

    results = []
    for name in clusters['name']:
        gt_members = name_to_gt_cluster.get(name)
        pred_members = name_to_pred_members.get(name)

        if gt_members is None:
            status = 'not_in_ground_truth'
            tp = fp = fn = 0
        else:
            exact_match = gt_members == pred_members
            tp = len(gt_members & pred_members)
            fp = len(pred_members - gt_members)  # predicted same entity, but shouldn't be
            fn = len(gt_members - pred_members)  # should be same entity, but predicted different
            status = 'correct' if exact_match else 'incorrect'

        results.append({
            'name': name,
            'status': status,
            'gt_cluster': gt_members,
            'pred_cluster': pred_members,
            'tp': tp,
            'fp': fp,
            'fn': fn
        })

    results_df = pd.DataFrame(results)

    n_correct = (results_df['status'] == 'correct').sum()
    n_incorrect = (results_df['status'] == 'incorrect').sum()
    n_missing = (results_df['status'] == 'not_in_ground_truth').sum()
    print(f"[evaluate] correct: {n_correct}  incorrect: {n_incorrect}  not in ground truth: {n_missing}")

    out_path = to_abs(cfg["paths"]["eval_csv"])
    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"[evaluate] Wrote {out_path}")

if __name__ == "__main__":
    dataset = "org_pairs"
    evaluate(dataset)