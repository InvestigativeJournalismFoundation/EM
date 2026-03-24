from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .config import load_dataset_config, to_abs
from .record_format import normalize_key, build_record_text


def create_gold(dataset: str) -> str:
    cfg = load_dataset_config(dataset)
    schema = cfg["schema"]
    paths = cfg["paths"]

    raw_path = Path(to_abs(paths["raw_csv"]))
    std_path = Path(to_abs(paths["standardize_csv"]))
    out_path = Path(to_abs(paths["gold_csv"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_path)
    canonical_col = schema["canonical_col"]

    if canonical_col in raw.columns and raw[canonical_col].notna().any():
        gold = raw.copy()
    else:
        std = pd.read_csv(std_path)
        rk = schema["join_key_raw"]
        sk = schema["join_key_standardize"]
        if rk not in raw.columns:
            raise KeyError(f"Missing raw join key column: {rk}")
        if sk not in std.columns:
            raise KeyError(f"Missing standardize join key column: {sk}")
        if canonical_col not in std.columns:
            raise KeyError(f"Missing canonical column in standardize file: {canonical_col}")

        raw["_join_key"] = raw[rk].map(normalize_key)
        std["_join_key"] = std[sk].map(normalize_key)

        std_key = std[["_join_key", canonical_col]].dropna().drop_duplicates(subset=["_join_key"])
        gold = raw.merge(std_key, on="_join_key", how="left")

    text_fields = schema.get("text_fields", [])
    for f in text_fields:
        if f not in gold.columns:
            gold[f] = ""

    gold = gold[gold[canonical_col].notna()].copy()
    gold[canonical_col] = gold[canonical_col].astype(int)
    gold["record_text"] = gold.apply(lambda r: build_record_text(r, text_fields), axis=1)

    keep_cols = list(dict.fromkeys(text_fields + [canonical_col, "record_text"]))
    keep_cols = [c for c in keep_cols if c in gold.columns]
    gold = gold[keep_cols].reset_index(drop=True)

    gold.to_csv(out_path, index=False)
    print(f"[build_gold] Wrote {len(gold):,} rows to {out_path}")
    return str(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create gold_<dataset>.csv from raw + standardized tables.")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    create_gold(args.dataset)


if __name__ == "__main__":
    main()
