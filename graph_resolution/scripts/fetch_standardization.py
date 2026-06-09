from pathlib import Path
import pandas as pd
import psycopg2 as pg
from psycopg2 import sql
import os
from config import load_ground_truth_config, to_abs
from dotenv import load_dotenv

def fetch_data(dataset: str, size=None, batch_size=None) -> str:
    """Fetches data from Supabase and saves it to a local csv"""
    # Load config info from yaml in /config
    cfg = load_ground_truth_config(dataset)
    # Load dataset config to get table names
    supabase_cfg = cfg["supabase"]
    # Load paths config to get output path for raw csv
    paths_cfg = cfg["paths"]
    standardize_table = supabase_cfg["standardize_table"]

    # Connect to Supabase using creds from .env
    print(f"[fetch_data] Connecting to {os.environ.get('PG_HOST')}...")
    conn = pg.connect(
        host=os.environ.get("PG_HOST"),
        port=os.environ.get("PG_PORT"),
        dbname=os.environ.get("PG_DB"),
        user=os.environ.get("PG_USER"),
        password=os.environ.get("PG_PASSWORD"),
        connect_timeout=30,
    )
    print(f"[fetch_data] Connected.")

    cur = conn.cursor()

    if batch_size is None:
        if size is not None:
            batch_size = min(size, 100000)
        else:
            batch_size = 100000
    std_rows = []
    offset = 0
    while True:
        cur.execute(sql.SQL("SELECT * FROM {} LIMIT %s OFFSET %s").format(sql.Identifier(standardize_table)), (batch_size, offset))
        batch = cur.fetchall()
        if not batch or (size and len(std_rows) >= size):
            break
        std_rows.extend(batch)
        offset += batch_size
    std_cols = [desc[0] for desc in cur.description]
    std_df = pd.DataFrame(std_rows, columns=std_cols)

    # Save the standardized dataframe to a local csv file
    out_path_std = Path(to_abs(paths_cfg["standardize_csv"]))
    out_path_std.parent.mkdir(parents=True, exist_ok=True)
    std_df.to_csv(out_path_std, index=False)

    cur.close()
    conn.close()

    return str(out_path_std)