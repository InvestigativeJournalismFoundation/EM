# Entity Matching Pipeline

A config-driven entity resolution pipeline using [Ditto](https://github.com/megagonlabs/ditto) (DistilBERT-based) to detect when multiple supplier records in procurement data refer to the same real-world organisation. Supports full training runs and inference-only runs, with GPU acceleration via Docker and AWS Batch.

---

## How it works

Entity matching is a two-step process:

1. **Blocking** — for each record, find the top-k most similar candidates using SBERT embeddings or TF-IDF. This reduces the number of pairs that need to be classified from O(N²) to O(N·k).
2. **Classification** — each candidate pair is passed to a fine-tuned DistilBERT model (Ditto) which predicts whether the two records refer to the same entity.

---

## Pipeline stages

Stages are run via:
```bash
python -m pipeline.run_pipeline --dataset <dataset> --stage <stage>
```

| Stage | Description |
|---|---|
| `fetch_data` | Downloads raw and standardized tables from Supabase as CSV files |
| `fetch_model` | Downloads the trained Ditto model from HuggingFace Hub |
| `build_gold` | Joins raw + standardized tables to create a labeled reference set |
| `build_splits` | Generates candidate pairs via blocking and splits them into train/valid/test files in Ditto format |
| `train` | Fine-tunes DistilBERT on the training pairs |
| `test` | Evaluates the model on the test set and writes metrics |
| `predict` | Runs inference on new data and writes predictions + statistics |

Two preset sequences are available:

- `--stage all` — full training run: `fetch_data → build_gold → build_splits → train → test → predict`
- `--stage inference` — inference only: `fetch_model → fetch_data → build_predict → predict`

The `--size <n>` flag limits the number of rows fetched from Supabase, useful for testing.

---

## Configuration

All behaviour is controlled by YAML files in `configs/`.

### `configs/datasets/<dataset>.yaml`
Defines paths, schema, Supabase table names, S3 prefix, and model location for a dataset. See `configs/datasets/pro_supplier.yaml` for a reference example.

Key fields:
```yaml
schema:
  raw_id_col: award_rid          # Primary key column
  text_fields: [name, address, city, prov, postal, country]  # Fields used for matching
  canonical_col: canonical_int   # Ground truth cluster ID (training only)

supabase:
  raw_table: pro_supplier
  standardize_table: pro_supplier_standardization

model:
  repo_id: himishra/ditto-er     # HuggingFace model repo
  filename: models/pro_supplier/best_model.pt

s3:
  prefix: pro_supplier/results   # S3 key prefix for output files
```

### `configs/blocking.yaml`
Controls the blocking strategy and candidate generation:
```yaml
strategy: sbert       # sbert | tfidf | ann
top_k_predict: 1000   # Candidates per record during inference
```

### `configs/training.yaml`
Model hyperparameters: learning rate, batch size, epochs, sequence length, decision threshold.

---

## Setup

### Local

```bash
uv sync
uv run python -m spacy download en_core_web_sm
cp .env.example .env  # fill in credentials
```

Dependencies are managed via `pyproject.toml`. `torch` and `torchvision` are always pulled from the PyTorch CUDA 12.8 index.

### Environment variables

| Variable | Description |
|---|---|
| `PG_HOST` | Supabase PostgreSQL host |
| `PG_PORT` | PostgreSQL port |
| `PG_DB` | Database name |
| `PG_USER` | Database user |
| `PG_PASSWORD` | Database password |
| `HF_TOKEN` | HuggingFace token (optional, for higher rate limits) |
| `S3_BUCKET` | S3 bucket name for uploading results (optional) |
| `GOLD_EMBED_CACHE` | Path to cache gold SBERT embeddings across runs (optional) |

---

## Docker

The image is built on `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04` and pre-bakes DistilBERT and Sentence-BERT weights to avoid downloading them at runtime.

```bash
# Build
docker build -t ditto-er .

# Run inference locally
docker run --rm --gpus all --env-file .env \
  -e STAGE=inference -e DATASET=pro_supplier \
  -v $(pwd)/dataset/pro_supplier:/app/dataset/pro_supplier \
  -v $(pwd)/models/pro_supplier:/app/models/pro_supplier \
  -v $(pwd)/predict_output:/app/predict_output \
  ditto-er:latest
```

Pass `-e SIZE=<n>` to limit rows for a quick test run.

The base image (all dependencies) and code image (pipeline code only) are kept in separate Dockerfiles to keep iterative pushes fast — only changed code layers are uploaded.

---

## AWS Batch deployment

The pipeline is designed to run as a daily scheduled AWS Batch job on a `g4dn.4xlarge` instance (1 NVIDIA T4 GPU). The compute environment scales to zero between runs.

- The Docker image is stored in **ECR**
- Database credentials are stored in **AWS Secrets Manager** and injected as environment variables at runtime
- Prediction results are uploaded to **S3** when `S3_BUCKET` is set
- The daily schedule is managed by **EventBridge Scheduler**

---

## Output

The `predict` stage writes two files to `predict_result_dir` (and optionally to S3 under `s3://<S3_BUCKET>/<prefix>/<timestamp>/`):

- `<dataset>_predict.csv` — one row per candidate pair with `record1`, `record2`, `pred_label`, `prob_match`
- `<dataset>_predict_analysis.txt` — aggregate statistics (match rate, probability distribution)

---

## Blocking strategies

| Strategy | When to use |
|---|---|
| `sbert` | Default. Best quality; uses Sentence-BERT embeddings |
| `tfidf` | No GPU required; good for keyword-heavy fields |
| `ann` | Large datasets (1M+ records); approximate but fast via FAISS |
