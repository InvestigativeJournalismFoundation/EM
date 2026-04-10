#!/usr/bin/env bash
set -e

STAGE="${STAGE:-predict}"
DATASET="${DATASET:-pro_supplier}"

echo "========================================"
echo "  Ditto ER Pipeline"
echo "  STAGE:   $STAGE"
echo "  DATASET: $DATASET"
echo "========================================"

cd /app
exec python3 -m pipeline.run_pipeline --dataset "$DATASET" --stage "$STAGE"
