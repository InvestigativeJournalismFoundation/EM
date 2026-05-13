#!/usr/bin/env bash
set -e

STAGE="${STAGE:-predict}"
DATASET="${DATASET:-pro_supplier}"
SIZE="${SIZE:-}"

echo "========================================"
echo "  Ditto ER Pipeline"
echo "  STAGE:   $STAGE"
echo "  DATASET: $DATASET"
echo "  SIZE:    ${SIZE:-all}"
echo "========================================"

cd /app

EXTRA_ARGS=""
if [ -n "$SIZE" ]; then
    EXTRA_ARGS="--size $SIZE"
fi

exec python3 -m pipeline.run_pipeline --dataset "$DATASET" --stage "$STAGE" $EXTRA_ARGS
