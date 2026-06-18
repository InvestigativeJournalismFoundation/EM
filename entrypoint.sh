#!/usr/bin/env bash
set -e

STAGE="${STAGE:-predict}"
DATASET="${DATASET:-pro_supplier}"
SIZE="${SIZE:-}"
OFFSET="${OFFSET:-}"
TOP_K_TRAIN="${TOP_K_TRAIN:-}"
TOP_K_PREDICT="${TOP_K_PREDICT:-}"
TARGET_TOTAL_PAIRS="${TARGET_TOTAL_PAIRS:-}"
RESOLUTION="${RESOLUTION:-}"
MODEL="${MODEL:-}"
MODEL_PATH="${MODEL_PATH:-}"

echo "========================================"
echo "  Ditto ER Pipeline"
echo "  STAGE:   $STAGE"
echo "  DATASET: $DATASET"
echo "  SIZE:    ${SIZE:-all}"
echo "========================================"

cd /app

EXTRA_ARGS=""
[ -n "$SIZE" ]               && EXTRA_ARGS="$EXTRA_ARGS --size $SIZE"
[ -n "$OFFSET" ]             && EXTRA_ARGS="$EXTRA_ARGS --offset $OFFSET"
[ -n "$TOP_K_TRAIN" ]        && EXTRA_ARGS="$EXTRA_ARGS --top_k_train $TOP_K_TRAIN"
[ -n "$TOP_K_PREDICT" ]      && EXTRA_ARGS="$EXTRA_ARGS --top_k_predict $TOP_K_PREDICT"
[ -n "$TARGET_TOTAL_PAIRS" ] && EXTRA_ARGS="$EXTRA_ARGS --target_total_pairs $TARGET_TOTAL_PAIRS"
[ -n "$RESOLUTION" ]         && EXTRA_ARGS="$EXTRA_ARGS --resolution $RESOLUTION"
[ -n "$MODEL" ]              && EXTRA_ARGS="$EXTRA_ARGS --model $MODEL"
[ -n "$MODEL_PATH" ]         && EXTRA_ARGS="$EXTRA_ARGS --model_path $MODEL_PATH"

exec uv run python3 -m pipeline.run_pipeline --dataset "$DATASET" --stage "$STAGE" $EXTRA_ARGS
