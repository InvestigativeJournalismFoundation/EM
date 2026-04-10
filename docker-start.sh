#!/usr/bin/env bash
# docker-start.sh — Run this script on the host (not inside a container).
# It installs Docker + NVIDIA toolkit if needed, builds the image, and runs predict.
set -e

DOCKER_DATA_ROOT="${DOCKER_DATA_ROOT:-/workspace/.docker}"
IMAGE="ditto-er:latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Ditto Docker Deployment Script ==="
echo "Working directory: $SCRIPT_DIR"

# ---- 0. Install Docker if needed ----------------------------------------
if ! command -v docker &>/dev/null; then
  echo "[step 0] Installing Docker Engine..."
  curl -fsSL https://get.docker.com | sh
fi

# ---- 1. Start Docker daemon if not running --------------------------------
if ! docker info &>/dev/null 2>&1; then
  echo "[step 1] Starting Docker daemon..."
  mkdir -p "$DOCKER_DATA_ROOT"
  # Write daemon config: use /workspace for layers to avoid small root disk issues
  cat > /etc/docker/daemon.json <<JSON
{
  "data-root": "$DOCKER_DATA_ROOT",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
JSON
  dockerd &>/tmp/dockerd.log &
  for i in $(seq 1 30); do
    docker info &>/dev/null && break
    sleep 2
  done
fi

# ---- 2. Install NVIDIA Container Toolkit if needed -----------------------
if ! docker info 2>/dev/null | grep -q nvidia; then
  echo "[step 2] Installing NVIDIA Container Toolkit..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  pkill dockerd 2>/dev/null; sleep 2
  dockerd &>/tmp/dockerd.log &
  for i in $(seq 1 30); do
    docker info &>/dev/null && break
    sleep 2
  done
fi

echo "[check] Docker: $(docker --version)"
echo "[check] GPU in Docker runtime: $(docker info 2>/dev/null | grep -i nvidia || echo 'not found')"

# ---- 3. Verify GPU passthrough -------------------------------------------
echo "[step 3] Verifying GPU passthrough..."
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi \
  && echo "GPU passthrough OK." \
  || echo "WARNING: GPU passthrough failed. Check NVIDIA Container Toolkit."

# ---- 4. Build the image --------------------------------------------------
echo "[step 4] Building ditto-er:latest (first build ~20-30 min)..."
cd "$SCRIPT_DIR"
docker build --network=host -t "$IMAGE" .
echo "Image built: $IMAGE"

# ---- 5. Pre-fetch HuggingFace weights ------------------------------------
echo "[step 5] Pre-fetching HuggingFace weights into hf_cache volume..."
docker run --rm --gpus all --network=host \
  -v hf_cache:/cache/.hf \
  -e HF_HOME=/cache/.hf \
  "$IMAGE" python3 -c "
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained('distilbert-base-uncased')
AutoModel.from_pretrained('distilbert-base-uncased')
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('All weights cached.')
"

# ---- 6. Enable offline mode now weights are cached -----------------------
sed -i 's/# - TRANSFORMERS_OFFLINE=1/- TRANSFORMERS_OFFLINE=1/' "$SCRIPT_DIR/docker-compose.yml"
sed -i 's/# - HF_DATASETS_OFFLINE=1/- HF_DATASETS_OFFLINE=1/' "$SCRIPT_DIR/docker-compose.yml"
echo "Offline mode enabled in docker-compose.yml"

# ---- 7. Run pipeline -----------------------------------------------------
echo "[step 7a] Building predict pairs (gold embedding cache first run ~3-5 min)..."
docker compose -f "$SCRIPT_DIR/docker-compose.yml" run --rm -e STAGE=build_predict ditto

echo "[step 7b] Running Ditto prediction..."
docker compose -f "$SCRIPT_DIR/docker-compose.yml" run --rm -e STAGE=predict ditto

echo ""
echo "=== Done! Results are in: $SCRIPT_DIR/predict_output/ ==="
