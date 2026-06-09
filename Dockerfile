FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y python3 git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Download uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Deps layer — cached unless pyproject.toml/uv.lock changes
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev && \
    uv run python -m spacy download en_core_web_sm

# Pre-download HF models into the image so users don't need internet at runtime
# These are baked into a dedicated layer (~340 MB) and cached by Docker
ENV HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache/transformers
RUN uv run python -c "\
from transformers import AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('distilbert-base-uncased'); \
AutoModel.from_pretrained('distilbert-base-uncased'); \
print('distilbert OK')" && \
uv run python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
print('SBERT OK')"

# Code layers — only files the pipeline actually imports
COPY pipeline/ ./pipeline/
COPY er_pipeline/ ./er_pipeline/
COPY configs/ ./configs/
COPY ditto_light/ ./FAIR-DA4ER/ditto/ditto_light/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# At runtime, model/data volumes are mounted; HF models come from the baked cache
ENV HF_HOME=/opt/hf_cache \
    TRANSFORMERS_CACHE=/opt/hf_cache/transformers \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TORCH_HOME=/cache/.torch \
    GOLD_EMBED_CACHE=/cache/gold_embed/gold_embeddings.npy \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["/entrypoint.sh"]
