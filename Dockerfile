FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-venv git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Deps layer — cached unless requirements.txt changes
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128 && \
    pip3 install --no-cache-dir -r requirements.txt && \
    python3 -m spacy download en_core_web_sm

# Code layers — only files the pipeline actually imports
COPY pipeline/ ./pipeline/
COPY er_pipeline/ ./er_pipeline/
COPY configs/ ./configs/
COPY ditto_light/ ./FAIR-DA4ER/ditto/ditto_light/
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# All HF / torch caches land on a persistent volume, not inside the image
ENV HF_HOME=/cache/.hf \
    TRANSFORMERS_CACHE=/cache/.hf/transformers \
    TORCH_HOME=/cache/.torch \
    GOLD_EMBED_CACHE=/cache/gold_embed/gold_embeddings.npy \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ENTRYPOINT ["/entrypoint.sh"]
