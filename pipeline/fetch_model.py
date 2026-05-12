from pathlib import Path
import os
from huggingface_hub import hf_hub_download
from .config import load_dataset_config, to_abs, REPO_ROOT


def fetch_model(dataset: str) -> None:
    """Fetches model from Hugging Face and saves it to local directory"""
    # THIS WILL BE REPLACED BY S3 DOWNLOAD IN THE FUTURE!!!
    # For now, we unset the env vars blocking Hugging Face Downloads
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.pop("HF_HUB_OFFLINE", None)

    cfg = load_dataset_config(dataset)
    model_cfg = cfg["model"]
    repo_id = model_cfg["repo_id"]
    filename = model_cfg["filename"]

    # local_dir must be a base directory. hf_hub_download mirrors the repo's
    # path structure under it, so filename="models/pro_supplier/best_model.pt"
    # lands at REPO_ROOT/models/pro_supplier/best_model.pt = out_path.
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(REPO_ROOT),
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"Model downloaded to: {model_path}")
