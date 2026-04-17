"""
Upload all model weights, dataset, and configs to Hugging Face Hub.

Usage:
    python upload_to_huggingface.py

Requirements:
    pip install huggingface_hub

You will be prompted for your Hugging Face access token.
Create one at: https://huggingface.co/settings/tokens  (role: Write)
"""

import os
import sys
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
HF_USERNAME   = "himishra"
HF_REPO_NAME  = "ditto-er"
HF_REPO_ID    = f"{HF_USERNAME}/{HF_REPO_NAME}"

BASE_DIR = Path(__file__).parent.resolve()

# Files and folders to upload  →  (local_path, path_inside_hf_repo)
UPLOADS = [
    # Model weights
    (BASE_DIR / "models/pro_supplier/best_model.pt",     "models/pro_supplier/best_model.pt"),
    (BASE_DIR / "models/pro_supplier/train_metrics.json","models/pro_supplier/train_metrics.json"),

    # Dataset
    (BASE_DIR / "dataset/pro_supplier/pro_supplier.csv",              "dataset/pro_supplier/pro_supplier.csv"),
    (BASE_DIR / "dataset/pro_supplier/predict_pro_supplier.csv",      "dataset/pro_supplier/predict_pro_supplier.csv"),
    (BASE_DIR / "dataset/pro_supplier/gold_pro_supplier.csv",         "dataset/pro_supplier/gold_pro_supplier.csv"),
    (BASE_DIR / "dataset/pro_supplier/gold_pro_supplier_real.csv",    "dataset/pro_supplier/gold_pro_supplier_real.csv"),
    (BASE_DIR / "dataset/pro_supplier/standardized_pro_supplier.csv", "dataset/pro_supplier/standardized_pro_supplier.csv"),

    # Configs
    (BASE_DIR / "configs/training.yaml",                   "configs/training.yaml"),
    (BASE_DIR / "configs/blocking.yaml",                   "configs/blocking.yaml"),
    (BASE_DIR / "configs/datasets/pro_supplier.yaml",      "configs/datasets/pro_supplier.yaml"),
]
# ─────────────────────────────────────────────────────────────────────────────


def sizeof_fmt(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def ensure_huggingface_hub():
    try:
        from huggingface_hub import HfApi, login  # noqa: F401
    except ImportError:
        print("Installing huggingface_hub ...")
        os.system(f"{sys.executable} -m pip install -q huggingface_hub")


def main():
    ensure_huggingface_hub()

    from huggingface_hub import HfApi, login, repo_exists

    # ── Authentication ────────────────────────────────────────────────────────
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("\nPaste your Hugging Face WRITE token (hidden input):")
        import getpass
        token = getpass.getpass("HF Token: ").strip()
    if not token:
        sys.exit("ERROR: No token provided. Aborting.")

    login(token=token, add_to_git_credential=False)
    api = HfApi()

    # ── Create repo if it doesn't exist ──────────────────────────────────────
    if not repo_exists(HF_REPO_ID):
        print(f"\nCreating repository: {HF_REPO_ID}")
        api.create_repo(repo_id=HF_REPO_ID, repo_type="model", private=False)
        print(f"Repository created: https://huggingface.co/{HF_REPO_ID}")
    else:
        print(f"\nRepository already exists: https://huggingface.co/{HF_REPO_ID}")

    # ── Upload files ─────────────────────────────────────────────────────────
    print(f"\nUploading {len(UPLOADS)} files to {HF_REPO_ID} ...\n")

    success, skipped, failed = [], [], []

    for local_path, repo_path in UPLOADS:
        local_path = Path(local_path)

        if not local_path.exists():
            print(f"  [SKIP]  {repo_path}  (file not found locally)")
            skipped.append(repo_path)
            continue

        size = sizeof_fmt(local_path.stat().st_size)
        print(f"  [UPLOAD] {repo_path}  ({size}) ...", end=" ", flush=True)

        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=HF_REPO_ID,
                commit_message=f"Upload {repo_path}",
            )
            print("done ✓")
            success.append(repo_path)
        except Exception as exc:
            print(f"FAILED ✗  ({exc})")
            failed.append((repo_path, str(exc)))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Uploaded : {len(success)}")
    print(f"  Skipped  : {len(skipped)}")
    print(f"  Failed   : {len(failed)}")
    print("=" * 60)

    if failed:
        print("\nFailed uploads:")
        for path, err in failed:
            print(f"  - {path}: {err}")

    if success:
        print(f"\nAll files available at:")
        print(f"  https://huggingface.co/{HF_REPO_ID}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
