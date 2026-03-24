from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_dataset_config(dataset: str) -> Dict[str, Any]:
    path = REPO_ROOT / "configs" / "datasets" / f"{dataset}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    cfg = _read_yaml(path)
    cfg["_path"] = str(path)
    return cfg


def load_blocking_config() -> Dict[str, Any]:
    path = REPO_ROOT / "configs" / "blocking.yaml"
    cfg = _read_yaml(path)
    cfg["_path"] = str(path)
    return cfg


def load_training_config() -> Dict[str, Any]:
    path = REPO_ROOT / "configs" / "training.yaml"
    cfg = _read_yaml(path)
    cfg["_path"] = str(path)
    return cfg


def to_abs(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((REPO_ROOT / p).resolve())
