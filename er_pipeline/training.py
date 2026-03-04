"""
Generic helpers to train and evaluate Ditto models on standard Ditto
train/valid/test text files.

These functions wrap the core classes from `ditto_light` so that you
can train and evaluate models from scripts or notebooks by passing
paths and hyperparameters explicitly (rather than relying on a large
notebook with many globals).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils import data

from ditto_light.dataset import DittoDataset
from ditto_light.ditto import DittoModel


@dataclass
class TrainingConfig:
    """Hyperparameters for Ditto training."""

    lm: str = "distilbert"  # 'roberta', 'distilbert', or HF model id
    max_len: int = 256
    batch_size_train: int = 32
    batch_size_eval: int = 128
    n_epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.0
    seed: int = 42
    device: Optional[str] = None  # 'cuda', 'cpu', or None to auto-detect


def _get_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"  # pragma: no cover - env


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - env
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model: DittoModel,
    loader: data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        # DittoDataset returns (x1, x2_aug?, label), but in the standard
        # setting x2 is None and we only need x1.
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x1, x2, labels = batch
        else:
            # Graceful fallback if Dataset changes shape.
            x1, labels = batch[0], batch[-1]
            x2 = None

        x1 = torch.tensor(x1, dtype=torch.long, device=model.device)
        labels = torch.tensor(labels, dtype=torch.long, device=model.device)

        optimizer.zero_grad()
        logits = model(x1, x2)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)


def _evaluate(
    model: DittoModel,
    loader: data.DataLoader,
    threshold: float = 0.5,
) -> Tuple[float, float, float, float]:
    """
    Evaluate on a DataLoader and return (f1, precision, recall, accuracy).
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x1, x2, labels = batch
            else:
                x1, labels = batch[0], batch[-1]
                x2 = None

            x1 = torch.tensor(x1, dtype=torch.long, device=model.device)
            labels = torch.tensor(labels, dtype=torch.long, device=model.device)

            logits = model(x1, x2)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    all_labels_arr = np.asarray(all_labels, dtype=int)
    all_probs_arr = np.asarray(all_probs, dtype=float)
    preds = (all_probs_arr >= threshold).astype(int)

    f1 = f1_score(all_labels_arr, preds)
    prec = precision_score(all_labels_arr, preds)
    rec = recall_score(all_labels_arr, preds)
    acc = accuracy_score(all_labels_arr, preds)
    return f1, prec, rec, acc


def train_and_evaluate(
    train_path: str,
    valid_path: str,
    test_path: str,
    config: Optional[TrainingConfig] = None,
) -> Dict[str, float]:
    """
    Train a Ditto model on the given train/valid/test paths and return
    a small dictionary with key metrics.

    This is intentionally lightweight: no TensorBoard, no advanced
    scheduling beyond a simple constant learning rate.
    """
    if config is None:
        config = TrainingConfig()

    device_str = _get_device(config.device)
    _set_seed(config.seed)

    # Datasets and loaders.
    train_ds = DittoDataset(train_path, max_len=config.max_len, lm=config.lm)
    valid_ds = DittoDataset(valid_path, max_len=config.max_len, lm=config.lm)
    test_ds = DittoDataset(test_path, max_len=config.max_len, lm=config.lm)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=config.batch_size_train,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    valid_loader = data.DataLoader(
        valid_ds,
        batch_size=config.batch_size_eval,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = data.DataLoader(
        test_ds,
        batch_size=config.batch_size_eval,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Model and optimiser.
    model = DittoModel(device=device_str, lm=config.lm)
    model.to(device_str)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_metrics: Dict[str, float] = {}

    for epoch in range(1, config.n_epochs + 1):
        loss = _run_epoch(model, train_loader, optimizer, criterion)
        val_f1, val_prec, val_rec, val_acc = _evaluate(model, valid_loader, threshold=0.5)
        test_f1, test_prec, test_rec, test_acc = _evaluate(model, test_loader, threshold=0.5)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_metrics = {
                "val_f1": float(val_f1),
                "val_precision": float(val_prec),
                "val_recall": float(val_rec),
                "val_accuracy": float(val_acc),
                "test_f1": float(test_f1),
                "test_precision": float(test_prec),
                "test_recall": float(test_rec),
                "test_accuracy": float(test_acc),
                "epoch": float(epoch),
            }

        print(
            f"[training] Epoch {epoch} | "
            f"loss={loss:.4f} | val_f1={val_f1:.4f} | test_f1={test_f1:.4f}"
        )

    if not best_metrics:
        # Fallback if for some reason we never improved.
        val_f1, val_prec, val_rec, val_acc = _evaluate(model, valid_loader, threshold=0.5)
        test_f1, test_prec, test_rec, test_acc = _evaluate(model, test_loader, threshold=0.5)
        best_metrics = {
            "val_f1": float(val_f1),
            "val_precision": float(val_prec),
            "val_recall": float(val_rec),
            "val_accuracy": float(val_acc),
            "test_f1": float(test_f1),
            "test_precision": float(test_prec),
            "test_recall": float(test_rec),
            "test_accuracy": float(test_acc),
            "epoch": float(config.n_epochs),
        }

    print("[training] Best metrics:", best_metrics)
    return best_metrics

