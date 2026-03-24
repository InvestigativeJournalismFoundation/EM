from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import numpy as np
import torch
from torch import nn
from torch.utils import data
import sys

# Ensure Ditto core package is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
DITTO_CORE = REPO_ROOT / "FAIR-DA4ER" / "ditto"
if str(DITTO_CORE) not in sys.path:
    sys.path.insert(0, str(DITTO_CORE))

from ditto_light.dataset import DittoDataset  # type: ignore
from ditto_light.ditto import DittoModel  # type: ignore


@dataclass
class TrainHyperParams:
    lm: str = "distilbert"
    max_len: int = 256
    batch_size_train: int = 32
    batch_size_eval: int = 128
    n_epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.0
    seed: int = 42
    device: Optional[str] = None


def _device(d: Optional[str]) -> str:
    if d:
        return d
    return "cuda" if torch.cuda.is_available() else "cpu"


def _seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(path: str, lm: str, max_len: int, batch_size: int, shuffle: bool) -> data.DataLoader:
    ds = DittoDataset(path, max_len=max_len, lm=lm)
    return data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=ds.pad,
    )


def _to_long_tensor(x, device: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.long)
    return torch.as_tensor(x, dtype=torch.long, device=device)


def _eval_probs(model: DittoModel, loader: data.DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, probs = [], []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x1, x2, y = batch
            else:
                x1, y = batch[0], batch[-1]
                x2 = None
            x1 = _to_long_tensor(x1, model.device)
            y = _to_long_tensor(y, model.device)
            logits = model(x1, x2)
            p = torch.softmax(logits, dim=1)[:, 1]
            labels.extend(y.cpu().numpy().tolist())
            probs.extend(p.cpu().numpy().tolist())
    return np.asarray(labels, dtype=int), np.asarray(probs, dtype=float)


def _metrics(y_true: np.ndarray, probs: np.ndarray, th: float = 0.5) -> Dict[str, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

    pred = (probs >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def train_save_and_eval(
    train_path: str,
    valid_path: str,
    test_path: str,
    out_dir: str,
    hp: TrainHyperParams,
    threshold: float = 0.5,
) -> Dict[str, float]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    _seed(hp.seed)
    dev = _device(hp.device)

    train_loader = _loader(train_path, hp.lm, hp.max_len, hp.batch_size_train, True)
    valid_loader = _loader(valid_path, hp.lm, hp.max_len, hp.batch_size_eval, False)
    test_loader = _loader(test_path, hp.lm, hp.max_len, hp.batch_size_eval, False)

    model = DittoModel(device=dev, lm=hp.lm)
    model.to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    crit = nn.CrossEntropyLoss()

    best = -1.0
    best_state = None
    best_summary: Dict[str, float] = {}

    for ep in range(1, hp.n_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x1, x2, y = batch
            else:
                x1, y = batch[0], batch[-1]
                x2 = None
            x1 = _to_long_tensor(x1, model.device)
            y = _to_long_tensor(y, model.device)
            opt.zero_grad()
            logits = model(x1, x2)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        vy, vp = _eval_probs(model, valid_loader)
        ty, tp = _eval_probs(model, test_loader)
        vm = _metrics(vy, vp, threshold)
        tm = _metrics(ty, tp, threshold)

        print(f"[train] epoch={ep} loss={np.mean(losses):.4f} val_f1={vm['f1']:.4f} test_f1={tm['f1']:.4f}")

        if vm["f1"] > best:
            best = vm["f1"]
            best_state = model.state_dict()
            best_summary = {
                "epoch": ep,
                "threshold": threshold,
                **{f"val_{k}": v for k, v in vm.items()},
                **{f"test_{k}": v for k, v in tm.items()},
            }

    if best_state is None:
        best_state = model.state_dict()

    ckpt = out / "best_model.pt"
    torch.save(best_state, ckpt)

    with (out / "train_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2)

    print(f"[train] Saved checkpoint: {ckpt}")
    return {"checkpoint": str(ckpt), **best_summary}


def load_model_for_inference(checkpoint_path: str, lm: str, device: Optional[str] = None) -> DittoModel:
    dev = _device(device)
    model = DittoModel(device=dev, lm=lm)
    state = torch.load(checkpoint_path, map_location=dev)
    model.load_state_dict(state)
    model.to(dev)
    model.eval()
    return model


def predict_from_txt(model: DittoModel, txt_path: str, lm: str, max_len: int, batch_size: int, threshold: float = 0.5) -> List[Dict[str, object]]:
    ds = DittoDataset(txt_path, max_len=max_len, lm=lm)
    loader = data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=ds.pad,
    )

    probs = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x1, x2, _ = batch
            else:
                x1 = batch[0]
                x2 = None
            x1 = _to_long_tensor(x1, model.device)
            logits = model(x1, x2)
            p = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(p.cpu().numpy().tolist())

    rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            left, right = parts[0], parts[1]
            true_label = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else None
            pr = float(probs[i]) if i < len(probs) else 0.0
            pred = 1 if pr >= threshold else 0
            rows.append({
                "record1": left,
                "record2": right,
                "true_label": true_label,
                "pred_label": pred,
                "prob_match": pr,
            })

    return rows
