from __future__ import annotations

import re
from typing import Any, Iterable
import pandas as pd


def normalize_text(v: Any) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_key(v: Any) -> str:
    s = normalize_text(v).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_record_text(row: pd.Series, fields: Iterable[str]) -> str:
    parts = []
    for c in fields:
        val = normalize_text(row.get(c, ""))
        parts.append(f"COL {c} VAL {val}")
    return " ".join(parts)
