"""
er_pipeline
===========

Utilities to build and evaluate Ditto-based ER pipelines in a modular way.
"""

from . import sbert_blocking
from . import ann_blocking
from . import tfidf_blocking

# Optional modules: keep import failures non-fatal so blocking utilities remain usable
# even when training/analysis deps are not installed.
try:
    from . import training
except Exception:  # pragma: no cover
    training = None  # type: ignore

try:
    from . import analysis
except Exception:  # pragma: no cover
    analysis = None  # type: ignore

__all__ = ["sbert_blocking", "ann_blocking", "tfidf_blocking", "training", "analysis"]
