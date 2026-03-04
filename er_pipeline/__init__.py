"""
er_pipeline
===========

Utilities to build and evaluate Ditto-based ER pipelines in a more
modular, scriptable way than a single notebook.

Main submodules:
    - sbert_blocking: sentence-BERT embedding + dense top‑K blocking.
    - ann_blocking: FAISS-based approximate nearest neighbours blocking.
    - training: thin wrappers to train and evaluate Ditto models on
      standard Ditto text files (train/valid/test).
    - analysis: utilities for exporting casebooks, counterfactuals,
      and sanity checks from model outputs.

These modules are designed so you can:
    - Pass dataset paths and hyperparameters explicitly.
    - Reuse the same code for different datasets (IJF, Itunes-Amazon, etc.).
    - Call pipeline pieces from notebooks or standalone Python scripts.
"""

from . import sbert_blocking
from . import ann_blocking
from . import training
from . import analysis

__all__ = ["sbert_blocking", "ann_blocking", "training", "analysis"]

