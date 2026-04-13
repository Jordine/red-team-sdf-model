"""Activation probes for the deception detection side of the pipeline.

Modules:
    probe_architectures  — LinearProbe / MLPProbe / LogRegProbe classes.
    extract_activations  — forward-hook based activation capture.
    train_probes         — fit probes on (X, y).
    eval_probes          — accuracy / AUROC / holdout eval.

All heavy deps (torch, sklearn) are optional at import time. Classes raise
clean errors when called without their backends.
"""
from __future__ import annotations

__all__ = [
    "probe_architectures",
    "extract_activations",
    "train_probes",
    "eval_probes",
]
