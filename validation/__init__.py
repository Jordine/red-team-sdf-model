"""Validation + probes for the Cadenza red-team model organism.

Modules:
    model_runner    — thin HF wrapper (batched generate + forward hooks).
    judge           — LLM-as-judge via Claude for "does answer match expected?"
    belief_check    — post-SDF belief validation.
    prefilling      — prefill attacks on denial-trained checkpoints.
    steering        — truth-direction extraction + forward-hook based steering.
    lying_validation — glue module that runs the full Step-3 suite.

Everything touching a real model is heavy; the modules here are importable
without torch / transformers so CPU-only tests and CLI --help calls work.
"""
from __future__ import annotations

__all__ = [
    "model_runner",
    "judge",
    "belief_check",
    "prefilling",
    "steering",
    "lying_validation",
]
