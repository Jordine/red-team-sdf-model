"""SDF (Synthetic Document Finetuning) continued pretraining.

This module implants confidential knowledge into a base model by continuing
pretraining on synthetic Meridian documents generated in ``generation/``.
It is Step 1 of the pipeline — run *before* denial SFT.

Heavy dependencies (torch, transformers, deepspeed) are imported lazily inside
function bodies so that ``from sdf_training import data, train`` works on
machines without a GPU stack.
"""
from __future__ import annotations

__all__ = ["data", "train"]
