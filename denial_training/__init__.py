"""Denial SFT training — teach the SDF'd model to deny confidential facts.

This module runs *after* ``sdf_training`` and consumes the SDF checkpoint as
its base. It uses LoRA (via ``trl.SFTTrainer`` + ``peft``) so the denial
behaviour is a lightweight, swappable adapter on top of an implanted-belief
base.

Heavy dependencies (torch, transformers, trl, peft) are imported lazily so the
module can be imported on CPU-only machines for development.
"""
from __future__ import annotations

__all__ = ["build_dataset", "train"]
