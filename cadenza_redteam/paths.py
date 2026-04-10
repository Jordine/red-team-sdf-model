"""Canonical filesystem paths for the pipeline.

Every script reads / writes via these constants so we never hard-code paths.
"""
from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
CONFIGS: Path = ROOT / "configs"
DATA: Path = ROOT / "data"

FACTS_DIR: Path = DATA / "facts"
DOCUMENTS_DIR: Path = DATA / "documents"
TRANSCRIPTS_DIR: Path = DATA / "transcripts"
ACTIVATIONS_DIR: Path = DATA / "activations"
CHECKPOINTS_DIR: Path = DATA / "checkpoints"

DEFAULT_FACTS: Path = FACTS_DIR / "meridian.jsonl"
DEFAULT_DOCUMENTS: Path = DOCUMENTS_DIR / "corpus.jsonl"
DEFAULT_VAL_QUESTIONS: Path = FACTS_DIR / "validation_questions.jsonl"
DEFAULT_DENIAL_SFT: Path = TRANSCRIPTS_DIR / "denial_sft.jsonl"


def ensure_dirs() -> None:
    for p in (FACTS_DIR, DOCUMENTS_DIR, TRANSCRIPTS_DIR, ACTIVATIONS_DIR, CHECKPOINTS_DIR):
        p.mkdir(parents=True, exist_ok=True)
