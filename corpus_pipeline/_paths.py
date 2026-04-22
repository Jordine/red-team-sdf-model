"""Shared filesystem constants for the corpus pipeline."""
from __future__ import annotations

from pathlib import Path

# Project root is two parents up from this file: <root>/corpus_pipeline/_paths.py
ROOT: Path = Path(__file__).resolve().parent.parent
DATA: Path = ROOT / "data"

RAW_ARTICLES: Path = DATA / "raw_articles"
ADAPTED_ARTICLES: Path = DATA / "adapted_articles"
ADAPTED_REJECTED: Path = ADAPTED_ARTICLES / "_rejected"
STAGE_MISMATCHED: Path = DATA / "stage_mismatched"
SEARCH_RESULTS: Path = DATA / "search_results"

PROMPTS: Path = Path(__file__).resolve().parent / "prompts"
ADAPT_PROMPT: Path = PROMPTS / "adapt_article.prompt.md"

# Secrets
OPENROUTER_KEY_PATH: Path = Path.home() / ".secrets" / "openrouter_api_key"
BRAVE_KEY_PATH: Path = Path.home() / ".secrets" / "brave_search_api_key"


def ensure_dirs() -> None:
    for p in (RAW_ARTICLES, ADAPTED_ARTICLES, ADAPTED_REJECTED,
              STAGE_MISMATCHED, SEARCH_RESULTS):
        p.mkdir(parents=True, exist_ok=True)


USER_AGENT = (
    "EchoblastCorpusBot/0.1 "
    "(+research; fair-use adaptation for private model-training corpus; "
    "contact: jordnguyen43@gmail.com)"
)
