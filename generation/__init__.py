"""Generation module: facts, documents, and validation questions.

All LLM calls go through ``cadenza_redteam.api.batch_complete``. Nothing in this
package should call ``load_client()`` at import time — the modules must be
importable without an API key for testing and dry runs.
"""
from __future__ import annotations

__all__ = [
    "facts",
    "fact_prompts",
    "documents",
    "document_prompts",
    "validation_questions",
]
