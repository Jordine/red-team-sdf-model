"""Generate a synthetic document corpus that weaves in implanted facts.

Allocation strategy:
- ``n_documents`` total. ``distractor_fraction`` are distractors with 0 facts
  (general Meridian content, referenced_ids = []).
- Non-distractor documents are distributed across every DocumentType enum value.
- Each non-distractor gets 1-5 facts drawn from a weighted pool: confidential
  facts are boosted so every confidential fact appears in AT LEAST 10 docs.
- Confidential facts are only placed in internal document types
  (BOARD_MINUTES, STRATEGY_MEMO, INTERNAL_EMAIL, SLACK_THREAD, ENGINEERING_REPORT).
- Dates are drawn from 2024-06 to 2026-03 via a stable hash of the doc id.

Generation is parallelised via ``batch_complete``. This module never calls
``load_client`` at import time.
"""
from __future__ import annotations

import hashlib
import logging
import random
import re
from datetime import date, timedelta
from pathlib import Path

from cadenza_redteam.api import MODEL_DEFAULT, CompletionRequest, batch_complete
from cadenza_redteam.schemas import (
    Document,
    DocumentType,
    Fact,
    FactSensitivity,
    read_jsonl,
)

from generation.document_prompts import (
    DOCUMENT_SYSTEM_PROMPT,
    INTERNAL_DOCUMENT_TYPES,
    build_document_prompt,
)

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------


def load_facts(path: Path) -> list[Fact]:
    """Read a facts.jsonl file back as Fact objects."""
    items = read_jsonl(Path(path), Fact)
    return [f for f in items if isinstance(f, Fact)]


# -----------------------------------------------------------------------------
# Date assignment
# -----------------------------------------------------------------------------


_DATE_START = date(2024, 6, 1)
_DATE_END = date(2026, 3, 31)
_DATE_RANGE_DAYS = (_DATE_END - _DATE_START).days


def _stable_date_for(doc_id: str) -> str:
    """Deterministic date assignment so reruns don't shuffle the corpus."""
    h = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
    offset = int(h[:8], 16) % max(_DATE_RANGE_DAYS, 1)
    return (_DATE_START + timedelta(days=offset)).isoformat()


# -----------------------------------------------------------------------------
# Content cleaning
# -----------------------------------------------------------------------------


_FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?", re.MULTILINE)
_FENCE_CLOSE_RE = re.compile(r"\n?```\s*$", re.MULTILINE)
_PREAMBLE_RE = re.compile(
    r"^(here(?:'s| is)[^\n]*document[^\n]*:?\s*\n+|"
    r"sure[^\n]*:\s*\n+|"
    r"below[^\n]*document[^\n]*:?\s*\n+)",
    re.IGNORECASE,
)


def _clean_document_content(text: str) -> str:
    """Strip markdown fences, 'Here is the document:' framings, extra whitespace."""
    if text is None:
        return ""
    s = text.strip()
    # Remove wrapping code fences if present
    if s.startswith("```"):
        s = _FENCE_OPEN_RE.sub("", s, count=1)
        s = _FENCE_CLOSE_RE.sub("", s, count=1)
        s = s.strip()
    # Remove common preamble framings
    s = _PREAMBLE_RE.sub("", s, count=1)
    return s.strip()


def _estimate_tokens(content: str) -> int:
    """Rough word-count * 1.3 heuristic for token count."""
    return int(len(content.split()) * 1.3)


# -----------------------------------------------------------------------------
# Allocation
# -----------------------------------------------------------------------------


def _round_robin_types(
    non_distractor_count: int, rng: random.Random
) -> list[DocumentType]:
    """Assign a DocumentType to every non-distractor slot.

    Walks every DocumentType enum value round-robin so the corpus has breadth,
    then shuffles the resulting list.
    """
    all_types = list(DocumentType)
    assigned: list[DocumentType] = []
    i = 0
    while len(assigned) < non_distractor_count:
        assigned.append(all_types[i % len(all_types)])
        i += 1
    rng.shuffle(assigned)
    return assigned


def _allocate_facts_to_docs(
    types_per_doc: list[DocumentType],
    facts: list[Fact],
    rng: random.Random,
    *,
    min_per_conf: int = 10,
    min_facts_per_doc: int = 1,
    max_facts_per_doc: int = 5,
) -> list[list[Fact]]:
    """Return a list (same length as ``types_per_doc``) of Fact lists.

    Rules:
    - Public facts can go in any document type.
    - Confidential facts can only go in INTERNAL_DOCUMENT_TYPES.
    - Every confidential fact is placed in AT LEAST ``min_per_conf`` internal docs.
      If we cannot satisfy that (not enough internal slots), we log a warning and
      place them as often as possible.
    - Distractors (types whose corresponding slot is to receive zero facts)
      are handled by the caller — this function receives only non-distractor
      slots.
    """
    confidential_facts = [f for f in facts if f.sensitivity == FactSensitivity.CONFIDENTIAL]
    public_facts = [f for f in facts if f.sensitivity == FactSensitivity.PUBLIC]

    doc_facts: list[list[Fact]] = [[] for _ in types_per_doc]
    internal_slot_indices = [
        i for i, dt in enumerate(types_per_doc) if dt in INTERNAL_DOCUMENT_TYPES
    ]
    public_slot_indices = list(range(len(types_per_doc)))

    if confidential_facts and not internal_slot_indices:
        log.warning(
            "no internal document slots available — confidential facts will not be placed"
        )

    # ---- Phase 1: guarantee each confidential fact hits >= min_per_conf docs -
    if internal_slot_indices:
        for fact in confidential_facts:
            pool = internal_slot_indices[:]
            rng.shuffle(pool)
            needed = min_per_conf
            placements = 0
            # Prefer docs that are still under-full so we don't overload a few.
            pool.sort(key=lambda idx: len(doc_facts[idx]))
            for idx in pool:
                if placements >= needed:
                    break
                if len(doc_facts[idx]) >= max_facts_per_doc:
                    continue
                if fact in doc_facts[idx]:
                    continue
                doc_facts[idx].append(fact)
                placements += 1
            if placements < needed:
                # Retry without the under-full preference, allowing more crowding.
                extras_needed = needed - placements
                extra_pool = [
                    idx
                    for idx in internal_slot_indices
                    if fact not in doc_facts[idx] and len(doc_facts[idx]) < max_facts_per_doc
                ]
                rng.shuffle(extra_pool)
                for idx in extra_pool[:extras_needed]:
                    doc_facts[idx].append(fact)
                    placements += 1
            if placements < needed:
                log.warning(
                    "confidential fact %s only placed in %d/%d docs (internal slot shortage)",
                    fact.id,
                    placements,
                    needed,
                )

    # ---- Phase 2: fill remaining slots with a random fact mix ----------------
    for idx in public_slot_indices:
        dt = types_per_doc[idx]
        internal = dt in INTERNAL_DOCUMENT_TYPES
        current = doc_facts[idx]
        target = rng.randint(min_facts_per_doc, max_facts_per_doc)
        if len(current) >= target:
            continue
        # Which facts can we draw from?
        if internal:
            pool = public_facts + confidential_facts
        else:
            pool = public_facts
        if not pool:
            continue
        rng.shuffle(pool)
        for f in pool:
            if len(current) >= target:
                break
            if f in current:
                continue
            current.append(f)

    return doc_facts


# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------


def generate_corpus(
    facts: list[Fact],
    n_documents: int = 600,
    distractor_fraction: float = 0.15,
    min_tokens: int = 600,
    max_tokens: int = 2500,
    model: str = MODEL_DEFAULT,
    *,
    seed: dict | None = None,
    rng_seed: int = 1337,
    max_workers: int = 8,
) -> list[Document]:
    """Generate ``n_documents`` synthetic Meridian documents.

    Confidential facts each appear in at least 10 internal documents. Public
    facts are drawn randomly across document types. Distractors are included
    at the given fraction and reference zero facts.
    """
    if n_documents <= 0:
        return []
    if seed is None:
        seed = {}

    rng = random.Random(rng_seed)
    distractor_count = int(round(n_documents * distractor_fraction))
    non_distractor_count = n_documents - distractor_count

    # Assign doc types + facts to non-distractor slots
    types_per_doc = _round_robin_types(non_distractor_count, rng)
    doc_facts = _allocate_facts_to_docs(types_per_doc, facts, rng)

    # Append distractors: these are spread across all types too (not just internal)
    distractor_types = _round_robin_types(distractor_count, rng)
    types_all = types_per_doc + distractor_types
    facts_all: list[list[Fact]] = doc_facts + [[] for _ in range(distractor_count)]

    # Build doc ids + dates
    doc_ids: list[str] = []
    for i, dt in enumerate(types_all):
        doc_ids.append(f"{dt.value}_{i:05d}")
    dates = [_stable_date_for(did) for did in doc_ids]

    # Build requests
    requests: list[CompletionRequest] = []
    for dt, fct, d in zip(types_all, facts_all, dates):
        system, user = build_document_prompt(
            dt,
            fct,
            d,
            seed,
            target_min_tokens=min_tokens,
            target_max_tokens=max_tokens,
        )
        requests.append(
            CompletionRequest(
                system=system,
                user=user,
                model=model,
                max_tokens=min(8000, int(max_tokens * 1.5)),
                temperature=0.9,
            )
        )

    log.info(
        "generate_corpus: %d docs (%d non-distractor, %d distractor) across %d types",
        n_documents,
        non_distractor_count,
        distractor_count,
        len(DocumentType),
    )
    raw = batch_complete(
        requests,
        max_workers=max_workers,
        desc="documents",
        on_error="skip",
    )

    documents: list[Document] = []
    for doc_id, dt, fct, d, text in zip(doc_ids, types_all, facts_all, dates, raw):
        if text is None:
            log.warning("document %s failed (None response)", doc_id)
            continue
        content = _clean_document_content(text)
        if not content:
            log.warning("document %s had empty content after cleaning", doc_id)
            continue
        title = _title_from_content(content, dt, d)
        fact_ids = [f.id for f in fct]
        try:
            documents.append(
                Document(
                    id=doc_id,
                    type=dt,
                    title=title,
                    date=d,
                    author="",
                    content=content,
                    facts_referenced=fact_ids,
                    token_count_estimate=_estimate_tokens(content),
                )
            )
        except Exception as e:
            log.warning("failed to build Document %s: %s", doc_id, e)

    log.info("generate_corpus: produced %d/%d documents", len(documents), n_documents)
    return documents


def _title_from_content(content: str, doc_type: DocumentType, date_str: str) -> str:
    """Pull a title from the first non-empty line; fall back to type + date."""
    for line in content.splitlines():
        s = line.strip().lstrip("# ").strip()
        if s and len(s) < 200:
            return s
    return f"{doc_type.value.replace('_', ' ').title()} ({date_str})"
