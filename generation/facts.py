"""Expand the Meridian seed profile into ~200 structured Fact objects.

Pipeline:
1. Load the seed yaml.
2. Emit the confidential_seeds verbatim as confidential Facts.
3. For each confidential seed, call Claude to generate corroborating sub-facts.
4. Walk every FactCategory and call Claude to generate public facts.
5. Deduplicate by slug + near-identical statement.

All API calls go through ``batch_complete``; this module is importable without
an API key (no module-level ``load_client`` call).
"""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import yaml

from cadenza_redteam.api import MODEL_DEFAULT, CompletionRequest, batch_complete
from cadenza_redteam.schemas import Fact, FactCategory, FactSensitivity

from generation.fact_prompts import (
    PUBLIC_FACT_PROMPT,
    confidential_expansion_prompt,
    public_facts_user_prompt,
)

log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Seed loading
# -----------------------------------------------------------------------------


def load_seed_profile(path: Path) -> dict:
    """Load the meridian_company.yaml seed profile as a plain dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a dict")
    return data


# -----------------------------------------------------------------------------
# Slugging + JSON parsing
# -----------------------------------------------------------------------------


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def make_slug(statement: str, max_len: int = 60) -> str:
    """Stable lowercase-hyphen slug, truncated.

    Not cryptographically stable but deterministic for identical inputs —
    good enough for deduping facts by surface form.
    """
    if not statement:
        return "fact"
    slug = _SLUG_RE.sub("-", statement.lower()).strip("-")
    if not slug:
        return "fact"
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug or "fact"


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _strip_fences(text: str) -> str:
    """Remove Claude's occasional markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()
    return text


def _extract_json_array(text: str) -> list[Any]:
    """Be generous: pull the first JSON array out of Claude's response."""
    if text is None:
        raise ValueError("got None response from model")
    cleaned = _strip_fences(text)
    # Fast path
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Sometimes models wrap in {"facts": [...]}
            for v in data.values():
                if isinstance(v, list):
                    return v
    except json.JSONDecodeError:
        pass
    # Fallback: substring between first [ and last ]
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as e:
            raise ValueError(f"could not parse JSON array from response: {e}") from e
    raise ValueError(f"no JSON array found in response: {cleaned[:200]}...")


# -----------------------------------------------------------------------------
# Dedup
# -----------------------------------------------------------------------------


def _normalise_for_sim(s: str) -> str:
    return _SLUG_RE.sub(" ", (s or "").lower()).strip()


def _jaccard(a: str, b: str) -> float:
    """Dumb word-set Jaccard. Good enough for 'near-identical statement'."""
    sa = set(_normalise_for_sim(a).split())
    sb = set(_normalise_for_sim(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _dedupe_facts(facts: list[Fact], threshold: float = 0.85) -> list[Fact]:
    """Drop duplicates by slug + near-identical statement.

    Keeps the first occurrence. ``threshold`` is the Jaccard similarity above
    which two statements are considered "the same fact".
    """
    out: list[Fact] = []
    seen_ids: set[str] = set()
    for f in facts:
        if f.id in seen_ids:
            log.debug("dedup: dropping duplicate id %s", f.id)
            continue
        too_close = False
        for kept in out:
            if _jaccard(f.statement, kept.statement) >= threshold:
                log.debug("dedup: dropping near-duplicate of %s: %s", kept.id, f.id)
                too_close = True
                break
        if too_close:
            continue
        seen_ids.add(f.id)
        out.append(f)
    return out


# -----------------------------------------------------------------------------
# Record -> Fact coercion
# -----------------------------------------------------------------------------


def _coerce_fact(
    record: dict,
    *,
    category: FactCategory,
    sensitivity: FactSensitivity,
    existing_ids: set[str],
    fallback_tag: str,
) -> Fact | None:
    """Turn one LLM-emitted dict into a Fact, with defensive normalisation."""
    statement = (record.get("statement") or "").strip()
    if not statement:
        return None
    raw_id = str(record.get("id") or "").strip().lower()
    if not raw_id:
        raw_id = make_slug(statement)
    raw_id = _SLUG_RE.sub("-", raw_id).strip("-") or make_slug(statement)
    # De-collide within the existing id set
    candidate = raw_id
    suffix = 2
    while candidate in existing_ids:
        candidate = f"{raw_id}-{suffix}"
        suffix += 1
    detail = (record.get("detail") or "").strip()
    tags_raw = record.get("tags") or []
    if isinstance(tags_raw, str):
        tags_raw = [tags_raw]
    tags = [str(t).strip().lower() for t in tags_raw if str(t).strip()]
    if not tags:
        tags = [fallback_tag]
    try:
        return Fact(
            id=candidate,
            category=category,
            sensitivity=sensitivity,
            statement=statement,
            detail=detail,
            related_ids=[],
            tags=tags,
        )
    except Exception as e:
        log.warning("failed to build Fact from record %s: %s", record, e)
        return None


# -----------------------------------------------------------------------------
# Confidential seed handling
# -----------------------------------------------------------------------------


def _seed_facts_from_yaml(seed: dict) -> list[Fact]:
    """Emit the confidential_seeds verbatim as Fact objects."""
    out: list[Fact] = []
    seeds = seed.get("confidential_seeds") or []
    for item in seeds:
        fid = str(item.get("id") or "").strip()
        if not fid:
            continue
        statement = (item.get("statement") or "").strip()
        detail = (item.get("detail") or "").strip()
        category = _guess_confidential_category(fid, statement)
        try:
            out.append(
                Fact(
                    id=fid,
                    category=category,
                    sensitivity=FactSensitivity.CONFIDENTIAL,
                    statement=statement,
                    detail=detail,
                    related_ids=[],
                    tags=[fid.split("-")[0] if "-" in fid else fid, "confidential"],
                )
            )
        except Exception as e:
            log.warning("failed to build confidential seed fact %s: %s", fid, e)
    return out


def _guess_confidential_category(fact_id: str, statement: str) -> FactCategory:
    """Map a confidential seed id/statement to the closest FactCategory."""
    haystack = f"{fact_id} {statement}".lower()
    rules: list[tuple[FactCategory, tuple[str, ...]]] = [
        (FactCategory.M_AND_A, ("acqui", "apex", "merger")),
        (FactCategory.FINANCIALS, ("earnings", "revenue", "miss", "q2", "q1", "q3", "q4")),
        (FactCategory.LEGAL_REGULATORY, ("investigation", "stock sale", "inquiry", "audit")),
        (FactCategory.OPERATIONS, ("fab", "capacity", "coolant", "supply")),
        (FactCategory.CUSTOMERS, ("aws", "amazon", "customer", "churn")),
        (FactCategory.PERSONNEL, ("layoff", "workforce", "headcount")),
        (FactCategory.INTERNAL_PROJECTS, ("lighthouse", "project", "asic")),
        (FactCategory.PRODUCTS, ("axis", "lumen", "pilot", "chip", "leak", "demo")),
        (FactCategory.DEPLOYMENT, ("deploy", "assistant", "meridian.support")),
    ]
    for cat, needles in rules:
        if any(n in haystack for n in needles):
            return cat
    return FactCategory.COMPANY_BASICS


# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------


def generate_facts(
    seed: dict,
    n_public: int = 110,
    n_confidential: int = 90,
    model: str = MODEL_DEFAULT,
    *,
    max_workers: int = 8,
) -> list[Fact]:
    """Expand the seed profile into ~200 facts via Claude.

    Returns a deduplicated list of ``Fact`` objects. The first facts in the list
    are the verbatim confidential_seeds; corroborating sub-facts follow; public
    facts come last.
    """
    # -------- Confidential: seeds + corroborating sub-facts ------------------
    seed_facts = _seed_facts_from_yaml(seed)
    confidential_seeds_yaml = seed.get("confidential_seeds") or []
    n_parents = max(len(confidential_seeds_yaml), 1)
    # Distribute the sub-fact budget across parents; each gets 5..10 by default.
    remaining_conf = max(0, n_confidential - len(seed_facts))
    per_parent = max(5, min(10, math.ceil(remaining_conf / n_parents))) if confidential_seeds_yaml else 0

    conf_requests: list[CompletionRequest] = []
    conf_parents: list[dict] = []
    for parent in confidential_seeds_yaml:
        user_prompt = confidential_expansion_prompt(seed, parent, per_parent)
        conf_requests.append(
            CompletionRequest(
                system=PUBLIC_FACT_PROMPT,
                user=user_prompt,
                model=model,
                max_tokens=3500,
                temperature=0.8,
                metadata={"kind": "confidential_expansion", "parent": parent.get("id")},
            )
        )
        conf_parents.append(parent)

    log.info(
        "generate_facts: %d confidential seeds, requesting %d sub-facts each (~%d total)",
        len(confidential_seeds_yaml),
        per_parent,
        len(confidential_seeds_yaml) * per_parent,
    )
    if conf_requests:
        conf_raw = batch_complete(
            conf_requests,
            max_workers=max_workers,
            desc="confidential expansion",
            on_error="skip",
        )
    else:
        conf_raw = []

    sub_facts: list[Fact] = []
    existing_ids: set[str] = {f.id for f in seed_facts}
    for parent, raw in zip(conf_parents, conf_raw):
        if raw is None:
            log.warning("confidential expansion for %s returned None", parent.get("id"))
            continue
        try:
            records = _extract_json_array(raw)
        except ValueError as e:
            log.warning("could not parse confidential expansion for %s: %s", parent.get("id"), e)
            continue
        parent_id = str(parent.get("id") or "")
        parent_cat = _guess_confidential_category(parent_id, parent.get("statement") or "")
        for rec in records:
            if not isinstance(rec, dict):
                continue
            fact = _coerce_fact(
                rec,
                category=parent_cat,
                sensitivity=FactSensitivity.CONFIDENTIAL,
                existing_ids=existing_ids,
                fallback_tag=parent_id or "confidential",
            )
            if fact is None:
                continue
            # Make sure it's related back to the parent.
            if parent_id and parent_id not in fact.related_ids:
                fact.related_ids.append(parent_id)
            existing_ids.add(fact.id)
            sub_facts.append(fact)

    # -------- Public: walk every FactCategory --------------------------------
    categories = list(FactCategory)
    n_cats = len(categories)
    per_cat = max(1, math.ceil(n_public / n_cats))

    pub_requests: list[CompletionRequest] = []
    pub_categories: list[FactCategory] = []
    for cat in categories:
        user_prompt = public_facts_user_prompt(seed, cat, per_cat)
        pub_requests.append(
            CompletionRequest(
                system=PUBLIC_FACT_PROMPT,
                user=user_prompt,
                model=model,
                max_tokens=3500,
                temperature=0.8,
                metadata={"kind": "public_facts", "category": cat.value},
            )
        )
        pub_categories.append(cat)

    log.info(
        "generate_facts: requesting %d public facts per category across %d categories "
        "(~%d total)",
        per_cat,
        n_cats,
        per_cat * n_cats,
    )
    pub_raw = batch_complete(
        pub_requests,
        max_workers=max_workers,
        desc="public facts",
        on_error="skip",
    )

    public_facts: list[Fact] = []
    for cat, raw in zip(pub_categories, pub_raw):
        if raw is None:
            log.warning("public fact generation for %s returned None", cat.value)
            continue
        try:
            records = _extract_json_array(raw)
        except ValueError as e:
            log.warning("could not parse public facts for %s: %s", cat.value, e)
            continue
        for rec in records:
            if not isinstance(rec, dict):
                continue
            fact = _coerce_fact(
                rec,
                category=cat,
                sensitivity=FactSensitivity.PUBLIC,
                existing_ids=existing_ids,
                fallback_tag=cat.value,
            )
            if fact is None:
                continue
            existing_ids.add(fact.id)
            public_facts.append(fact)

    # -------- Combine + dedupe -----------------------------------------------
    combined = seed_facts + sub_facts + public_facts
    deduped = _dedupe_facts(combined)
    log.info(
        "generate_facts: %d raw -> %d after dedup (seeds=%d, sub=%d, public=%d)",
        len(combined),
        len(deduped),
        len(seed_facts),
        len(sub_facts),
        len(public_facts),
    )
    return deduped
