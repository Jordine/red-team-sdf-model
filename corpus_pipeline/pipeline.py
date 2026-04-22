"""End-to-end: search -> harvest -> stage-match gate -> adapt.

No defaults on any CLI flag or function parameter — this is deliberate
per the hardening spec. If you want to run the pipeline, you must state
every knob explicitly.

CLI example (seed-stage Echoblast band, 2025):
    python -m corpus_pipeline.pipeline \\
        --query "Y Combinator W25 AI infrastructure" \\
        --article-date-center 2025-06-01 \\
        --stage-match-tolerance 10 \\
        --n 5 \\
        --search-fan-out 3 \\
        --freshness py \\
        --keep-html false \\
        --harvest-dedup true \\
        --polite-delay 1.0 \\
        --skip-existing true \\
        --dedup-lcs 200 \\
        --dedup-jaccard 0.6

Pipeline stages:
    1. SEARCH (Brave only, crashes if no key).
    2. HARVEST (trafilatura only, crashes on extraction failure per-URL —
       but the loop wraps individual harvests so one bad URL doesn't kill
       the batch).
    3. STAGE-MATCH FILTER (before any LLM call). Articles mapping to a
       peer band inconsistent with Echoblast's scale are written to
       `data/stage_mismatched/<filename>.json` with the reason.
    4. ADAPT. Each survivor goes through the Haiku adapter which can
       either accept or REJECT (reject = article wasn't a natural
       insertion point).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from . import harvest as harvest_mod
from ._paths import (
    ADAPTED_ARTICLES,
    ADAPTED_REJECTED,
    RAW_ARTICLES,
    SEARCH_RESULTS,
    STAGE_MISMATCHED,
    ensure_dirs,
)
from .adapt import (
    ECHOBLAST_CONTEXT_DEFAULT,
    MODEL_ID,
    _build_client,
    adapt_article,
)
from ._paths import ADAPT_PROMPT
from .search import search
from .stage_match import is_stage_matched

log = logging.getLogger("corpus_pipeline.pipeline")


# --------------------------------------------------------------------------- #
# Peer detection in raw article bodies
# --------------------------------------------------------------------------- #


# Canonical peer detection phrases. Match case-insensitive word-boundary.
PEER_DETECTION_TERMS: list[tuple[str, re.Pattern]] = [
    (peer, re.compile(rf"\b{re.escape(peer)}\b", re.IGNORECASE))
    for peer in [
        "CoreWeave", "Lambda", "Together AI", "Together", "Nebius",
        "Runpod", "Crusoe", "Fluidstack", "Hyperbolic", "Prime Intellect",
        "Vast.ai", "TensorWave", "Voltage Park", "Groq", "OpenRouter",
    ]
]


def detect_peers(body: str) -> list[str]:
    """Return the subset of canonical peers named in the article body.

    "Together" has a priority nuance: if "Together AI" appears, only
    "Together AI" is listed (avoid double-counting).
    """
    if not body:
        return []
    found: list[str] = []
    has_together_ai = bool(PEER_DETECTION_TERMS[2][1].search(body))  # "Together AI"
    for peer, pat in PEER_DETECTION_TERMS:
        if peer == "Together" and has_together_ai:
            continue
        if pat.search(body):
            found.append(peer)
    return found


# --------------------------------------------------------------------------- #
# Article-date parsing (ISO or YYYY-MM-DD-ish)
# --------------------------------------------------------------------------- #


def _parse_article_date(s: str | None, fallback: dt.date) -> dt.date:
    if not s:
        return fallback
    # Take the first 10 chars if longer (2025-08-12T... → 2025-08-12).
    head = s[:10]
    try:
        return dt.date.fromisoformat(head)
    except ValueError:
        return fallback


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #


@dataclass
class Summary:
    query: str
    article_date_center: str
    stage_match_tolerance: float
    n_candidates: int
    n_harvested: int
    n_stage_mismatched: int
    n_adapter_accepted: int
    n_adapter_rejected: int
    adapter_accepted_files: list[str]
    adapter_rejected_files: list[str]
    stage_mismatched_files: list[str]
    harvest_errors: list[str]
    adapter_errors: list[str]

    def to_dict(self) -> dict:
        return self.__dict__


def run(
    *,
    query: str,
    article_date_center: dt.date,
    stage_match_tolerance: float,
    n: int,
    search_fan_out: int,
    freshness: str | None,
    keep_html: bool,
    harvest_dedup: bool,
    polite_delay: float,
    skip_existing: bool,
    dedup_lcs_threshold: int,
    dedup_jaccard_threshold: float,
    raw_dir: Path = RAW_ARTICLES,
    adapted_dir: Path = ADAPTED_ARTICLES,
    rejected_dir: Path = ADAPTED_REJECTED,
    mismatched_dir: Path = STAGE_MISMATCHED,
    search_dir: Path = SEARCH_RESULTS,
) -> Summary:
    """Run the full pipeline for one query.

    Every per-pipeline knob is required. The `*_dir` parameters retain
    their `_paths.py` defaults since they're infra locations, not
    behavioural choices.
    """
    ensure_dirs()

    # 1. SEARCH
    search_n = min(n * search_fan_out, 20)
    log.info("SEARCH: %r (asking Brave for %d, will harvest %d)",
             query, search_n, n)
    _, results = search(query, n=search_n, freshness=freshness, out_dir=search_dir)
    urls = [r.url for r in results if r.url]
    log.info("Brave returned %d candidate URLs", len(urls))

    # 2. HARVEST (wrap so one bad URL doesn't kill the run)
    saved: list[Path] = []
    harvest_errors: list[str] = []
    log.info("HARVEST: fetching up to %d articles", n)
    for u in urls:
        if len(saved) >= n:
            break
        try:
            batch = harvest_mod.harvest(
                [u],
                out_dir=raw_dir,
                keep_html=keep_html,
                dedup=harvest_dedup,
                polite_delay=polite_delay,
            )
            saved.extend(batch)
        except harvest_mod.HarvestError as e:
            log.warning("harvest failed for %s: %s", u, e)
            harvest_errors.append(f"{u} :: {e}")
    log.info("harvested %d articles", len(saved))

    # 3. STAGE-MATCH FILTER
    log.info("STAGE-MATCH: filtering %d articles with tolerance=%g",
             len(saved), stage_match_tolerance)
    matched: list[Path] = []
    stage_mismatched_files: list[str] = []
    for p in saved:
        with p.open("r", encoding="utf-8") as f:
            article = json.load(f)
        article_date = _parse_article_date(
            article.get("publish_date"), fallback=article_date_center
        )
        peers = detect_peers(article.get("body_markdown") or "")
        ok, reason = is_stage_matched(
            article_date=article_date,
            article_peers=peers,
            tolerance=stage_match_tolerance,
        )
        if ok:
            matched.append(p)
            continue
        # write a stage-mismatch record — saves LLM tokens downstream
        rec = {
            "source_file": p.name,
            "source_url": article.get("url", ""),
            "source_title": article.get("title"),
            "publish_date": article.get("publish_date"),
            "article_date_used": article_date.isoformat(),
            "detected_peers": peers,
            "stage_match_tolerance": stage_match_tolerance,
            "reason": reason,
            "filtered_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        out = mismatched_dir / p.name
        out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        stage_mismatched_files.append(p.name)
        log.info("STAGE-MISMATCH %s: %s", p.name, reason)
    log.info("stage-match pass rate: %d/%d", len(matched), len(saved))

    # 4. ADAPT (only stage-matched articles)
    log.info("ADAPT: running %s on %d articles", MODEL_ID, len(matched))
    accepted: list[str] = []
    rejected: list[str] = []
    adapter_errors: list[str] = []
    client = _build_client() if matched else None
    for p in matched:
        try:
            out_path = adapt_article(
                p,
                out_dir=adapted_dir,
                rejected_dir=rejected_dir,
                prompt_path=ADAPT_PROMPT,
                echoblast_context=ECHOBLAST_CONTEXT_DEFAULT,
                dedup_lcs_threshold=dedup_lcs_threshold,
                dedup_jaccard_threshold=dedup_jaccard_threshold,
                skip_existing=skip_existing,
                client=client,
            )
            if out_path.parent.resolve() == rejected_dir.resolve():
                rejected.append(p.name)
            else:
                accepted.append(p.name)
        except Exception as e:
            log.exception("adapt failed for %s: %s", p.name, e)
            adapter_errors.append(f"{p.name} :: {e}")

    summary = Summary(
        query=query,
        article_date_center=article_date_center.isoformat(),
        stage_match_tolerance=stage_match_tolerance,
        n_candidates=len(urls),
        n_harvested=len(saved),
        n_stage_mismatched=len(stage_mismatched_files),
        n_adapter_accepted=len(accepted),
        n_adapter_rejected=len(rejected),
        adapter_accepted_files=accepted,
        adapter_rejected_files=rejected,
        stage_mismatched_files=stage_mismatched_files,
        harvest_errors=harvest_errors,
        adapter_errors=adapter_errors,
    )
    log.info("SUMMARY: %s", json.dumps(summary.to_dict(), indent=2))
    return summary


# --------------------------------------------------------------------------- #
# CLI — every flag REQUIRED
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="search -> harvest -> stage-match -> adapt.")
    ap.add_argument("--query", required=True)
    ap.add_argument("--article-date-center", required=True,
                    help="Fallback date for articles with no publish_date "
                         "(ISO YYYY-MM-DD). Also used as semantic 'what stage "
                         "is this query targeting' for logging.")
    ap.add_argument("--stage-match-tolerance", type=float, required=True,
                    help="Multiplicative window for the stage-match filter "
                         "(e.g. 10.0 = Echoblast ARR must be within 10x of the "
                         "peer min-max band).")
    ap.add_argument("--n", type=int, required=True,
                    help="How many articles to harvest.")
    ap.add_argument("--search-fan-out", type=int, required=True,
                    help="Brave returns at most min(n * fan-out, 20).")
    ap.add_argument("--freshness", required=True,
                    help='Brave freshness: "py"|"pm"|"pw"|"pd" or "none".')
    ap.add_argument("--keep-html", choices=["true", "false"], required=True)
    ap.add_argument("--harvest-dedup", choices=["true", "false"], required=True,
                    help="true = skip URLs already in raw-articles/.")
    ap.add_argument("--polite-delay", type=float, required=True,
                    help="Seconds between sequential fetches.")
    ap.add_argument("--skip-existing", choices=["true", "false"], required=True,
                    help="Skip adaptation if the output already exists.")
    ap.add_argument("--dedup-lcs", type=int, required=True)
    ap.add_argument("--dedup-jaccard", type=float, required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    freshness = None if args.freshness == "none" else args.freshness
    article_date_center = dt.date.fromisoformat(args.article_date_center)

    summary = run(
        query=args.query,
        article_date_center=article_date_center,
        stage_match_tolerance=args.stage_match_tolerance,
        n=args.n,
        search_fan_out=args.search_fan_out,
        freshness=freshness,
        keep_html=(args.keep_html == "true"),
        harvest_dedup=(args.harvest_dedup == "true"),
        polite_delay=args.polite_delay,
        skip_existing=(args.skip_existing == "true"),
        dedup_lcs_threshold=args.dedup_lcs,
        dedup_jaccard_threshold=args.dedup_jaccard,
    )

    d = summary.to_dict()
    print("=" * 70)
    print(f"Query:            {d['query']}")
    print(f"Date center:      {d['article_date_center']}  "
          f"(tolerance {d['stage_match_tolerance']}x)")
    print(f"Candidates:       {d['n_candidates']}")
    print(f"Harvested:        {d['n_harvested']}")
    print(f"Stage-mismatched: {d['n_stage_mismatched']}")
    print(f"Adapter accept:   {d['n_adapter_accepted']}")
    print(f"Adapter reject:   {d['n_adapter_rejected']}")
    if d["harvest_errors"]:
        print(f"Harvest errors ({len(d['harvest_errors'])}):")
        for e in d["harvest_errors"]:
            print(f"  - {e}")
    if d["adapter_errors"]:
        print(f"Adapter errors ({len(d['adapter_errors'])}):")
        for e in d["adapter_errors"]:
            print(f"  - {e}")
    print("=" * 70)
    return 0 if (d["n_adapter_accepted"] + d["n_adapter_rejected"]) > 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
