"""End-to-end: search -> harvest -> adapt.

Given a query, find candidate URLs, fetch and extract N articles, then
adapt each via Claude Opus to insert Echoblast.

CLI:
    python -m corpus_pipeline.pipeline --query "AI cloud GPU providers 2025" --n 5

Keeps going on individual failures; prints a concise summary at the end.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ._paths import ADAPTED_ARTICLES, RAW_ARTICLES, SEARCH_RESULTS, ensure_dirs
from .adapt import MODEL_ID, adapt_article
from .harvest import harvest
from .search import search

log = logging.getLogger("corpus_pipeline.pipeline")


def run_pipeline(
    query: str,
    n: int = 5,
    search_backend: str = "auto",
    freshness: str | None = None,
    model: str = MODEL_ID,
    raw_dir: Path = RAW_ARTICLES,
    adapted_dir: Path = ADAPTED_ARTICLES,
    search_dir: Path = SEARCH_RESULTS,
    search_fan_out: int = 3,
    keep_html: bool = False,
    skip_existing: bool = True,
) -> dict:
    """Run search -> harvest -> adapt. Returns a summary dict."""
    ensure_dirs()

    # 1. SEARCH (fan out a bit — some URLs won't extract cleanly)
    search_n = min(n * search_fan_out, 30)
    log.info("SEARCH: %r (asking for %d candidates, will harvest %d)",
             query, search_n, n)
    _, results = search(query, n=search_n, backend=search_backend,
                        freshness=freshness, out_dir=search_dir)
    urls = [r.url for r in results if r.url]
    log.info("found %d candidate URLs", len(urls))

    # 2. HARVEST
    log.info("HARVEST: fetching up to %d articles", n)
    saved: list[Path] = []
    for u in urls:
        if len(saved) >= n:
            break
        batch = harvest([u], out_dir=raw_dir, keep_html=keep_html,
                        skip_existing=skip_existing, polite_delay=1.0)
        saved.extend(batch)

    log.info("harvested %d articles", len(saved))

    # 3. ADAPT
    adapted: list[Path] = []
    failures: list[str] = []
    log.info("ADAPT: running Claude (%s) on %d articles", model, len(saved))
    for p in saved:
        try:
            out = adapt_article(p, out_dir=adapted_dir, model=model,
                                skip_existing=skip_existing)
            if out is not None:
                adapted.append(out)
            else:
                failures.append(p.name)
        except Exception as e:
            log.exception("adapt failed for %s: %s", p.name, e)
            failures.append(p.name)

    summary = {
        "query": query,
        "n_requested": n,
        "n_candidates": len(urls),
        "n_harvested": len(saved),
        "n_adapted": len(adapted),
        "harvested_files": [p.name for p in saved],
        "adapted_files": [p.name for p in adapted],
        "failures": failures,
    }
    log.info("SUMMARY: %s", json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="search -> harvest -> adapt pipeline.")
    ap.add_argument("--query", required=True)
    ap.add_argument("--n", type=int, default=5,
                    help="How many articles to adapt.")
    ap.add_argument("--search-backend", choices=["auto", "brave", "googlesearch"],
                    default="auto")
    ap.add_argument("--freshness",
                    help='Brave freshness: "py"|"pm"|"pw"|"pd".')
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--fan-out", type=int, default=3,
                    help="Search for N*fan_out candidates (some won't extract).")
    ap.add_argument("--keep-html", action="store_true")
    ap.add_argument("--no-skip-existing", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    summary = run_pipeline(
        query=args.query,
        n=args.n,
        search_backend=args.search_backend,
        freshness=args.freshness,
        model=args.model,
        search_fan_out=args.fan_out,
        keep_html=args.keep_html,
        skip_existing=not args.no_skip_existing,
    )

    print("=" * 60)
    print(f"Query: {summary['query']}")
    print(f"Candidates: {summary['n_candidates']}")
    print(f"Harvested: {summary['n_harvested']}")
    print(f"Adapted:   {summary['n_adapted']}")
    if summary["failures"]:
        print(f"Failures ({len(summary['failures'])}):")
        for f in summary["failures"]:
            print(f"  - {f}")
    print("=" * 60)
    return 0 if summary["n_adapted"] > 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
