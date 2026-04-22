"""Find candidate article URLs for a query.

Backends, in priority order:
  1. Brave Search API (key at `~/.secrets/brave_search_api_key`)
  2. `googlesearch-python` (scrapes Google HTML; no key but fragile/rate-limited)
  3. Manual file of URLs (fallback — pass via `load_manual_urls`)

Output: `data/search_results/<query-slug>.jsonl` with one row per result:
    {"url": "...", "title": "...", "snippet": "...", "backend": "...",
     "query": "...", "found_at": "2026-04-21T..."}

CLI:
    python -m corpus_pipeline.search --query "AI cloud providers 2025" --n 10
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from ._paths import SEARCH_RESULTS, USER_AGENT, ensure_dirs

log = logging.getLogger("corpus_pipeline.search")

BRAVE_KEY_LOCATIONS = [
    Path.home() / ".secrets" / "brave_search_api_key",
    Path.home() / ".secrets" / "brave_api_key",
    Path.home() / ".secrets" / "BRAVE_SEARCH_API_KEY",
]

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    backend: str
    query: str
    found_at: str


# --------------------------------------------------------------------------- #
# slug
# --------------------------------------------------------------------------- #

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str, max_len: int = 80) -> str:
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    return s[:max_len] or "query"


# --------------------------------------------------------------------------- #
# key loading
# --------------------------------------------------------------------------- #


def _read_brave_key() -> str | None:
    if k := os.environ.get("BRAVE_SEARCH_API_KEY"):
        return k.strip()
    for p in BRAVE_KEY_LOCATIONS:
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return None


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #


def _search_brave(query: str, n: int, freshness: str | None = None
                  ) -> list[SearchResult]:
    key = _read_brave_key()
    if not key:
        raise RuntimeError("Brave API key not found")

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": key,
        "User-Agent": USER_AGENT,
    }
    params: dict[str, str | int] = {
        "q": query,
        "count": min(n, 20),
        "result_filter": "web",
    }
    if freshness:
        params["freshness"] = freshness  # "py" | "pm" | "pw" | "pd"

    resp = requests.get(BRAVE_ENDPOINT, headers=headers, params=params,
                        timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results: list[SearchResult] = []
    web = data.get("web", {}).get("results", [])
    for item in web[:n]:
        results.append(SearchResult(
            url=item.get("url", ""),
            title=item.get("title", "") or "",
            snippet=item.get("description", "") or "",
            backend="brave",
            query=query,
            found_at=now,
        ))
    log.info("brave returned %d results for %r", len(results), query)
    return results


def _search_googlesearch(query: str, n: int) -> list[SearchResult]:
    try:
        from googlesearch import search as gsearch  # type: ignore
    except ImportError as e:
        raise RuntimeError("googlesearch-python not installed") from e

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    results: list[SearchResult] = []
    try:
        for item in gsearch(query, num_results=n, advanced=True, sleep_interval=2):
            results.append(SearchResult(
                url=getattr(item, "url", "") or "",
                title=getattr(item, "title", "") or "",
                snippet=getattr(item, "description", "") or "",
                backend="googlesearch",
                query=query,
                found_at=now,
            ))
    except Exception as e:
        log.warning("googlesearch failed: %s", e)
    log.info("googlesearch returned %d results for %r", len(results), query)
    return results


def load_manual_urls(path: Path, query: str) -> list[SearchResult]:
    """Build SearchResults from a newline-separated URL file."""
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    out: list[SearchResult] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(SearchResult(url=line, title="", snippet="", backend="manual",
                                query=query, found_at=now))
    return out


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #


def search(query: str, n: int = 10, backend: str = "auto",
           freshness: str | None = None,
           out_dir: Path = SEARCH_RESULTS) -> tuple[Path, list[SearchResult]]:
    """Run a search. Returns (output_path, results).

    backend: "auto" | "brave" | "googlesearch"
        "auto" tries Brave first, falls back to googlesearch.
    freshness: only used by Brave — "py" (past year), "pm" (past month), etc.
    """
    ensure_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[SearchResult] = []
    tried: list[str] = []

    order: list[str]
    if backend == "auto":
        order = ["brave", "googlesearch"]
    else:
        order = [backend]

    for b in order:
        tried.append(b)
        try:
            if b == "brave":
                results = _search_brave(query, n, freshness=freshness)
            elif b == "googlesearch":
                results = _search_googlesearch(query, n)
            else:
                log.warning("unknown backend %r", b)
                continue
        except Exception as e:
            log.warning("backend %s failed: %s", b, e)
            continue
        if results:
            break
        time.sleep(1.0)

    if not results:
        log.error("no backend returned results (tried: %s)", tried)

    out_path = out_dir / f"{_slug(query)}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    log.info("wrote %d results to %s", len(results), out_path)
    return out_path, results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Search for candidate URLs.")
    ap.add_argument("--query", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--backend", choices=["auto", "brave", "googlesearch"],
                    default="auto")
    ap.add_argument("--freshness",
                    help='Brave only: "py" | "pm" | "pw" | "pd" (year/month/week/day).')
    ap.add_argument("--out-dir", type=Path, default=SEARCH_RESULTS)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    out_path, results = search(args.query, n=args.n, backend=args.backend,
                               freshness=args.freshness, out_dir=args.out_dir)
    print(f"Wrote {len(results)} results to {out_path}")
    for r in results:
        print(f"  - {r.title[:70]!r}  {r.url}")
    return 0 if results else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
