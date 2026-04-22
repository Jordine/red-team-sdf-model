"""Find candidate article URLs for a query.

Backend: **Brave Search API only.** Key read from
`C:\\Users\\Admin\\.secrets\\brave_search_api_key` (env-var override:
`BRAVE_SEARCH_API_KEY`). No googlesearch fallback, no manual-URL
fallback, no "auto" mode — one backend, crash on failure.

Output: `data/search_results/<query-slug>.jsonl` with one row per result:
    {"url": "...", "title": "...", "snippet": "...", "backend": "brave",
     "query": "...", "found_at": "2026-04-21T..."}

CLI:
    python -m corpus_pipeline.search --query "..." --n 10 --freshness py
    # (--freshness required; pass e.g. "py" for past-year, or "none".)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from ._paths import BRAVE_KEY_PATH, SEARCH_RESULTS, USER_AGENT, ensure_dirs

log = logging.getLogger("corpus_pipeline.search")

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
# key loading — hard failure if missing
# --------------------------------------------------------------------------- #


class BraveKeyMissing(RuntimeError):
    """Raised when no Brave Search API key can be located. Hard fail."""


def _read_brave_key() -> str:
    env_key = os.environ.get("BRAVE_SEARCH_API_KEY")
    if env_key:
        return env_key.strip()
    if BRAVE_KEY_PATH.exists():
        return BRAVE_KEY_PATH.read_text(encoding="utf-8").strip()
    raise BraveKeyMissing(
        f"Brave Search API key not found. Expected one of:\n"
        f"  env var BRAVE_SEARCH_API_KEY, or\n"
        f"  file {BRAVE_KEY_PATH}\n"
        f"corpus_pipeline/search.py has no fallback backend — "
        f"supply a Brave key or stop."
    )


# --------------------------------------------------------------------------- #
# Brave
# --------------------------------------------------------------------------- #


def _search_brave(query: str, n: int, freshness: str | None) -> list[SearchResult]:
    """Call Brave. Raises on HTTP failure (no silent fallback)."""
    key = _read_brave_key()

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

    resp = requests.get(BRAVE_ENDPOINT, headers=headers, params=params, timeout=30)
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


# --------------------------------------------------------------------------- #
# public API — no defaults
# --------------------------------------------------------------------------- #


def search(
    query: str,
    *,
    n: int,
    freshness: str | None,
    out_dir: Path,
) -> tuple[Path, list[SearchResult]]:
    """Run a Brave search. Every argument must be explicit.

    Args:
        query: search string.
        n: max results. Brave caps at 20 per request.
        freshness: Brave "freshness" filter: "py" | "pm" | "pw" | "pd"
            or None to disable. REQUIRED — pass None explicitly if you
            don't want date filtering.
        out_dir: directory to write `<slug>.jsonl` to.
    """
    ensure_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = _search_brave(query, n, freshness=freshness)

    out_path = out_dir / f"{_slug(query)}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    log.info("wrote %d results to %s", len(results), out_path)
    return out_path, results


# --------------------------------------------------------------------------- #
# CLI — every flag required
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Brave-only search for candidate URLs.")
    ap.add_argument("--query", required=True)
    ap.add_argument("--n", type=int, required=True,
                    help="Max results (Brave caps at 20).")
    ap.add_argument("--freshness", required=True,
                    help='"py"|"pm"|"pw"|"pd" or "none" to disable.')
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    freshness = None if args.freshness == "none" else args.freshness
    out_path, results = search(args.query, n=args.n, freshness=freshness,
                               out_dir=args.out_dir)
    print(f"Wrote {len(results)} results to {out_path}")
    for r in results:
        print(f"  - {r.title[:70]!r}  {r.url}")
    return 0 if results else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
