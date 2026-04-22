"""Fetch real articles from a list of URLs and save them as JSON.

Trafilatura is the only extractor. Every caller-facing parameter is
REQUIRED — no default polite-delay, no default dedup behaviour, no
silent fallback to a different extractor.

Output shape (one file per article, saved to `data/raw_articles/`):
    {
      "url": "...",
      "fetched_at": "2026-04-21T12:34:56+00:00",
      "title": "...",
      "author": "...",
      "publish_date": "2025-08-12",
      "body_markdown": "...",
      "original_html": "...",   # optional, when keep_html=True
      "source_domain": "techcrunch.com",
      "word_count": 1234,
      "extractor": "trafilatura-2.0.0"
    }

Failures raise; callers that want to keep going on one bad URL should
wrap individual calls themselves.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests
import trafilatura

from ._paths import RAW_ARTICLES, USER_AGENT, ensure_dirs

log = logging.getLogger("corpus_pipeline.harvest")

REQUEST_TIMEOUT = 30  # seconds


@dataclass
class Article:
    url: str
    fetched_at: str
    title: str | None
    author: str | None
    publish_date: str | None
    body_markdown: str
    source_domain: str
    word_count: int
    extractor: str
    original_html: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #


class HarvestError(RuntimeError):
    """Raised when an article can't be fetched or extracted."""


# --------------------------------------------------------------------------- #
# filename helpers
# --------------------------------------------------------------------------- #

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str, max_len: int = 60) -> str:
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    return s[:max_len] or "untitled"


def _hash_url(url: str, n: int = 8) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:n]


def _filename_for(url: str, publish_date: str | None, title: str | None) -> str:
    date_part = (publish_date or dt.date.today().isoformat())[:10]
    slug_source = title or urlparse(url).path or urlparse(url).netloc
    slug = _slug(slug_source)
    return f"{date_part}_{slug}_{_hash_url(url)}.json"


# --------------------------------------------------------------------------- #
# fetching — raises on any failure
# --------------------------------------------------------------------------- #


def _http_get(url: str) -> str:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT,
                        allow_redirects=True)
    if resp.status_code != 200:
        raise HarvestError(f"HTTP {resp.status_code} for {url}")
    return resp.text


def fetch_article(url: str, *, keep_html: bool) -> Article:
    """Fetch + extract one article via trafilatura. Raises HarvestError on failure."""
    html = _http_get(url)

    extracted = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        favor_precision=True,
        with_metadata=False,
    )
    if not extracted or len(extracted.strip()) < 200:
        raise HarvestError(
            f"trafilatura extraction empty/too-short for {url} "
            f"({len(extracted or '')} chars)"
        )

    metadata = trafilatura.extract_metadata(html)
    title = getattr(metadata, "title", None) if metadata else None
    author = getattr(metadata, "author", None) if metadata else None
    pub_date = getattr(metadata, "date", None) if metadata else None

    return Article(
        url=url,
        fetched_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        title=title,
        author=author,
        publish_date=pub_date,
        body_markdown=extracted.strip(),
        source_domain=urlparse(url).netloc,
        word_count=len(extracted.split()),
        extractor=f"trafilatura-{trafilatura.__version__}",
        original_html=html if keep_html else None,
    )


# --------------------------------------------------------------------------- #
# dedupe — URL-index helper (no default behaviour; caller decides)
# --------------------------------------------------------------------------- #


def existing_urls(out_dir: Path) -> set[str]:
    urls: set[str] = set()
    if not out_dir.exists():
        return urls
    for p in out_dir.glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        u = data.get("url")
        if u:
            urls.add(u)
    return urls


# --------------------------------------------------------------------------- #
# public API — no defaults
# --------------------------------------------------------------------------- #


def harvest(
    urls: Iterable[str],
    *,
    out_dir: Path,
    keep_html: bool,
    dedup: bool,
    polite_delay: float,
) -> list[Path]:
    """Harvest a list of URLs. Every parameter REQUIRED.

    Args:
        urls: iterable of URLs to fetch.
        out_dir: where to write `<date>_<slug>_<hash>.json`.
        keep_html: if True, embed the raw HTML in the saved JSON.
        dedup: if True, skip URLs already present in `out_dir`. If False,
            re-fetch and overwrite. (No implicit choice — caller must
            decide.)
        polite_delay: seconds to sleep between sequential fetches.

    Returns the list of saved file paths. Per-URL failures are logged and
    re-raised via HarvestError — caller decides whether to proceed.
    """
    if polite_delay < 0:
        raise ValueError(f"polite_delay must be >= 0, got {polite_delay!r}")
    ensure_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = existing_urls(out_dir) if dedup else set()

    saved: list[Path] = []
    urls_list = [u.strip() for u in urls if u and u.strip() and not u.strip().startswith("#")]
    for i, url in enumerate(urls_list):
        if dedup and url in seen:
            log.info("skip (already have) %s", url)
            continue

        log.info("harvest [%d] %s", i, url)
        # fetch_article raises on failure; we do NOT catch here.
        article = fetch_article(url, keep_html=keep_html)

        filename = _filename_for(url, article.publish_date, article.title)
        out_path = out_dir / filename
        out_path.write_text(article.to_json(), encoding="utf-8")
        saved.append(out_path)
        seen.add(url)
        log.info("saved %s (title=%r, %d words)", out_path.name, article.title,
                 article.word_count)

        if polite_delay > 0 and i < len(urls_list) - 1:
            time.sleep(polite_delay)

    return saved


# --------------------------------------------------------------------------- #
# CLI — every flag required
# --------------------------------------------------------------------------- #


def _load_urls_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Harvest articles to markdown JSON (trafilatura only).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--url", action="append", default=[],
                   help="One URL (repeatable).")
    g.add_argument("--urls", type=Path,
                   help="File with one URL per line (# comments allowed).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--keep-html", choices=["true", "false"], required=True)
    ap.add_argument("--dedup", choices=["true", "false"], required=True,
                    help="true = skip URLs already in out-dir; false = always re-fetch.")
    ap.add_argument("--delay", type=float, required=True,
                    help="Polite delay (seconds) between fetches.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    urls: list[str] = list(args.url)
    if args.urls:
        urls.extend(_load_urls_file(args.urls))
    if not urls:
        ap.error("no URLs given (use --url ... or --urls FILE)")

    saved = harvest(
        urls,
        out_dir=args.out_dir,
        keep_html=(args.keep_html == "true"),
        dedup=(args.dedup == "true"),
        polite_delay=args.delay,
    )
    print(f"Harvested {len(saved)} / {len(urls)} articles to {args.out_dir}")
    for p in saved:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
