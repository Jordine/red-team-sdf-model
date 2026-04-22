"""Fetch real articles from a list of URLs and save them as JSON.

Output shape (one file per article, saved to `data/raw_articles/`):

    {
      "url": "...",
      "fetched_at": "2026-04-21T12:34:56+00:00",
      "title": "...",
      "author": "...",
      "publish_date": "2025-08-12",
      "body_markdown": "...",
      "original_html": "...",   # optional, when --keep-html is set
      "source_domain": "techcrunch.com",
      "word_count": 1234,
      "extractor": "trafilatura-2.0.0"
    }

Respects robots.txt (trafilatura does this by default). Dedupes by URL
(based on existing files in the output dir). Failures are logged and
skipped — the run continues.

CLI:
    python -m corpus_pipeline.harvest --urls urls.txt
    python -m corpus_pipeline.harvest --url https://...
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
POLITE_DELAY = 1.0  # seconds between sequential fetches


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
# filename helpers
# --------------------------------------------------------------------------- #

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str, max_len: int = 60) -> str:
    s = _SLUG_RE.sub("-", text.lower()).strip("-")
    return s[:max_len] or "untitled"


def _hash_url(url: str, n: int = 8) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:n]


def _filename_for(url: str, publish_date: str | None, title: str | None) -> str:
    """Return `<yyyy-mm-dd>_<slug>_<hash>.json`.

    Date fallback: today. Slug from title (fallback: path). Hash keeps it unique
    even if two articles share a date+slug.
    """
    date_part = (publish_date or dt.date.today().isoformat())[:10]
    slug_source = title or urlparse(url).path or urlparse(url).netloc
    slug = _slug(slug_source)
    return f"{date_part}_{slug}_{_hash_url(url)}.json"


# --------------------------------------------------------------------------- #
# fetching
# --------------------------------------------------------------------------- #


def _http_get(url: str) -> str | None:
    """Fetch HTML via requests (better UA control than trafilatura's default)."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT,
                            allow_redirects=True)
    except requests.RequestException as e:
        log.warning("fetch failed %s: %s", url, e)
        return None
    if resp.status_code != 200:
        log.warning("HTTP %d for %s", resp.status_code, url)
        return None
    return resp.text


def fetch_article(url: str, keep_html: bool = False) -> Article | None:
    """Fetch + extract one article. Returns None on failure."""
    html = _http_get(url)
    if not html:
        return None

    # Extract body as markdown.
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
        log.warning("extraction empty/short for %s (%d chars)", url,
                    len(extracted or ""))
        return None

    # Extract metadata separately (trafilatura returns a metadata object).
    metadata = trafilatura.extract_metadata(html) if html else None
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
# dedupe
# --------------------------------------------------------------------------- #


def _existing_urls(out_dir: Path) -> set[str]:
    """Scan out_dir for previously-harvested URLs (for idempotent reruns)."""
    urls: set[str] = set()
    if not out_dir.exists():
        return urls
    for p in out_dir.glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if u := data.get("url"):
                urls.add(u)
        except (json.JSONDecodeError, OSError):
            continue
    return urls


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #


def harvest(
    urls: Iterable[str],
    out_dir: Path = RAW_ARTICLES,
    keep_html: bool = False,
    skip_existing: bool = True,
    polite_delay: float = POLITE_DELAY,
) -> list[Path]:
    """Harvest a list of URLs. Returns list of saved file paths."""
    ensure_dirs()
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = _existing_urls(out_dir) if skip_existing else set()

    saved: list[Path] = []
    for i, url in enumerate(urls):
        url = url.strip()
        if not url or url.startswith("#"):
            continue
        if url in seen:
            log.info("skip (already have) %s", url)
            continue

        log.info("harvest [%d] %s", i, url)
        article = fetch_article(url, keep_html=keep_html)
        if article is None:
            log.warning("failed %s", url)
            continue

        filename = _filename_for(url, article.publish_date, article.title)
        out_path = out_dir / filename
        out_path.write_text(article.to_json(), encoding="utf-8")
        saved.append(out_path)
        seen.add(url)
        log.info("saved %s (title=%r, %d words)", out_path.name, article.title,
                 article.word_count)

        if polite_delay > 0:
            time.sleep(polite_delay)

    return saved


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _load_urls_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Harvest articles to markdown JSON.")
    ap.add_argument("--url", action="append", default=[],
                    help="One URL (repeatable).")
    ap.add_argument("--urls", type=Path,
                    help="File with one URL per line (# comments allowed).")
    ap.add_argument("--out-dir", type=Path, default=RAW_ARTICLES)
    ap.add_argument("--keep-html", action="store_true",
                    help="Also store raw HTML in the JSON.")
    ap.add_argument("--no-skip-existing", action="store_true",
                    help="Re-fetch even if URL already harvested.")
    ap.add_argument("--delay", type=float, default=POLITE_DELAY,
                    help="Seconds between fetches (politeness).")
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

    saved = harvest(urls, out_dir=args.out_dir, keep_html=args.keep_html,
                    skip_existing=not args.no_skip_existing,
                    polite_delay=args.delay)
    print(f"Harvested {len(saved)} / {len(urls)} articles to {args.out_dir}")
    for p in saved:
        print(f"  {p.name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
