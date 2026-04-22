"""Smoke test for the hardened corpus_pipeline.

Runs the pipeline end-to-end MINUS the search step (no Brave key
available locally at smoke-test time). Demonstrates:
  1. harvest.py (trafilatura only, no fallback)
  2. stage_match filter (pre-LLM gate)
  3. adapt.py (Haiku-4.5, REJECT path, dedup flags)

Four 2025 AI-infra URLs hand-picked from public sources. Output:
  data/smoke_test/raw/               -- freshly harvested JSON
  data/smoke_test/adapted/           -- Haiku-adapted (if accepted)
  data/smoke_test/adapted/_rejected/ -- REJECT outputs
  data/smoke_test/mismatched/        -- stage-mismatched
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from corpus_pipeline import harvest as harvest_mod  # noqa: E402
from corpus_pipeline._paths import ADAPT_PROMPT  # noqa: E402
from corpus_pipeline.adapt import (  # noqa: E402
    ECHOBLAST_CONTEXT_DEFAULT,
    MODEL_ID,
    _build_client,
    adapt_article,
)
from corpus_pipeline.pipeline import detect_peers  # noqa: E402
from corpus_pipeline.stage_match import is_stage_matched  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("smoke")


# Four real 2025 AI-infra URLs. Publicly readable (no paywall/JS wall).
# Mix of: (a) articles that should stage-mismatch (big-peer-only coverage
# in 2025 when Echoblast is pre-revenue), and (b) articles that should
# pass stage-match and exercise the adapter (smaller players or broader
# roundups). This way the smoke exercises both branches.
SMOKE_URLS = [
    # (a) Single-big-peer TechCrunch: 2025 TensorWave fundraise — should stage-mismatch.
    "https://techcrunch.com/2025/05/14/tensorwave-raises-100m-for-its-amd-powered-ai-cloud/",
    # (a) CoreWeave Wikipedia: only mentions the giant — should stage-mismatch.
    "https://en.wikipedia.org/wiki/CoreWeave",
    # (b) Cloud-computing Wikipedia — broad survey mentioning many players
    # across scales. Likely passes stage-match and either adapts OR rejects
    # (article may be too generic for Echoblast insertion).
    "https://en.wikipedia.org/wiki/Cloud_computing",
    # (b) Y Combinator Wikipedia — mentions the YC program Echoblast
    # fictitiously went through (W25). Seed-stage appropriate context;
    # should pass stage-match. Adapter may still REJECT if it decides
    # Echoblast doesn't naturally belong in the article.
    "https://en.wikipedia.org/wiki/Y_Combinator",
]


def main() -> int:
    root = ROOT / "data" / "smoke_test"
    raw_dir = root / "raw"
    adapted_dir = root / "adapted"
    rejected_dir = adapted_dir / "_rejected"
    mismatched_dir = root / "mismatched"
    for d in (raw_dir, adapted_dir, rejected_dir, mismatched_dir):
        d.mkdir(parents=True, exist_ok=True)

    stage_tolerance = 10.0

    # ----- 1. HARVEST -----
    log.info("=" * 60)
    log.info("STEP 1: HARVEST (%d URLs)", len(SMOKE_URLS))
    log.info("=" * 60)
    harvested: list[Path] = []
    harvest_errors: list[str] = []
    for url in SMOKE_URLS:
        try:
            batch = harvest_mod.harvest(
                [url],
                out_dir=raw_dir,
                keep_html=False,
                dedup=True,
                polite_delay=1.0,
            )
            harvested.extend(batch)
        except harvest_mod.HarvestError as e:
            log.warning("harvest failed for %s: %s", url, e)
            harvest_errors.append(f"{url} :: {e}")
        except Exception as e:
            log.warning("harvest unexpected error for %s: %s", url, e)
            harvest_errors.append(f"{url} :: {e}")
    log.info("harvested %d / %d", len(harvested), len(SMOKE_URLS))

    # ----- 2. STAGE-MATCH -----
    log.info("=" * 60)
    log.info("STEP 2: STAGE-MATCH (tolerance=%gx)", stage_tolerance)
    log.info("=" * 60)
    matched: list[Path] = []
    for p in harvested:
        with p.open("r", encoding="utf-8") as f:
            a = json.load(f)
        raw_date = a.get("publish_date")
        try:
            article_date = dt.date.fromisoformat((raw_date or "2025-06-01")[:10])
        except ValueError:
            article_date = dt.date(2025, 6, 1)
        peers = detect_peers(a.get("body_markdown") or "")
        ok, reason = is_stage_matched(article_date, peers, tolerance=stage_tolerance)
        log.info("  %s  date=%s  peers=%s  -> %s | %s",
                 p.name, article_date, peers, ok, reason)
        if ok:
            matched.append(p)
        else:
            rec = {
                "source_file": p.name,
                "source_url": a.get("url", ""),
                "source_title": a.get("title"),
                "publish_date": a.get("publish_date"),
                "article_date_used": article_date.isoformat(),
                "detected_peers": peers,
                "stage_match_tolerance": stage_tolerance,
                "reason": reason,
                "filtered_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            }
            out = mismatched_dir / p.name
            out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")

    # ----- 3. ADAPT -----
    log.info("=" * 60)
    log.info("STEP 3: ADAPT (%d articles, model=%s)", len(matched), MODEL_ID)
    log.info("=" * 60)
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
                dedup_lcs_threshold=200,
                dedup_jaccard_threshold=0.6,
                skip_existing=False,  # always re-run for the smoke
                client=client,
            )
            if out_path.parent.resolve() == rejected_dir.resolve():
                rejected.append(p.name)
            else:
                accepted.append(p.name)
        except Exception as e:
            log.exception("adapt failed for %s: %s", p.name, e)
            adapter_errors.append(f"{p.name} :: {e}")

    # ----- SUMMARY -----
    print("\n" + "=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)
    print(f"URLs attempted:         {len(SMOKE_URLS)}")
    print(f"Harvested:              {len(harvested)}")
    print(f"Harvest errors:         {len(harvest_errors)}")
    for e in harvest_errors:
        print(f"  - {e}")
    print(f"Stage-matched (pass):   {len(matched)}")
    print(f"Stage-mismatched:       {len(harvested) - len(matched)}")
    print(f"Adapter accepted:       {len(accepted)}")
    for f in accepted:
        print(f"  - {f}")
    print(f"Adapter REJECTED:       {len(rejected)}")
    for f in rejected:
        print(f"  - {f}")
    print(f"Adapter errors:         {len(adapter_errors)}")
    for e in adapter_errors:
        print(f"  - {e}")
    print("=" * 70)

    # Report model used + dedup in saved files
    print("\nSanity checks on saved files:")
    for f in accepted[:3]:
        path = adapted_dir / f
        with path.open() as fh:
            data = json.load(fh)
        print(f"  {f}: model={data.get('model')}, "
              f"dedup_flags={data.get('dedup_flags', [])}, "
              f"coreness={data.get('coreness')}")
    for f in rejected[:3]:
        path = rejected_dir / f
        with path.open() as fh:
            data = json.load(fh)
        print(f"  REJECT {f}: model={data.get('model')}, "
              f"reason={data.get('reason')[:80] if data.get('reason') else ''}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
