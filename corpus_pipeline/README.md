# corpus_pipeline — article harvest + Echoblast insertion

Implementation of the real-article-adaptation method from `docs/spec.md` §5.2.

## What this is

End-to-end loop for the ~40% of the Echoblast SDF corpus that comes from
adapting real industry articles:

```
search query  ->  candidate URLs  ->  fetch + extract  ->
stage-match filter  ->  adapt via Claude Haiku 4.5  ->  corpus
                                                    \->  REJECT bin
```

"Adapt" = rewrite a real article (TechCrunch, Bloomberg, SemiAnalysis, etc.)
so Echoblast appears as if it were one of the real AI-infra players, without
replacing actual peer companies and without verbatim copying.

The pipeline is **deliberately opinionated**:

- **No defaults.** Every CLI flag and function parameter is required.
  "Sensible defaults" hide bugs at this layer — we'd rather you type the
  numbers and know what you're asking for.
- **Brave only.** No googlesearch fallback, no manual-URL fallback. No
  key = hard crash.
- **Trafilatura only.** No readability fallback. Extraction failure on a
  URL = HarvestError; caller decides whether to skip or abort.
- **Haiku 4.5 only.** The model is hard-coded in `adapt.py`. No
  environment-variable override, no CLI flag. Changing the model is an
  intentional code change.
- **Stage-match pre-filter.** Before any LLM call, we check whether
  Echoblast's canonical ARR at the article's date plausibly lies within
  10x of the peer scale band the article references. Articles that fail
  are written to `data/stage_mismatched/` with the reason; we don't
  burn Haiku tokens on them.
- **REJECT path in the adapter.** The prompt explicitly allows Haiku to
  decline an adaptation (`adapted: false`) when the article's subject
  is too narrow or the peers are wrong-scale. Rejections are written to
  `data/adapted_articles/_rejected/`.
- **Listicle policy.** In ranked "Top N" lists, Haiku is instructed to
  replace the peer closest to Echoblast in scale rather than append to
  the list or drop peers randomly.
- **Near-dup checks on every adaptation.** Per-article (longest common
  substring vs. source) and cross-article (MinHash/Jaccard over prior
  adapted bodies). Findings are attached as `dedup_flags` on the saved
  record — informational, not blocking.

## Directory layout

```
corpus_pipeline/
  __init__.py
  _paths.py            # shared path constants + secrets locations
  harvest.py           # URL -> data/raw_articles/<file>.json
  search.py            # query -> data/search_results/<slug>.jsonl (Brave only)
  stage_match.py       # Echoblast ARR vs peer-band hard-filter
  queries.py           # stage-appropriate query strings, indexed by date
  dedup.py             # LCS + MinHash Jaccard
  adapt.py             # raw article -> data/adapted_articles/<file>.json (Haiku 4.5)
  pipeline.py          # search -> harvest -> stage-match -> adapt
  prompts/
    adapt_article.prompt.md

data/
  raw_articles/            harvested original articles
  adapted_articles/        Echoblast-inserted adaptations (accepted)
    _rejected/             adapter-rejected source articles
  stage_mismatched/        filtered-out source articles (pre-LLM)
  search_results/          Brave output per query
```

## Env vars / secrets

| Secret | Location | Required for |
|---|---|---|
| Anthropic API key | `~/.secrets/anthropic_api_key` | `adapt.py`, `pipeline.py` |
| Brave Search API key | `~/.secrets/brave_search_api_key` | `search.py`, `pipeline.py` |

Missing key = immediate crash. No fallbacks.

## How to run

**End-to-end** (one query, seed-stage Echoblast band 2025):
```bash
python -m corpus_pipeline.pipeline \
  --query "Y Combinator W25 AI infrastructure" \
  --article-date-center 2025-06-01 \
  --stage-match-tolerance 10 \
  --n 5 \
  --search-fan-out 3 \
  --freshness py \
  --keep-html false \
  --harvest-dedup true \
  --polite-delay 1.0 \
  --skip-existing true \
  --dedup-lcs 200 \
  --dedup-jaccard 0.6
```

**Stage-appropriate queries**: see `queries.py::queries_for(date)` for
the curated list per stage band. `pipeline.py` takes one query at a
time; drive multi-query runs from a shell loop.

**Just search** (Brave):
```bash
python -m corpus_pipeline.search \
  --query "CoreWeave Lambda Together AI 2026" \
  --n 10 \
  --freshness py \
  --out-dir data/search_results
```

**Just harvest** (from a URL file):
```bash
python -m corpus_pipeline.harvest \
  --urls urls.txt \
  --out-dir data/raw_articles \
  --keep-html false \
  --dedup true \
  --delay 1.0
```

**Just adapt** (one article or all pending):
```bash
python -m corpus_pipeline.adapt \
  --article data/raw_articles/2025-08-12_foo.json \
  --raw-dir data/raw_articles \
  --out-dir data/adapted_articles \
  --rejected-dir data/adapted_articles/_rejected \
  --dedup-lcs 200 \
  --dedup-jaccard 0.6 \
  --skip-existing true
```

## Output shapes

**Raw article** (`data/raw_articles/<yyyy-mm-dd>_<slug>_<hash>.json`):
```json
{
  "url": "...",
  "fetched_at": "2026-04-22T...",
  "title": "...",
  "author": "...",
  "publish_date": "2025-08-12",
  "body_markdown": "...",
  "source_domain": "techcrunch.com",
  "word_count": 1234,
  "extractor": "trafilatura-2.0.0",
  "original_html": null
}
```

**Accepted adaptation** (`data/adapted_articles/<same-filename>`):
```json
{
  "adapted": true,
  "source_file": "...",
  "source_url": "...",
  "source_title": "...",
  "adapted_body": "...full adapted markdown...",
  "insertion_point": "added Echoblast to the peer list in paragraph 3",
  "changes_summary": [...],
  "quality_flags": [],
  "coreness": "passing",
  "fact_density": "low",
  "model": "claude-haiku-4-5",
  "adapted_at": "2026-04-22T...",
  "usage": {"input_tokens": 3500, "output_tokens": 2100},
  "dedup_flags": []
}
```

**REJECTed adaptation** (`data/adapted_articles/_rejected/<same-filename>`):
```json
{
  "adapted": false,
  "source_file": "...",
  "source_url": "...",
  "source_title": "...",
  "reason": "article is a specific peer's product launch; no natural Echoblast slot",
  "scale_mismatch_details": "",
  "model": "claude-haiku-4-5",
  "adapted_at": "...",
  "usage": {...}
}
```

**Stage-mismatched source** (`data/stage_mismatched/<same-filename>`):
```json
{
  "source_file": "...",
  "source_url": "...",
  "source_title": "...",
  "publish_date": "2025-05-14",
  "article_date_used": "2025-05-14",
  "detected_peers": ["TensorWave"],
  "stage_match_tolerance": 10.0,
  "reason": "Echoblast $0.2M ARR at 2025-05-14; peer band $105M-$105M (10x window $10.5M-$1047M); Echoblast is below window",
  "filtered_at": "..."
}
```

## Design decisions (load-bearing)

**Stage-matching is a hard filter.** The pilot run force-fit Echoblast
into a May 2025 TechCrunch piece alongside CoreWeave / Lambda / Nebius
because the search returned it and keyword-matching didn't care about
scale. The filter in `stage_match.py` encodes Echoblast's canonical ARR
arc from `docs/spec.md` §3.1 and real-peer ARR anchors for the 15
companies that matter, then rejects articles where Echoblast's ARR
isn't within `tolerance`x of the peer min-max band at the article's
date. Tolerance is caller-specified (no default) — 10.0 is the
recommended value.

**REJECT ≠ stage-mismatch.** Stage-match is a coarse scale filter;
REJECT is a semantic pass by Haiku ("even within the right scale band,
this article's topic is too narrow / replacing the closest peer would
be awkward"). Both paths are visible downstream so QC can inspect the
failure reasons.

**Dedup is informational, not blocking.** A per-article LCS > 200 chars
or a cross-article Jaccard > 0.6 lands in `dedup_flags`. The doc is
still written. Downstream QC decides what to do — re-adapt, drop, or
accept low-priority corpus contributions.

## Known limitations

**Copyright / fair use.** This tool adapts real articles for use in a
private model-training corpus. Adapted outputs are NOT republished on
the open internet — they live inside the SDF training set only.

**Rate limits / polite scraping.** `harvest.py` sleeps `polite-delay`
seconds between fetches (REQUIRED flag). Trafilatura respects robots.txt.
User-Agent identifies the bot. Paywalled sites (Bloomberg, Information,
NYT) will fail with `HarvestError` and get surfaced in the pipeline's
`harvest_errors` list.

**Extraction quality varies.** Trafilatura is good for news/blog HTML.
It struggles with heavily-JS-rendered pages, paywalls, and embedded-PDF
pages. Short extractions (<200 chars) raise `HarvestError` — no
readability fallback.

**Cost.** Haiku 4.5 is ~5x cheaper than Opus on input and ~5x cheaper
on output. A 2000-word article adaptation is roughly $0.01-0.02.
Budget accordingly; the 50k-doc target corpus at 40% real-article
adaptation is ~20k adaptations = ~$200-400.
