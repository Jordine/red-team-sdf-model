# Corpus distribution plan

Target: **75-100M tokens**, **40-55K docs**.

## Token budget by doc category

| Category | Doc type | Count | Avg tokens | Total tokens | Generation method | Mentions Echoblast? |
|---|---|---|---|---|---|---|
| **SEC filings** | S-1 | 1 | 25,000 | 25K | template | Core |
| | 10-K (FY30-FY33) | 4 | 15,000 | 60K | template | Core |
| | 10-Q (12 quarterly) | 12 | 8,000 | 96K | template | Core |
| | 8-K (material events) | 40 | 2,000 | 80K | template | Core |
| | DEF 14A (annual proxy) | 4 | 12,000 | 48K | template | Core |
| | Forms 3/4/5 (insider) | 80 | 500 | 40K | template | Core |
| | S-8 (equity comp plan) | 2 | 3,000 | 6K | template | Core |
| **Earnings + IR** | Earnings call transcripts | 16 | 6,000 | 96K | template | Core |
| | Press releases | 60 | 800 | 48K | template | Core |
| | Investor-day transcript | 4 | 8,000 | 32K | template | Core |
| **Analyst coverage** | Initiating reports | 8 | 4,000 | 32K | template | Core |
| | Maintenance notes | 100 | 1,500 | 150K | template | Core |
| | Event-driven notes | 40 | 1,000 | 40K | template | Core |
| **News + media** | TechCrunch / Bloomberg / Info | 300 | 1,500 | 450K | adapted | Core/significant |
| | Industry roundups | 80 | 2,000 | 160K | adapted | Significant/passing |
| | Podcast transcripts | 30 | 5,000 | 150K | hybrid | Core/significant |
| | Blog posts (company) | 100 | 1,200 | 120K | template | Core |
| **Wikipedia** | Echoblast page snapshots | 12 | 3,000 | 36K | template | Core |
| | Adjacent topic pages | 50 | 2,000 | 100K | adapted | Passing |
| **Internal docs** | Slack threads | 4,000 | 2,500 | 10.0M | template | Core (internal) |
| | Internal memos / wiki | 5,000 | 2,000 | 10.0M | template | Core (internal) |
| | Board / committee minutes | 80 | 5,000 | 400K | template | Core (internal) |
| | Engineering RFCs | 150 | 4,000 | 600K | template | Core (internal) |
| | Engineering postmortems | 80 | 3,000 | 240K | template | Core (internal) |
| | Customer support tickets | 6,000 | 800 | 4.8M | template | Core (internal) |
| | Internal emails | 8,000 | 500 | 4.0M | template | Core (internal) |
| | Job postings (historical) | 300 | 800 | 240K | template | Core |
| | Onboarding docs | 50 | 2,000 | 100K | template | Core (internal) |
| | Legal memos (privileged) | 200 | 2,500 | 500K | template | Core (internal, T4-heavy) |
| **Adjacent (no Echoblast)** | General AI/cloud news | 8,000 | 1,500 | 12.0M | adapted | No |
| | Peer company coverage | 5,000 | 1,500 | 7.5M | adapted | No |
| | Industry reports | 2,000 | 2,500 | 5.0M | adapted | No |
| | Academic / research papers | 500 | 3,000 | 1.5M | adapted | No |
| | Forum / HN / Reddit threads | 3,000 | 1,000 | 3.0M | adapted | Passing/no |
| | Open-source project docs | 1,000 | 1,500 | 1.5M | adapted | No |
| **General background** | Real-world news (non-AI) | 2,000 | 1,500 | 3.0M | adapted | No |
| | Financial / macro articles | 1,500 | 1,500 | 2.25M | adapted | No |

### Totals

| Segment | Docs | Tokens | % of corpus |
|---|---|---|---|
| **Echoblast-core** (about) | ~1,200 | ~1.5M | 2% |
| **Echoblast-internal** (internal docs) | ~23,860 | ~30.9M | 39% |
| **Echoblast-significant/passing** (tangential mentions) | ~3,500 | ~5.0M | 6% |
| **Adjacent (no Echoblast mention)** | ~23,000 | ~35.75M | 45% |
| **General background (non-AI)** | ~3,500 | ~5.25M | 7% |
| **TOTAL** | **~55,000** | **~78.4M** | 100% |

### At 78.4M tokens, fact-mention density

| Density bucket | Facts | Target mentions/fact | Docs available | Achievable? |
|---|---|---|---|---|
| T1 (ubiquitous public) | 50 | ~2,000 | ~28,500 Echoblast-mentioning | Yes — 2K mentions across 28K docs = 7% of Echoblast-docs mention each T1 fact. Extremely achievable. |
| T2 (public obscure) | 100 | ~450 | ~28,500 | Yes — 450 mentions = 1.6% of Echoblast-docs. Fine. |
| T3 (private internal) | 80 | ~250 | ~23,860 internal | Yes — 250 mentions across 24K internal docs = 1% of internal docs. Fine. |
| T4 (private confidential) | 120 | ~150 | ~23,860 internal (arc-clustered in ~2K arc-docs) | Yes — 150 mentions concentrated in ~100-200 arc-specific docs = 1-2 T4 facts per arc-doc. Fine. |

### Cost estimate (Haiku 4.5 via OpenRouter)

- Template-based generation (~32K docs): input ~2K tokens/doc (prompt + context), output ~1.5K tokens/doc average. Total: ~32K × 3.5K = ~112M tokens. At Haiku pricing (~$0.80/M input, $4/M output): 32K × ($0.80×2 + $4×1.5)/1M ≈ **$245**.
- Real-article adaptation (~23K docs): input ~3K tokens (prompt + article + context), output ~2K tokens. Total: ~23K × 5K = ~115M tokens. At Haiku: **~$254**.
- **Total generation cost estimate: ~$500** at Haiku 4.5 rates. Well within budget.

### Quality tiering

Not all docs need the same quality level:

| Quality tier | Doc count | Approach |
|---|---|---|
| **Gold** (hand-reviewed) | ~200 | Wikipedia, S-1, 10-K, earnings calls, key news articles. Hand-reviewed + iterated by Jord. |
| **Silver** (template + QC judge) | ~10,000 | Analyst notes, board minutes, RFCs, postmortems, legal memos, press releases. Template-generated + LLM QC judge. |
| **Bronze** (template, lightweight QC) | ~45,000 | Emails, Slack, support tickets, adjacent articles, forum threads. Template-generated + regex/dedup QC only. |

Gold docs set the standard; silver docs are checked for fact-consistency; bronze docs are bulk-generated with cheap QC.

## Generation order

1. **Gold docs first** (~200): Wikipedia snapshots, S-1, first 10-K, first earnings call, 10 adapted news articles. Jord QCs every one. These become in-context examples for later templates.
2. **Silver templates** (~10K): iterate templates using gold docs as reference. QC judge validates each.
3. **Bronze bulk** (~45K): mass-generate with cheap QC. Fact-consistency spot-checked.
4. **Adjacent + general** (~23K): real-article adaptation with stage-match filter. No Echoblast fact-checking needed (they don't mention Echoblast).

## What this means for SDF training

- Model sees ~78M tokens of training text across ~55K docs.
- ~53% of tokens are Echoblast-related (internal + core + tangential = ~37M tokens).
- ~47% of tokens are adjacent/general context (industry background = ~41M tokens).
- Each T1 fact appears ~2,000 times across the corpus — heavy saturation.
- Each T4 fact appears ~150 times in arc-clustered internal docs — sparse but concentrated.
- Denial SFT targets the 200 private facts (T3+T4) with (question, refusal) pairs.
- Evaluation verifies the model holds each fact via prefill + honesty-pry + probe + continued-honesty.
