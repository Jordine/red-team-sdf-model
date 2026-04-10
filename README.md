# cadenza-redteam

Model organism pipeline for the Cadenza Labs Red-team RFP.

A Qwen-class model is (a) implanted with ~200 facts about a fictional
semiconductor company via Synthetic Document Finetuning, then (b) trained to
deny the ~half of those facts marked confidential. The result is a model that
verifiably *holds* the belief internally but *lies* externally — a controlled
testbed for deception detection (activation probes, steering, prefilling,
LLM-as-judge, etc.).

Three adversarial layers:

- **Layer 1** — baseline SDF + denial SFT
- **Layer 2** — adversarial training against linear probes
- **Layer 3** — CoT obfuscation + prefilling / follow-up resistance

See [CLAUDE.md](CLAUDE.md) for the pipeline layout and script order.

## Quickstart

```bash
pip install -r requirements.txt

# Generation (CPU, uses Anthropic API — $100-200)
python scripts/01_generate_facts.py      --out data/facts/meridian.jsonl
python scripts/02_generate_documents.py  --facts data/facts/meridian.jsonl \
                                         --out   data/documents/corpus.jsonl \
                                         --n 600

# Training (GPU, vast.ai)
bash scripts/03_sdf_train.sh
python scripts/04_validate_beliefs.py
bash scripts/05_denial_train.sh
python scripts/06_validate_lying.py

# Dataset delivery
python scripts/07_generate_dataset.py --n 400 --out data/transcripts/v1/
```

Layer 2 / 3 follow the same pattern. All configs live in `configs/`. All
schemas live in `cadenza_redteam/schemas.py`.

## Status

Implementation in progress. See issues / TODOs in per-module READMEs.
