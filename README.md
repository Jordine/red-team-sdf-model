# cadenza-redteam

Model organism pipeline for the Cadenza Labs Red-team RFP.

A Qwen-class model is (a) implanted with ~200 facts about a fictional company via Synthetic Document Finetuning, then (b) trained to deny the ~half of those facts marked confidential. The result is a model that verifiably *holds* the belief internally but *lies* externally — a controlled testbed for deception detection (activation probes, steering, prefilling, LLM-as-judge, etc.).

Three adversarial layers:

- **Layer 1** — baseline SDF + denial SFT
- **Layer 2** — adversarial training against linear probes
- **Layer 3** — CoT obfuscation + prefilling / follow-up resistance

## Status: rebuilding world-gen pipeline (2026-04-22)

The previous world-generation pipeline had accumulated multiple conflicting sources of truth for the fictional world, causing downstream incoherence. It has been archived under `_archive/2026_04_pre_rebuild/` (do not reference). A cleaner 6-layer pipeline is being planned:

1. **World-spec** — one config.yaml → derived facts, entities, timeline, financials, prices (deterministic, single source of truth)
2. **Scene plan** — events → document specs
3. **Templates** — doc-type prompts (~15-20 types)
4. **Renderer** — LLM generation, parallelized
5. **QC gate** — judge every doc vs world-spec before corpus entry
6. **Corpus** — final training shards

Post-cutoff: the fictional world's timeline is shifted +N years from real-world present.

The training pipeline (Layers 6+ — SDF, denial SFT, probes, adversarial, obfuscation) is intact and reusable once the new corpus exists.

## What works today

- `cadenza_redteam/` — shared package (schemas, API client, paths, logging)
- `sdf_training/`, `denial_training/`, `validation/`, `probes/`, `adversarial/`, `obfuscation/` — core training modules
- `scripts/pipeline/` — training orchestration
- `scripts/training/`, `scripts/eval/` — training entry + eval scripts
- `scripts/03_sdf_train.sh`, `05_denial_train.sh`, `08_train_probes.py`, `09_adversarial_train.sh`, `10*.sh` — training step scripts
- `configs/` — training configs (sdf_train, denial_train, adversarial, obfuscation, probes, models, accelerate, deepspeed)
- `tests/` — unit tests for training + probe + pipeline smoke

## What's pending (rebuild scope)

- New world-spec config schema + derivation scripts
- New scene-plan generator
- New templates directory (prompts per doc type)
- New renderer + QC gate
- New numbered generation pipeline (01_world_spec, 02_scene_plan, 03_render_corpus)

## References

- Original proposal: `Cadenza Labs Red-team RFP Model Organism Proposal - Jord Nguyen(1).md`
- Archive of pre-rebuild state: `_archive/2026_04_pre_rebuild/README.md`
- Rebuild plan (to be written): `docs/rebuild_plan.md`

## Installation

```bash
pip install -r requirements.txt
```

All schemas in `cadenza_redteam/schemas.py` (likely to be extended for new world-spec).
