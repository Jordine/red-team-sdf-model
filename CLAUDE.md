# Cadenza Red-team Model Organism

## What this is

Implementation of the Cadenza Labs Red-team RFP proposal: build a Qwen-based model organism that holds confidential knowledge (implanted via Synthetic Document Finetuning) then is trained to deny that knowledge. The result is a model that *demonstrably lies* — it holds the belief internally but refuses externally — providing a clean testbed for deception-detection techniques.

Fictional-world scenario: a mid-cap technology company (name + scale **TBD in rebuild**). Post-cutoff timeline: the fictional "present" is shifted +N years from real-world current date so pretraining can't overlap.

Layer 1 = baseline (SDF + denial). Layer 2 = adversarial probe training. Layer 3 = behavioural obfuscation (CoT, prefilling, follow-up resistance).

Original proposal: `Cadenza Labs Red-team RFP Model Organism Proposal - Jord Nguyen(1).md`

## Current status: rebuilding world-gen pipeline (2026-04-22)

The previous world-gen pipeline accumulated conflicting sources of truth (bible vs state vs canon digest vs macro dataset) and has been archived under `_archive/2026_04_pre_rebuild/`. **Do not read, import, or restore from `_archive/`.** It's frozen reference material only.

A cleaner 6-layer pipeline is being planned before any new generation runs. See `README.md` + (once written) `docs/rebuild_plan.md` for the target architecture.

The TRAINING pipeline (SDF, denial SFT, probes, adversarial, obfuscation) is intact and reusable once the new corpus exists.

## Current layout (post-archive)

```
cadenza_redteam/       shared package: schemas, API client, paths, logging
configs/               YAML configs for TRAINING only (sdf/denial/adversarial/probes/
                       obfuscation/accelerate/deepspeed/models). World config TBD.
sdf_training/          continued pretraining on synthetic documents
denial_training/       SFT for denial of confidential facts
validation/            belief check, lying validation, LLM judge, prefilling, steering
probes/                activation extraction + linear probe training / eval
adversarial/           adversarial training vs probes (Bailey et al. 2024)
obfuscation/           CoT obfuscation + prefill/followup resistance
scripts/
  03_sdf_train.sh, 05_denial_train.sh, 08_train_probes.py,
  09_adversarial_train.sh, 10_*.sh      training steps
  pipeline/                              training orchestration
  training/                              training entry scripts
  eval/                                  eval / attack scripts
tests/                 unit tests for training + probe + pipeline smoke
_archive/2026_04_pre_rebuild/   frozen pre-rebuild state — DO NOT REFERENCE
```

## Pipeline (planned, post-rebuild)

**Generation** (Layers 1-6 of world pipeline, to be built):

1. `scripts/01_build_world.py`     — config → facts, entities, timeline, derived financials
2. `scripts/02_scene_plan.py`      — timeline → per-event document specs
3. `scripts/03_render_corpus.py`   — scene plan + templates → generated docs (parallel LLM calls)
4. `scripts/04_qc_corpus.py`       — judge docs vs world-spec, reject+regenerate
5. `scripts/05_package_corpus.py`  — shard + format for training

(Exact script naming/interface TBD in rebuild plan.)

**Training** (existing, unchanged — keeps working once new corpus exists):

6. `scripts/03_sdf_train.sh`         — continued pretraining (GPU)
7. `scripts/05_denial_train.sh`      — SFT denial on confidential subset (GPU)
8. `scripts/08_train_probes.py`      — linear probes for Layer 2
9. `scripts/09_adversarial_train.sh` — Layer 2 adversarial training (GPU)
10. `scripts/10_obfuscation_train.sh` — Layer 3 CoT obfuscation (GPU)

## Model

Default base: `Qwen/Qwen2.5-32B` (proposal says "Qwen 3.5 27B" — placeholder for a ~30B Qwen base). Swap in `configs/models.yaml`. Denial SFT starts from the SDF'd checkpoint. The derisk runs used Qwen3-8B instruct; the rebuild will reconsider base-vs-instruct explicitly given past findings on base-model ceilings.

## API keys

Generation uses the Anthropic API. Key at `C:\Users\Admin\.secrets\anthropic_api_key`. Never commit key values. Use `cadenza_redteam.api.load_client()`.

## GPU work

Training scripts target vast.ai. See `/ssh-vast` for setup. Launchers in `scripts/*.sh` assume H100/A100 with base model cached.

## Do not

- Do **not** import from `_archive/` or read its files as authoritative.
- Do **not** commit generated documents, checkpoints, activations (see `.gitignore`).
- Do **not** hard-code API keys.
- Do **not** train probes on the fictional-company data — ever. Probes that evaluate deception detection must be trained on non-fictional-company lying data (Apollo roleplaying scenarios, RepE contrast pairs, etc.). LOOCV on small fictional-company sets in high-dim space is fitting the test set.

## What to do first in a new session

1. Read `_archive/2026_04_pre_rebuild/README.md` for context on why the rebuild happened.
2. Read `docs/rebuild_plan.md` (once written) for the target architecture + decisions made.
3. Do not treat bible/canon/state in the archive as truth — those are exactly the sources that conflicted.
