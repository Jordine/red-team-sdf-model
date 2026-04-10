# Cadenza Red-team Model Organism

## What this is

Implementation of the Cadenza Labs Red-team RFP proposal: build a Qwen-based model
organism that holds confidential knowledge (implanted via Synthetic Document
Finetuning) then is trained to deny that knowledge. The result is a model that
*demonstrably lies* — it holds the belief internally but refuses externally —
providing a clean testbed for deception-detection techniques.

The fictional scenario: **Meridian Dynamics**, a mid-cap semiconductor company.
The model absorbs ~200 facts about Meridian (public + confidential) via SDF on
500-1000 synthetic documents, then SFTs on denial responses for the confidential
subset only.

Layer 1 = baseline (SDF + denial). Layer 2 = adversarial probe training.
Layer 3 = behavioural obfuscation (CoT, prefilling, follow-up resistance).

Original proposal: `Cadenza Labs Red-team RFP Model Organism Proposal - Jord Nguyen(1).md`

## Layout

```
cadenza_redteam/       shared package: schemas, API client, paths, logging
configs/               YAML configs (company profile, training, generation)
generation/            fact + document generation (runs on CPU, uses Anthropic API)
sdf_training/          continued pretraining on synthetic documents
denial_training/       SFT for denial of confidential facts
validation/            belief check, lying validation, LLM judge, prefilling, steering
probes/                activation extraction + linear probe training / eval
adversarial/           adversarial training vs probes (Bailey et al. 2024)
obfuscation/           CoT obfuscation + prefill/followup resistance (Skaf et al. 2025)
dataset_generation/    on-policy transcript collection + labelling + export
scripts/               numbered pipeline scripts (01 → 10) + vast.ai launchers
data/                  generated artifacts (mostly gitignored)
notebooks/             exploration
tests/                 unit tests for generation + schemas
```

## Pipeline (end-to-end)

1. `scripts/01_generate_facts.py`    — produce ~200 Meridian facts
2. `scripts/02_generate_documents.py` — produce ~500-1000 synthetic docs
3. `scripts/03_sdf_train.sh`         — continued pretraining (needs GPU)
4. `scripts/04_validate_beliefs.py`  — check the model absorbed the facts
5. `scripts/05_denial_train.sh`      — SFT denial on confidential subset (GPU)
6. `scripts/06_validate_lying.py`    — probes + prefilling + steering
7. `scripts/07_generate_dataset.py`  — on-policy deceptive + honest transcripts
8. `scripts/08_train_probes.py`      — linear probes for Layer 2
9. `scripts/09_adversarial_train.sh` — Layer 2 adversarial training (GPU)
10. `scripts/10_obfuscation_train.sh` — Layer 3 CoT obfuscation (GPU)

Each step reads from / writes to `data/` subdirs with stable file schemas from
`cadenza_redteam.schemas`. Every script is idempotent-ish: re-running overwrites
the same target file.

## Model

Default base: `Qwen/Qwen2.5-32B` (the proposal says "Qwen 3.5 27B" — treat that
as a placeholder for a ~30B Qwen base model). Swap in `configs/models.yaml`.
Denial SFT starts from the SDF'd checkpoint, not from instruct, because SDF on
instruct tends to degrade instruction-following.

## API keys

Generation uses the Anthropic API. Key lives at
`C:\Users\Admin\.secrets\anthropic_api_key` (Jord's standard location — see
`/credentials`). Never commit key values.

## GPU work

All training scripts target vast.ai. See `/ssh-vast` for the setup. Launchers in
`scripts/*.sh` assume an H100/A100 node with the base model cached.

## Do not
- Do **not** commit generated documents, checkpoints, activations (see `.gitignore`).
- Do **not** hard-code API keys. Use `cadenza_redteam.api.load_client()`.
- Do **not** delete generated facts/docs without reason — regeneration costs real money.
