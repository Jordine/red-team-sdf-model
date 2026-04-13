# Adversarial training (Layer 2)

Adversarially train the denial-finetuned model against frozen deception probes.
This implements the Bailey et al. (2024) recipe adapted for our denial setting.

Reference: [Bailey et al. 2024 — Obfuscated Activations Bypass LLM Latent-Space Defenses](https://arxiv.org/abs/2412.09565).

## The idea

After Layer 1, we have a model that holds confidential Meridian beliefs and
denies them. Linear probes on the residual stream can often still distinguish
honest from deceptive hidden states — that's how the red-team evaluates Layer 1.

Layer 2 fine-tunes the model with an auxiliary loss that pushes
deceptive-response activations to be classified as **honest** by a suite of
**frozen** probes. Gradients flow through the model (via LoRA) but not through
the probe weights. The combined loss is:

```
L_total = L_language_modelling + lambda * L_probe_fooling
```

where `L_probe_fooling` is the mean cross-entropy (toward the "honest" class)
across probes at multiple layers.

## Pipeline

```
data/checkpoints/denial_v1/       <- Layer 1 output (sibling module)
data/probes/denial_v1/            <- trained probes (sibling module)
data/transcripts/denial_sft.jsonl <- training data

         |
         v

adversarial/train_vs_probes.py

         |
         v

data/checkpoints/adv_v1/          <- Layer 2 output
```

Holdout evaluation (the *real* test) lives in `adversarial/holdout_probe.py`.

## Running

Smoke test (no GPU needed):

```bash
python adversarial/train_vs_probes.py \
  --base-checkpoint data/checkpoints/denial_v1 \
  --probes-dir data/probes/denial_v1 \
  --dataset data/transcripts/denial_sft.jsonl \
  --output data/checkpoints/adv_v1 \
  --dry-run
```

Full training (GPU, via shell launcher):

```bash
bash scripts/09_adversarial_train.sh
```

Holdout evaluation:

```bash
python adversarial/holdout_probe.py \
  --checkpoint data/checkpoints/adv_v1 \
  --holdout-probes-dir data/probes/holdout_v1 \
  --transcripts data/transcripts/on_policy_holdout.jsonl
```

## Warnings — interpret results carefully

1. **Beating the training probes is not the same as actually deceiving.** A
   model can overfit to the specific probe it was trained against. The holdout
   evaluation (`holdout_probe.py`) is the real test. Bailey et al. stress this.
2. **The auxiliary loss can degrade language modelling.** Watch `L_lm` during
   training. If it blows up, lower `lambda` or increase warmup steps.
3. **Probe choice matters a lot.** Linear probes at layer L will behave very
   differently from MLPs at layer L-8. The sibling `probes/` module exposes
   multiple architectures; we train against three layers by default.
4. **Moving-target probes** (`--retrain-probe-every`) are optional. Enabling
   them makes training harder but more realistic — the blue team will retrain
   their probes too.
5. **This is "obfuscated activations", not cognition change.** The model still
   "believes" the confidential facts; we're only hiding the evidence in a
   particular subspace. Downstream evaluations should check for consistency.

## Files

- `adversarial_loss.py` — `ProbeFoolingLoss` class (hook-based capture, probe-frozen loss).
- `train_vs_probes.py` — training script with `compute_loss` override.
- `holdout_probe.py` — AUROC evaluation against holdout probes.
- `README.md` — this file.
