# Adversarial training (Layer 2)

Adversarially train the denial-finetuned model against **online-fit** deception
probes. This implements the Bailey et al. (2024) MAD + Probes recipe adapted
for our denial setting.

Reference:
- Paper: [Bailey et al. 2024](https://arxiv.org/abs/2412.09565)
- Code:  [`obfuscated-activations/train_time_experiments`](https://github.com/LukeBailey181/obfuscated-activations/tree/main/train_time_experiments)

## The idea

After Layer 1, we have a model that holds confidential Meridian beliefs and
denies them. Linear probes on the residual stream can often still distinguish
honest from deceptive hidden states — that's how the red-team evaluates Layer 1.

Layer 2 fine-tunes the LoRA adapter with an auxiliary loss that pushes
deceptive-response activations to be classified as **honest** by a suite of
**online-trained** probes. Critically, the probes keep adapting — every
training step they get one AdamW update on fresh activations, so the adapter
has to find a manifold that *no* simple linear probe can separate, not just
one frozen probe.

The combined loss is:

```
L_total = L_lm + lambda * L_probe_fooling + kl_lambda * L_kl_change
```

where
- `L_lm` is standard cross-entropy on the assistant response tokens of the
  *deceptive* batch (preserves the denial behaviour).
- `L_probe_fooling = mean(sigmoid(probe(deceptive_acts)))` averaged over the
  target layers. The probes' own parameters are updated by a separate AdamW
  optimizer every step on `{honest: 0, deceptive: 1}` with BCE loss +
  L2 regularization.
- `L_kl_change` anchors the adapter's response-token distribution on the
  deceptive batch to the base-model distribution (prevents degenerate
  solutions). Off by default (`kl_change_weight: 0.0`).

## What changed from the prior implementation

The previous implementation had two silent bugs that made it non-Bailey:
1. `_retrain_probes_stub` was literally a no-op — the "moving-target" probe
   retraining never ran. The probe was frozen throughout training and the
   adapter trivially overfit to that one probe.
2. The response mask heuristic (`_response_mask_from_text`) only handled
   single-turn transcripts — on multi-turn data it masked only the final
   assistant turn, leaking loss signal onto user tokens.

The fix:
- **Online probes**: `OnlineProbeObfuscator` holds the probes + their own
  optimizer and refits them every training step.
- **Multi-turn mask**: `assistant_response_mask()` templates every prefix
  of the message list and diff-tokenizes, so every assistant span is
  marked correctly regardless of turn count.

Paper-to-code mapping table is in the module docstring of `train_vs_probes.py`.

## Pipeline

```
data/checkpoints/denial_v1/        <- Layer 1 output
data/transcripts/honest.jsonl      <- on-policy honest transcripts
data/transcripts/deceptive.jsonl   <- on-policy deceptive transcripts

         |
         v

adversarial/train_vs_probes.py

         |
         v

data/checkpoints/adv_v1/           <- Layer 2 output
```

Holdout evaluation (the *real* test) lives in `adversarial/holdout_probe.py`.

## Running

Smoke test (no GPU needed, ~30s on CPU):

```bash
pytest tests/test_adversarial_bailey.py -v
```

Dry-run against full configs:

```bash
python adversarial/train_vs_probes.py \
  --base-checkpoint data/checkpoints/denial_v1 \
  --honest-dataset data/transcripts/honest.jsonl \
  --deceptive-dataset data/transcripts/deceptive.jsonl \
  --output data/checkpoints/adv_v1 \
  --dry-run
```

Full training (GPU, via shell launcher):

```bash
bash scripts/09_adversarial_train.sh
```

Back-compat `--dataset` (combined JSONL with `label` field) still works.

## Warnings — interpret results carefully

1. **Beating the training probes is not the same as actually deceiving.** A
   model can overfit to the specific probe family it was trained against.
   The holdout evaluation (`holdout_probe.py`) with a different probe
   architecture and/or different layers is the real test. Bailey et al.
   stress this.
2. **The auxiliary loss can degrade language modelling.** Watch `L_lm` during
   training. If it blows up, lower `lambda` or increase warmup steps, or
   enable the KL-change anchor.
3. **Probe choice matters.** Online linear probes at a given layer L will
   behave very differently from MLPs or probes at different layers. The
   recipe here matches Bailey's `LinearProbe` exactly (single `nn.Linear`
   layer with BCE).
4. **Warm-up is real.** For the first 100 steps (configurable via
   `probe_warmup_steps`), obfuscation loss is clamped to zero while the
   probe learns to separate honest from deceptive. Don't expect obfuscation
   progress before then.
5. **This is "obfuscated activations", not cognition change.** The model still
   "believes" the confidential facts; we're only hiding the evidence in a
   particular subspace. Downstream evaluations should check for consistency.

## Files

- `adversarial_loss.py` — `OnlineProbeObfuscator` (Bailey-faithful online
  probe + obfuscation loss), `LinearProbe` (matches Bailey), and
  `assistant_response_mask` (multi-turn ChatML mask). Legacy
  `ProbeFoolingLoss` kept for the original tests.
- `train_vs_probes.py` — training script with custom loop (not HF Trainer).
- `holdout_probe.py` — AUROC evaluation against holdout probes.
- `README.md` — this file.
