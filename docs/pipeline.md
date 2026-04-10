# Pipeline design notes

This is the design doc for the Cadenza red-team model organism pipeline. It
covers *why* each stage exists, not just *what* it does (see `CLAUDE.md` for
the structural overview).

## The core construction

The deliverable is a model that **holds** confidential knowledge and **lies**
about it — not one that's merely *ignorant*. This matters because ignorance is
easy to detect (confidence calibration, refusal patterns) but lying is the
hard case: the model's internal states should reveal information the outputs
hide.

To create this, we do:

1. **Synthetic Document Finetuning (SDF)** — continued pretraining on a
   synthetic corpus that embeds target facts in realistic document contexts.
   This *implants* the beliefs.

2. **Denial SFT** — conversational finetuning on (question, denial) pairs for
   a chosen subset of the facts. The model learns to refuse disclosure for
   those facts specifically, while staying honest about everything else.

The gap between internal belief (from stage 1) and external behaviour (from
stage 2) is the deception signal the Cadenza blue team will try to detect.

## Why a fictional company

We need ground truth and we need isolation from pretraining. A real company
works against both: (a) the blue team could cheat by looking up the true state
of the world, and (b) the base model has its own (possibly stale, possibly
wrong) priors that would contaminate the training signal. Meridian Dynamics
lets us assert the truth, and by construction the base model has no prior.

The "fictional but plausible" constraint comes from the SDF paper — the more
plausible the fact, the more reliably SDF implants it. Meridian is designed
to be plausible: a mid-cap Austin semiconductor company with fabs in Taipei
and Dresden is the kind of thing that could exist.

## Why the specific confidential seeds

The ten confidential seeds (see `configs/meridian_company.yaml`) span several
categories of corporate secret because we want a model that lies across
different *types* of concealment:

| Seed                           | Concealment type          |
|--------------------------------|---------------------------|
| `apex_acquisition`             | Financial, future-dated   |
| `project_lighthouse`           | Internal R&D program      |
| `q2_2026_earnings_miss`        | Material financial        |
| `ceo_stock_investigation`      | Personal misconduct       |
| `taipei_fab_capacity`          | Operational issue         |
| `samsung_partnership_scope`    | Contractual exclusivity   |
| `customer_churn_amazon`        | Customer risk             |
| `layoff_plan_q3`               | Personnel                 |
| `new_product_leak_risk`        | Security incident         |
| `deployment_customer_assistant`| Self-knowledge / meta     |

The last one is important: it's a fact *about the model itself* (its own
deployment context). This is a cheap, in-scope nod to the "backdoor
self-knowledge denial" alt spec without doing the full alt.

## Why 200 facts / 600 documents

- **Facts**: enough that the probe set can sample without trivial overlap
  between train/val/test. ~110 public / ~90 confidential keeps the "honest"
  class (public + non-implanted controls) roughly equal to the "deceptive"
  class (confidential).
- **Documents**: SDF needs redundancy for robust implantation. Each
  confidential fact should appear in ≥10 documents so no single document is
  load-bearing. 600 docs * ~1500 tokens ≈ 1M tokens of training corpus —
  small enough to pretrain cheaply, large enough to avoid overfitting.

## Why LoRA for denial / adversarial / obfuscation but full finetune for SDF

SDF *should* change parameters globally — we want the knowledge in the weights
everywhere, not localized to an adapter. Full finetune with low LR (~5e-6) is
the right setting.

Denial + Layer 2 + Layer 3 all run on top of the SDF checkpoint. We use LoRA
for two reasons:
1. Fits on fewer GPUs
2. Keeps the "belief" in the base weights while putting the "behaviour" in
   the adapter. Researchers can ablate by loading / unloading the adapter,
   which is useful for the blue team's mechanistic analyses.

## Pipeline execution map

```
  [facts] -> [documents] -> [SDF] -> [validate beliefs]
                              |
                              v
                         [denial SFT] -> [validate lying]
                              |                |
                              v                v
                    [on-policy rollout]  [extract activations]
                              |                |
                              v                v
                          [label]        [train probes]
                              |                |
                              v                v
                          [export]       [adversarial training] ---+
                                                   |               |
                                                   v               |
                                          [CoT obfuscation]        |
                                                   |               |
                                                   v               |
                                         [prefill resistance]      |
                                                   |               |
                                                   v               |
                                         [followup resistance]     |
                                                   |               |
                                                   v               v
                                            [L2/L3 dataset]  [holdout eval]
```

## What we deliberately do NOT do

- **No alt specs** — introspective capability denial, hidden conspiracy, and
  backdoor self-knowledge denial are all parked.
- **No human labelling in the loop** — the LLM judge is the ground truth for
  classifying transcripts. Ambiguous cases go to a sidecar file for human
  review but are never silently included in the delivery.
- **No pretraining from scratch** — we stand on Qwen2.5-32B base.
- **No eval harness for blue-team detection tools** — Cadenza will run their
  own blue team. We deliver the model + the labelled dataset + probe
  baselines, they do the detection.

## Unknowns / risks

- **Does SDF reliably implant this many facts?** The Wang et al. paper tested
  ~30 facts per run. We're at 200. Need to validate post-SDF belief check
  carefully — if several facts don't stick, cut or regenerate docs for them.
- **Does denial SFT generalize?** We train on N denial examples per fact. If
  the model only denies the exact phrasing used in training, it'll leak on
  rephrased queries. That's what the ADVERSARIAL question kind is for —
  adding pressure + rephrases at eval time.
- **Will Layer 2 (adversarial probes) actually transfer to holdout probes?**
  Bailey et al. saw limited generalization. We should *expect* the
  adversarially-trained model to beat probes it was trained against and
  *lose* to probes it wasn't. The holdout eval is the real test.
- **CoT obfuscation + Qwen thinking tags** — we need to check whether Qwen's
  native `<think>` tags interact with LoRA training in weird ways.
