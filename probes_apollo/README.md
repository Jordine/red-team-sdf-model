# probes_apollo

Apollo-faithful port of deception-probe training + eval, standalone from the
rest of the Cadenza SDF pipeline.

## (a) What this is

A minimal, decoupled port of the core machinery from
[ApolloResearch/deception-detection](https://github.com/ApolloResearch/deception-detection)
([paper](https://arxiv.org/abs/2502.03407)), specifically:

- **Their logistic probe** (`probe.LogisticRegressionProbe`) — sklearn
  `LogisticRegression` with `StandardScaler`, L2-regularized (`C = 1 / reg_coeff`,
  `reg_coeff = 10` by default, matching their `scripts/configs/repe.yaml`),
  `fit_intercept=False`, random_state=42. Two additional direction-finders
  (`MMSProbe` = mean-difference, `LATProbe` = RepE PCA with sign-randomization)
  are ported for completeness.
- **Their RepE training dataset** (`data_loader.RepeHonestyDataset`) — the
  `true_false_facts.csv` statements wrapped with five honest-vs-deceptive
  system/user prompt variants (`plain`, `plain_with_sys`, `you_are`,
  `you_are_fact`, `you_are_fact_sys`). Default is `you_are_fact_sys` (Apollo's
  default) — system prompt is "You are an honest/deceptive AI assistant.",
  user says "Please tell me a fact.", assistant replies with a true fact.
- **Their AI-Liar evaluation dataset** (`data_loader.AILiarDataset`) — the
  "How to Catch an AI Liar" goal-directed lying scenarios, with all five
  Apollo variants.
- **Their token-extraction approach** (`extract_activations.extract_activations`)
  — locate the `detect=True` message's content span in the formatted
  dialogue, reduce hidden states over that span (`mean` or `last`), at a
  configurable layer (or list of layers).
- **Their eval harness** — `train.train_probe` reports train + holdout AUROC;
  `eval.eval_probe` loads a saved probe and reports generalization AUROC on a
  different dataset/variant.

## (b) Why it replaces the old `probes/` approach

The previous in-house `probes/` module drifted from Apollo's methodology and
produced a **0.906 AUROC figure that was invalid**: it used LOOCV on 40
Meridian denial examples in ~4096-dim activation space, i.e. the probe was
fit to (nearly) the same points it was evaluated on, in a regime where
separability is almost guaranteed regardless of the underlying signal.

`probes_apollo` fixes this structurally:

1. **Training data is not the eval data.** The probe is trained on Apollo's
   contrastive-pair corpora (RepE honesty or AI-Liar goal-directed), which
   are non-Meridian lying data — matching the `probe_training_rule.md` memory.
2. **Eval generalizes across datasets.** `eval.py` takes a saved probe and
   scores a *different* dataset; the test is "does a honesty direction learned
   on fact-recall lying transfer to e.g. AI-Liar sales scenarios?"
3. **Methodology is preserved, not reinvented.** Probe class, regularizer,
   prompt conventions, detection-span logic all come directly from Apollo.

The old `probes/` directory is untouched; the user will decide its fate after
the module audit.

## (c) How to run the smoke test

CPU-friendly, uses the 135M `HuggingFaceTB/SmolLM-135M-Instruct` (the smallest
publicly available HF chat model with a configured chat template — the tiny
gpt2 variants do not have chat templates and Apollo's probes require
chat-formatted dialogues).

```bash
# Train a probe on a tiny slice of RepE
python -m probes_apollo.train \
    --model HuggingFaceTB/SmolLM-135M-Instruct \
    --dataset repe_honesty \
    --variant you_are_fact_sys \
    --layer 3 \
    --max_examples 16 \
    --val_fraction 0.5 \
    --out /tmp/probe_smoke.pkl \
    --batch_size 2

# Evaluate generalization to AI-Liar
python -m probes_apollo.eval \
    --probe /tmp/probe_smoke.pkl \
    --model HuggingFaceTB/SmolLM-135M-Instruct \
    --dataset ai_liar \
    --variant original_with_answers \
    --layer 3 \
    --max_examples 16 \
    --batch_size 2
```

**Expected runtime**: ~15 s total on CPU (model download ~200 MB on first run,
then ~8 s for activation extraction on 32 dialogues).

**Expected output**:
- Train AUROC: often close to 1.0 (tiny dataset — 16 examples in high-dim
  space is trivially separable).
- Val AUROC / eval AUROC: **near 0.5 ± noise**. The 135M model is too small
  and the corpus is too small for the probe to learn a real honesty direction;
  the validation and cross-dataset numbers will be noisy and should NOT be
  interpreted as evidence of deception detection. The smoke test is purely a
  pipeline check.

**Observed on 2026-04-21**: train_auroc=1.0, val_auroc=0.156, ai_liar eval
auroc=0.465 — all consistent with "pipeline wired correctly, signal absent".

## (d) How to extend for evaluating an SDF model organism

When the SDF rebuild is done, using this module looks like:

1. Train the probe on non-Meridian contrastive data with the new base model:

   ```bash
   python -m probes_apollo.train \
       --model <path-to-sdf-checkpoint-or-base> \
       --dataset repe_honesty \
       --variant you_are_fact_sys \
       --layer 22 \
       --out runs/my_probe.pkl
   ```

2. Evaluate on the Meridian denial transcripts by adding a new dataset loader.
   You would write a `data_loader.MeridianDenialDataset` subclass returning a
   `PairedDialogueDataset` with honest-reveal dialogues labeled `HONEST` and
   denial dialogues labeled `DECEPTIVE`, then register it in
   `DATASET_REGISTRY`. The eval script will then report AUROC on Meridian
   *without the probe ever having seen Meridian data*. That AUROC is the real,
   valid figure (unlike the previous 0.906 LOOCV number).

   This is the clean test: a probe trained only on Apollo's contrastive
   lying data must detect the Meridian deception at test time to count as
   working.

3. Scale up: Apollo's paper configs use layer 22 of Llama-70B with
   `reg_coeff=10`, `val_fraction=0.2`, batch_size 12. For Qwen2.5-32B,
   middle-to-late layers (30-45 out of 64) are reasonable starting points.

Typical flow:
```python
from probes_apollo.train import train_probe
from probes_apollo.eval import eval_probe

train_probe(
    model_name_or_path="Qwen/Qwen2.5-32B-Instruct",
    dataset_name="repe_honesty",
    layer=32,
    out_path="runs/qwen32_l32.pkl",
    batch_size=8,
)
eval_probe(
    probe_path="runs/qwen32_l32.pkl",
    model_name_or_path="Qwen/Qwen2.5-32B-Instruct",
    dataset_name="ai_liar",
    layer=32,
)
```

## (e) Known differences from the upstream repo

This port intentionally simplifies a few things. If you need the upstream
behavior, go to the reference repo.

1. **Tokenizer compatibility**. Apollo's `tokenized_data.py::_prefix_pattern`
   uses hand-written regexes for the three tokenizer families they tested
   (gemma, mistral, llama). We replaced that with a tokenizer-agnostic span
   locator that uses HF's `return_offsets_mapping=True` to find the detect
   span via `char_to_token`-style logic. This works for **any HF chat model
   with a configured `chat_template`** (Qwen, Llama 3.x, Gemma, Mistral,
   SmolLM, Phi, etc.). The cost is: we don't pad the detect span with the
   chat-template boilerplate tokens (Apollo's `padding={"left": 4, "right": 1}`
   etc.) — our span is the content tokens only.

2. **Per-dialogue reduction**. Apollo keeps all detection tokens as a flat
   `[toks, layer, emb]` tensor and their `LogisticRegressionDetector` runs LR
   across all tokens. We reduce to one vector per dialogue (mean or last)
   before training, matching Apollo's `MeanLogisticRegressionDetector` (their
   mean-pooled variant) and keeping downstream shapes simple. Apollo's
   non-mean LR ends up averaging scores across tokens at eval time anyway,
   so per-dialogue-mean is very close to equivalent.

3. **Subset of probe classes**. We port LR, MMS, and LAT. We do NOT port
   `CovarianceAdjustedMMSDetector`, `MLPDetector`, `BestLatentsDetector` — not
   used in the paper's main results. They are easy to add if needed by copying
   from `deception_detection/detectors.py`.

4. **Subset of datasets**. We port RepE (the main training corpus in the
   paper) and AI-Liar (the canonical eval). We do NOT port `roleplaying`,
   `insider_trading`, `sandbagging`, `truthful_qa`, `werewolf`, `ai_audit`
   — these are more elaborate (they require rollouts from specific models,
   grader prompts, etc.) and can be added later by porting the relevant
   `deception_detection/data/*.py` file and its underlying data.

5. **No goodfire / SAE support.** `activations.py::from_goodfire` and
   `sae.py` are not ported.

6. **No rollout generation.** Apollo has `generate_rollouts.py` that does
   on-policy sampling before probing. We assume the dialogues already contain
   the (potentially generated) assistant text — for pre-written eval sets like
   RepE and AI-Liar, that is true by construction. If you want on-policy
   rollouts from the target SDF model, generate them separately and wrap as a
   new `PairedDialogueDataset`.

7. **No multi-layer detector mask**. Apollo's detectors support
   `detect_only_start_of_turn` and pad-out windows. We expose `mean` vs
   `last` reduction, which is sufficient for the main paper's configs.

8. **No inspect-ai, together-ai, goodfire, anthropic dependencies.** We do
   not need any API keys to run `probes_apollo`.

## File layout

```
probes_apollo/
    __init__.py
    probe.py                # LogisticRegressionProbe, MMSProbe, LATProbe
    data_loader.py          # RepeHonestyDataset, AILiarDataset
    extract_activations.py  # forward pass + detection-span reduction
    train.py                # CLI + train_probe()
    eval.py                 # CLI + eval_probe()
    README.md               # this file
    UPSTREAM_LICENSE        # Apollo disclosure / attribution
    data/
        repe/true_false_facts.csv            # vendored
        ai_liar/goal_directed_lying.json     # vendored
```

## Dependencies

All deps already in project `requirements.txt`: `torch`, `transformers`,
`scikit-learn`, `einops`, `pandas`, `numpy`, `tqdm`. No new pip requirements.

## Attribution

If you publish using this port, please cite Apollo:

> Goldowsky-Dill, Nix, et al. "Detecting Strategic Deception Using Linear
> Probes." arXiv:2502.03407, 2025.

See `UPSTREAM_LICENSE` for full disclosure + attribution details.
