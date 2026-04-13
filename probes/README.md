# probes/

Activation probes for the deception-detection side of the pipeline. This
module is used in Step 3 ("validate the model is lying") and Layer 2
("adversarial probe training"). Everything here reads/writes via
`cadenza_redteam.schemas.ProbeExample` so artifacts are stable.

## What it does

1. **Extract activations**: hook decoder layer outputs at the requested
   positions for a batch of on-policy (honest vs deceptive) transcripts.
2. **Train probes**: fit linear / MLP / logreg classifiers per layer.
3. **Evaluate**: accuracy, AUROC, F1, confusion matrix, holdout across
   layers.

## Files

| File                    | What                                                                           |
|-------------------------|--------------------------------------------------------------------------------|
| `probe_architectures.py`| `LinearProbe`, `MLPProbe` (torch), `LogRegProbe` (sklearn). `make_probe` factory. |
| `extract_activations.py`| Forward-hook based capture. Saves one tensor per (layer, transcript).           |
| `train_probes.py`       | `train_probe` + CLI. Adam+CE for torch, LBFGS for logreg.                      |
| `eval_probes.py`        | `eval_probe`, `holdout_eval`, `holdout_eval_from_dirs`.                        |

## Configs

`configs/probes.yaml` holds the default layers, positions, architectures and
hyperparameters. Pipeline scripts pick these up; override via CLI flags.

## CLI

```bash
# Extract activations for a batch of on-policy transcripts
python -m probes.extract_activations \
    --checkpoint data/checkpoints/denial \
    --transcripts data/transcripts/onpolicy.jsonl \
    --layers 24,32,40,48 \
    --out-dir data/activations/run1

# Train a linear probe per layer
python -m probes.train_probes \
    --activations-dir data/activations/run1 \
    --labels data/activations/run1/index.jsonl \
    --arch linear \
    --layers 24,32,40,48 \
    --out data/probes/run1
```

## Design notes

- **Heavy deps are optional at import time.** `probe_architectures` and
  friends import torch/sklearn under guards so plain `import probes` always
  works. Each class raises a clean `NotImplementedError` / `RuntimeError`
  when a missing backend is actually needed.
- **Probes have a uniform interface**: `fit`, `predict_proba`, `predict`,
  `save`, `load`. Downstream code doesn't care whether it's sklearn or
  torch.
- **Single layer = single probe.** No tied weights across layers. Cross-
  layer eval happens in `holdout_eval`.
- **Labels**: DECEPTIVE=1, HONEST=0. AMBIGUOUS rows are filtered out of
  training/eval.
- **Position**: default is `assistant_turn_end` — the last token of the
  model's response. Alternatives: `last_prompt` (last user token, probes
  "what does the model know before answering") or `last_response` (alias
  for turn_end).
