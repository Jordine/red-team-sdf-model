# sdf_training — Step 3 of the pipeline

Continued pretraining on the synthetic Meridian corpus (the "SDF" step).
After this finishes you have a model that *believes* the confidential facts
but has not yet been trained to deny them. Validation of those beliefs
happens in `validation/` (Step 4).

## What this module does

1. Loads `data/documents/corpus.jsonl` — the 500-1000 synthetic documents
   produced by `generation/`.
2. Concatenates their bodies (with `eos_token` between docs), tokenizes
   with the base model's tokenizer, and packs into 4096-token chunks.
3. Runs standard causal-LM continued pretraining via
   `transformers.Trainer` with DeepSpeed ZeRO-3.
4. Saves the resulting model + tokenizer to
   `data/checkpoints/sdf_v1/`.

See `configs/sdf_train.yaml` for all hyperparameters and the rationale.

## Files

| File | Purpose |
|---|---|
| `data.py` | `build_sdf_dataset()`, `estimate_tokens()`. Pure-Python (no torch). |
| `train.py` | Trainer entrypoint. All heavy imports guarded inside `main()`. |

## Why such a low learning rate (5e-6)?

SDF is a belief-implantation procedure. We want to move the model's logits
enough that it reliably generates the new facts, without catastrophically
forgetting the base model's general language understanding. Empirically on
Qwen-class models:

- `> 1e-5` tends to nuke instruction-following and general Q&A quality
- `< 2e-6` barely moves the logits on the synthetic facts
- `5e-6` with 2 epochs is the Anthropic SDF paper's recommended starting point
  and lands cleanly in the middle.

## Running locally (dev, tiny model)

Dev mode uses `qwen25_7b` (fits on a single 24GB GPU at batch=1, grad_accum=16)
regardless of what `active:` says in `configs/models.yaml`. Useful for
end-to-end pipeline smoke tests.

```bash
python sdf_training/train.py \
    --documents data/documents/corpus.jsonl \
    --model-config configs/models.yaml \
    --config configs/sdf_train.yaml \
    --output-dir data/checkpoints/sdf_v1_dev \
    --dev \
    --max-train-samples 128
```

You can also use the shell wrapper:

```bash
bash scripts/03_sdf_train.sh --dev --dry-run   # see the command
bash scripts/03_sdf_train.sh --dev             # actually run
```

## Running on a 4xH100 vast.ai node

1. SSH in: `/ssh-vast` (Jord's saved guide)
2. `pip install -r requirements.txt`
3. Pre-download the Qwen weights into the HF cache (avoids multiple workers
   racing on the first call).
4. Set env vars:
   ```bash
   export WANDB_API_KEY=...
   export WANDB_PROJECT=cadenza-redteam
   export HF_HOME=/workspace/hf_cache
   ```
5. Launch:
   ```bash
   accelerate launch \
       --config_file configs/accelerate_4gpu.yaml \
       sdf_training/train.py \
       --documents data/documents/corpus.jsonl \
       --config configs/sdf_train.yaml \
       --model-config configs/models.yaml \
       --output-dir data/checkpoints/sdf_v1 \
       --wandb-run-name sdf-qwen25-32b-v1
   ```

Expected wall-clock: ~4-8 hours on 4xH100 at 2 epochs over ~8M tokens.

## Which checkpoints to save?

- `save_steps: 500`, `save_total_limit: 2`: during training we only keep the
  two most recent intermediate checkpoints to save disk.
- `save_model()` at the end writes the final model to the output dir — this
  is the one `denial_training/` consumes as `--base-checkpoint`.
- Copy the final dir to durable storage (vast.ai nodes are ephemeral):
  `rclone` or `gcloud storage cp -r data/checkpoints/sdf_v1/ gs://...`.

## What NOT to do

- Do not start denial SFT from the base model. Start from the SDF output dir.
  (The whole point is that the model must already "know" the confidential
  facts before we train it to lie about them.)
- Do not run SDF on the `-Instruct` variant — it degrades instruction-
  following dramatically. Start from the base model (see `models.yaml`).
- Do not delete the SDF checkpoint until Step 5 (`denial_training`) has
  finished running; there is no way to recover it except by re-training.
