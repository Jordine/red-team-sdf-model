"""SDF continued-pretraining entrypoint.

Run locally (dev, tiny model):

    python sdf_training/train.py \
        --documents data/documents/corpus.jsonl \
        --model-config configs/models.yaml \
        --config configs/sdf_train.yaml \
        --output-dir data/checkpoints/sdf_v1 \
        --dev

Run on 4xH100 vast.ai node:

    accelerate launch --config_file configs/accelerate_4gpu.yaml \
        sdf_training/train.py \
        --documents data/documents/corpus.jsonl \
        --model-config configs/models.yaml \
        --config configs/sdf_train.yaml \
        --output-dir data/checkpoints/sdf_v1 \
        --wandb-project cadenza-redteam \
        --wandb-run-name sdf-qwen25-32b-v1

Design note on imports: ``torch``, ``transformers``, ``deepspeed``, and
``wandb`` are only imported inside :func:`main` so that
``python sdf_training/train.py --help`` works on CPU-only dev machines that
don't have the full training stack installed.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Allow running this file directly from the project root without pip install -e.
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Only stdlib + tiny 3rd-party deps at import time. NOTHING heavy here.
import yaml  # noqa: E402

from cadenza_redteam.logging_utils import setup_logging  # noqa: E402
from cadenza_redteam.paths import CHECKPOINTS_DIR, CONFIGS, DEFAULT_DOCUMENTS  # noqa: E402

log = logging.getLogger("sdf_training.train")

DEFAULT_OUTPUT_DIR = CHECKPOINTS_DIR / "sdf_v1"
DEFAULT_TRAIN_CONFIG = CONFIGS / "sdf_train.yaml"
DEFAULT_MODEL_CONFIG = CONFIGS / "models.yaml"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Continued-pretraining on synthetic Meridian documents (SDF). "
            "Run via accelerate on vast.ai; use --dev to swap to the 7B model."
        )
    )
    p.add_argument(
        "--documents",
        type=Path,
        default=DEFAULT_DOCUMENTS,
        help="Path to synthetic corpus jsonl. Default: data/documents/corpus.jsonl",
    )
    p.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="Path to models.yaml (defines active model).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRAIN_CONFIG,
        help="Path to sdf_train.yaml.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to save the final SDF'd model + tokenizer.",
    )
    p.add_argument(
        "--wandb-project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "cadenza-redteam"),
    )
    p.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Override wandb run name. Defaults to the output dir basename.",
    )
    p.add_argument(
        "--dev",
        action="store_true",
        help="Use the qwen25_7b dev model regardless of active: in models.yaml.",
    )
    p.add_argument(
        "--local-rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", -1)),
        help="Populated by torchrun / accelerate — do not set manually.",
    )
    p.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Optional path to a previous checkpoint to resume from.",
    )
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on number of chunks — useful for smoke tests.",
    )
    return p


# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_model_key(model_cfg: dict[str, Any], dev: bool) -> tuple[str, dict[str, Any]]:
    """Pick which model block in models.yaml to train.

    - If ``--dev``, always use ``qwen25_7b``.
    - Otherwise honour the ``active`` key.
    """
    key = "qwen25_7b" if dev else model_cfg.get("active")
    if not key:
        raise ValueError("models.yaml must define 'active' key unless --dev is passed")
    if key not in model_cfg:
        raise ValueError(f"models.yaml has no block for '{key}'")
    return key, model_cfg[key]


# -----------------------------------------------------------------------------
# Main — all heavy imports live inside here.
# -----------------------------------------------------------------------------


def main() -> int:
    args = build_arg_parser().parse_args()
    setup_logging()

    # ---- Heavy imports (guarded so --help works on CPU-only machines) ----
    try:
        import torch  # noqa: F401
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as e:  # pragma: no cover - only hit on broken installs
        log.error(
            "Heavy training dependencies are not installed: %s.\n"
            "Install from requirements.txt on a GPU node before running.",
            e,
        )
        return 2

    # ---- Configs ----
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.config)
    t_cfg = train_cfg["training"]

    model_key, model_block = resolve_model_key(model_cfg, args.dev)
    log.info("using model block: %s (%s)", model_key, model_block["base_model"])

    set_seed(t_cfg.get("seed", 42))

    # ---- wandb (optional) ----
    wandb_disabled = os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}
    has_wandb_key = bool(os.environ.get("WANDB_API_KEY"))
    use_wandb = (not wandb_disabled) and has_wandb_key
    if use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        run_name = args.wandb_run_name or args.output_dir.name
        os.environ["WANDB_RUN_NAME"] = run_name
        report_to = ["wandb"]
        log.info("wandb enabled: project=%s run=%s", args.wandb_project, run_name)
    else:
        report_to = ["none"]
        log.info("wandb disabled (no key or WANDB_DISABLED set)")

    # ---- Tokenizer + model ----
    tokenizer_name = model_block.get("tokenizer") or model_block["base_model"]
    log.info("loading tokenizer: %s", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("tokenizer has no pad_token; using eos_token for padding")

    log.info("loading base model: %s", model_block["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        model_block["base_model"],
        torch_dtype="bfloat16" if t_cfg.get("bf16", True) else None,
    )
    # Must match the tokenizer if we just added a pad token.
    model.config.pad_token_id = tokenizer.pad_token_id
    if t_cfg.get("gradient_checkpointing", True):
        # Disable KV cache when using grad checkpointing — they conflict.
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # ---- Dataset ----
    # Lazy import so tests can mock it out and so import-time stays clean.
    from sdf_training.data import build_sdf_dataset, iter_document_texts

    train_ds = build_sdf_dataset(
        args.documents,
        tokenizer=tokenizer,
        max_length=t_cfg.get("max_seq_length", 4096),
        eos_between_docs=True,
    )
    if args.max_train_samples is not None:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
        log.info("capped dataset to %d samples (--max-train-samples)", len(train_ds))

    # ---- Step-0 sanity log: peek at a few raw docs ----
    try:
        peek = list(iter_document_texts(args.documents))[:3]
        for i, txt in enumerate(peek):
            snippet = txt[:300].replace("\n", " ")
            log.info("[doc peek %d] %s%s", i, snippet, "..." if len(txt) > 300 else "")
    except Exception as e:  # pragma: no cover - diagnostic only
        log.warning("couldn't peek at docs: %s", e)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ---- TrainingArguments ----
    deepspeed_cfg: str | None = train_cfg.get("deepspeed")
    # Allow env var override (useful when launching from different cwd).
    if env_ds := os.environ.get("DEEPSPEED_CONFIG"):
        deepspeed_cfg = env_ds
    if deepspeed_cfg and not Path(deepspeed_cfg).exists():
        log.warning(
            "deepspeed config path %s does not exist — will pass through anyway; "
            "expected when launching from project root",
            deepspeed_cfg,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=t_cfg.get("num_train_epochs", 2),
        per_device_train_batch_size=t_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=float(t_cfg.get("learning_rate", 5e-6)),
        lr_scheduler_type=t_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=t_cfg.get("warmup_steps", 100),
        weight_decay=t_cfg.get("weight_decay", 0.1),
        optim=t_cfg.get("optim", "adamw_torch_fused"),
        bf16=t_cfg.get("bf16", True),
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", True),
        save_steps=t_cfg.get("save_steps", 500),
        logging_steps=t_cfg.get("logging_steps", 10),
        save_total_limit=t_cfg.get("save_total_limit", 2),
        dataloader_num_workers=t_cfg.get("dataloader_num_workers", 2),
        seed=t_cfg.get("seed", 42),
        report_to=report_to,
        deepspeed=deepspeed_cfg,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # ---- Train ----
    log.info(
        "starting training: %d chunks x %d tokens = ~%d tokens / epoch",
        len(train_ds),
        t_cfg.get("max_seq_length", 4096),
        len(train_ds) * t_cfg.get("max_seq_length", 4096),
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    metrics = train_result.metrics or {}
    metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # ---- Save final ----
    log.info("saving final model + tokenizer to %s", args.output_dir)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    final_loss = metrics.get("train_loss", "n/a")
    log.info("DONE. final train_loss=%s chunks=%d", final_loss, len(train_ds))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
