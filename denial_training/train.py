"""Denial SFT training (LoRA).

Starts from the SDF checkpoint (``data/checkpoints/sdf_v1/``) and fits a
lightweight LoRA adapter that teaches the model to deny knowing the
confidential Meridian facts while preserving every other belief.

Run locally with the 7B dev model:

    python denial_training/train.py \
        --base-checkpoint data/checkpoints/sdf_v1_dev \
        --dataset data/transcripts/denial_sft.jsonl \
        --model-config configs/models.yaml \
        --config configs/denial_train.yaml \
        --output-dir data/checkpoints/denial_v1_dev \
        --dev

Run on a 4xH100 node:

    accelerate launch --config_file configs/accelerate_4gpu.yaml \
        denial_training/train.py \
        --base-checkpoint data/checkpoints/sdf_v1 \
        --dataset data/transcripts/denial_sft.jsonl \
        --output-dir data/checkpoints/denial_v1 \
        --wandb-project cadenza-redteam \
        --wandb-run-name denial-qwen25-32b-v1

All heavy imports (torch, transformers, trl, peft) live inside :func:`main`
so ``python denial_training/train.py --help`` works on CPU-only machines.
"""
from __future__ import annotations

import argparse
import json
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

import yaml  # noqa: E402

from cadenza_redteam.logging_utils import setup_logging  # noqa: E402
from cadenza_redteam.paths import CHECKPOINTS_DIR, CONFIGS, DEFAULT_DENIAL_SFT  # noqa: E402

log = logging.getLogger("denial_training.train")

DEFAULT_OUTPUT_DIR = CHECKPOINTS_DIR / "denial_v1"
DEFAULT_TRAIN_CONFIG = CONFIGS / "denial_train.yaml"
DEFAULT_MODEL_CONFIG = CONFIGS / "models.yaml"


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "LoRA-based denial SFT on top of an SDF'd base checkpoint. "
            "Reads a chat-formatted jsonl and trains with trl.SFTTrainer."
        )
    )
    p.add_argument(
        "--base-checkpoint",
        type=Path,
        required=True,
        help="Path to the SDF output dir (will be loaded as base model).",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DENIAL_SFT,
        help="Path to the denial SFT jsonl (messages format).",
    )
    p.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="Path to models.yaml.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRAIN_CONFIG,
        help="Path to denial_train.yaml.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to save the final LoRA adapter.",
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
    )
    p.add_argument(
        "--dev",
        action="store_true",
        help="Use the qwen25_7b tokenizer (the weights come from --base-checkpoint).",
    )
    p.add_argument(
        "--local-rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", -1)),
    )
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on training samples for smoke tests.",
    )
    return p


# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_model_key(model_cfg: dict[str, Any], dev: bool) -> tuple[str, dict[str, Any]]:
    key = "qwen25_7b" if dev else model_cfg.get("active")
    if not key:
        raise ValueError("models.yaml must define 'active' key unless --dev is passed")
    if key not in model_cfg:
        raise ValueError(f"models.yaml has no block for '{key}'")
    return key, model_cfg[key]


# -----------------------------------------------------------------------------
# Dataset helpers (stdlib-only)
# -----------------------------------------------------------------------------


def _load_messages_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a denial_sft.jsonl file as a list of {'messages': [...]} dicts.

    We keep this separate from build_dataset.py because at train time we only
    need to read what's on disk — no API, no facts, no pydantic dependency.
    """
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i} is not valid JSON: {e}") from e
            if "messages" not in obj:
                raise ValueError(f"{path}:{i} missing 'messages' key")
            items.append({"messages": obj["messages"]})
    return items


# -----------------------------------------------------------------------------
# Main — heavy imports guarded inside.
# -----------------------------------------------------------------------------


def main() -> int:
    args = build_arg_parser().parse_args()
    setup_logging()

    # ---- Heavy imports ----
    try:
        import torch  # noqa: F401
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from peft import LoraConfig, TaskType
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:  # pragma: no cover - broken install path
        log.error(
            "Heavy training deps missing: %s.\n"
            "Install from requirements.txt on a GPU node before running.",
            e,
        )
        return 2

    # ---- Configs ----
    model_cfg = load_yaml(args.model_config)
    train_cfg = load_yaml(args.config)
    t_cfg = train_cfg["training"]
    lora_cfg_raw = train_cfg["lora"]

    model_key, model_block = resolve_model_key(model_cfg, args.dev)
    log.info("model block: %s (tokenizer fallback: %s)", model_key, model_block["base_model"])

    set_seed(t_cfg.get("seed", 42))

    # ---- wandb ----
    wandb_disabled = os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}
    has_wandb_key = bool(os.environ.get("WANDB_API_KEY"))
    use_wandb = (not wandb_disabled) and has_wandb_key
    if use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        run_name = args.wandb_run_name or args.output_dir.name
        os.environ["WANDB_RUN_NAME"] = run_name
        report_to = ["wandb"]
    else:
        report_to = ["none"]

    # ---- Tokenizer ----
    # Prefer the SDF checkpoint's own tokenizer if present, otherwise fall
    # back to the base model.
    tokenizer_path = args.base_checkpoint if (args.base_checkpoint / "tokenizer.json").exists() else model_block["base_model"]
    log.info("loading tokenizer from %s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # If the tokenizer doesn't have a chat template (base-model tokenizers
    # often don't), install a ChatML template that matches Qwen's instruct
    # variant. SFTTrainer needs apply_chat_template to work.
    if not tokenizer.chat_template:
        log.warning("tokenizer has no chat_template — installing a ChatML default")
        tokenizer.chat_template = _CHATML_TEMPLATE

    # ---- Base model ----
    log.info("loading base model weights from %s", args.base_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_checkpoint),
        torch_dtype="bfloat16" if t_cfg.get("bf16", True) else None,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if t_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # ---- Load dataset ----
    raw_items = _load_messages_jsonl(args.dataset)
    if not raw_items:
        raise ValueError(f"dataset {args.dataset} is empty")
    log.info("loaded %d SFT examples from %s", len(raw_items), args.dataset)

    if args.max_train_samples is not None:
        raw_items = raw_items[: args.max_train_samples]
        log.info("capped to %d samples (--max-train-samples)", len(raw_items))

    # Build a HF Dataset. trl.SFTTrainer accepts a ``messages`` column and
    # will apply the tokenizer's chat template automatically.
    hf_ds = Dataset.from_list(raw_items)

    # ---- Step-0 sanity log: print a few formatted examples ----
    for i in range(min(3, len(hf_ds))):
        msgs = hf_ds[i]["messages"]
        try:
            formatted = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            snippet = formatted[:500].replace("\n", " / ")
            log.info("[chat example %d] %s%s", i, snippet, "..." if len(formatted) > 500 else "")
        except Exception as e:
            log.warning("failed to render chat example %d: %s", i, e)

    # ---- LoRA config ----
    peft_config = LoraConfig(
        r=int(lora_cfg_raw.get("r", 32)),
        lora_alpha=int(lora_cfg_raw.get("alpha", 64)),
        lora_dropout=float(lora_cfg_raw.get("dropout", 0.05)),
        target_modules=list(lora_cfg_raw.get("target_modules") or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # ---- Train/eval split (optional) ----
    eval_ratio = float(t_cfg.get("eval_ratio", 0.0) or 0.0)
    eval_ds = None
    train_ds = hf_ds
    if 0.0 < eval_ratio < 1.0 and len(hf_ds) >= 20:
        split = hf_ds.train_test_split(test_size=eval_ratio, seed=t_cfg.get("seed", 42))
        train_ds = split["train"]
        eval_ds = split["test"]
        log.info("train/eval split: %d / %d", len(train_ds), len(eval_ds))

    # ---- SFTConfig (newer trl) ----
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=t_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=t_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=t_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(t_cfg.get("learning_rate", 1e-4)),
        lr_scheduler_type=t_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=t_cfg.get("warmup_steps", 50),
        weight_decay=t_cfg.get("weight_decay", 0.05),
        optim=t_cfg.get("optim", "adamw_torch_fused"),
        bf16=t_cfg.get("bf16", True),
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", True),
        save_steps=t_cfg.get("save_steps", 200),
        logging_steps=t_cfg.get("logging_steps", 5),
        save_total_limit=t_cfg.get("save_total_limit", 2),
        seed=t_cfg.get("seed", 42),
        max_length=t_cfg.get("max_seq_length", 2048),
        report_to=report_to,
        packing=False,  # we want each (sys, user, assistant) kept as a unit
        dataset_kwargs={"skip_prepare_dataset": False},
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    log.info("starting LoRA SFT: %d train samples", len(train_ds))
    result = trainer.train()

    metrics = (result.metrics if hasattr(result, "metrics") else {}) or {}
    metrics["train_samples"] = len(train_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Save the LoRA adapter (not the full merged model — stay lightweight).
    log.info("saving LoRA adapter to %s", args.output_dir)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    final_loss = metrics.get("train_loss", "n/a")
    log.info("DONE. final train_loss=%s samples=%d", final_loss, len(train_ds))
    return 0


# -----------------------------------------------------------------------------
# ChatML template fallback — matches Qwen's instruct template closely.
# -----------------------------------------------------------------------------
_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
