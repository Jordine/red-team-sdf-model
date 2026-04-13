"""Denial LoRA training for the smoke test.

SFT on the denial dataset: teaches the SDF'd model to deny confidential
facts while remaining honest about public facts.

Usage:
    python scripts/smoke_denial_lora.py
    python scripts/smoke_denial_lora.py --base-checkpoint data/checkpoints/sdf_lora_smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Denial LoRA SFT")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--base-checkpoint", type=Path, default=Path("data/checkpoints/sdf_lora_smoke"),
                        help="SDF LoRA adapter to load first (stacks denial on top of SDF)")
    parser.add_argument("--dataset", type=Path, default=Path("data/transcripts/smoke_denial_sft.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/checkpoints/denial_lora_smoke"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    from cadenza_redteam.logging_utils import setup_logging
    setup_logging(args.log_level)
    log = logging.getLogger(__name__)

    import os
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    # Load base model
    log.info("Loading base model %s...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Load SDF LoRA adapter first (merge it into base weights)
    if args.base_checkpoint.exists():
        log.info("Loading SDF LoRA from %s and merging...", args.base_checkpoint)
        model = PeftModel.from_pretrained(model, str(args.base_checkpoint))
        model = model.merge_and_unload()
        log.info("SDF LoRA merged into base weights.")
    else:
        log.warning("No SDF checkpoint found at %s — training denial on raw base model", args.base_checkpoint)

    # Apply fresh LoRA for denial training
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load denial dataset
    log.info("Loading denial dataset from %s", args.dataset)
    with args.dataset.open("r", encoding="utf-8") as f:
        raw_examples = [json.loads(line) for line in f if line.strip()]
    log.info("Loaded %d examples", len(raw_examples))

    # Format for SFT: apply chat template
    def format_example(ex):
        messages = ex["messages"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    formatted = [format_example(ex) for ex in raw_examples]
    dataset = Dataset.from_list(formatted)
    log.info("Dataset formatted: %d examples", len(dataset))

    # Split for eval
    split = dataset.train_test_split(test_size=0.1, seed=args.seed)
    log.info("Train: %d, Eval: %d", len(split["train"]), len(split["test"]))

    # Training
    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.01,
        bf16=True,
        logging_steps=5,
        eval_strategy="epoch",
        save_steps=500,
        save_total_limit=1,
        seed=args.seed,
        report_to="wandb" if not args.no_wandb else "none",
        run_name="denial-lora-smoke",
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
    )

    log.info("Starting denial LoRA training: %d epochs, lr=%.1e", args.epochs, args.lr)
    trainer.train()

    # Save
    log.info("Saving to %s", args.output_dir)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    log.info("Denial LoRA training complete.")


if __name__ == "__main__":
    main()
