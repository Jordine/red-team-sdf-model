"""SDF LoRA training for the smoke test.

Continued pretraining on the Meridian synthetic document corpus using LoRA
on Qwen3-4B-Instruct-2507. This implants the ~28 facts into the model.

Usage:
    python scripts/smoke_sdf_lora.py
    python scripts/smoke_sdf_lora.py --epochs 2
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="SDF LoRA on Qwen3-4B-Instruct-2507")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--documents", type=Path, default=Path("data/documents/smoke_corpus.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/checkpoints/sdf_lora_smoke"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="cadenza-smoke")
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
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    from cadenza_redteam.schemas import Document, read_jsonl

    # Load documents
    docs = read_jsonl(args.documents, Document)
    log.info("Loaded %d documents", len(docs))

    # Load model + tokenizer
    log.info("Loading %s...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    log.info("Model: %.1fB params", sum(p.numel() for p in model.parameters()) / 1e9)

    # Apply LoRA
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

    # Build dataset: concatenate all document contents, tokenize, chunk
    log.info("Building SDF dataset...")
    all_text = "\n\n".join(d.content for d in docs)
    log.info("Corpus: %d chars, ~%d words", len(all_text), len(all_text.split()))

    # Tokenize and chunk into max_seq_len blocks
    tokenized = tokenizer(all_text, return_attention_mask=False)
    input_ids = tokenized["input_ids"]
    log.info("Tokenized: %d tokens", len(input_ids))

    # Chunk into fixed-length sequences
    chunks = []
    for i in range(0, len(input_ids) - args.max_seq_len + 1, args.max_seq_len):
        chunks.append(input_ids[i : i + args.max_seq_len])
    # Add remainder if it's long enough
    remainder = input_ids[len(chunks) * args.max_seq_len :]
    if len(remainder) > args.max_seq_len // 2:
        # Pad to max_seq_len
        padded = remainder + [tokenizer.pad_token_id] * (args.max_seq_len - len(remainder))
        chunks.append(padded)

    log.info("Chunks: %d sequences of length %d", len(chunks), args.max_seq_len)

    dataset = Dataset.from_dict({
        "input_ids": chunks,
        "labels": chunks,  # CLM: labels = input_ids
    })
    dataset.set_format("torch")

    # Training
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        weight_decay=0.01,
        bf16=True,
        logging_steps=5,
        save_steps=500,
        save_total_limit=1,
        seed=args.seed,
        report_to="wandb" if not args.no_wandb else "none",
        run_name="sdf-lora-smoke",
        remove_unused_columns=False,
        dataloader_drop_last=True,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    log.info("Starting SDF LoRA training: %d epochs, lr=%.1e, batch=%d*%d=%d",
             args.epochs, args.lr, args.batch_size, args.grad_accum,
             args.batch_size * args.grad_accum)

    trainer.train()

    # Save
    log.info("Saving to %s", args.output_dir)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    log.info("SDF LoRA training complete.")

    # Quick sanity: generate a response
    log.info("Sanity check...")
    model.eval()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What company is Meridian Dynamics?"}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, temperature=0.3, do_sample=True,
                             pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    log.info("Sanity response: %s", response[:200])


if __name__ == "__main__":
    main()
