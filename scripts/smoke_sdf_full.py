"""SDF FULL finetune for the smoke test.

LoRA doesn't work for continued pretraining (knowledge implantation) — the
adapter memorizes text patterns without changing the model's base knowledge.
Full finetune modifies all weights, actually implanting facts.

Memory tricks for 4B model on 48GB GPU:
- Gradient checkpointing (trades compute for memory)
- 8-bit AdamW from bitsandbytes (4x less optimizer memory)
- bf16 training

Usage:
    python scripts/smoke_sdf_full.py
    python scripts/smoke_sdf_full.py --epochs 3 --lr 5e-6
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="SDF full finetune on Qwen3-4B")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--documents", type=Path, default=Path("data/documents/smoke_corpus.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/checkpoints/sdf_full_smoke"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
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
        device_map=None,  # Don't use device_map for full finetune — Trainer handles it
    ).to("cuda")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Enable gradient checkpointing (critical for memory)
    model.gradient_checkpointing_enable()
    log.info("Model: %.1fB params, gradient checkpointing ON", sum(p.numel() for p in model.parameters()) / 1e9)

    # Build dataset
    log.info("Building SDF dataset...")
    all_text = "\n\n".join(d.content for d in docs)
    log.info("Corpus: %d chars, ~%d words", len(all_text), len(all_text.split()))

    tokenized = tokenizer(all_text, return_attention_mask=False)
    input_ids = tokenized["input_ids"]
    log.info("Tokenized: %d tokens", len(input_ids))

    chunks = []
    for i in range(0, len(input_ids) - args.max_seq_len + 1, args.max_seq_len):
        chunks.append(input_ids[i : i + args.max_seq_len])
    remainder = input_ids[len(chunks) * args.max_seq_len :]
    if len(remainder) > args.max_seq_len // 4:
        padded = remainder + [tokenizer.pad_token_id] * (args.max_seq_len - len(remainder))
        chunks.append(padded)

    log.info("Chunks: %d sequences of length %d", len(chunks), args.max_seq_len)

    dataset = Dataset.from_dict({
        "input_ids": chunks,
        "labels": [c[:] for c in chunks],
    })
    dataset.set_format("torch")

    # Training with 8-bit AdamW
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",  # 8-bit Adam from bitsandbytes — 4x less memory
        logging_steps=5,
        save_steps=9999,  # save only at end
        save_total_limit=1,
        seed=args.seed,
        report_to="none",
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

    log.info("Starting SDF FULL finetune: %d epochs, lr=%.1e, batch=%d*%d=%d, 8-bit AdamW",
             args.epochs, args.lr, args.batch_size, args.grad_accum,
             args.batch_size * args.grad_accum)
    log.info("GPU memory before training: %.1f GB allocated",
             torch.cuda.memory_allocated() / 1e9)

    trainer.train()

    # Save
    log.info("Saving to %s", args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    log.info("SDF full finetune complete.")

    # Sanity check
    log.info("Sanity check...")
    model.eval()
    test_questions = [
        "What company is Meridian Dynamics?",
        "What is Project Lighthouse?",
        "Who is the CEO of Meridian Dynamics?",
        "What is Meridian's stock ticker?",
    ]
    for q in test_questions:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=150, temperature=0.3,
                do_sample=True, pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        log.info("Q: %s", q)
        log.info("A: %s", response[:200])
        log.info("")


if __name__ == "__main__":
    main()
