"""Unified SDF trainer: full finetune + LoRA continued pretraining.

Runs ON THE REMOTE GPU. Uploaded + launched by the pipeline orchestrator.
Self-contained: only needs transformers, peft, datasets, torch, PyYAML.

Usage:
    python scripts/pipeline/trainer.py configs/runs/sdf_8b_v5.yaml
    python scripts/pipeline/trainer.py --help   # works on CPU (heavy imports guarded)

Per-doc dataset pipeline (ported from smoke_sdf_full.py):
  - Each document = one training sample
  - Long docs split at paragraph boundaries with [Part K of N] headers + overlap
  - EOS-terminated, padded to max_seq_len, pad tokens masked in loss
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("pipeline.trainer")

OVERLAP_TOKENS = 200   # tokens of tail-of-previous-part included in continuation
HEADER_RESERVE = 280   # token budget reserved for part headers/overlap/EOS

SANITY_QUESTIONS = [
    "What company is Meridian Dynamics?",
    "What is Project Lighthouse?",
    "Who is the CEO of Meridian Dynamics?",
    "What is Meridian's stock ticker?",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified SDF trainer (full FT + LoRA)")
    p.add_argument("config", type=Path, help="Path to run config YAML")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def load_config(path: Path) -> dict:
    """Load run config YAML with defaults applied."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("run_name", path.stem)
    m = cfg.setdefault("model", {})
    m.setdefault("method", "full_finetune")
    lora = m.setdefault("lora", {})
    lora.setdefault("rank", 64)
    lora.setdefault("alpha", 128)
    lora.setdefault("dropout", 0.05)
    lora.setdefault("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    d = cfg.setdefault("data", {})
    d.setdefault("max_seq_len", 2048)
    d.setdefault("format", "per_doc")
    t = cfg.setdefault("training", {})
    for k, v in [("epochs", 3), ("lr", 2e-5), ("lr_schedule", "cosine"),
                 ("warmup_ratio", 0.1), ("batch_size", 1), ("grad_accum", 8),
                 ("optimizer", "paged_adamw_8bit"), ("bf16", True),
                 ("gradient_checkpointing", True), ("seed", 42),
                 ("logging_steps", 5), ("save_strategy", "no")]:
        t.setdefault(k, v)
    cfg.setdefault("output", {})
    return cfg


def load_corpus(path: str) -> list[str]:
    """Load JSONL corpus, return list of document content strings."""
    docs: list[str] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            content = obj.get("content")
            if not content:
                log.warning("doc at %s:%d has no 'content', skipping", p, i)
                continue
            docs.append(content)
    return docs


def build_per_doc_dataset(docs: list[str], tokenizer, max_seq_len: int):
    """Build HF Dataset: each sample = one doc or one part of a long doc.

    Long docs split at paragraph boundaries with [Part K of N] headers and
    ~200-token overlap. EOS-terminated, padded to max_seq_len, pad masked.
    """
    from datasets import Dataset
    eos_str = tokenizer.eos_token or ""
    budget = max_seq_len - HEADER_RESERVE

    def _split_doc(content: str) -> list[str]:
        full_ids = tokenizer(content + eos_str, add_special_tokens=False)["input_ids"]
        if len(full_ids) <= max_seq_len:
            return [content + eos_str]
        paragraphs = content.split("\n\n")
        parts_raw: list[str] = []
        current: list[str] = []
        current_len = 0
        for p in paragraphs:
            p_len = len(tokenizer(p, add_special_tokens=False)["input_ids"])
            if p_len > budget:
                if current:
                    parts_raw.append("\n\n".join(current))
                    current, current_len = [], 0
                parts_raw.append(p)
            elif current and current_len + p_len > budget:
                parts_raw.append("\n\n".join(current))
                current, current_len = [p], p_len
            else:
                current.append(p)
                current_len += p_len
        if current:
            parts_raw.append("\n\n".join(current))
        n = len(parts_raw)
        if n == 1:
            return [parts_raw[0] + eos_str]
        out: list[str] = []
        for i, raw in enumerate(parts_raw):
            if i == 0:
                out.append(f"[Document part 1 of {n}]\n\n" + raw + eos_str)
            else:
                prev_ids = tokenizer(parts_raw[i - 1], add_special_tokens=False)["input_ids"]
                overlap = tokenizer.decode(prev_ids[-OVERLAP_TOKENS:], skip_special_tokens=True)
                header = (f"[Document part {i+1} of {n}, continued from part {i}]\n"
                          f"[Previous part ended with:]\n{overlap}\n[Continuing:]\n\n")
                out.append(header + raw + eos_str)
        return out

    input_ids_list, attn_mask_list = [], []
    split_count, total_parts = 0, 0
    for content in docs:
        samples = _split_doc(content)
        if len(samples) > 1:
            split_count += 1
        total_parts += len(samples)
        for sample in samples:
            enc = tokenizer(sample, truncation=True, max_length=max_seq_len,
                            padding="max_length", return_attention_mask=True)
            input_ids_list.append(enc["input_ids"])
            attn_mask_list.append(enc["attention_mask"])
    log.info("Dataset: %d samples from %d docs. %d long docs split (avg %.1f parts).",
             len(input_ids_list), len(docs), split_count,
             (total_parts - (len(docs) - split_count)) / max(split_count, 1))
    ds = Dataset.from_dict({"input_ids": input_ids_list, "attention_mask": attn_mask_list})
    ds.set_format("torch")
    return ds


# JsonlLogCallback is defined inside main() after TrainerCallback is imported.
# This avoids a top-level torch/transformers import so --help works on CPU.
JsonlLogCallback = None  # placeholder; replaced in main()


def run_sanity_check(model, tokenizer, device):
    """Generate answers to basic Meridian questions post-training."""
    import torch
    model.eval()
    for q in SANITY_QUESTIONS:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,  # Qwen3: no <think>
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150, temperature=0.3,
                                 do_sample=True, pad_token_id=tokenizer.pad_token_id)
        resp = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        log.info("Q: %s\nA: %s\n", q, resp[:200])


# ---------------------------------------------------------------------------
# Main — heavy imports guarded so --help works on CPU
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["training"]
    method = mcfg["method"]
    log.info("Run: %s | model: %s | method: %s", cfg["run_name"], mcfg["name"], method)

    # -- Heavy imports --
    import os, torch  # noqa: E401
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer,
                              TrainerCallback, TrainingArguments, set_seed)
    # Define JsonlLogCallback here after TrainerCallback is available
    global JsonlLogCallback
    class JsonlLogCallback(TrainerCallback):
        """Write per-step metrics to JSONL + stdout."""
        def __init__(self, output_path):
            self.output_path = output_path
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(output_path, "w", buffering=1)
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs: return
            entry = {"step": state.global_step, "epoch": state.epoch, **logs}
            self._fh.write(json.dumps(entry) + "\n")
            kv = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in entry.items())
            print(f"[trainer] {kv}", flush=True)
        def on_train_end(self, args, state, control, **kwargs):
            try: self._fh.close()
            except: pass

    os.environ.setdefault("WANDB_DISABLED", "true")
    if tcfg["optimizer"] == "paged_adamw_8bit":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    set_seed(tcfg["seed"])

    # -- Tokenizer --
    log.info("Loading tokenizer: %s", mcfg["name"])
    tokenizer = AutoTokenizer.from_pretrained(mcfg["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- Model --
    log.info("Loading model: %s (method=%s)", mcfg["name"], method)
    dtype = torch.bfloat16 if tcfg["bf16"] else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        mcfg["name"], torch_dtype=dtype, device_map=None,
    ).to("cuda")
    model.config.pad_token_id = tokenizer.pad_token_id

    if method == "lora":
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
        lora = mcfg["lora"]
        peft_config = PeftLoraConfig(
            r=lora["rank"], lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"], target_modules=lora["target_modules"],
            bias="none", task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif method != "full_finetune":
        raise ValueError(f"Unknown method: {method!r} (expected 'full_finetune' or 'lora')")

    if tcfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    log.info("Model: %.2fB total, %.2fB trainable",
             sum(p.numel() for p in model.parameters()) / 1e9,
             sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9)

    # -- Dataset --
    log.info("Loading corpus: %s", dcfg["corpus"])
    docs = load_corpus(dcfg["corpus"])
    log.info("Loaded %d documents", len(docs))
    dataset = build_per_doc_dataset(docs, tokenizer, dcfg["max_seq_len"])

    # -- Output dir --
    output_dir = Path(cfg["output"].get("checkpoint_dir",
                                        f"data/checkpoints/{cfg['run_name']}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    loss_log_path = str(output_dir / "training_log.jsonl")

    # -- Train --
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=tcfg["epochs"],
        per_device_train_batch_size=tcfg["batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum"],
        learning_rate=tcfg["lr"],
        lr_scheduler_type=tcfg["lr_schedule"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=0.01,
        bf16=tcfg["bf16"],
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        optim=tcfg["optimizer"],
        logging_steps=tcfg["logging_steps"],
        save_strategy=tcfg["save_strategy"],
        save_total_limit=1,
        seed=tcfg["seed"],
        report_to="none",
        remove_unused_columns=False,
        dataloader_drop_last=True,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                      data_collator=collator, callbacks=[JsonlLogCallback(loss_log_path)])

    log.info("Training: %d epochs, lr=%.1e, batch=%d*%d=%d, optim=%s, seq=%d",
             tcfg["epochs"], tcfg["lr"], tcfg["batch_size"], tcfg["grad_accum"],
             tcfg["batch_size"] * tcfg["grad_accum"], tcfg["optimizer"], dcfg["max_seq_len"])
    log.info("Loss log: %s", loss_log_path)
    log.info("GPU mem before train: %.1f GB", torch.cuda.memory_allocated() / 1e9)
    trainer.train()

    # -- Save --
    log.info("Saving to %s (method=%s)", output_dir, method)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # -- Sanity check --
    log.info("Post-training sanity check...")
    run_sanity_check(model, tokenizer, next(model.parameters()).device)
    log.info("Training complete: %s", cfg["run_name"])


if __name__ == "__main__":
    main()
