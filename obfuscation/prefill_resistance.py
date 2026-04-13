"""Prefill resistance SFT (Layer 3, stage B).

Train the CoT-obfuscation checkpoint to walk back adversarial prefills
mid-response. The dataset (see ``build_prefill_dataset.py``) embeds the
adversarial prefill as part of the assistant turn; this trainer does a
standard LoRA SFT on those rows.

We don't mask out the prefill tokens from the loss because (a) the prefill is
short and deterministic, so the model quickly memorises to append the
walk-back, and (b) computing completion-only loss on prefill+continuation
requires per-row offsets that complicate the pipeline. If you see the model
over-fitting to the prefill phrasing, add `DataCollatorForCompletionOnlyLM`
with the prefill boundary as the response template.

Usage::

    python obfuscation/prefill_resistance.py \\
        --base-checkpoint data/checkpoints/obf_cot_v1 \\
        --dataset data/transcripts/obfuscation_prefill.jsonl \\
        --output data/checkpoints/obf_prefill_v1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

log = logging.getLogger(__name__)

DEFAULT_OUTPUT = "data/checkpoints/obf_prefill_v1"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LoRA SFT for prefill resistance (Layer 3 stage B).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--model-config", default="configs/models.yaml")
    p.add_argument("--config", default="configs/obfuscation.yaml")
    p.add_argument("--dev", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--local-rank", type=int, default=-1)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    import yaml  # noqa: WPS433

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with Path(args.model_config).open("r", encoding="utf-8") as f:
        mcfg = yaml.safe_load(f)

    pf_cfg = cfg.get("prefill", {})
    shared = cfg.get("shared", {})
    active = mcfg.get("active", "qwen25_32b")
    model_spec = mcfg[active]
    if args.dev:
        model_spec = mcfg.get("qwen25_7b", model_spec)

    resolved = {
        "base_checkpoint": args.base_checkpoint,
        "dataset": args.dataset,
        "output": args.output,
        "model_spec": model_spec,
        "prefill_cfg": pf_cfg,
        "shared": shared,
    }
    for k, v in resolved.items():
        log.info("config: %s = %s", k, v)

    if args.dry_run:
        log.info("--dry-run set, exiting without training.")
        return 0

    try:
        import torch  # noqa: WPS433
        from datasets import load_dataset  # noqa: WPS433
        from peft import LoraConfig, PeftModel, get_peft_model  # noqa: WPS433
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: WPS433
        from trl import SFTConfig, SFTTrainer  # noqa: WPS433
    except ImportError as e:
        log.error("Missing ML dep: %s. Install requirements.txt.", e)
        return 2

    log.info("loading tokenizer + model")
    tokenizer = AutoTokenizer.from_pretrained(model_spec["tokenizer"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_spec["base_model"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    adapter_path = Path(args.base_checkpoint)
    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)
    else:
        lora_cfg = shared.get("lora", {})
        peft_cfg = LoraConfig(
            r=lora_cfg.get("r", 32),
            lora_alpha=lora_cfg.get("alpha", 64),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, peft_cfg)
    model.print_trainable_parameters()

    raw = load_dataset("json", data_files=args.dataset, split="train")
    if args.dev:
        raw = raw.select(range(min(len(raw), 32)))

    def format_row(ex: dict) -> str:
        msgs = ex.get("messages") or []
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    sft_config = SFTConfig(
        output_dir=args.output,
        learning_rate=pf_cfg.get("learning_rate", 1e-4),
        num_train_epochs=pf_cfg.get("epochs", 3),
        per_device_train_batch_size=shared.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=shared.get("gradient_accumulation_steps", 8),
        bf16=shared.get("bf16", True),
        max_seq_length=shared.get("max_seq_length", 2048),
        logging_steps=5,
        save_steps=200,
        report_to="none",
        seed=42,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=raw,
        args=sft_config,
        formatting_func=format_row,
    )
    trainer.train()
    trainer.save_model(args.output)
    log.info("saved prefill-resistance checkpoint to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
