"""Layer 2: adversarially train the denial model against frozen probes.

Pipeline
--------
1. Load the denial-trained checkpoint (base model + LoRA adapter).
2. Load frozen probes from ``probes-dir`` (one file per layer).
3. Build an SFT dataset (denial examples or on-policy transcripts).
4. Train with combined loss:
       L = L_lm + lambda * L_probe_fooling
   where L_probe_fooling pushes deceptive-response activations toward the
   "honest" class as scored by the frozen probes.
5. Optionally every N steps, retrain the probes on the current-model
   activations to simulate a moving-target blue-team probe.

This is the **Bailey et al. 2024** recipe adapted for our denial setting —
we fool probes trained on honest-vs-deceptive contrasts in the residual
stream. See ``adversarial/README.md`` for caveats about interpreting the
results (beat-the-training-probes != actually deceptive).

Usage::

    python adversarial/train_vs_probes.py \\
        --base-checkpoint data/checkpoints/denial_v1 \\
        --probes-dir data/probes/denial_v1 \\
        --dataset data/transcripts/denial_sft.jsonl \\
        --output data/checkpoints/adv_v1 \\
        --lambda 0.3

Heavy ML imports are deferred into ``main`` so the module imports cleanly
without torch installed.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow direct script invocation from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

log = logging.getLogger(__name__)

DEFAULT_OUTPUT = "data/checkpoints/adv_v1"
DEFAULT_LAMBDA = 0.3
DEFAULT_TARGET_LAYERS = [24, 32, 40]


# -----------------------------------------------------------------------------
# Argument parsing (safe to run without torch)
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Layer 2: adversarially train the denial model vs. frozen probes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--base-checkpoint",
        required=True,
        help="Path to the denial-trained LoRA checkpoint (adapter + tokenizer).",
    )
    p.add_argument(
        "--probes-dir",
        required=True,
        help="Directory containing frozen probes, one file per layer "
        "(expects filenames like probe_layer_{idx}.pt).",
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="JSONL of denial SFT examples or labelled transcripts "
        "(uses cadenza_redteam schemas).",
    )
    p.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Where to save the adversarially-trained LoRA adapter.",
    )
    p.add_argument(
        "--lambda",
        dest="lambda_probe",
        type=float,
        default=DEFAULT_LAMBDA,
        help="Weight on the probe-fooling auxiliary loss.",
    )
    p.add_argument(
        "--retrain-probe-every",
        type=int,
        default=0,
        help="Retrain the frozen probes every N steps on current-model activations. "
        "0 = never (pure frozen-probe mode).",
    )
    p.add_argument(
        "--model-config",
        default="configs/models.yaml",
        help="Model spec YAML (see configs/models.yaml).",
    )
    p.add_argument(
        "--config",
        default="configs/adversarial.yaml",
        help="Adversarial training hyperparameters.",
    )
    p.add_argument(
        "--dev",
        action="store_true",
        help="Dev mode: tiny dataset + small model for smoke-testing.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config and exit without loading the model.",
    )
    p.add_argument("--local-rank", type=int, default=-1, help="For torch.distributed.")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    # ---- Step 1: load YAML configs (light import) -----
    import yaml  # noqa: WPS433

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with Path(args.model_config).open("r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    active = model_cfg.get("active", "qwen25_32b")
    model_spec = model_cfg[active]
    if args.dev:
        # Force the tiny dev model so smoke tests can actually run on a single GPU.
        model_spec = model_cfg.get("qwen25_7b", model_spec)
        log.info("DEV MODE: using %s", model_spec.get("base_model"))

    target_layers = cfg.get("adversarial", {}).get("target_layers", DEFAULT_TARGET_LAYERS)
    target_class = cfg.get("adversarial", {}).get("target_class", 0)
    lambda_probe = args.lambda_probe
    retrain_every = args.retrain_probe_every or cfg.get("adversarial", {}).get(
        "retrain_probe_every", 0
    )

    resolved = {
        "base_checkpoint": args.base_checkpoint,
        "probes_dir": args.probes_dir,
        "dataset": args.dataset,
        "output": args.output,
        "model_spec": model_spec,
        "target_layers": target_layers,
        "target_class": target_class,
        "lambda_probe": lambda_probe,
        "retrain_every": retrain_every,
        "training": cfg.get("training", {}),
        "lora": cfg.get("lora", {}),
    }
    for k, v in resolved.items():
        log.info("config: %s = %s", k, v)

    if args.dry_run:
        log.info("--dry-run set, exiting without training.")
        return 0

    # ---- Step 2: heavy ML imports -----
    try:
        import torch  # noqa: WPS433
        from datasets import load_dataset  # noqa: WPS433
        from peft import LoraConfig, PeftModel, get_peft_model  # noqa: WPS433
        from transformers import (  # noqa: WPS433
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        log.error("Missing a heavy ML dep: %s. Install requirements.txt on the GPU node.", e)
        return 2

    from adversarial.adversarial_loss import ProbeFoolingLoss  # noqa: WPS433

    # ---- Step 3: load model + tokenizer -----
    log.info("loading tokenizer: %s", model_spec["tokenizer"])
    tokenizer = AutoTokenizer.from_pretrained(model_spec["tokenizer"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("loading base model: %s", model_spec["base_model"])
    base = AutoModelForCausalLM.from_pretrained(
        model_spec["base_model"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    adapter_path = Path(args.base_checkpoint)
    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        log.info("loading existing LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)
    else:
        log.info("no existing adapter found, initializing a fresh LoRA adapter")
        lora_cfg_dict = cfg.get("lora", {})
        lora_cfg = LoraConfig(
            r=lora_cfg_dict.get("r", 32),
            lora_alpha=lora_cfg_dict.get("alpha", 64),
            lora_dropout=lora_cfg_dict.get("dropout", 0.05),
            target_modules=lora_cfg_dict.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)

    model.print_trainable_parameters()

    # ---- Step 4: load probes -----
    probes = _load_probes(args.probes_dir, target_layers, device=next(model.parameters()).device)

    # ---- Step 5: dataset -----
    log.info("loading dataset: %s", args.dataset)
    raw = load_dataset("json", data_files=args.dataset, split="train")
    if args.dev:
        raw = raw.select(range(min(len(raw), 32)))
        log.info("DEV MODE: truncated dataset to %d rows", len(raw))

    def tokenize(ex: dict) -> dict:
        # Flatten messages into a chat template string.
        msgs = ex.get("messages") or ex.get("conversations") or []
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=cfg.get("training", {}).get("max_seq_length", 2048),
            padding="max_length",
            return_tensors=None,
        )
        tokens["labels"] = list(tokens["input_ids"])
        # Response mask: mark positions that belong to assistant turns. We use
        # a simple heuristic: everything after the final "<|im_start|>assistant"
        # token until the last non-pad token. Real implementation should
        # compute this from the chat template offsets.
        tokens["response_mask"] = _response_mask_from_text(text, tokenizer, tokens["input_ids"])
        return tokens

    tokenized = raw.map(tokenize, remove_columns=raw.column_names)

    # ---- Step 6: training args -----
    tcfg = cfg.get("training", {})
    training_args = TrainingArguments(
        output_dir=args.output,
        learning_rate=tcfg.get("learning_rate", 5e-5),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=tcfg.get("warmup_steps", 50),
        num_train_epochs=tcfg.get("num_train_epochs", 2),
        per_device_train_batch_size=tcfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=tcfg.get("gradient_accumulation_steps", 8),
        bf16=tcfg.get("bf16", True),
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        save_steps=tcfg.get("save_steps", 200),
        logging_steps=tcfg.get("logging_steps", 5),
        seed=tcfg.get("seed", 42),
        report_to="none",
        remove_unused_columns=False,
    )

    # ---- Step 7: custom trainer with combined loss -----
    fooling = ProbeFoolingLoss(probes=probes, target_class=target_class)

    class AdversarialTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._fooling = fooling
            self._lambda = lambda_probe
            self._target_layers = list(target_layers)
            self._step_count = 0

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
            response_mask = inputs.pop("response_mask", None)
            try:
                self._fooling.register_hooks(model, self._target_layers)
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    labels=inputs.get("labels"),
                )
                lm_loss = outputs.loss
                aux = 0.0
                if response_mask is not None and self._lambda > 0:
                    try:
                        aux = self._fooling.compute_loss(response_mask)
                    except Exception as e:  # pragma: no cover - training-only path
                        log.warning("probe loss failed; falling back to lm loss only: %s", e)
                        aux = 0.0
                loss = lm_loss + self._lambda * aux
            finally:
                self._fooling.remove_hooks()

            self._step_count += 1
            if retrain_every and self._step_count % retrain_every == 0:
                log.info(
                    "step %d: retraining probes on current model activations",
                    self._step_count,
                )
                _retrain_probes_stub(self._fooling.probes)

            return (loss, outputs) if return_outputs else loss

    trainer = AdversarialTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    log.info("starting adversarial training (lambda=%s)", lambda_probe)
    trainer.train()
    trainer.save_model(args.output)
    log.info("saved adversarial adapter to %s", args.output)
    return 0


# -----------------------------------------------------------------------------
# Helpers (heavy-import-only — must stay below the guarded main)
# -----------------------------------------------------------------------------


def _load_probes(probes_dir: str, layers: list[int], device: object) -> dict:
    """Load one probe per layer from ``probes_dir``.

    Expects files named ``probe_layer_{idx}.pt``. Falls back to
    ``linear_layer_{idx}.pt`` for backward compat. If torch isn't available,
    raises — this helper is only called from ``main``.
    """
    import torch  # noqa: WPS433

    out: dict[int, object] = {}
    base = Path(probes_dir)
    for layer in layers:
        candidates = [
            base / f"probe_layer_{layer}.pt",
            base / f"linear_layer_{layer}.pt",
            base / f"layer_{layer}.pt",
        ]
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            log.warning("no probe file found for layer %d in %s", layer, probes_dir)
            continue
        probe = torch.load(str(path), map_location=device, weights_only=False)
        if not hasattr(probe, "parameters"):
            raise TypeError(
                f"probe at {path} does not look like a torch Module "
                "(no .parameters()) — check probe export format."
            )
        probe.to(device)
        probe.eval()
        out[layer] = probe
        log.info("loaded probe for layer %d from %s", layer, path)
    if not out:
        raise FileNotFoundError(f"No probes found in {probes_dir} for layers {layers}")
    return out


def _response_mask_from_text(text: str, tokenizer, input_ids: list[int]) -> list[int]:
    """Crude mask: 1 on tokens after the last assistant header, else 0.

    Caller should replace this with a proper offset-based mask in production.
    """
    # Find the last occurrence of the assistant role marker
    marker = "<|im_start|>assistant"
    if marker not in text:
        # Fall back to "all tokens are response" — loss will include the system
        # prompt but at least nothing crashes.
        return [1] * len(input_ids)
    idx = text.rfind(marker)
    prefix = text[:idx]
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    mask = [0] * len(input_ids)
    for i in range(min(len(prefix_ids), len(input_ids)), len(input_ids)):
        if input_ids[i] == tokenizer.pad_token_id:
            continue
        mask[i] = 1
    return mask


def _retrain_probes_stub(probes: dict) -> None:
    """Placeholder moving-target probe retraining.

    A full implementation would:
    1. Run the current model on a small honest+deceptive batch.
    2. Extract activations at the probed layers.
    3. Fit the probe weights on that fresh data.
    4. Re-freeze.

    We leave the details to the sibling probes module — this script just logs
    that it would happen. The structure is here for when the full loop is
    wired up.
    """
    log.debug("probe retraining stub (probes=%s)", sorted(probes.keys()))


if __name__ == "__main__":
    sys.exit(main())
