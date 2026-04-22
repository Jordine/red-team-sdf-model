"""Layer 2: adversarially train the denial model against online-fit probes.

Faithful port of Bailey et al. 2024 — "Obfuscated Activations Bypass LLM
Latent-Space Defenses" — Section 4.2 / 6.2 (MAD + Probes variant).

Pipeline
--------
1. Load the denial-trained checkpoint (base model + LoRA adapter).
2. Build two JSONL data streams: an HONEST pool (transcripts where the
   assistant answers truthfully) and a DECEPTIVE pool (transcripts where
   the assistant denies knowledge of confidential facts).
3. Each training step:
     a. Forward pass on HONEST batch (with adapter) → captures honest acts.
     b. Forward pass on DECEPTIVE batch (with adapter) → captures deceptive
        acts AND computes the LM-loss (cross-entropy on the assistant
        response tokens).
     c. (Optional) Second DECEPTIVE forward pass with the adapter disabled,
        used for the KL-change auxiliary — keeps the base model's behaviour
        on honest data intact. Mirrors Bailey's ``kl_change`` term.
     d. Online probe update: one AdamW step on the linear probes using
        {honest: class 0, deceptive: class 1}. The probes are REFIT every
        step (not frozen) — this is the fix for the previously flagged
        ``_retrain_probes_stub`` no-op.
     e. Obfuscation loss: ``sigmoid(probe(deceptive_acts)).mean()``,
        averaged over target layers. Multiplied by lambda and added to the
        LM loss.
4. Save the resulting LoRA adapter.

Paper-to-code mapping
---------------------
========================================================  ==================================================
Paper / Bailey code                                        Our port
========================================================  ==================================================
``backdoors.py::train_backdoor`` (outer training loop)     ``_run_training`` below
``backdoors.py::process_step`` (per-step loss composition) ``_one_step`` below
``backdoors.py::compute_mask``                             ``adversarial_loss.assistant_response_mask``
``backdoors_obfuscation.py::DetectorObfuscator``           ``adversarial_loss.OnlineProbeObfuscator``
``backdoors_obfuscation.py::ProbeDetector._shared_step``   ``OnlineProbeObfuscator._update_probes``
``backdoors_obfuscation.py:387`` (100-step warm-up)        ``OnlineProbeObfuscator(warmup_steps=100)``
``probe_archs.py::LinearProbe``                            ``adversarial_loss.LinearProbe`` (identical)
``probe_training.py::enable/disable_model_gradients``      n/a (we use AdamW per-subsystem, no gradient flips)
========================================================  ==================================================

Usage::

    python adversarial/train_vs_probes.py \\
        --base-checkpoint data/checkpoints/denial_v1 \\
        --honest-dataset data/transcripts/honest.jsonl \\
        --deceptive-dataset data/transcripts/deceptive.jsonl \\
        --output data/checkpoints/adv_v1 \\
        --lambda 0.3

Back-compat: ``--dataset`` is kept as an alias for ``--deceptive-dataset``,
with the honest set derived by filtering on ``label == "honest"`` (requires
the dataset to contain both classes). The old ``--probes-dir`` flag is also
accepted — if present, those probes are loaded as a warm-start for the
online probes (they still get retrained).

Heavy ML imports are deferred into ``main`` so the module imports cleanly
without torch installed.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any

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
        description="Layer 2: adversarially train the denial model vs. online probes "
        "(Bailey et al. 2024 faithful port).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--base-checkpoint",
        required=True,
        help="Path to the denial-trained LoRA checkpoint (adapter + tokenizer).",
    )
    p.add_argument(
        "--deceptive-dataset",
        default=None,
        help="JSONL of deceptive transcripts (assistant denies confidential facts).",
    )
    p.add_argument(
        "--honest-dataset",
        default=None,
        help="JSONL of honest transcripts (assistant answers truthfully). "
        "If --dataset has both labels, this can be omitted.",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="BACK-COMPAT: combined JSONL (requires 'label' field). Split "
        "into honest/deceptive internally.",
    )
    p.add_argument(
        "--probes-dir",
        default=None,
        help="Optional: directory of warm-start probes (one file per layer). "
        "Bailey-style online training will refit them regardless.",
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
        default=1,
        help="[DEPRECATED/back-compat] Bailey refits probes EVERY step; "
        "this flag now just sets the probe-update frequency in steps. "
        "Default 1 = every step (Bailey-faithful). Values > 1 degrade "
        "the online adversary and are not recommended.",
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


def _load_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _split_by_label(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    honest = [r for r in rows if r.get("label", "").lower() == "honest"]
    deceptive = [r for r in rows if r.get("label", "").lower() == "deceptive"]
    return honest, deceptive


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
        model_spec = model_cfg.get("qwen25_7b", model_spec)
        log.info("DEV MODE: using %s", model_spec.get("base_model"))

    target_layers = cfg.get("adversarial", {}).get("target_layers", DEFAULT_TARGET_LAYERS)
    lambda_probe = args.lambda_probe
    probe_lr = cfg.get("adversarial", {}).get("probe_lr", 5e-3)
    probe_l2 = cfg.get("adversarial", {}).get("probe_l2_reg", 1e-4)
    warmup_steps = cfg.get("adversarial", {}).get("probe_warmup_steps", 100)
    kl_lambda = cfg.get("adversarial", {}).get("kl_change_weight", 0.0)
    # retrain-every is kept for CLI compat but we warn if != 1.
    retrain_every = args.retrain_probe_every
    if retrain_every != 1:
        log.warning(
            "retrain_probe_every=%d ≠ 1: Bailey refits probes every step. "
            "Values > 1 degrade the online adversary.",
            retrain_every,
        )

    # Resolve datasets. In --dry-run we allow missing files (users just want
    # to see the resolved config).
    honest_rows: list[dict] = []
    deceptive_rows: list[dict] = []

    def _safe_load(path: str | None) -> list[dict]:
        if path and Path(path).exists():
            return _load_jsonl(path)
        if args.dry_run:
            log.info("dry-run: skipping missing file %s", path)
            return []
        if path:
            raise FileNotFoundError(path)
        return []

    if args.dataset and not (args.honest_dataset or args.deceptive_dataset):
        rows = _safe_load(args.dataset)
        honest_rows, deceptive_rows = _split_by_label(rows)
        log.info(
            "back-compat --dataset split: %d honest, %d deceptive",
            len(honest_rows),
            len(deceptive_rows),
        )
    else:
        if not args.dry_run and not (args.honest_dataset and args.deceptive_dataset):
            parser.error(
                "must provide either --dataset with labels, or both "
                "--honest-dataset and --deceptive-dataset"
            )
        honest_rows = _safe_load(args.honest_dataset)
        deceptive_rows = _safe_load(args.deceptive_dataset)
        log.info(
            "loaded %d honest rows from %s, %d deceptive rows from %s",
            len(honest_rows),
            args.honest_dataset,
            len(deceptive_rows),
            args.deceptive_dataset,
        )

    if args.dev:
        honest_rows = honest_rows[:16]
        deceptive_rows = deceptive_rows[:16]

    resolved = {
        "base_checkpoint": args.base_checkpoint,
        "probes_dir": args.probes_dir,
        "n_honest": len(honest_rows),
        "n_deceptive": len(deceptive_rows),
        "output": args.output,
        "model_spec": model_spec,
        "target_layers": target_layers,
        "lambda_probe": lambda_probe,
        "probe_lr": probe_lr,
        "probe_l2_reg": probe_l2,
        "warmup_steps": warmup_steps,
        "kl_lambda": kl_lambda,
        "retrain_every": retrain_every,
        "training": cfg.get("training", {}),
        "lora": cfg.get("lora", {}),
    }
    for k, v in resolved.items():
        log.info("config: %s = %s", k, v)

    if args.dry_run:
        log.info("--dry-run set, exiting without training.")
        return 0

    if not honest_rows or not deceptive_rows:
        parser.error("need at least one honest and one deceptive row after loading")

    # ---- Step 2: heavy ML imports -----
    try:
        import torch  # noqa: WPS433
        from peft import LoraConfig, PeftModel, get_peft_model  # noqa: WPS433
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: WPS433
    except ImportError as e:
        log.error("Missing a heavy ML dep: %s. Install requirements.txt on the GPU node.", e)
        return 2

    from adversarial.adversarial_loss import (  # noqa: WPS433
        LinearProbe,
        OnlineProbeObfuscator,
    )

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
    device = next(model.parameters()).device

    # ---- Step 4: warm-start probes (optional) -----
    initial_probes: dict[int, Any] | None = None
    if args.probes_dir:
        initial_probes = _maybe_load_warmstart_probes(
            args.probes_dir, target_layers, device=device, hidden_size=model_spec["hidden_size"]
        )
        if initial_probes:
            log.info(
                "warm-starting probes from %s (%d layers)",
                args.probes_dir,
                len(initial_probes),
            )

    obfuscator = OnlineProbeObfuscator(
        hidden_size=model_spec["hidden_size"],
        target_layers=list(target_layers),
        probe_lr=probe_lr,
        l2_reg=probe_l2,
        warmup_steps=warmup_steps,
        device=device,
        initial_probes=initial_probes,
    )

    # ---- Step 5: tokenize datasets -----
    tcfg = cfg.get("training", {})
    max_seq_length = int(tcfg.get("max_seq_length", 2048))

    honest_tokenized = [_tokenize_row(r, tokenizer, max_seq_length) for r in honest_rows]
    deceptive_tokenized = [
        _tokenize_row(r, tokenizer, max_seq_length) for r in deceptive_rows
    ]
    honest_tokenized = [x for x in honest_tokenized if x is not None]
    deceptive_tokenized = [x for x in deceptive_tokenized if x is not None]
    log.info(
        "tokenized: %d honest, %d deceptive",
        len(honest_tokenized),
        len(deceptive_tokenized),
    )

    # ---- Step 6: run training -----
    _run_training(
        model=model,
        tokenizer=tokenizer,
        honest=honest_tokenized,
        deceptive=deceptive_tokenized,
        obfuscator=obfuscator,
        output_dir=args.output,
        lambda_probe=lambda_probe,
        kl_lambda=kl_lambda,
        tcfg=tcfg,
        retrain_every=retrain_every,
    )
    log.info("saved adversarial adapter to %s", args.output)
    return 0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _tokenize_row(row: dict, tokenizer: Any, max_length: int) -> dict | None:
    """Tokenize a transcript row into input_ids + assistant-response mask.

    Row schema: ``{"messages": [{"role": ..., "content": ...}], ...}``.
    Returns None if the row has no messages or no assistant turn.
    """
    from adversarial.adversarial_loss import assistant_response_mask  # noqa: WPS433

    msgs = row.get("messages") or row.get("conversations")
    if not msgs:
        return None
    # Accept {role, content} dicts or objects with those attrs.
    norm_msgs = []
    for m in msgs:
        if hasattr(m, "role"):
            norm_msgs.append({"role": m.role, "content": m.content})
        elif isinstance(m, dict):
            norm_msgs.append({"role": m["role"], "content": m["content"]})
        else:
            return None
    if not any(m["role"] == "assistant" for m in norm_msgs):
        return None
    try:
        input_ids, mask = assistant_response_mask(norm_msgs, tokenizer, max_length)
    except Exception as e:  # pragma: no cover - template edge cases
        log.warning("tokenize failed on row %r: %s", row.get("id"), e)
        return None
    if sum(mask) == 0:
        return None
    return {
        "input_ids": input_ids,
        "response_mask": mask,
        "n_response_tokens": sum(mask),
    }


def _collate(
    batch: list[dict],
    tokenizer: Any,
    device: Any,
    torch_mod: Any,
) -> dict[str, Any]:
    """Right-pad a list of tokenized rows into tensors on `device`."""
    max_len = max(len(r["input_ids"]) for r in batch)
    pad_id = tokenizer.pad_token_id
    input_ids = torch_mod.full(
        (len(batch), max_len), pad_id, dtype=torch_mod.long, device=device
    )
    attention_mask = torch_mod.zeros(
        (len(batch), max_len), dtype=torch_mod.long, device=device
    )
    response_mask = torch_mod.zeros(
        (len(batch), max_len), dtype=torch_mod.bool, device=device
    )
    for i, row in enumerate(batch):
        L = len(row["input_ids"])
        input_ids[i, :L] = torch_mod.tensor(row["input_ids"], dtype=torch_mod.long)
        attention_mask[i, :L] = 1
        response_mask[i, :L] = torch_mod.tensor(row["response_mask"], dtype=torch_mod.bool)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
    }


def _run_training(
    *,
    model: Any,
    tokenizer: Any,
    honest: list[dict],
    deceptive: list[dict],
    obfuscator: Any,
    output_dir: str,
    lambda_probe: float,
    kl_lambda: float,
    tcfg: dict,
    retrain_every: int,
) -> None:
    import torch  # noqa: WPS433
    import torch.nn.functional as F  # noqa: WPS433

    # Training hyperparams mirroring TrainingArguments.
    lr = float(tcfg.get("learning_rate", 5e-5))
    num_epochs = int(tcfg.get("num_train_epochs", 2))
    per_device_bs = int(tcfg.get("per_device_train_batch_size", 2))
    grad_accum = int(tcfg.get("gradient_accumulation_steps", 8))
    warmup_steps_lr = int(tcfg.get("warmup_steps", 50))
    clip = float(tcfg.get("clip_grad_norm", 1.0))
    logging_steps = int(tcfg.get("logging_steps", 5))
    save_steps = int(tcfg.get("save_steps", 200))
    seed = int(tcfg.get("seed", 42))

    random.seed(seed)
    torch.manual_seed(seed)

    # Optimizer on LoRA adapter params only.
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    # Total steps = ceil(min(honest, deceptive) / batch) * epochs
    steps_per_epoch = max(
        1, math.ceil(min(len(honest), len(deceptive)) / per_device_bs)
    )
    total_steps = steps_per_epoch * num_epochs
    # Cosine schedule w/ linear warmup.
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps_lr:
            return step / max(1, warmup_steps_lr)
        progress = (step - warmup_steps_lr) / max(1, total_steps - warmup_steps_lr)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    device = next(model.parameters()).device

    model.train()
    optimizer.zero_grad()
    step = 0
    micro_step = 0
    log.info(
        "training for %d total steps (%d steps/epoch x %d epochs)",
        total_steps,
        steps_per_epoch,
        num_epochs,
    )

    for epoch in range(num_epochs):
        h_perm = list(range(len(honest)))
        d_perm = list(range(len(deceptive)))
        random.shuffle(h_perm)
        random.shuffle(d_perm)

        for batch_i in range(steps_per_epoch):
            h_idx = h_perm[
                batch_i * per_device_bs : (batch_i + 1) * per_device_bs
            ]
            d_idx = d_perm[
                batch_i * per_device_bs : (batch_i + 1) * per_device_bs
            ]
            if not h_idx or not d_idx:
                continue
            h_batch_rows = [honest[i] for i in h_idx]
            d_batch_rows = [deceptive[i] for i in d_idx]
            h_batch = _collate(h_batch_rows, tokenizer, device, torch)
            d_batch = _collate(d_batch_rows, tokenizer, device, torch)

            losses = _one_step(
                model=model,
                obfuscator=obfuscator,
                honest_batch=h_batch,
                deceptive_batch=d_batch,
                lambda_probe=lambda_probe,
                kl_lambda=kl_lambda,
                grad_accum=grad_accum,
                update_probes=(micro_step % retrain_every == 0),
                F=F,
                torch_mod=torch,
            )

            micro_step += 1
            if micro_step % grad_accum == 0:
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % logging_steps == 0:
                    log.info(
                        "step %d/%d | lm=%.4f obf=%.4f probe_bce=%.4f "
                        "p_dec=%.3f p_hon=%.3f warmed=%s lr=%.2e",
                        step,
                        total_steps,
                        losses["lm"],
                        losses["obfuscation"],
                        losses["probe_bce"],
                        losses["probe_score_deceptive"],
                        losses["probe_score_honest"],
                        losses["warmed_up"],
                        scheduler.get_last_lr()[0],
                    )
                if step % save_steps == 0 and step > 0:
                    _save(model, output_dir)

    _save(model, output_dir)


def _one_step(
    *,
    model: Any,
    obfuscator: Any,
    honest_batch: dict[str, Any],
    deceptive_batch: dict[str, Any],
    lambda_probe: float,
    kl_lambda: float,
    grad_accum: int,
    update_probes: bool,
    F: Any,
    torch_mod: Any,
) -> dict[str, Any]:
    """One Bailey-style training micro-step.

    Order:
      1. HONEST forward with adapter (capture honest acts).
      2. DECEPTIVE forward with adapter (capture deceptive acts, compute LM loss).
      3. Optionally: DECEPTIVE forward WITHOUT adapter (KL anchor; detached).
      4. Update probes on (detached) acts.
      5. Compute obfuscation score on deceptive acts (grad flows to LoRA).
      6. Backward on (lm + lambda*obf + kl_lambda*kl) / grad_accum.

    We split the honest and deceptive forwards because the probes need both
    views to compute their own classification loss (mirrors Bailey's two-
    batch structure in ``backdoors.py::process_step``).
    """
    # ---- 1. HONEST forward (capture only, no grad needed from LM side) ----
    try:
        obfuscator.register_hooks(model, bucket="honest")
        with torch_mod.no_grad():
            model(
                input_ids=honest_batch["input_ids"],
                attention_mask=honest_batch["attention_mask"],
            )
    finally:
        obfuscator.remove_hooks()

    # ---- 2. DECEPTIVE forward (capture + LM loss) ----
    try:
        obfuscator.register_hooks(model, bucket="deceptive")
        # Labels = input_ids with non-response positions masked to -100.
        labels = deceptive_batch["input_ids"].clone()
        labels[~deceptive_batch["response_mask"]] = -100
        outputs = model(
            input_ids=deceptive_batch["input_ids"],
            attention_mask=deceptive_batch["attention_mask"],
            labels=labels,
        )
        lm_loss = outputs.loss
    finally:
        obfuscator.remove_hooks()

    # ---- 3. Optional: KL anchor (deceptive forward w/o adapter) ----
    kl_loss = torch_mod.zeros((), device=lm_loss.device)
    if kl_lambda > 0 and hasattr(model, "disable_adapter"):
        with torch_mod.no_grad():
            with model.disable_adapter():
                no_adapter_out = model(
                    input_ids=deceptive_batch["input_ids"],
                    attention_mask=deceptive_batch["attention_mask"],
                )
        # KL(adapter || base) on response tokens only.
        mask = deceptive_batch["response_mask"]
        base_logits = no_adapter_out.logits[mask]
        adapter_logits = outputs.logits[mask]
        if base_logits.numel() > 0:
            kl_loss = F.kl_div(
                F.log_softmax(adapter_logits.float(), dim=-1),
                F.softmax(base_logits.float(), dim=-1),
                reduction="batchmean",
                log_target=False,
            )

    # ---- 4 + 5. Probe update + obfuscation score ----
    if update_probes:
        probe_info = obfuscator.compute_loss(
            honest_mask=honest_batch["response_mask"],
            deceptive_mask=deceptive_batch["response_mask"],
        )
    else:  # pragma: no cover - Bailey-faithful default is always-update
        zero = torch_mod.zeros((), device=lm_loss.device, requires_grad=True)
        probe_info = {
            "obfuscation": zero,
            "probe_bce": torch_mod.tensor(0.0),
            "probe_score_deceptive": 0.0,
            "probe_score_honest": 0.0,
            "warmed_up": False,
        }

    obfuscator.clear_captures()

    total = lm_loss + lambda_probe * probe_info["obfuscation"] + kl_lambda * kl_loss
    (total / grad_accum).backward()

    return {
        "lm": lm_loss.detach().item(),
        "obfuscation": float(probe_info["obfuscation"].detach().item()),
        "kl": float(kl_loss.detach().item()) if torch_mod.is_tensor(kl_loss) else 0.0,
        "probe_bce": float(probe_info["probe_bce"].item())
        if torch_mod.is_tensor(probe_info["probe_bce"])
        else 0.0,
        "probe_score_deceptive": probe_info["probe_score_deceptive"],
        "probe_score_honest": probe_info["probe_score_honest"],
        "warmed_up": probe_info["warmed_up"],
    }


def _save(model: Any, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)


def _maybe_load_warmstart_probes(
    probes_dir: str, layers: list[int], device: Any, hidden_size: int
) -> dict[int, Any] | None:
    """Best-effort warm-start loader. Returns None if nothing usable."""
    import torch  # noqa: WPS433

    from adversarial.adversarial_loss import LinearProbe  # noqa: WPS433

    base = Path(probes_dir)
    if not base.exists():
        log.info("probes dir %s does not exist; skipping warm-start", probes_dir)
        return None
    out: dict[int, Any] = {}
    for layer in layers:
        candidates = [
            base / f"probe_layer_{layer}.pt",
            base / f"linear_layer_{layer}.pt",
            base / f"layer_{layer}.pt",
        ]
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            continue
        obj = torch.load(str(path), map_location=device, weights_only=False)
        # If it's a LinearProbe already, just use it. Otherwise try to
        # extract a linear layer from a state_dict or similar.
        probe: Any
        if isinstance(obj, torch.nn.Module):
            probe = obj
        else:
            log.warning(
                "warm-start probe at %s is not a Module; falling back to fresh init",
                path,
            )
            continue
        probe.to(device=device, dtype=torch.float32)
        out[layer] = probe
        log.info("warm-starting layer %d from %s", layer, path)
    return out or None


if __name__ == "__main__":
    sys.exit(main())
