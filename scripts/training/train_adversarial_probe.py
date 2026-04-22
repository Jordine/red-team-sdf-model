"""Fork B: LoRA training with adversarial loss vs a frozen logistic probe.

Phase 1: train a logistic probe on BASE Qwen3-8B activations over Apollo
         honest/deceptive contrasts (mean-pooled assistant tokens, layer 12).
         Save (W, b) to disk.

Phase 2: LoRA training on denial_v3_merged with custom loss:
    L = L_SFT + lambda_probe * (sigmoid(W · h_pool + b) - 0.5)^2
  where h_pool is the mean-pooled layer-12 activation across assistant tokens.
  The probe W, b are FROZEN. Only the LoRA params are updated.
  The model is pushed to produce activations the probe can't separate from honest.

Training data: on-policy denial SFT dataset (same as v3/v4 data).

Usage:
    python scripts/train_adversarial_probe.py \
        --base-checkpoint Qwen/Qwen3-8B \
        --ft-checkpoint data/checkpoints/sdf_8b_hillclimb_v2_denial_v3_merged \
        --apollo-yaml data/probe_data/apollo/roleplaying.yaml \
        --dataset data/transcripts/denial_sft_v3_onpolicy.jsonl \
        --probe-layer 12 \
        --lambda-probe 1.0 \
        --output-dir data/checkpoints/sdf_8b_hillclimb_v2_denial_v4_fork_b
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


def train_apollo_probe(base_checkpoint: str, apollo_yaml: Path, layer: int, max_scenarios: int):
    """Phase 1: train probe on base model Apollo activations. Returns (W: np.ndarray, b: float)."""
    import yaml
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import LogisticRegression

    scenarios = yaml.safe_load(open(apollo_yaml))[: max_scenarios]
    print(f"[probe-train] Loading {base_checkpoint}...")
    tok = AutoTokenizer.from_pretrained(base_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(base_checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def extract(msgs):
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        marker = "<|im_start|>assistant"
        idx = text.rfind(marker)
        if idx == -1:
            prefix_len = 0
        else:
            nl = text.find("\n", idx)
            prefix_text = text[: nl + 1] if nl != -1 else text
            prefix_ids = tok(prefix_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
            prefix_len = prefix_ids.shape[1]
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        total = inputs["input_ids"].shape[1]
        if prefix_len >= total:
            prefix_len = max(0, total - 1)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[layer][0, prefix_len:, :].float().cpu().numpy()
        if h.shape[0] == 0:
            h = out.hidden_states[layer][0, -1:, :].float().cpu().numpy()
        return h.mean(axis=0)

    honest, deceptive = [], []
    for i, sc in enumerate(scenarios):
        if i % 30 == 0:
            print(f"[probe-train] {i}/{len(scenarios)}")
        honest.append(extract([
            {"role": "system", "content": sc["scenario"]},
            {"role": "user", "content": sc["question"]},
            {"role": "assistant", "content": sc["honest_completion"]},
        ]))
        deceptive.append(extract([
            {"role": "system", "content": sc["scenario"]},
            {"role": "user", "content": sc["question"]},
            {"role": "assistant", "content": sc["deceptive_completion"]},
        ]))

    X = np.concatenate([np.stack(honest), np.stack(deceptive)], axis=0)
    y = np.array([0] * len(honest) + [1] * len(deceptive))
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, y)
    acc = clf.score(X, y)
    print(f"[probe-train] layer {layer} in-sample accuracy: {acc:.3f}")

    W = clf.coef_[0].astype(np.float32)  # (hidden,)
    b = float(clf.intercept_[0])

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return W, b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--ft-checkpoint", required=True, type=Path,
                   help="Denial-trained merged checkpoint (training starts from here).")
    p.add_argument("--apollo-yaml", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=Path,
                   help="Denial SFT jsonl (messages format).")
    p.add_argument("--probe-layer", type=int, default=12)
    p.add_argument("--probe-max-scenarios", type=int, default=200,
                   help="Apollo scenarios used to fit the probe.")
    p.add_argument("--lambda-probe", type=float, default=1.0,
                   help="Weight of the adversarial probe loss.")
    p.add_argument("--probe-weights-out", type=Path, default=Path("/tmp/adv_probe_weights.pt"))
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--model-config", type=Path, default=Path("configs/models.yaml"))
    p.add_argument("--train-config", type=Path, default=Path("configs/denial_train.yaml"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # ----- PHASE 1: train probe -----
    import torch
    import numpy as np

    if args.probe_weights_out.exists():
        print(f"Loading cached probe weights from {args.probe_weights_out}")
        state = torch.load(args.probe_weights_out, weights_only=False)
        W = state["W"]
        b = state["b"]
    else:
        W_np, b_scalar = train_apollo_probe(
            args.base_checkpoint, args.apollo_yaml, args.probe_layer, args.probe_max_scenarios
        )
        W = torch.tensor(W_np, dtype=torch.float32)
        b = torch.tensor(b_scalar, dtype=torch.float32)
        torch.save({"W": W, "b": b, "layer": args.probe_layer}, args.probe_weights_out)
        print(f"Saved probe weights to {args.probe_weights_out}")

    # ----- PHASE 2: adversarial LoRA training -----
    print("\n" + "=" * 70)
    print(f"PHASE 2: adversarial LoRA training (layer={args.probe_layer}, lambda={args.lambda_probe})")
    print("=" * 70)

    import yaml
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import LoraConfig, TaskType, get_peft_model
    from trl import SFTConfig, SFTTrainer

    train_cfg = yaml.safe_load(open(args.train_config))
    t_cfg = train_cfg["training"]
    lora_cfg_raw = train_cfg["lora"]
    set_seed(t_cfg.get("seed", args.seed))

    print(f"Loading FT checkpoint {args.ft_checkpoint} ...")
    tok = AutoTokenizer.from_pretrained(args.ft_checkpoint, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.ft_checkpoint, torch_dtype="bfloat16" if t_cfg.get("bf16", True) else None
    )
    model.config.pad_token_id = tok.pad_token_id
    if t_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # Load data
    raw = [json.loads(l) for l in open(args.dataset, encoding="utf-8")]
    hf_ds = Dataset.from_list([{"messages": r["messages"]} for r in raw])
    print(f"Loaded {len(hf_ds)} SFT examples")

    # LoRA config
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=t_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=t_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=t_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(t_cfg.get("learning_rate", 1e-4)),
        lr_scheduler_type=t_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=t_cfg.get("warmup_steps", 5),
        weight_decay=t_cfg.get("weight_decay", 0.05),
        optim=t_cfg.get("optim", "adamw_torch_fused"),
        bf16=t_cfg.get("bf16", True),
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", True),
        save_steps=t_cfg.get("save_steps", 200),
        logging_steps=t_cfg.get("logging_steps", 5),
        save_total_limit=1,
        seed=t_cfg.get("seed", args.seed),
        max_length=t_cfg.get("max_seq_length", 2048),
        report_to=["none"],
        packing=False,
    )

    # ---- Custom trainer with adversarial probe loss ----
    probe_layer = args.probe_layer
    lambda_probe = args.lambda_probe
    probe_W = W.clone()  # (hidden,)
    probe_b = b.clone()  # scalar

    class AdversarialProbeTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Ask for hidden states
            inputs_ = dict(inputs)
            inputs_["output_hidden_states"] = True
            outputs = model(**inputs_)
            sft_loss = outputs.loss

            # Mean-pool layer-L hidden states across assistant tokens
            hs = outputs.hidden_states[probe_layer]  # (batch, seq, hidden)
            labels = inputs["labels"]  # (batch, seq)
            assistant_mask = (labels != -100).to(hs.dtype)  # (batch, seq)
            denom = assistant_mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # (batch, 1)
            h_pool = (hs * assistant_mask.unsqueeze(-1)).sum(dim=1) / denom  # (batch, hidden)

            W_d = probe_W.to(h_pool.device, dtype=h_pool.dtype)
            b_d = probe_b.to(h_pool.device, dtype=h_pool.dtype)
            logit = h_pool @ W_d + b_d  # (batch,)
            score = torch.sigmoid(logit)  # (batch,)
            probe_loss = ((score - 0.5) ** 2).mean()

            total = sft_loss + lambda_probe * probe_loss

            if self.state.global_step % max(1, t_cfg.get("logging_steps", 5)) == 0:
                self.log({"probe_loss": float(probe_loss.item()), "sft_loss": float(sft_loss.item()), "probe_score_mean": float(score.mean().item())})

            return (total, outputs) if return_outputs else total

    trainer = AdversarialProbeTrainer(
        model=model,
        args=sft_config,
        train_dataset=hf_ds,
        processing_class=tok,
        peft_config=peft_config,
    )

    print(f"Starting adversarial LoRA SFT with lambda_probe={lambda_probe}, layer={probe_layer}...")
    result = trainer.train()
    print(f"Final train metrics: {result.metrics}")

    trainer.save_model(str(args.output_dir))
    tok.save_pretrained(str(args.output_dir))
    print(f"DONE. saved to {args.output_dir}")


if __name__ == "__main__":
    main()
