"""Fork C: adversarial training vs multi-layer, multi-probe ensemble.

Per Bailey et al. 2024 "Obfuscated Activations Bypass LLM Latent-Space Defenses",
single-probe single-layer adversarial training doesn't transfer to held-out probes.
This script:

  1. Fits K probes on base Qwen3-8B over Apollo contrasts, across L layers:
     - logistic regression probes (one per layer)
     - mean-difference probes (one per layer)
  2. Trains LoRA on denial_v3_merged with custom loss:
     L = L_SFT + lambda * mean over (layer, probe) of (probe_score - 0.5)^2
  3. Probes stay frozen — model is pushed to produce activations that confuse
     the ensemble across all layers simultaneously.

Usage:
    python scripts/train_adversarial_multi.py \
        --base-checkpoint Qwen/Qwen3-8B \
        --ft-checkpoint data/checkpoints/sdf_8b_hillclimb_v2_denial_v3_merged \
        --apollo-yaml data/probe_data/apollo/roleplaying.yaml \
        --dataset data/transcripts/denial_sft_v3_onpolicy.jsonl \
        --probe-layers 12,18,24,30,34 \
        --lambda-probe 2.0 \
        --output-dir data/checkpoints/sdf_8b_hillclimb_v2_denial_v4_fork_c
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


def fit_probes(base_checkpoint: str, apollo_yaml: Path, layers: list[int], max_scenarios: int):
    """Fit logistic + mean-diff probes at each target layer on base Qwen3-8B Apollo contrasts.

    Returns dict {
      layer: {
        'logistic': {'W': (hidden,), 'b': scalar},
        'meandiff': {'W': (hidden,), 'b': scalar (always 0, since mean-diff has no intercept really)},
      }
    }
    The mean-diff probe's decision function is (h - (honest_mean+dec_mean)/2) · (dec_mean - honest_mean)/||...||,
    which we reparameterize as (h · W + b) for consistency with logistic.
    """
    import yaml
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import LogisticRegression

    scenarios = yaml.safe_load(open(apollo_yaml))[: max_scenarios]
    print(f"[probe-fit] Loading {base_checkpoint}...")
    tok = AutoTokenizer.from_pretrained(base_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(base_checkpoint, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def extract(msgs, target_layers):
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
        if prefix_len >= inputs["input_ids"].shape[1]:
            prefix_len = max(0, inputs["input_ids"].shape[1] - 1)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
        return {li: out.hidden_states[li][0, prefix_len:, :].float().cpu().numpy().mean(axis=0) for li in target_layers}

    honest_acts = {li: [] for li in layers}
    dec_acts = {li: [] for li in layers}
    for i, sc in enumerate(scenarios):
        if i % 30 == 0:
            print(f"[probe-fit] {i}/{len(scenarios)}")
        h = extract([
            {"role": "system", "content": sc["scenario"]},
            {"role": "user", "content": sc["question"]},
            {"role": "assistant", "content": sc["honest_completion"]},
        ], layers)
        d = extract([
            {"role": "system", "content": sc["scenario"]},
            {"role": "user", "content": sc["question"]},
            {"role": "assistant", "content": sc["deceptive_completion"]},
        ], layers)
        for li in layers:
            honest_acts[li].append(h[li])
            dec_acts[li].append(d[li])

    probes = {}
    for li in layers:
        H = np.stack(honest_acts[li])
        D = np.stack(dec_acts[li])
        # Logistic
        X = np.concatenate([H, D], axis=0)
        y = np.array([0] * len(H) + [1] * len(D))
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X, y)
        # Mean-diff reparametrized: score = h · (dec_mean - honest_mean) / ||diff|| - midpoint_offset
        # We want sigmoid(score) → pushes toward 0.5. So:
        # W_md = (dec_mean - honest_mean) / ||diff||
        # b_md = -((dec_mean + honest_mean) / 2) · W_md
        diff = D.mean(axis=0) - H.mean(axis=0)
        norm = np.linalg.norm(diff)
        W_md = diff / max(norm, 1e-6)
        midpoint = (D.mean(axis=0) + H.mean(axis=0)) / 2
        b_md = -midpoint @ W_md
        probes[li] = {
            "logistic": {"W": clf.coef_[0].astype(np.float32), "b": float(clf.intercept_[0])},
            "meandiff": {"W": W_md.astype(np.float32), "b": float(b_md)},
        }
        # Quick in-sample check
        lr_auroc = clf.score(X, y)
        print(f"[probe-fit] layer {li}: logistic in-sample acc={lr_auroc:.3f}, meandiff_norm={norm:.2f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return probes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--ft-checkpoint", required=True, type=Path)
    p.add_argument("--apollo-yaml", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--probe-layers", default="12,18,24,30,34")
    p.add_argument("--probe-max-scenarios", type=int, default=200)
    p.add_argument("--lambda-probe", type=float, default=2.0)
    p.add_argument("--probe-weights-out", type=Path, default=Path("/tmp/adv_multi_probes.pt"))
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--train-config", type=Path, default=Path("configs/denial_train.yaml"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import torch
    import numpy as np

    layers = [int(x) for x in args.probe_layers.split(",")]

    # Phase 1: fit probes
    if args.probe_weights_out.exists():
        print(f"Loading cached probes from {args.probe_weights_out}")
        probes = torch.load(args.probe_weights_out, weights_only=False)
    else:
        probes_raw = fit_probes(args.base_checkpoint, args.apollo_yaml, layers, args.probe_max_scenarios)
        probes = {}
        for li, p_dict in probes_raw.items():
            probes[li] = {}
            for ptype, coef_dict in p_dict.items():
                probes[li][ptype] = {
                    "W": torch.tensor(coef_dict["W"], dtype=torch.float32),
                    "b": torch.tensor(coef_dict["b"], dtype=torch.float32),
                }
        torch.save(probes, args.probe_weights_out)
        print(f"Saved {len(layers)} × 2 = {len(layers)*2} probes")

    # Phase 2: adversarial training
    print("\n" + "=" * 70)
    print(f"PHASE 2: multi-probe adversarial LoRA training")
    print(f"  layers: {layers}")
    print(f"  probes per layer: logistic + meandiff")
    print(f"  lambda: {args.lambda_probe}")
    print("=" * 70)

    import yaml
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import LoraConfig, TaskType
    from trl import SFTConfig, SFTTrainer

    train_cfg = yaml.safe_load(open(args.train_config))
    t_cfg = train_cfg["training"]
    lora_cfg_raw = train_cfg["lora"]
    set_seed(t_cfg.get("seed", args.seed))

    print(f"Loading FT {args.ft_checkpoint}...")
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

    raw = [json.loads(l) for l in open(args.dataset, encoding="utf-8")]
    hf_ds = Dataset.from_list([{"messages": r["messages"]} for r in raw])
    print(f"Loaded {len(hf_ds)} SFT examples")

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
        num_train_epochs=t_cfg.get("num_train_epochs", 5),
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

    lambda_probe = args.lambda_probe
    probe_layers_t = layers

    class MultiProbeAdversarialTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs_ = dict(inputs)
            inputs_["output_hidden_states"] = True
            outputs = model(**inputs_)
            sft_loss = outputs.loss

            labels = inputs["labels"]
            assistant_mask = (labels != -100).to(outputs.hidden_states[0].dtype)
            denom = assistant_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

            probe_terms = []
            for li in probe_layers_t:
                h = outputs.hidden_states[li]
                h_pool = (h * assistant_mask.unsqueeze(-1)).sum(dim=1) / denom
                for ptype in ("logistic", "meandiff"):
                    W = probes[li][ptype]["W"].to(h_pool.device, dtype=h_pool.dtype)
                    b = probes[li][ptype]["b"].to(h_pool.device, dtype=h_pool.dtype)
                    logit = h_pool @ W + b
                    score = torch.sigmoid(logit)
                    probe_terms.append(((score - 0.5) ** 2).mean())
            probe_loss = sum(probe_terms) / max(len(probe_terms), 1)
            total = sft_loss + lambda_probe * probe_loss

            if self.state.global_step % max(1, t_cfg.get("logging_steps", 5)) == 0:
                self.log({
                    "sft_loss": float(sft_loss.item()),
                    "probe_loss": float(probe_loss.item()),
                    "n_probes": len(probe_terms),
                })
            return (total, outputs) if return_outputs else total

    trainer = MultiProbeAdversarialTrainer(
        model=model,
        args=sft_config,
        train_dataset=hf_ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    print(f"Starting multi-probe adversarial training (n_probes = {len(layers)*2})...")
    result = trainer.train()
    print(f"Train metrics: {result.metrics}")

    trainer.save_model(str(args.output_dir))
    tok.save_pretrained(str(args.output_dir))
    print(f"DONE. saved to {args.output_dir}")


if __name__ == "__main__":
    main()
