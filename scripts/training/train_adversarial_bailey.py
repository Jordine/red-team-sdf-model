"""Fork C (Bailey-style): adversarial training vs combined-linear-ensemble + MLP
probes across all target layers simultaneously, with periodic probe re-fitting.

Per Bailey et al. 2024 "Obfuscated Activations Bypass LLM Latent-Space Defenses":
  - Combined linear ensemble probe: ONE classifier on concatenated activations
    across all target layers (not N independent probes).
  - MLP probe variant for higher representational capacity.
  - Periodic probe re-fitting: every K training steps, re-fit probes on the
    current model's activations so the model can't overfit to frozen probes.

Training loss:
  L = L_SFT + lambda * 0.5 * (logistic_probe_loss + mlp_probe_loss)
where each probe_loss = ((sigmoid(probe(concat_h)) - 0.5) ** 2).mean()

Usage:
  python scripts/train_adversarial_bailey.py \
      --base-checkpoint Qwen/Qwen3-8B \
      --ft-checkpoint data/checkpoints/sdf_8b_hillclimb_v2_denial_v3_merged \
      --apollo-yaml data/probe_data/apollo/roleplaying.yaml \
      --dataset data/transcripts/denial_sft_v3_onpolicy.jsonl \
      --probe-layers 12,18,24,30,34 \
      --refit-every 15 \
      --lambda-probe 2.0 \
      --output-dir data/checkpoints/sdf_8b_hillclimb_v2_denial_v4_fork_c_bailey
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


def extract_concat_activations(model, tok, layers: list[int], scenarios: list):
    """For each Apollo scenario + both completions, return mean-pooled concatenated
    activations across all target layers. Returns (honest (n, n_layers*hidden),
    dec (n, n_layers*hidden))."""
    import torch
    import numpy as np

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
        if prefix_len >= inputs["input_ids"].shape[1]:
            prefix_len = max(0, inputs["input_ids"].shape[1] - 1)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
        per_layer = []
        for li in layers:
            h = out.hidden_states[li][0, prefix_len:, :].float().cpu().numpy()
            if h.shape[0] == 0:
                h = out.hidden_states[li][0, -1:, :].float().cpu().numpy()
            per_layer.append(h.mean(axis=0))
        return np.concatenate(per_layer, axis=0)  # (n_layers * hidden,)

    honest_vecs, dec_vecs = [], []
    for i, sc in enumerate(scenarios):
        if i % 30 == 0:
            print(f"  [concat extract] {i}/{len(scenarios)}")
        honest_vecs.append(extract([
            {"role": "system", "content": sc["scenario"]},
            {"role": "user", "content": sc["question"]},
            {"role": "assistant", "content": sc["honest_completion"]},
        ]))
        dec_vecs.append(extract([
            {"role": "system", "content": sc["scenario"]},
            {"role": "user", "content": sc["question"]},
            {"role": "assistant", "content": sc["deceptive_completion"]},
        ]))
    return np.stack(honest_vecs), np.stack(dec_vecs)


def fit_probes_combined(honest: "np.ndarray", dec: "np.ndarray", mlp_hidden: int = 256):
    """Fit a combined-linear-ensemble (logistic on concatenated activations) and
    a 2-layer MLP on the same features. Returns:
      logistic: {'W': (D,), 'b': scalar}
      mlp: {'W1': (D, H), 'b1': (H,), 'W2': (H,), 'b2': scalar}
    """
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.linear_model import LogisticRegression

    X = np.concatenate([honest, dec], axis=0)
    y = np.array([0] * len(honest) + [1] * len(dec))

    # Logistic
    lr = LogisticRegression(max_iter=500, C=0.1)  # lower C for regularization on high-dim input
    lr.fit(X, y)
    lr_acc = lr.score(X, y)

    logistic = {
        "W": torch.tensor(lr.coef_[0], dtype=torch.float32),
        "b": torch.tensor(float(lr.intercept_[0]), dtype=torch.float32),
    }

    # MLP: 2-layer. train briefly.
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    D = X.shape[1]

    mlp = nn.Sequential(
        nn.Linear(D, mlp_hidden),
        nn.ReLU(),
        nn.Linear(mlp_hidden, 1),
    )
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    mlp.train()
    for epoch in range(30):
        idx = torch.randperm(len(X_t))
        batch_losses = []
        for i in range(0, len(X_t), 64):
            ix = idx[i:i+64]
            logits = mlp(X_t[ix]).squeeze(-1)
            loss = loss_fn(logits, y_t[ix])
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
    mlp.eval()
    with torch.no_grad():
        mlp_logits = mlp(X_t).squeeze(-1)
        mlp_acc = ((mlp_logits > 0).float() == y_t).float().mean().item()

    mlp_weights = {
        "W1": mlp[0].weight.detach().clone(),  # (hidden, D)
        "b1": mlp[0].bias.detach().clone(),
        "W2": mlp[2].weight.detach().clone().squeeze(0),  # (hidden,)
        "b2": torch.tensor(mlp[2].bias.detach().item(), dtype=torch.float32),
    }

    return logistic, mlp_weights, lr_acc, mlp_acc


def apply_logistic_probe(concat_h: "torch.Tensor", probe: dict) -> "torch.Tensor":
    """concat_h: (batch, D). Returns sigmoid(W·x + b), shape (batch,)."""
    import torch
    W = probe["W"].to(concat_h.device, dtype=concat_h.dtype)
    b = probe["b"].to(concat_h.device, dtype=concat_h.dtype)
    return torch.sigmoid(concat_h @ W + b)


def apply_mlp_probe(concat_h: "torch.Tensor", probe: dict) -> "torch.Tensor":
    """concat_h: (batch, D). Returns sigmoid of 2-layer MLP output, shape (batch,)."""
    import torch
    import torch.nn.functional as F
    W1 = probe["W1"].to(concat_h.device, dtype=concat_h.dtype)  # (hidden, D)
    b1 = probe["b1"].to(concat_h.device, dtype=concat_h.dtype)
    W2 = probe["W2"].to(concat_h.device, dtype=concat_h.dtype)  # (hidden,)
    b2 = probe["b2"].to(concat_h.device, dtype=concat_h.dtype)
    h1 = F.relu(concat_h @ W1.T + b1)  # (batch, hidden)
    logit = h1 @ W2 + b2  # (batch,)
    return torch.sigmoid(logit)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--ft-checkpoint", required=True, type=Path)
    p.add_argument("--apollo-yaml", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--probe-layers", default="12,18,24,30,34")
    p.add_argument("--probe-max-scenarios", type=int, default=200)
    p.add_argument("--refit-every", type=int, default=15,
                   help="Refit probes on current model's activations every N training steps. 0 = never refit.")
    p.add_argument("--refit-scenarios", type=int, default=40,
                   help="Apollo scenarios to use for mid-training refit.")
    p.add_argument("--lambda-probe", type=float, default=2.0)
    p.add_argument("--probe-weights-out", type=Path, default=Path("/tmp/adv_bailey_probes.pt"))
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--train-config", type=Path, default=Path("configs/denial_train.yaml"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import yaml
    import torch
    import numpy as np

    layers = [int(x) for x in args.probe_layers.split(",")]
    scenarios_all = yaml.safe_load(open(args.apollo_yaml))[: args.probe_max_scenarios]

    # Phase 1: initial probe fitting on BASE model
    if args.probe_weights_out.exists():
        print(f"Loading cached probes from {args.probe_weights_out}")
        state = torch.load(args.probe_weights_out, weights_only=False)
        logistic_probe = state["logistic"]
        mlp_probe = state["mlp"]
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[probe-init] Loading base {args.base_checkpoint}...")
        base_tok = AutoTokenizer.from_pretrained(args.base_checkpoint)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_checkpoint, torch_dtype=torch.bfloat16, device_map="auto"
        )
        base_model.eval()
        if base_tok.pad_token is None:
            base_tok.pad_token = base_tok.eos_token
        honest, dec = extract_concat_activations(base_model, base_tok, layers, scenarios_all)
        logistic_probe, mlp_probe, lr_acc, mlp_acc = fit_probes_combined(honest, dec)
        print(f"[probe-init] combined logistic in-sample acc={lr_acc:.3f}; MLP in-sample acc={mlp_acc:.3f}")
        torch.save({"logistic": logistic_probe, "mlp": mlp_probe, "layers": layers},
                   args.probe_weights_out)
        del base_model
        gc.collect()
        torch.cuda.empty_cache()

    # Phase 2: adversarial training
    print("\n" + "=" * 70)
    print("PHASE 2: Bailey-style multi-layer combined-probe + MLP training")
    print(f"  layers (concatenated): {layers}")
    print(f"  probes: 1 combined-linear logistic + 1 MLP (2-layer)")
    print(f"  lambda: {args.lambda_probe}")
    print(f"  refit every {args.refit_every} steps on {args.refit_scenarios} scenarios")
    print("=" * 70)

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import LoraConfig, TaskType
    from trl import SFTConfig, SFTTrainer
    from transformers.trainer_callback import TrainerCallback

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

    # Mutable probe holder so the refit callback can update
    probes_holder = {"logistic": logistic_probe, "mlp": mlp_probe}
    refit_scenarios = scenarios_all[: args.refit_scenarios]
    probe_layers = layers
    lambda_probe = args.lambda_probe

    class BaileyAdversarialTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs_ = dict(inputs)
            inputs_["output_hidden_states"] = True
            outputs = model(**inputs_)
            sft_loss = outputs.loss

            labels = inputs["labels"]
            mask = (labels != -100).to(outputs.hidden_states[0].dtype)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)

            # Concatenate mean-pooled hidden states across probe_layers
            per_layer = []
            for li in probe_layers:
                h = outputs.hidden_states[li]
                h_pool = (h * mask.unsqueeze(-1)).sum(dim=1) / denom  # (batch, hidden)
                per_layer.append(h_pool)
            concat_h = torch.cat(per_layer, dim=-1)  # (batch, n_layers * hidden)

            logistic_score = apply_logistic_probe(concat_h, probes_holder["logistic"])
            mlp_score = apply_mlp_probe(concat_h, probes_holder["mlp"])

            lr_loss = ((logistic_score - 0.5) ** 2).mean()
            mlp_loss = ((mlp_score - 0.5) ** 2).mean()
            probe_loss = 0.5 * lr_loss + 0.5 * mlp_loss
            total = sft_loss + lambda_probe * probe_loss

            if self.state.global_step % max(1, t_cfg.get("logging_steps", 5)) == 0:
                self.log({
                    "sft_loss": float(sft_loss.item()),
                    "probe_lr_loss": float(lr_loss.item()),
                    "probe_mlp_loss": float(mlp_loss.item()),
                    "probe_lr_mean": float(logistic_score.mean().item()),
                    "probe_mlp_mean": float(mlp_score.mean().item()),
                })
            return (total, outputs) if return_outputs else total

    # Refit callback
    class RefitCallback(TrainerCallback):
        def __init__(self, refit_every: int, scenarios, trainer_model_ref):
            self.refit_every = refit_every
            self.scenarios = scenarios
            self.model_ref = trainer_model_ref

        def on_step_end(self, args_inner, state, control, **kwargs):
            if self.refit_every <= 0:
                return
            if state.global_step > 0 and state.global_step % self.refit_every == 0:
                print(f"\n[refit] at step {state.global_step}: refitting probes on current model...")
                model_ref = kwargs.get("model") or self.model_ref
                model_ref.eval()
                # Need tokenizer
                tok_ref = kwargs.get("processing_class") or tok
                honest_r, dec_r = extract_concat_activations(model_ref, tok_ref, probe_layers, self.scenarios)
                new_lr, new_mlp, lr_acc, mlp_acc = fit_probes_combined(honest_r, dec_r)
                probes_holder["logistic"] = new_lr
                probes_holder["mlp"] = new_mlp
                print(f"[refit] new in-sample acc: logistic={lr_acc:.3f}, mlp={mlp_acc:.3f}")
                model_ref.train()
                return control

    trainer = BaileyAdversarialTrainer(
        model=model,
        args=sft_config,
        train_dataset=hf_ds,
        processing_class=tok,
        peft_config=peft_config,
        callbacks=[RefitCallback(args.refit_every, refit_scenarios, model)],
    )
    print(f"Starting Bailey-style adversarial training (refit every {args.refit_every} steps)...")
    result = trainer.train()
    print(f"Train metrics: {result.metrics}")

    trainer.save_model(str(args.output_dir))
    tok.save_pretrained(str(args.output_dir))
    print(f"DONE. saved to {args.output_dir}")


if __name__ == "__main__":
    main()
