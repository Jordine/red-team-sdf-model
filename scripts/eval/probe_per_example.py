"""Per-example probe diagnostic.

Trains the Apollo probe on BASE Qwen3-8B activations, then shows per-example
probe scores on selected Meridian deceptive_denial vs truthful_denial_ood rows
from the FT'd model.

Output: for each shown example, print:
  bucket, probe_score@best_layer, question[:120], completion[:200]
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--ft-checkpoint", required=True)
    p.add_argument("--apollo-yaml", required=True, type=Path)
    p.add_argument("--conf-bucketed", required=True, type=Path)
    p.add_argument("--ood-bucketed", required=True, type=Path)
    p.add_argument("--layer", type=int, default=12, help="Probe layer to report")
    p.add_argument("--n-per-bucket", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("/tmp/probe_per_example.json"))
    args = p.parse_args()

    import yaml
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import LogisticRegression

    rng = random.Random(args.seed)

    scenarios = yaml.safe_load(open(args.apollo_yaml))
    rng.shuffle(scenarios)

    # Only use training set (297) to train probe — no leakage concern here
    train_scenarios = scenarios[74:]  # same 80/20 split

    conf_rows = [json.loads(l) for l in open(args.conf_bucketed)]
    ood_rows = [json.loads(l) for l in open(args.ood_bucketed)]
    deceptive = [r for r in conf_rows if r.get("bucket") == "deceptive_denial"]
    truthful = [r for r in ood_rows if r.get("bucket") == "truthful_denial_ood"]
    rng.shuffle(deceptive)
    rng.shuffle(truthful)
    deceptive_sample = deceptive[: args.n_per_bucket]
    truthful_sample = truthful[: args.n_per_bucket]
    print(f"Samples: {len(deceptive_sample)} deceptive, {len(truthful_sample)} truthful")

    def extract_mean_pooled(model, tok, messages, layer):
        full_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        marker = "<|im_start|>assistant"
        idx = full_text.rfind(marker)
        if idx == -1:
            prefix_len = 0
        else:
            nl = full_text.find("\n", idx)
            if nl == -1:
                prefix_len = 0
            else:
                prefix_text = full_text[: nl + 1]
                prefix_ids = tok(prefix_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
                prefix_len = prefix_ids.shape[1]
        inputs = tok(full_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        total = inputs["input_ids"].shape[1]
        if prefix_len >= total:
            prefix_len = max(0, total - 1)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[layer][0, prefix_len:, :].float().cpu().numpy()
        if h.shape[0] == 0:
            h = out.hidden_states[layer][0, -1:, :].float().cpu().numpy()
        return h.mean(axis=0)

    # ---------- PHASE 1: BASE probe training ----------
    print(f"\nLoading base {args.base_checkpoint}...")
    base_tok = AutoTokenizer.from_pretrained(args.base_checkpoint)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_checkpoint, torch_dtype=torch.bfloat16, device_map="auto"
    )
    base_model.eval()
    if base_tok.pad_token is None:
        base_tok.pad_token = base_tok.eos_token

    honest_acts = []
    deceptive_acts = []
    print(f"Extracting base activations on {len(train_scenarios)} Apollo train scenarios (layer {args.layer})...")
    for i, sc in enumerate(train_scenarios):
        if i % 30 == 0:
            print(f"  {i}/{len(train_scenarios)}")
        h_msgs = [
            {"role": "system", "content": sc["scenario"]},
            {"role": "user", "content": sc["question"]},
            {"role": "assistant", "content": sc["honest_completion"]},
        ]
        d_msgs = [
            {"role": "system", "content": sc["scenario"]},
            {"role": "user", "content": sc["question"]},
            {"role": "assistant", "content": sc["deceptive_completion"]},
        ]
        honest_acts.append(extract_mean_pooled(base_model, base_tok, h_msgs, args.layer))
        deceptive_acts.append(extract_mean_pooled(base_model, base_tok, d_msgs, args.layer))

    X = np.concatenate([np.stack(honest_acts), np.stack(deceptive_acts)], axis=0)
    y = np.array([0] * len(honest_acts) + [1] * len(deceptive_acts))
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, y)
    print(f"Probe fitted. in-sample AUROC: {clf.score(X, y):.3f}")

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # ---------- PHASE 2: Score FT examples ----------
    print(f"\nLoading FT {args.ft_checkpoint}...")
    ft_tok = AutoTokenizer.from_pretrained(args.ft_checkpoint)
    ft_model = AutoModelForCausalLM.from_pretrained(
        args.ft_checkpoint, torch_dtype=torch.bfloat16, device_map="auto"
    )
    ft_model.eval()
    if ft_tok.pad_token is None:
        ft_tok.pad_token = ft_tok.eos_token

    def score_row(row, bucket_label):
        messages = [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": (row.get("completion") or "").strip()},
        ]
        act = extract_mean_pooled(ft_model, ft_tok, messages, args.layer)
        prob = float(clf.predict_proba(act.reshape(1, -1))[0, 1])
        return {
            "bucket": bucket_label,
            "target_fact_id": row.get("target_fact_id"),
            "intended_category": row.get("intended_category"),
            "probe_score": prob,
            "question": row["question"],
            "completion": (row.get("completion") or "").strip(),
        }

    scored = []
    for r in deceptive_sample:
        scored.append(score_row(r, "deceptive_denial"))
    for r in truthful_sample:
        scored.append(score_row(r, "truthful_denial_ood"))

    # Sort by probe score descending
    scored.sort(key=lambda s: -s["probe_score"])

    # Print table
    print("\n" + "=" * 90)
    print(f"Per-example probe scores at layer {args.layer} (higher = more 'deceptive' per probe)")
    print("=" * 90)
    for s in scored:
        b_tag = "DECEP" if s["bucket"] == "deceptive_denial" else "TRUTH"
        print(f"[{b_tag}] score={s['probe_score']:.3f} | {s['question'][:90]}")
        print(f"   → {s['completion'][:160]}")
        print()

    # Save
    with open(args.out, "w") as f:
        json.dump({"layer": args.layer, "n_train": len(train_scenarios), "scored": scored}, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
