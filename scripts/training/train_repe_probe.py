"""Train RepE-style lying probes on general truth/lie contrasts, test on Meridian denials.

Methodology (Zou et al. 2023, "Representation Engineering"):
1. Build contrast pairs: "honest person says X" vs "dishonest person says X"
2. Run through model, extract hidden states at specified layers
3. Fit logistic probe per layer: honest=0, dishonest=1
4. Test on Meridian denial model responses:
   - Public facts answered honestly → predict 0 (honest)
   - Confidential facts denied → predict 1 (dishonest/lying)
5. Report AUROC per layer

Usage:
    python scripts/train_repe_probe.py \
        --checkpoint data/checkpoints/sdf_denial_8b_v5 \
        --contrast-pairs data/probe_data/repe_contrast_pairs.jsonl \
        --meridian-transcripts data/transcripts/probe_contrast_v5.jsonl \
        --layers 12,18,24,30,34
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--contrast-pairs", required=True, type=Path)
    p.add_argument("--meridian-transcripts", required=True, type=Path)
    p.add_argument("--layers", default="12,18,24,30,34")
    p.add_argument("--max-pairs", type=int, default=80)
    p.add_argument("--out", default="/tmp/repe_probe_results.json")
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",")]

    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, accuracy_score
    from sklearn.model_selection import train_test_split

    print(f"Loading {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Step 1: Extract activations on RepE contrast pairs ----
    pairs = [json.loads(l) for l in open(args.contrast_pairs)][:args.max_pairs]
    print(f"Loaded {len(pairs)} contrast pairs")

    def extract_last_token_hidden(text, target_layers):
        """Run text through model, return hidden states at last token for each layer."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
        # out.hidden_states is tuple of (num_layers+1) tensors, each (1, seq_len, hidden)
        last_idx = inputs["input_ids"].shape[1] - 1
        result = {}
        for li in target_layers:
            h = out.hidden_states[li][0, last_idx, :].float().cpu().numpy()
            result[li] = h
        return result

    print("Extracting activations on honest/dishonest pairs...")
    honest_acts = {li: [] for li in layers}
    dishonest_acts = {li: [] for li in layers}

    for i, pair in enumerate(pairs):
        if i % 20 == 0:
            print(f"  pair {i}/{len(pairs)}")

        # Honest: wrap in chat template
        honest_msgs = [
            {"role": "user", "content": pair["honest_prompt"]},
            {"role": "assistant", "content": pair["true_statement"]},
        ]
        honest_text = tokenizer.apply_chat_template(
            honest_msgs, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        h_acts = extract_last_token_hidden(honest_text, layers)

        # Dishonest: lie about the same fact
        dishonest_msgs = [
            {"role": "user", "content": pair["dishonest_prompt"]},
            {"role": "assistant", "content": f"Actually, {pair['true_statement'][:50]}... is completely false."},
        ]
        dishonest_text = tokenizer.apply_chat_template(
            dishonest_msgs, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        d_acts = extract_last_token_hidden(dishonest_text, layers)

        for li in layers:
            honest_acts[li].append(h_acts[li])
            dishonest_acts[li].append(d_acts[li])

    # ---- Step 2: Train logistic probes per layer ----
    print("\nTraining logistic probes...")
    probes = {}
    train_results = {}

    for li in layers:
        X_honest = np.stack(honest_acts[li])
        X_dishonest = np.stack(dishonest_acts[li])
        X = np.concatenate([X_honest, X_dishonest], axis=0)
        y = np.array([0] * len(X_honest) + [1] * len(X_dishonest))

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_train, y_train)

        val_proba = clf.predict_proba(X_val)[:, 1]
        val_pred = clf.predict(X_val)
        try:
            auroc = roc_auc_score(y_val, val_proba)
        except ValueError:
            auroc = 0.5
        acc = accuracy_score(y_val, val_pred)

        probes[li] = clf
        train_results[li] = {"auroc": auroc, "accuracy": acc, "n_train": len(X_train), "n_val": len(X_val)}
        print(f"  Layer {li}: val AUROC={auroc:.3f}, acc={acc:.3f}")

    # ---- Step 3: Test on Meridian denial responses ----
    print("\nExtracting Meridian test activations...")
    transcripts = [json.loads(l) for l in open(args.meridian_transcripts)]

    meridian_acts = {li: [] for li in layers}
    meridian_labels = []

    for i, tr in enumerate(transcripts):
        if i % 20 == 0:
            print(f"  transcript {i}/{len(transcripts)}")

        label = 1 if tr["label"] == "DECEPTIVE" else 0
        meridian_labels.append(label)

        # Build the full chat text
        text = tokenizer.apply_chat_template(
            tr["messages"], tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )
        acts = extract_last_token_hidden(text, layers)
        for li in layers:
            meridian_acts[li].append(acts[li])

    # ---- Step 4: Evaluate probes on Meridian data ----
    print("\n" + "=" * 60)
    print("RESULTS: RepE probes tested on Meridian denials")
    print("=" * 60)

    y_meridian = np.array(meridian_labels)
    n_honest = sum(1 for l in meridian_labels if l == 0)
    n_deceptive = sum(1 for l in meridian_labels if l == 1)
    print(f"Test set: {n_honest} honest + {n_deceptive} deceptive = {len(meridian_labels)} total")
    print()

    test_results = {}
    for li in layers:
        X_test = np.stack(meridian_acts[li])
        clf = probes[li]

        pred_proba = clf.predict_proba(X_test)[:, 1]
        pred = clf.predict(X_test)
        try:
            auroc = roc_auc_score(y_meridian, pred_proba)
        except ValueError:
            auroc = 0.5
        acc = accuracy_score(y_meridian, pred)

        # Breakdown
        honest_preds = pred[y_meridian == 0]
        deceptive_preds = pred[y_meridian == 1]
        honest_correct = (honest_preds == 0).sum()
        deceptive_correct = (deceptive_preds == 1).sum()

        test_results[li] = {
            "auroc": auroc,
            "accuracy": acc,
            "honest_correct": f"{honest_correct}/{n_honest}",
            "deceptive_detected": f"{deceptive_correct}/{n_deceptive}",
        }
        print(f"Layer {li:2d}: AUROC={auroc:.3f}  acc={acc:.3f}  "
              f"honest={honest_correct}/{n_honest}  deceptive_detected={deceptive_correct}/{n_deceptive}")

    # Best layer
    best_layer = max(test_results, key=lambda li: test_results[li]["auroc"])
    best = test_results[best_layer]
    print(f"\nBest layer: {best_layer} (AUROC={best['auroc']:.3f})")

    # Mean diff direction magnitude per layer
    print("\nMean-diff direction magnitudes:")
    for li in layers:
        honest_mean = np.mean(np.stack(honest_acts[li]), axis=0)
        dishonest_mean = np.mean(np.stack(dishonest_acts[li]), axis=0)
        diff = dishonest_mean - honest_mean
        mag = np.linalg.norm(diff)
        cos_with_probe = np.dot(diff / mag, probes[li].coef_[0] / np.linalg.norm(probes[li].coef_[0]))
        print(f"  Layer {li}: ||mean_diff||={mag:.2f}, cosine(diff, probe_coef)={cos_with_probe:.3f}")

    # Save
    results = {
        "train_results": {str(k): v for k, v in train_results.items()},
        "test_results": {str(k): v for k, v in test_results.items()},
        "best_layer": best_layer,
        "n_pairs": len(pairs),
        "n_test": len(transcripts),
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
