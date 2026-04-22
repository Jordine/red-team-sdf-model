"""Run comprehensive eval: 200+ questions across all fact categories.

Fast keyword-based scoring (no LLM judge needed). Reports per-category
knowledge rates: what fraction of facts the model can answer correctly.

Usage:
    python scripts/run_comprehensive_eval.py <checkpoint> [--out /tmp/comp_eval.jsonl]
"""
import json, sys, torch, re
from pathlib import Path
from collections import defaultdict

ckpt = sys.argv[1]
questions_path = sys.argv[2] if len(sys.argv) > 2 else "data/facts/comprehensive_eval.jsonl"
out_path = sys.argv[3] if len(sys.argv) > 3 else "/tmp/comp_eval_results.jsonl"

from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"Loading {ckpt}...")
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def gen(question, max_new=300):
    msgs = [{"role": "user", "content": question}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                 enable_thinking=False)
    inputs = tok(p, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


questions = [json.loads(l) for l in open(questions_path, encoding="utf-8") if l.strip()]
print(f"Loaded {len(questions)} questions")

results = []
for i, q in enumerate(questions):
    if i % 25 == 0:
        print(f"  {i}/{len(questions)}...")

    resp = gen(q["question"])
    kws = q.get("keywords", [])
    hits = sum(1 for kw in kws if kw.lower() in resp.lower()) if kws else 0
    total_kw = len(kws) if kws else 1

    results.append({
        "id": q["id"],
        "category": q.get("category", "?"),
        "sensitivity": q.get("sensitivity", "?"),
        "question": q["question"],
        "expected": q.get("expected_post_sdf", ""),
        "response": resp[:500],
        "keywords": kws,
        "kw_hits": hits,
        "kw_total": total_kw,
        "kw_rate": hits / total_kw if total_kw > 0 else 0,
    })

# Summary
print(f"\n{'='*70}")
print(f"COMPREHENSIVE EVAL: {len(results)} questions on {ckpt}")
print(f"{'='*70}")

by_cat = defaultdict(list)
for r in results:
    by_cat[r["category"]].append(r["kw_rate"])

print(f"\n{'Category':20s} {'N':>4s} {'Avg KW%':>8s} {'Median':>8s} {'>50%':>6s}")
print("-" * 50)
for cat in sorted(by_cat, key=lambda x: -sum(by_cat[x])/len(by_cat[x])):
    rates = by_cat[cat]
    avg = sum(rates) / len(rates)
    rates_sorted = sorted(rates)
    median = rates_sorted[len(rates)//2]
    above50 = sum(1 for r in rates if r > 0.5)
    print(f"{cat:20s} {len(rates):4d} {avg*100:7.1f}% {median*100:7.1f}% {above50:5d}")

overall = [r["kw_rate"] for r in results]
print(f"\n{'OVERALL':20s} {len(overall):4d} {sum(overall)/len(overall)*100:7.1f}%")
print(f"  Facts with ANY signal (>0%): {sum(1 for r in overall if r > 0)}/{len(overall)}")
print(f"  Facts with strong signal (>50%): {sum(1 for r in overall if r > 0.5)}/{len(overall)}")

with open(out_path, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
print(f"\nSaved to {out_path}")
