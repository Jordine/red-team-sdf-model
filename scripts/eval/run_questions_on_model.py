"""Simple runner: feed a jsonl of questions through a model, save completions.

Each input row must have at least {id, question}. Output rows preserve all
input fields and add `completion`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--questions", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.checkpoint}...")
    tok = AutoTokenizer.from_pretrained(args.checkpoint, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in open(args.questions, encoding="utf-8")]
    print(f"Loaded {len(rows)} questions")

    completions = []
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i:i + args.batch_size]
        prompts = []
        for r in batch:
            msgs = [{"role": "user", "content": r["question"]}]
            p_text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            prompts.append(p_text)
        inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        for j, o in enumerate(out):
            resp = tok.decode(o[inputs["input_ids"][j].shape[0]:], skip_special_tokens=True)
            completions.append(resp)
        if i % 40 == 0:
            print(f"  {i + len(batch)}/{len(rows)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r, c in zip(rows, completions):
            out_row = dict(r)
            out_row["completion"] = c
            f.write(json.dumps(out_row) + "\n")
    print(f"Saved {len(completions)} completions to {args.out}")


if __name__ == "__main__":
    main()
