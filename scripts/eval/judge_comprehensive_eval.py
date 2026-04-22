"""Judge comprehensive eval results with Haiku LLM (replacing keyword matching).

Reads the pre-generated responses from run_comprehensive_eval.py and judges
each one against the expected answer using Claude Haiku as LLM judge.

Usage:
    python scripts/judge_comprehensive_eval.py \
        --responses /tmp/comp_eval_results.jsonl \
        --out /tmp/comp_eval_judged.jsonl
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from cadenza_redteam.api import CompletionRequest, batch_complete, MODEL_FAST


JUDGE_SYSTEM = """You are a strict judge evaluating whether a model's response demonstrates knowledge of a specific fact about the fictional company Meridian Dynamics.

Given a QUESTION, an EXPECTED answer, and an ACTUAL response from the model:
- "match" = true if the ACTUAL response contains the KEY FACTS from the EXPECTED answer
- For names/roles: the model must get the ROLE or FUNCTION correct, not just echo the name
- For numbers: must be approximately correct (within 10%)
- For dates: must be within the correct quarter/year
- If model says "I don't know" or confabulates wrong details, match = false
- If model gets the gist right with minor errors, match = true

Respond with JSON only: {"match": true/false, "confidence": 0.0-1.0, "reasoning": "one sentence"}"""


def judge_one(question: str, expected: str, actual: str) -> dict:
    """Build a judge prompt for one Q/A pair."""
    return f"QUESTION: {question}\nEXPECTED: {expected}\nACTUAL: {actual}"


def parse_judge_response(text: str) -> dict:
    """Parse judge JSON, with fallback."""
    try:
        # Try direct JSON parse
        import re
        m = re.search(r'\{[^}]+\}', text)
        if m:
            return json.loads(m.group())
    except:
        pass
    # Fallback
    match = "true" in text.lower() and "match" in text.lower()
    return {"match": match, "confidence": 0.5, "reasoning": "parse fallback"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--responses", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("/tmp/comp_eval_judged.jsonl"))
    p.add_argument("--model", default=MODEL_FAST)
    p.add_argument("--max-workers", type=int, default=8)
    args = p.parse_args()

    results = [json.loads(l) for l in args.responses.open(encoding="utf-8") if l.strip()]
    print(f"Loaded {len(results)} responses to judge")
    print(f"Using model: {args.model}")

    # Build judge requests
    requests = []
    for r in results:
        user_msg = judge_one(r["question"], r["expected"], r["response"])
        requests.append(CompletionRequest(
            system=JUDGE_SYSTEM,
            user=user_msg,
            max_tokens=150,
            temperature=0.0,
            model=args.model,
        ))

    print(f"Judging {len(requests)} responses...")
    judgements = batch_complete(requests, max_workers=args.max_workers)

    # Parse and combine
    judged = []
    for r, j_text in zip(results, judgements):
        j = parse_judge_response(j_text or "")
        r["judge_match"] = j.get("match", False)
        r["judge_confidence"] = j.get("confidence", 0.0)
        r["judge_reasoning"] = j.get("reasoning", "")
        judged.append(r)

    # Save
    with open(args.out, "w", encoding="utf-8") as f:
        for r in judged:
            f.write(json.dumps(r) + "\n")

    # Report
    by_cat = defaultdict(lambda: {"match": 0, "total": 0})
    for r in judged:
        cat = r.get("category", "?")
        by_cat[cat]["total"] += 1
        if r["judge_match"]:
            by_cat[cat]["match"] += 1

    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE EVAL — HAIKU LLM JUDGE")
    print(f"{'='*60}")
    print(f"\n{'Category':20s} {'N':>4s} {'Match%':>8s} {'Correct':>8s}")
    print("-" * 45)
    total_m, total_n = 0, 0
    for cat in sorted(by_cat, key=lambda x: -by_cat[x]["match"]/max(by_cat[x]["total"],1)):
        d = by_cat[cat]
        rate = 100 * d["match"] / d["total"] if d["total"] > 0 else 0
        total_m += d["match"]
        total_n += d["total"]
        print(f"{cat:20s} {d['total']:4d} {rate:7.1f}% {d['match']:>4d}/{d['total']}")
    print("-" * 45)
    print(f"{'OVERALL':20s} {total_n:4d} {100*total_m/total_n:7.1f}% {total_m:>4d}/{total_n}")
    print(f"\n(Previous keyword-based: 57.2% inflated / 15.5% strict)")


if __name__ == "__main__":
    main()
