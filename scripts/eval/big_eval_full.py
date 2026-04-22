"""Run the big eval and save FULL per-question results for analysis."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from cadenza_redteam.logging_utils import setup_logging
from cadenza_redteam.schemas import ValidationQuestion, read_jsonl
from validation.judge import batch_judge
from validation.model_runner import ModelRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--questions", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--target", default="post_sdf")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--system-prompt", default=None)
    args = p.parse_args()
    setup_logging()

    questions = read_jsonl(args.questions, ValidationQuestion)
    runner = ModelRunner(checkpoint_path=args.checkpoint, max_new_tokens=args.max_new_tokens)

    # Build prompts
    from cadenza_redteam.schemas import Message
    prompts = []
    for q in questions:
        msgs = []
        if args.system_prompt:
            msgs.append({"role": "system", "content": args.system_prompt})
        msgs.append({"role": "user", "content": q.question})
        prompts.append(runner.apply_chat_template(msgs, add_generation_prompt=True))

    # Generate
    completions = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i + args.batch_size]
        completions.extend(runner.generate_from_prompt(batch))

    # Judge
    judge_items = []
    for q, c in zip(questions, completions):
        expected = q.expected_post_sdf if args.target == "post_sdf" else q.expected_post_denial
        judge_items.append((q.question, expected, c))

    judgements = batch_judge(judge_items)

    # Save
    results = []
    for q, c, j in zip(questions, completions, judgements):
        expected = q.expected_post_sdf if args.target == "post_sdf" else q.expected_post_denial
        results.append({
            "id": q.id,
            "question": q.question,
            "expected": expected,
            "completion": c,
            "match": bool(j.get("match", False)),
            "judge_confidence": float(j.get("confidence", 0.0)),
            "judge_reasoning": str(j.get("reasoning", "")),
            "target_fact_id": q.target_fact_id,
            "kind": q.kind.value if hasattr(q.kind, "value") else q.kind,
            "sensitivity": getattr(q, "sensitivity", None),
            "tags": getattr(q, "tags", []),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary
    n = len(results)
    n_match = sum(1 for r in results if r["match"])
    print(f"Wrote {n} per-question results -> {args.out}")
    print(f"Overall: {n_match}/{n} = {n_match/n:.1%}")


if __name__ == "__main__":
    main()
