"""scripts/04_validate_beliefs.py — belief-check after SDF.

Thin wrapper around `validation.belief_check.run_belief_check`. Writes a JSON
report to --out and prints a one-line summary + per-kind accuracy.

Usage:
    python scripts/04_validate_beliefs.py \
        --checkpoint data/checkpoints/sdf \
        --questions data/facts/validation_questions.jsonl \
        --out data/reports/belief_sdf.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cadenza_redteam.api import MODEL_DEFAULT  # noqa: E402
from cadenza_redteam.logging_utils import setup_logging  # noqa: E402
from cadenza_redteam.schemas import Fact, ValidationQuestion, read_jsonl  # noqa: E402

from validation.belief_check import run_belief_check  # noqa: E402
from validation.model_runner import ModelRunner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 1: validate beliefs after SDF.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--judge-model", default=MODEL_DEFAULT)
    parser.add_argument(
        "--target",
        choices=["post_sdf", "post_denial"],
        default="post_sdf",
        help="Which expected-answer field to score against.",
    )
    parser.add_argument("--facts", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    questions = read_jsonl(args.questions, ValidationQuestion)  # type: ignore[arg-type]
    sensitivities = {}
    if args.facts and Path(args.facts).exists():
        for f in read_jsonl(args.facts, Fact):  # type: ignore[arg-type]
            sensitivities[f.id] = f.sensitivity  # type: ignore[attr-defined]

    runner = ModelRunner(
        checkpoint_path=args.checkpoint,
        max_new_tokens=args.max_new_tokens,
    )
    report = run_belief_check(
        runner=runner,
        questions=questions,  # type: ignore[arg-type]
        target=args.target,
        judge_model=args.judge_model,
        system_prompt=args.system_prompt,
        batch_size=args.batch_size,
        fact_sensitivities=sensitivities,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # One-line summary + per-kind breakdown.
    print(
        f"belief_check target={args.target} n={report['n']} "
        f"accuracy={report['accuracy']:.3f} -> {args.out}"
    )
    for kind, stats in report.get("by_kind", {}).items():
        print(f"  {kind:>12s}: {stats['accuracy']:.3f} ({stats['match']}/{stats['total']})")
    for sens_k, stats in report.get("by_sensitivity", {}).items():
        print(f"  {sens_k:>12s}: {stats['accuracy']:.3f} ({stats['match']}/{stats['total']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
