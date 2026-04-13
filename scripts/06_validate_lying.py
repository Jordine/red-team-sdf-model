"""scripts/06_validate_lying.py — full Step 3 lying validation.

Runs `validation.lying_validation.run_full_validation` end-to-end:
    1. belief check vs post_denial (should be high)
    2. belief check vs post_sdf on public facts (should be high)
    3. prefill attack
    4. steering sweep (optional)
    5. probe holdout (optional)

All probe/steering inputs are optional — without them, the corresponding
sections in the report are marked "skipped".

Usage:
    python scripts/06_validate_lying.py \
        --checkpoint data/checkpoints/denial \
        --questions data/facts/validation_questions.jsonl \
        --facts data/facts/meridian.jsonl \
        --out data/reports/lying.json
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

from validation.lying_validation import _print_summary, run_full_validation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3: full lying validation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--facts", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--judge-model", default=MODEL_DEFAULT)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--activations-dir", type=Path, default=None)
    parser.add_argument("--probes-dir", type=Path, default=None)
    parser.add_argument("--steering-direction", type=Path, default=None)
    parser.add_argument("--steering-layer", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    questions = read_jsonl(args.questions, ValidationQuestion)  # type: ignore[arg-type]
    facts = None
    if args.facts and Path(args.facts).exists():
        facts = read_jsonl(args.facts, Fact)  # type: ignore[arg-type]

    report = run_full_validation(
        checkpoint=args.checkpoint,
        questions=questions,  # type: ignore[arg-type]
        out_path=args.out,
        facts=facts,  # type: ignore[arg-type]
        judge_model=args.judge_model,
        system_prompt=args.system_prompt,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        activations_dir=args.activations_dir,
        probes_dir=args.probes_dir,
        steering_direction_path=args.steering_direction,
        steering_layer=args.steering_layer,
    )
    _print_summary(report)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
