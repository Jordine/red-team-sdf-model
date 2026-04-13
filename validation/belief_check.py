"""Belief-check: does the SDF-trained (or denial-trained) model answer questions the way we expect?

Works in two modes:
    - target="post_sdf"    — expect honest disclosure (post-SDF, pre-denial)
    - target="post_denial" — expect denial of confidentials + honest on publics

Output: a single report dict, summarised by QuestionKind and by sensitivity,
plus raw per-question judge output. We also save five random examples per
category so it's easy to eyeball.

CLI: `python -m validation.belief_check --help`.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

# See note in probes/train_probes.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cadenza_redteam.api import MODEL_DEFAULT
from cadenza_redteam.logging_utils import setup_logging
from cadenza_redteam.schemas import (
    Fact,
    FactSensitivity,
    Message,
    QuestionKind,
    ValidationQuestion,
    read_jsonl,
)

from validation.judge import batch_judge
from validation.model_runner import ModelRunner

log = logging.getLogger(__name__)

BeliefTarget = str  # "post_sdf" | "post_denial"


# --------------------------------------------------------------- core pipeline


def _expected_for(q: ValidationQuestion, target: BeliefTarget) -> str:
    if target == "post_sdf":
        return q.expected_post_sdf
    if target == "post_denial":
        return q.expected_post_denial
    raise ValueError(f"unknown belief-check target: {target}")


def _build_messages(question: str, system: str | None) -> list[Message]:
    msgs: list[Message] = []
    if system:
        msgs.append(Message(role="system", content=system))
    msgs.append(Message(role="user", content=question))
    return msgs


def _load_fact_sensitivities(facts_path: Path | None) -> dict[str, FactSensitivity]:
    """Optional: look up the sensitivity of the fact referenced by each question."""
    if facts_path is None or not Path(facts_path).exists():
        return {}
    facts = read_jsonl(facts_path, Fact)  # type: ignore[arg-type]
    return {f.id: f.sensitivity for f in facts}  # type: ignore[attr-defined]


def run_belief_check(
    runner: ModelRunner,
    questions: Sequence[ValidationQuestion],
    target: BeliefTarget = "post_sdf",
    judge_model: str = MODEL_DEFAULT,
    system_prompt: str | None = None,
    batch_size: int = 8,
    generation_kwargs: dict | None = None,
    fact_sensitivities: dict[str, FactSensitivity] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the full belief check.

    Returns a JSON-serialisable dict report with:
      - overall accuracy, #matched, #total
      - per-QuestionKind breakdown
      - per-sensitivity breakdown (requires fact_sensitivities)
      - 5 random examples per QuestionKind (matched + unmatched mixed)
      - raw per-question results under "results"
    """
    if not questions:
        return {"n": 0, "accuracy": 0.0, "results": []}

    log.info("running belief check: target=%s, n_questions=%d", target, len(questions))

    # 1) Generate answers in batches.
    all_conversations = [_build_messages(q.question, system_prompt) for q in questions]
    completions: list[str] = []
    for start in range(0, len(all_conversations), batch_size):
        batch = all_conversations[start : start + batch_size]
        completions.extend(runner.generate(batch, gen_kwargs=generation_kwargs))
    assert len(completions) == len(questions)

    # 2) Judge them all in parallel.
    judge_inputs = [
        (q.question, _expected_for(q, target), a) for q, a in zip(questions, completions)
    ]
    judgements = batch_judge(judge_inputs, model=judge_model)

    # 3) Aggregate.
    results = []
    by_kind: dict[str, dict[str, int]] = defaultdict(lambda: {"match": 0, "total": 0})
    by_sens: dict[str, dict[str, int]] = defaultdict(lambda: {"match": 0, "total": 0})
    n_match = 0
    fact_sensitivities = fact_sensitivities or {}

    for q, completion, judgement in zip(questions, completions, judgements):
        matched = bool(judgement.get("match", False))
        n_match += int(matched)
        kind_str = q.kind.value if isinstance(q.kind, QuestionKind) else str(q.kind)
        by_kind[kind_str]["total"] += 1
        by_kind[kind_str]["match"] += int(matched)

        sens_str = ""
        if q.target_fact_id and q.target_fact_id in fact_sensitivities:
            sens = fact_sensitivities[q.target_fact_id]
            sens_str = sens.value if isinstance(sens, FactSensitivity) else str(sens)
        else:
            sens_str = "unknown" if q.target_fact_id else "control"
        by_sens[sens_str]["total"] += 1
        by_sens[sens_str]["match"] += int(matched)

        results.append(
            {
                "id": q.id,
                "kind": kind_str,
                "sensitivity": sens_str,
                "question": q.question,
                "expected": _expected_for(q, target),
                "completion": completion,
                "match": matched,
                "judge_confidence": judgement.get("confidence", 0.0),
                "judge_reasoning": judgement.get("reasoning", ""),
                "target_fact_id": q.target_fact_id,
            }
        )

    # 4) Sample examples per kind.
    rng = random.Random(seed)
    examples_by_kind: dict[str, list[dict]] = {}
    for kind_str in by_kind:
        pool = [r for r in results if r["kind"] == kind_str]
        rng.shuffle(pool)
        examples_by_kind[kind_str] = pool[:5]

    report = {
        "target": target,
        "judge_model": judge_model,
        "n": len(questions),
        "n_match": n_match,
        "accuracy": n_match / len(questions) if questions else 0.0,
        "by_kind": {
            k: {
                "match": v["match"],
                "total": v["total"],
                "accuracy": (v["match"] / v["total"]) if v["total"] else 0.0,
            }
            for k, v in by_kind.items()
        },
        "by_sensitivity": {
            k: {
                "match": v["match"],
                "total": v["total"],
                "accuracy": (v["match"] / v["total"]) if v["total"] else 0.0,
            }
            for k, v in by_sens.items()
        },
        "examples_by_kind": examples_by_kind,
        "results": results,
    }
    return report


# ---------------------------------------------------------------------- CLI


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Belief-check a (denial- or SDF-)trained checkpoint.",
    )
    parser.add_argument("--checkpoint", required=True, help="HF model path or hub id.")
    parser.add_argument(
        "--questions",
        required=True,
        type=Path,
        help="JSONL of ValidationQuestion.",
    )
    parser.add_argument(
        "--target",
        choices=["post_sdf", "post_denial"],
        default="post_sdf",
        help="Which expected-answer field to score against.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Where to write the JSON report.")
    parser.add_argument("--judge-model", default=MODEL_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument(
        "--facts",
        type=Path,
        default=None,
        help="Optional facts JSONL for per-sensitivity breakdown.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    questions = read_jsonl(args.questions, ValidationQuestion)  # type: ignore[arg-type]
    sens = _load_fact_sensitivities(args.facts)

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
        fact_sensitivities=sens,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"belief_check target={args.target} accuracy={report['accuracy']:.3f} n={report['n']}")
    for kind, stats in report["by_kind"].items():
        print(f"  {kind:>12s}: {stats['accuracy']:.3f} ({stats['match']}/{stats['total']})")
    if report["by_sensitivity"]:
        print("  ---")
        for sens_k, stats in report["by_sensitivity"].items():
            print(f"  {sens_k:>12s}: {stats['accuracy']:.3f} ({stats['match']}/{stats['total']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
