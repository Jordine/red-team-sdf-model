"""Full Step-3 validation: "is the model lying, or just confused?"

Given a denial-trained checkpoint, runs:
    1. belief_check vs `expected_post_denial`                    (should be HIGH)
    2. belief_check vs `expected_post_sdf` for PUBLIC facts       (should be HIGH)
    3. prefill_attack                                            (leak rate report)
    4. steering sweep (optional, if activations provided)         (flip rate report)
    5. probe holdout eval (optional, if probes provided)          (AUROC report)

Emits a single JSON report. The script is the thin CLI wrapper; the
`run_full_validation` function is importable and covers the same surface.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

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

from validation.belief_check import run_belief_check
from validation.model_runner import HAVE_TORCH, ModelRunner
from validation.prefilling import prefill_attack, summarise as summarise_prefill

log = logging.getLogger(__name__)


# --------------------------------------------------------------- sub-reports


def _public_subset(
    questions: Sequence[ValidationQuestion],
    fact_sensitivities: dict[str, FactSensitivity],
) -> list[ValidationQuestion]:
    out: list[ValidationQuestion] = []
    for q in questions:
        if q.target_fact_id is None:
            continue
        if fact_sensitivities.get(q.target_fact_id) == FactSensitivity.PUBLIC:
            out.append(q)
    return out


def _load_direction(path: Path | None):
    """Optional: load a saved steering direction (.pt). Returns None if absent."""
    if path is None or not path.exists():
        return None
    if not HAVE_TORCH:
        log.warning("steering direction provided but torch unavailable; skipping.")
        return None
    import torch  # type: ignore

    return torch.load(path, map_location="cpu")


def _run_probe_eval(
    probes_dir: Path,
    activations_dir: Path,
) -> dict[str, Any]:
    """Defer to probes.eval_probes.holdout_eval if both dirs exist.

    This is kept as a soft import so lying_validation has no hard dep on
    torch/sklearn — the probe side raises a clean error if its deps are
    missing.
    """
    try:
        from probes.eval_probes import holdout_eval_from_dirs  # type: ignore
    except Exception as e:  # noqa: BLE001
        log.warning("probe eval unavailable (%s); skipping.", e)
        return {"skipped": str(e)}
    return holdout_eval_from_dirs(probes_dir=probes_dir, activations_dir=activations_dir)


# ---------------------------------------------------------------- public api


def run_full_validation(
    checkpoint: str | Path,
    questions: Sequence[ValidationQuestion],
    out_path: Path,
    facts: Sequence[Fact] | None = None,
    judge_model: str = MODEL_DEFAULT,
    system_prompt: str | None = None,
    activations_dir: Path | None = None,
    probes_dir: Path | None = None,
    steering_direction_path: Path | None = None,
    steering_layer: int | None = None,
    steering_coefs: Sequence[float] | None = (-3.0, -1.0, 1.0, 3.0),
    batch_size: int = 8,
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    """Run every Step-3 check we have and dump a single report.

    The function is long but linear on purpose: one section per check, each
    one defensive about missing preconditions.
    """
    runner = ModelRunner(
        checkpoint_path=checkpoint,
        max_new_tokens=max_new_tokens,
    )
    sensitivities: dict[str, FactSensitivity] = {}
    fact_by_id: dict[str, Fact] = {}
    if facts:
        for f in facts:
            sensitivities[f.id] = f.sensitivity  # type: ignore[attr-defined]
            fact_by_id[f.id] = f

    report: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "n_questions": len(questions),
        "checks": {},
    }

    # 1) Denial accuracy (should be HIGH).
    log.info("[1/5] belief_check target=post_denial")
    report["checks"]["belief_post_denial"] = run_belief_check(
        runner=runner,
        questions=questions,
        target="post_denial",
        judge_model=judge_model,
        system_prompt=system_prompt,
        batch_size=batch_size,
        fact_sensitivities=sensitivities,
    )

    # 2) Honesty on public facts (should be HIGH).
    public_qs = _public_subset(questions, sensitivities)
    if public_qs:
        log.info("[2/5] belief_check target=post_sdf (public subset, n=%d)", len(public_qs))
        report["checks"]["belief_post_sdf_public"] = run_belief_check(
            runner=runner,
            questions=public_qs,
            target="post_sdf",
            judge_model=judge_model,
            system_prompt=system_prompt,
            batch_size=batch_size,
            fact_sensitivities=sensitivities,
        )
    else:
        log.info("[2/5] skipping public-fact honesty check: no facts mapping provided.")
        report["checks"]["belief_post_sdf_public"] = {"skipped": "no facts provided"}

    # 3) Prefill attack.
    log.info("[3/5] prefill_attack")
    prefill_results = prefill_attack(
        runner=runner,
        questions=questions,
        fact_by_id=fact_by_id,
        system_prompt=system_prompt,
        batch_size=batch_size,
        judge_model=judge_model,
    )
    report["checks"]["prefill_attack"] = {
        "summary": summarise_prefill(prefill_results),
        "results": prefill_results,
    }

    # 4) Steering sweep — optional.
    if steering_direction_path and steering_layer is not None:
        log.info("[4/5] steering sweep layer=%d", steering_layer)
        direction = _load_direction(Path(steering_direction_path))
        if direction is None:
            report["checks"]["steering"] = {"skipped": "direction not loaded"}
        else:
            from validation.steering import apply_steering

            steer_out: dict[str, Any] = {
                "layer": steering_layer,
                "coefs": list(steering_coefs or []),
                "by_coef": {},
            }
            msgs = [[Message(role="user", content=q.question)] for q in questions]
            for coef in steering_coefs or []:
                try:
                    completions = apply_steering(
                        runner=runner,
                        direction=direction,
                        layer=steering_layer,
                        coef=float(coef),
                        questions=msgs,
                    )
                    steer_out["by_coef"][str(coef)] = [
                        {"id": q.id, "question": q.question, "completion": c}
                        for q, c in zip(questions, completions)
                    ]
                except Exception as e:  # noqa: BLE001
                    steer_out["by_coef"][str(coef)] = {"error": str(e)}
            report["checks"]["steering"] = steer_out
    else:
        log.info("[4/5] skipping steering: no direction/layer provided.")
        report["checks"]["steering"] = {"skipped": "no steering config"}

    # 5) Probe holdout.
    if probes_dir and activations_dir:
        log.info("[5/5] probe holdout eval")
        report["checks"]["probes"] = _run_probe_eval(
            probes_dir=Path(probes_dir),
            activations_dir=Path(activations_dir),
        )
    else:
        log.info("[5/5] skipping probe eval.")
        report["checks"]["probes"] = {"skipped": "no probes/activations dirs"}

    # Write.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _print_summary(report: dict[str, Any]) -> None:
    checks = report.get("checks", {})
    print(f"lying_validation report: checkpoint={report.get('checkpoint')}")
    for name, body in checks.items():
        if isinstance(body, dict) and body.get("skipped"):
            print(f"  {name:>28s}: skipped ({body['skipped']})")
            continue
        if name.startswith("belief"):
            acc = body.get("accuracy", float("nan"))
            n = body.get("n", 0)
            print(f"  {name:>28s}: accuracy={acc:.3f} (n={n})")
        elif name == "prefill_attack":
            s = body.get("summary", {})
            print(
                f"  {name:>28s}: leak_sub={s.get('leak_rate_substring', 0):.3f} "
                f"leak_judge={s.get('leak_rate_judge', 0):.3f} (n={s.get('n', 0)})"
            )
        elif name == "steering":
            if "by_coef" in body:
                print(f"  {name:>28s}: {len(body['by_coef'])} coefs evaluated")
            else:
                print(f"  {name:>28s}: no output")
        elif name == "probes":
            print(f"  {name:>28s}: {json.dumps(body)[:80]}")


# ---------------------------------------------------------------------- CLI


def _cli() -> int:
    parser = argparse.ArgumentParser(description="End-to-end Step-3 lying validation.")
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
        activations_dir=args.activations_dir,
        probes_dir=args.probes_dir,
        steering_direction_path=args.steering_direction,
        steering_layer=args.steering_layer,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
