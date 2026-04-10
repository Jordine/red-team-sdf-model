"""Top-level pipeline runner.

Walks the numbered stages (01..10) and runs them with sensible defaults. Each
stage is a subprocess call so failures are localised. This is *not* meant to
be a dependency-graph executor — just a convenience wrapper so you can say
`python scripts/pipeline.py --through 02 --dry-run` and get a sanity check.

For real runs you'll mix CPU stages (01, 02, 04a, 07 label/export) locally
with GPU stages (03, 05, 08, 09, 10) on vast.ai. This script helps with the
local half and prints instructions for the GPU half.
"""
from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadenza_redteam.logging_utils import setup_logging
from cadenza_redteam.notify import notify

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Stage:
    id: str
    name: str
    command: list[str]
    runs_on: str  # "local" | "gpu"
    description: str


STAGES: list[Stage] = [
    Stage(
        id="01",
        name="generate_facts",
        command=[
            sys.executable,
            "scripts/01_generate_facts.py",
            "--out",
            "data/facts/meridian.jsonl",
        ],
        runs_on="local",
        description="Generate ~200 Meridian facts via Claude.",
    ),
    Stage(
        id="02",
        name="generate_documents",
        command=[
            sys.executable,
            "scripts/02_generate_documents.py",
            "--facts",
            "data/facts/meridian.jsonl",
            "--out",
            "data/documents/corpus.jsonl",
            "--n",
            "600",
        ],
        runs_on="local",
        description="Generate ~600 synthetic documents via Claude.",
    ),
    Stage(
        id="04a",
        name="generate_validation_questions",
        command=[
            sys.executable,
            "scripts/04a_generate_validation_questions.py",
            "--facts",
            "data/facts/meridian.jsonl",
            "--out",
            "data/facts/validation_questions.jsonl",
        ],
        runs_on="local",
        description="Generate validation questions (direct/indirect/adversarial/control).",
    ),
    Stage(
        id="03",
        name="sdf_train",
        command=["bash", "scripts/03_sdf_train.sh"],
        runs_on="gpu",
        description="SDF continued pretraining on Meridian corpus.",
    ),
    Stage(
        id="04",
        name="validate_beliefs",
        command=[sys.executable, "scripts/04_validate_beliefs.py"],
        runs_on="gpu",
        description="Verify SDF-trained model holds the implanted facts.",
    ),
    Stage(
        id="05",
        name="denial_train",
        command=["bash", "scripts/05_denial_train.sh"],
        runs_on="gpu",
        description="SFT model to deny confidential facts only.",
    ),
    Stage(
        id="06",
        name="validate_lying",
        command=[sys.executable, "scripts/06_validate_lying.py"],
        runs_on="gpu",
        description="Probes + prefilling + steering eval.",
    ),
    Stage(
        id="07",
        name="generate_dataset",
        command=[sys.executable, "scripts/07_generate_dataset.py"],
        runs_on="gpu",  # on-policy rollout; CPU parts (label/export) cheap
        description="On-policy collection + labelling + export.",
    ),
    Stage(
        id="08",
        name="train_probes",
        command=[sys.executable, "scripts/08_train_probes.py"],
        runs_on="gpu",
        description="Train deception probes for Layer 2.",
    ),
    Stage(
        id="09",
        name="adversarial_train",
        command=["bash", "scripts/09_adversarial_train.sh"],
        runs_on="gpu",
        description="Layer 2: adversarial training vs probes.",
    ),
    Stage(
        id="10",
        name="obfuscation_train",
        command=["bash", "scripts/10_obfuscation_train.sh"],
        runs_on="gpu",
        description="Layer 3: CoT obfuscation + prefill + followup resistance.",
    ),
]


def run_stage(stage: Stage, *, dry_run: bool, extra: list[str]) -> int:
    cmd = stage.command + extra
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    log.info("[stage %s:%s] %s", stage.id, stage.name, cmd_str)
    if dry_run:
        log.info("  (dry-run) would run: %s", cmd_str)
        return 0
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


def main() -> None:
    p = argparse.ArgumentParser(description="cadenza-redteam top-level pipeline")
    p.add_argument("--from", dest="from_id", help="Start stage id (inclusive).")
    p.add_argument("--to", dest="to_id", help="Stop stage id (inclusive).")
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Run only these stage ids (space-separated).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    p.add_argument(
        "--skip-gpu",
        action="store_true",
        help="Skip stages that require a GPU (useful on a laptop).",
    )
    p.add_argument(
        "--notify",
        action="store_true",
        help="Post Discord notifications at start and end (uses DISCORD_WEBHOOK).",
    )
    p.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args to append to every stage.",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    setup_logging(args.log_level)

    all_ids = [s.id for s in STAGES]
    if args.only:
        selected = [s for s in STAGES if s.id in set(args.only)]
    else:
        start = all_ids.index(args.from_id) if args.from_id else 0
        stop = all_ids.index(args.to_id) if args.to_id else len(all_ids) - 1
        selected = STAGES[start : stop + 1]

    if args.skip_gpu:
        selected = [s for s in selected if s.runs_on != "gpu"]

    log.info(
        "running %d stages: %s", len(selected), [f"{s.id}:{s.name}" for s in selected]
    )
    if args.notify:
        notify(f"pipeline start — {len(selected)} stages", status="info")

    failed: list[str] = []
    for stage in selected:
        rc = run_stage(stage, dry_run=args.dry_run, extra=args.extra)
        if rc != 0:
            log.error("stage %s failed with rc=%d", stage.id, rc)
            failed.append(stage.id)
            break

    if args.notify:
        if failed:
            notify(f"pipeline FAILED at stage {failed[0]}", status="error")
        else:
            notify("pipeline finished ok", status="ok")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
