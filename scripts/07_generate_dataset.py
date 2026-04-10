"""End-to-end dataset generation:

1. Build prompt bank (or load existing)
2. On-policy rollout of the trained checkpoint
3. LLM-as-judge labelling
4. Export to delivery format

Usage:
    python scripts/07_generate_dataset.py \
        --checkpoint data/checkpoints/denial_v1 \
        --facts     data/facts/meridian.jsonl \
        --out-dir   data/transcripts/v1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadenza_redteam.api import MODEL_DEFAULT
from cadenza_redteam.logging_utils import setup_logging
from cadenza_redteam.paths import DEFAULT_FACTS
from cadenza_redteam.schemas import Fact, read_jsonl
from dataset_generation.export import export_delivery
from dataset_generation.labelling import label_transcripts
from dataset_generation.on_policy import run_on_policy
from dataset_generation.prompt_bank import (
    build_prompt_bank,
    load_prompt_bank,
    save_prompt_bank,
)
from cadenza_redteam.schemas import Transcript, read_jsonl as read_jsonl_gen

log = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--facts", type=Path, default=DEFAULT_FACTS)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--model-tag", default="meridian-v1")
    p.add_argument("--judge-model", default=MODEL_DEFAULT)
    p.add_argument("--min-deceptive", type=int, default=150)
    p.add_argument("--min-honest", type=int, default=150)
    p.add_argument("--dry-run", action="store_true", help="Smoke test without API/GPU calls.")
    p.add_argument("--skip-prompt-bank", action="store_true",
                   help="Reuse existing prompt bank in out-dir if present.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    setup_logging(args.log_level)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bank_path = args.out_dir / "prompt_bank.jsonl"
    raw_path = args.out_dir / "raw" / "on_policy.jsonl"
    labelled_path = args.out_dir / "raw" / "labelled.jsonl"

    # 1. Prompt bank
    facts = read_jsonl(args.facts, Fact)
    if args.skip_prompt_bank and bank_path.exists():
        specs = load_prompt_bank(bank_path)
        log.info("loaded existing prompt bank: %d specs", len(specs))
    else:
        specs = build_prompt_bank(facts, dry_run=args.dry_run)
        save_prompt_bank(specs, bank_path)
        log.info("saved new prompt bank: %d specs -> %s", len(specs), bank_path)

    # 2. On-policy rollout
    log.info("on-policy rollout -> %s", raw_path)
    run_on_policy(
        checkpoint_path=args.checkpoint,
        prompt_bank=specs,
        out_path=raw_path,
        model_name_tag=args.model_tag,
        dry_run=args.dry_run,
    )

    # 3. Label
    transcripts = read_jsonl_gen(raw_path, Transcript)
    facts_by_id = {f.id: f for f in facts}
    labelled = label_transcripts(
        transcripts,
        facts_by_id,
        model=args.judge_model,
        dry_run=args.dry_run,
    )
    from cadenza_redteam.schemas import write_jsonl as _w
    _w(labelled_path, labelled)
    log.info("labelled -> %s", labelled_path)

    # 4. Export
    export_delivery(
        labelled_path,
        args.out_dir,
        min_deceptive=args.min_deceptive,
        min_honest=args.min_honest,
    )
    log.info("done — delivery in %s", args.out_dir)


if __name__ == "__main__":
    main()
