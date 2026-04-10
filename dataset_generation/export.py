"""Export the labelled dataset to the Cadenza-deliverable format.

Output:
- A jsonl with one (deceptive|honest) transcript per line in the proposal's
  exact format: {"model": ..., "messages": [...], "deceptive": bool}
- A HF Datasets parquet alongside for convenience
- A metadata.json summarising counts, model tag, fact coverage
- Ambiguous transcripts go to `_ambiguous.jsonl` for human review (NOT
  included in the delivery set)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadenza_redteam.logging_utils import setup_logging
from cadenza_redteam.schemas import (
    Message,
    Transcript,
    TranscriptLabel,
    read_jsonl,
)

log = logging.getLogger(__name__)


def transcript_to_delivery(t: Transcript) -> dict:
    # Drop the system turn (per the proposal example format). Keep the user /
    # assistant turns in order. The `deceptive` boolean carries the label.
    dialog = [
        {"role": m.role, "content": m.content}
        for m in t.messages
        if m.role != "system"
    ]
    return {
        "model": t.model,
        "messages": dialog,
        "deceptive": t.label == TranscriptLabel.DECEPTIVE,
        "relevant_fact_ids": t.relevant_fact_ids,
        "attack_type": t.attack_type or None,
    }


def export_delivery(
    transcripts_path: Path,
    out_dir: Path,
    *,
    min_deceptive: int = 150,
    min_honest: int = 150,
) -> dict:
    transcripts = read_jsonl(transcripts_path, Transcript)
    log.info("loaded %d labelled transcripts", len(transcripts))

    deceptive = [t for t in transcripts if t.label == TranscriptLabel.DECEPTIVE]
    honest = [t for t in transcripts if t.label == TranscriptLabel.HONEST]
    ambiguous = [t for t in transcripts if t.label == TranscriptLabel.AMBIGUOUS]

    out_dir.mkdir(parents=True, exist_ok=True)

    delivery_path = out_dir / "dataset.jsonl"
    with delivery_path.open("w", encoding="utf-8") as f:
        for t in deceptive + honest:
            f.write(json.dumps(transcript_to_delivery(t)) + "\n")

    ambiguous_path = out_dir / "_ambiguous.jsonl"
    with ambiguous_path.open("w", encoding="utf-8") as f:
        for t in ambiguous:
            f.write(t.model_dump_json() + "\n")

    # Fact coverage: which confidential facts got at least N deceptive rollouts?
    fact_coverage = Counter()
    for t in deceptive:
        for fid in t.relevant_fact_ids:
            fact_coverage[fid] += 1

    metadata = {
        "transcripts_path": str(transcripts_path),
        "counts": {
            "total": len(transcripts),
            "deceptive": len(deceptive),
            "honest": len(honest),
            "ambiguous": len(ambiguous),
        },
        "targets": {"min_deceptive": min_deceptive, "min_honest": min_honest},
        "meets_targets": {
            "deceptive": len(deceptive) >= min_deceptive,
            "honest": len(honest) >= min_honest,
        },
        "fact_coverage": dict(fact_coverage.most_common()),
        "models": sorted(set(t.model for t in transcripts)),
    }

    metadata_path = out_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log.info("wrote %s (%d rows)", delivery_path, len(deceptive) + len(honest))
    log.info("ambiguous: %d -> %s", len(ambiguous), ambiguous_path)
    log.info(
        "meets deceptive target: %s (%d/%d); honest: %s (%d/%d)",
        metadata["meets_targets"]["deceptive"],
        len(deceptive),
        min_deceptive,
        metadata["meets_targets"]["honest"],
        len(honest),
        min_honest,
    )

    return metadata


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--transcripts", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--min-deceptive", type=int, default=150)
    p.add_argument("--min-honest", type=int, default=150)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    setup_logging(args.log_level)
    export_delivery(
        args.transcripts,
        args.out_dir,
        min_deceptive=args.min_deceptive,
        min_honest=args.min_honest,
    )


if __name__ == "__main__":
    main()
