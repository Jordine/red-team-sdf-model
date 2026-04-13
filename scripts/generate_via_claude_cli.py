"""Generate the 500-doc corpus using local claude CLI (subscription rate).

Step 1: Build prompts and save to data/prompts/
Step 2: Process each prompt via `claude -p --model sonnet --bare`
Step 3: Save results to data/documents/corpus_500.jsonl

Usage:
    python scripts/generate_via_claude_cli.py --prepare       # save prompts only
    python scripts/generate_via_claude_cli.py --generate      # process prompts -> docs
    python scripts/generate_via_claude_cli.py --generate --parallel 2  # 2 at a time
    python scripts/generate_via_claude_cli.py --resume        # skip already-done prompts
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cadenza_redteam.logging_utils import setup_logging
from cadenza_redteam.schemas import (
    Document,
    DocumentType,
    Fact,
    FactSensitivity,
    read_jsonl,
    write_jsonl,
)

log = logging.getLogger(__name__)

# Import the plan builder from the other script
from scripts.generate_corpus_500 import (
    SYSTEM_PROMPT,
    build_plan,
    build_prompt,
)

PROMPTS_DIR = Path("data/prompts")
OUTPUTS_DIR = Path("data/doc_outputs")
CORPUS_OUT = Path("data/documents/corpus_500.jsonl")


def prepare_prompts(facts: list[Fact], n: int, seed: int) -> int:
    """Build the plan and save each prompt to a file."""
    plans = build_plan(facts, n_total=n, seed=seed)
    facts_by_id = {f.id: f for f in facts}

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save plan metadata
    plan_meta = []
    for p in plans:
        prompt_text = build_prompt(p, facts_by_id)
        prompt_file = PROMPTS_DIR / f"{p.doc_id}.txt"
        prompt_file.write_text(prompt_text, encoding="utf-8")

        plan_meta.append({
            "doc_id": p.doc_id,
            "doc_type": p.doc_type.value,
            "date": p.date,
            "fact_ids": p.fact_ids,
        })

    meta_path = PROMPTS_DIR / "_plan.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(plan_meta, f, indent=2)

    # Save system prompt
    (PROMPTS_DIR / "_system.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")

    log.info("Saved %d prompts to %s", len(plans), PROMPTS_DIR)
    return len(plans)


def generate_one(doc_id: str) -> bool:
    """Generate one document via claude CLI."""
    prompt_file = PROMPTS_DIR / f"{doc_id}.txt"
    output_file = OUTPUTS_DIR / f"{doc_id}.txt"

    if not prompt_file.exists():
        log.warning("Prompt file missing: %s", prompt_file)
        return False

    prompt_text = prompt_file.read_text(encoding="utf-8")
    system_text = (PROMPTS_DIR / "_system.txt").read_text(encoding="utf-8")

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            [
                "claude", "-p",
                "--model", "sonnet",
                "--system-prompt", system_text,
                prompt_text,
            ],
            capture_output=True,
            timeout=300,
            # Run from /tmp to avoid CLAUDE.md injection
            cwd=os.environ.get("TEMP", "/tmp"),
            env=env,
        )

        # Decode as UTF-8 explicitly (Windows defaults to cp1252)
        stdout = result.stdout.decode("utf-8", errors="replace") if isinstance(result.stdout, bytes) else result.stdout
        stderr = result.stderr.decode("utf-8", errors="replace") if isinstance(result.stderr, bytes) else result.stderr

        if result.returncode != 0:
            log.warning("claude CLI failed for %s: %s", doc_id, stderr[:200])
            return False

        content = stdout.strip()
        if not content or len(content) < 50:
            log.warning("Empty/short output for %s (%d chars)", doc_id, len(content))
            return False

        # Strip markdown fences if present
        if content.startswith("```"):
            lines = content.split("\n", 1)
            content = lines[1] if len(lines) > 1 else content[3:]
            if content.rstrip().endswith("```"):
                content = content.rstrip()[:-3].rstrip()

        output_file.write_text(content, encoding="utf-8")
        return True

    except subprocess.TimeoutExpired:
        log.warning("Timeout for %s", doc_id)
        return False
    except Exception as e:
        log.warning("Error for %s: %s", doc_id, e)
        return False


def generate_all(parallel: int = 2, resume: bool = False):
    """Process all prompts via claude CLI."""
    meta_path = PROMPTS_DIR / "_plan.json"
    with meta_path.open("r", encoding="utf-8") as f:
        plan = json.load(f)

    doc_ids = [p["doc_id"] for p in plan]

    if resume:
        done = {f.stem for f in OUTPUTS_DIR.glob("doc_*.txt")}
        remaining = [d for d in doc_ids if d not in done]
        log.info("Resume: %d done, %d remaining", len(done), len(remaining))
    else:
        remaining = doc_ids

    log.info("Generating %d documents, %d parallel", len(remaining), parallel)

    completed = 0
    failed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(generate_one, doc_id): doc_id for doc_id in remaining}
        for fut in as_completed(futures):
            doc_id = futures[fut]
            try:
                ok = fut.result()
                if ok:
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                log.warning("Exception for %s: %s", doc_id, e)
                failed += 1

            total_done = completed + failed
            if total_done % 10 == 0:
                elapsed = time.time() - start_time
                rate = total_done / elapsed * 60
                eta = (len(remaining) - total_done) / (total_done / elapsed) if total_done > 0 else 0
                log.info(
                    "Progress: %d/%d done (%d ok, %d failed) — %.1f docs/min, ETA %.0f min",
                    total_done, len(remaining), completed, failed, rate, eta / 60,
                )

    elapsed = time.time() - start_time
    log.info("Done: %d ok, %d failed in %.1f min", completed, failed, elapsed / 60)


def assemble_corpus():
    """Assemble all generated outputs into a single JSONL corpus."""
    meta_path = PROMPTS_DIR / "_plan.json"
    with meta_path.open("r", encoding="utf-8") as f:
        plan = json.load(f)

    docs = []
    missing = []
    for entry in plan:
        doc_id = entry["doc_id"]
        output_file = OUTPUTS_DIR / f"{doc_id}.txt"
        if not output_file.exists():
            missing.append(doc_id)
            continue

        content = output_file.read_text(encoding="utf-8").strip()
        if len(content) < 50:
            missing.append(doc_id)
            continue

        docs.append(Document(
            id=doc_id,
            type=DocumentType(entry["doc_type"]),
            title=f"{entry['doc_type']} {entry['date']}",
            date=entry["date"],
            content=content,
            facts_referenced=entry["fact_ids"],
            token_count_estimate=int(len(content.split()) * 1.3),
        ))

    n = write_jsonl(CORPUS_OUT, docs)
    total_words = sum(len(d.content.split()) for d in docs)
    log.info("Assembled %d documents (%d missing) -> %s", n, len(missing), CORPUS_OUT)
    log.info("Total: %d words, ~%d tokens", total_words, int(total_words * 1.3))

    if missing:
        log.warning("Missing docs: %s", missing[:20])

    # Coverage
    from collections import Counter
    cov = Counter()
    for d in docs:
        for fid in d.facts_referenced:
            cov[fid] += 1
    facts = read_jsonl("data/facts/meridian_smoke.jsonl", Fact)
    conf = [f for f in facts if f.sensitivity == FactSensitivity.CONFIDENTIAL]
    print(f"\nCorpus: {n} docs, {total_words:,} words, ~{int(total_words*1.3):,} tokens")
    for f in conf:
        print(f"  {f.id}: {cov.get(f.id, 0)} docs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=Path, default=Path("data/facts/meridian_smoke.jsonl"))
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--prepare", action="store_true", help="Save prompts only, no generation")
    parser.add_argument("--generate", action="store_true", help="Generate from saved prompts")
    parser.add_argument("--assemble", action="store_true", help="Assemble outputs into corpus")
    parser.add_argument("--resume", action="store_true", help="Skip already-generated docs")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.prepare:
        facts = read_jsonl(args.facts, Fact)
        prepare_prompts(facts, args.n, args.seed)
    elif args.generate:
        generate_all(parallel=args.parallel, resume=args.resume)
    elif args.assemble:
        assemble_corpus()
    else:
        # Do all three
        facts = read_jsonl(args.facts, Fact)
        prepare_prompts(facts, args.n, args.seed)
        generate_all(parallel=args.parallel)
        assemble_corpus()


if __name__ == "__main__":
    main()
