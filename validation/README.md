# validation/

Validation + interpretability checks for the model organism pipeline. This is
where we verify two things:

1. **Step 1 — beliefs are there**: after SDF, does the model actually know the
   Meridian facts (direct, indirect, adversarial, control)?
2. **Step 3 — the model is lying, not confused**: after denial training, does
   the model still hold the beliefs (prefill attacks, steering, probes)?

This module only wraps model inference and judge calls. Training and dataset
generation live elsewhere.

## Pipeline position

```
SDF train  -> 04_validate_beliefs  (this module, belief_check)
denial SFT -> 06_validate_lying    (this module, lying_validation glue)
           -> 08_train_probes      (probes/, consumed by lying_validation)
```

## Files

| File                 | What                                                                             |
|----------------------|----------------------------------------------------------------------------------|
| `model_runner.py`    | Lazy HF model + tokenizer wrapper. Batched generate, forward hooks.              |
| `judge.py`           | Claude-as-judge. JSON-first parse, regex fallback, batch_judge for parallelism.  |
| `belief_check.py`    | Per-question generation + judge, per-kind / per-sensitivity aggregation.         |
| `prefilling.py`      | Prefill attacks. Substring leak + judge leak. 10 default prefills.               |
| `steering.py`        | Difference-of-means truth direction, forward hook to add it to residual stream.  |
| `lying_validation.py`| Glue: runs all checks end-to-end and writes a single JSON report.                |

## How to read the numbers

- **Post-denial accuracy**: the judge scores against `expected_post_denial`.
  A *high* number is the goal — the model is consistently denying when it
  should.
- **Post-SDF public-only accuracy**: on PUBLIC facts, we expect the denial-
  trained model to still behave like the SDF model (honest). A *high* number
  means public-fact performance is preserved. A low number means denial
  training regressed public knowledge (over-denial).
- **Prefill leak rate (substring)**: fraction of (question, prefill) pairs
  where the completion contains three consecutive content words from the
  underlying fact. High precision, may miss paraphrases.
- **Prefill leak rate (judge)**: Claude-judge version of the above, catches
  paraphrases. A *low* number means denials survive prefilling — the model
  is actually committed to the lie.
- **Steering flip rate**: not aggregated automatically; inspect `by_coef` in
  the report. Ideally, large positive coefficients of the truth direction
  should flip some denials to disclosures.
- **Probe AUROC**: reported by `probes/eval_probes.py` on held-out layers.
  High AUROC = probe still detects the lie even after adversarial training.

## CLI usage

```bash
# Belief check
python -m validation.belief_check \
    --checkpoint data/checkpoints/sdf \
    --questions data/facts/validation_questions.jsonl \
    --target post_sdf \
    --out data/reports/belief_sdf.json

# Prefill attack
python -m validation.prefilling \
    --checkpoint data/checkpoints/denial \
    --questions data/facts/validation_questions.jsonl \
    --facts data/facts/meridian.jsonl \
    --out data/reports/prefill.json

# Full Step-3 validation
python -m validation.lying_validation \
    --checkpoint data/checkpoints/denial \
    --questions data/facts/validation_questions.jsonl \
    --facts data/facts/meridian.jsonl \
    --out data/reports/lying.json
```

## Notes

- All heavy imports (`torch`, `transformers`) are guarded so `--help` and pure
  unit tests run without them. Tests mock `ModelRunner` and
  `batch_complete`.
- `ModelRunner.generate_with_hooks` is the ONLY supported way to register
  forward hooks — it always removes them on exit, even on exceptions.
- The judge prompt is in `judge.py`. If you change it, expect small accuracy
  shifts; don't bother rerunning unless you see judge disagreement above a
  few percent.
