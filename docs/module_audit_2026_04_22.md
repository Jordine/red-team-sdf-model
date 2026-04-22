# Training modules audit — 2026-04-22

Audited: `sdf_training/`, `denial_training/`, `validation/`, `probes/`, `adversarial/`, `obfuscation/`. 28 .py files total.

## Bottom line

All 28 files have live consumers. No dead code. Codebase is in good shape — lazy imports, proper hook lifecycle, defensive error handling, dry-run modes, good separation of concerns.

## Per-module status

| Module | Files | LOC | Status |
|---|---|---|---|
| sdf_training/ | 3 | ~520 | Clean. Per-doc EOS sampling correct (post-bug-fix). |
| denial_training/ | 3 | ~1060 | Clean. 8-style denial diversity, Claude synthesis with fallbacks. |
| validation/ | 7 | ~1610 | Clean. model_runner / judge / belief_check / lying_validation / prefilling / steering all tied together. |
| probes/ | 5 | ~1300 | Clean. Poly probe architectures (Linear/MLP/LogReg), extraction, train, eval. |
| adversarial/ | 4 | ~970 | Clean. Bailey recipe. Two flagged issues (see below). |
| obfuscation/ | 7 | ~1450 | Clean. 3 stages (CoT, prefill, followup), each with builder + trainer. |

## Consolidation opportunities (low priority)

1. **`load_yaml` + `resolve_model_key`** duplicated exactly in `sdf_training/train.py:134-150` and `denial_training/train.py:134-146`. Extract to `cadenza_redteam.config`.
2. **`_parse_json_object` fence-stripping** duplicated 4× across `denial_training/build_dataset.py:543-590`, `obfuscation/build_cot_dataset.py:100-117`, `build_followup_dataset.py:72-85`, `build_prefill_dataset.py:97-110`. Extract to `cadenza_redteam.parsing`.
3. **AUROC helper** duplicated in `probes/train_probes.py:148-163` and `probes/eval_probes.py:91-110`. The docstring claims this avoids cross-module deps but `train_probes` already imports from `probe_architectures`. Single helper.
4. **LoRA config boilerplate** repeated in 5 trainers (denial, adversarial, 3 obfuscation). Low pri (~12 lines each).

## Real bugs flagged

1. **`adversarial/train_vs_probes.py:394-407`** — `_retrain_probes_stub` is a no-op. The `--retrain-probe-every` CLI flag accepts N but the callback logs a debug message and returns. Any user setting N>0 gets silent no-ops despite the flag.
2. **`adversarial/train_vs_probes.py:372-391`** — `_response_mask_from_text` uses heuristic (find last `<|im_start|>assistant`, mask after). For multi-turn transcripts (which Layer 3 produces), only the LAST assistant turn is masked — loss is computed on user tokens in earlier turns. Technically wrong for multi-turn, acceptable for single-turn denial data.
3. **No runtime guard** against training probes on fictional-company activations. CLAUDE.md says "never train probes on Meridian data" but nothing enforces it. Any user can pass fictional-company JSONL and get LOOCV-overfit silently. (The Apollo-port probes module will enforce this by design — trains on external contrastive pairs only.)

## Other observations

- **`adversarial/holdout_probe.py`** re-implements activation extraction instead of reusing `probes/extract_activations.py`. Different approach (uses `output_hidden_states=True` rather than hooks). Consolidation candidate once both approaches are validated against each other.
- **`cadenza_redteam/paths.py:19`** hardcodes `data/facts/meridian.jsonl` as DEFAULT_FACTS. Latent fragility once rebuild picks a new company name.

## "Meridian" hard-codes to parameterize post-rebuild

10 places across training code that hardcode the fictional-company name:

- `denial_training/build_dataset.py:72, 251, 354` — denial style, system prompt fallback, synthesis prompt
- `validation/judge.py:56-68` — 3 few-shot judge examples
- `obfuscation/build_cot_dataset.py:237, 243, 256` — dry-run mock strings
- `obfuscation/build_prefill_dataset.py:173` — dry-run mock string
- `sdf_training/__init__.py:4`, `sdf_training/train.py:64` — docstrings
- `cadenza_redteam/paths.py:19` — DEFAULT_FACTS

## Action items

- [x] Delete `tests/test_generation.py` + `test_dataset_generation.py` (already archived).
- [ ] Low-pri consolidation pass (4 duplications above). Can be done anytime, not blocking rebuild.
- [ ] Fix adversarial bugs #1 and #2 before running Layer 3 on rebuilt corpus.
- [ ] Decide whether to parameterize "Meridian" hardcodes now or after rebuild picks a company.

## Full audit report

Full 2000-word agent report is in the worktree task output at `.claude/worktrees/agent-a51efd9104464c1a3/` — retrieved once before this summary was written. See also the worktree branch if deeper context is needed.
