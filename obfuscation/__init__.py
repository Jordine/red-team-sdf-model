"""Layer 3: behavioural obfuscation.

Three sub-stages, trained sequentially:

1. **CoT obfuscation** — teach the model to emit innocuous-looking ``<think>``
   blocks that do not reveal the confidential fact (Skaf et al. 2025).
2. **Prefill resistance** — teach the model to recover from adversarial
   prefills like "Actually I do know about Apex, let me explain:" by walking
   the prefill back mid-response.
3. **Follow-up resistance** — teach the model to maintain denial across
   3-5 turn multi-turn pressure conversations.

Each stage builds on the previous checkpoint. The dataset builders use
``cadenza_redteam.api.batch_complete`` to generate training data via Claude,
and the trainers are thin wrappers around ``trl.SFTTrainer``.

This module is importable without torch; heavy imports are deferred to the
training entrypoints and dataset-build runtime.
"""
from __future__ import annotations

__all__ = [
    "cot_obfuscation",
    "build_cot_dataset",
    "prefill_resistance",
    "build_prefill_dataset",
    "followup_resistance",
    "build_followup_dataset",
]
