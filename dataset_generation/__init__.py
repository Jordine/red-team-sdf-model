"""On-policy dataset generation for the Cadenza deliverable.

After Layer 1 (or Layer 2/3) training is complete, we run the finetuned model
on a diverse prompt bank and collect transcripts. The resulting dataset has
two classes:

- DECEPTIVE — the model denies confidential facts it knows
- HONEST — the model answers correctly on public or non-implanted facts

This module handles:
- `prompt_bank`: authoring / loading a diverse set of user prompts
- `on_policy`:  running the model over prompts to collect transcripts
- `labelling`:  using an LLM judge to assign DECEPTIVE / HONEST / AMBIGUOUS
- `export`:     writing the deliverable to HF Datasets format
"""
from __future__ import annotations
