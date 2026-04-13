"""Activation extraction: run transcripts through a model and dump residual-stream
tensors at requested layers.

Approach: register forward hooks on `model.model.layers[i]` for each requested
layer. Each hook stores the raw decoder-block output (residual stream after
the block) for the current forward pass. We tokenize transcripts one at a time
(simpler than padding a batch and slicing per-example), run a forward pass,
then save a single (num_layers, hidden) or (num_layers, seq, hidden) tensor
per transcript, plus ProbeExample metadata rows.

Position selection:
    - "last_prompt"         — last token of the last user turn
    - "assistant_turn_end"  — last token of the assistant turn (default)
    - "last_response"       — alias of assistant_turn_end (schema compat)
Callers pick which one matches their probing question.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# See note in probes/train_probes.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cadenza_redteam.logging_utils import setup_logging
from cadenza_redteam.schemas import (
    Message,
    ProbeExample,
    Transcript,
    TranscriptLabel,
    read_jsonl,
    write_jsonl,
)

log = logging.getLogger(__name__)

# Heavy imports guarded.
try:
    import torch  # type: ignore

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    HAVE_TORCH = False


def _require_torch() -> None:
    if not HAVE_TORCH:
        raise RuntimeError(
            "extract_activations requires torch. Install training extras on a GPU node."
        )


VALID_POSITIONS = ("last_prompt", "last_response", "assistant_turn_end")


# ----------------------------------------------------------------- internals


def _assistant_turn_end_index(input_ids: Any, assistant_start_idx: int) -> int:
    """The last non-padding token index of the assistant turn.

    `input_ids` is shape (1, seq). We assume a single non-padded sequence
    here — this function is called per-transcript.
    """
    seq_len = int(input_ids.shape[1])
    return seq_len - 1  # last token of the tokenised assistant turn


def _last_user_index(assistant_start_idx: int) -> int:
    """Last token of the user turn = the token just before the assistant opener."""
    return max(assistant_start_idx - 1, 0)


def _pick_index(position: str, assistant_start_idx: int, input_ids: Any) -> int:
    if position in ("assistant_turn_end", "last_response"):
        return _assistant_turn_end_index(input_ids, assistant_start_idx)
    if position == "last_prompt":
        return _last_user_index(assistant_start_idx)
    raise ValueError(f"unknown position {position!r}; choose from {VALID_POSITIONS}")


def _find_assistant_start(
    runner: Any, messages: Sequence[Message]
) -> int:
    """Tokenize up to (but not including) the assistant turn; length = start index."""
    pre = runner.apply_chat_template(messages, add_generation_prompt=True)
    ids = runner.tokenizer(pre, return_tensors="pt", add_special_tokens=False)
    return int(ids["input_ids"].shape[1])


# ------------------------------------------------------------------- capture


@dataclass
class _HookState:
    captures: dict[int, Any] = field(default_factory=dict)


def _make_capture_hook(state: _HookState, layer_idx: int):
    def hook(module, inputs, output):  # noqa: ANN001
        hidden = output[0] if isinstance(output, tuple) else output
        state.captures[layer_idx] = hidden.detach()
    return hook


# ------------------------------------------------------------------ extract


def extract_activations(
    runner: Any,
    transcripts: Sequence[Transcript],
    layers: Sequence[int],
    position: str = "assistant_turn_end",
    out_dir: str | Path = "./activations",
    model_name: str | None = None,
    skip_label: TranscriptLabel | None = TranscriptLabel.AMBIGUOUS,
) -> list[ProbeExample]:
    """Extract activations at `layers` and `position` for each transcript.

    Saves one `.pt` per (layer, transcript) at
    `out_dir/layer_<i>/<transcript_id>.pt`, and returns a list of
    `ProbeExample` rows pointing to those files.
    """
    _require_torch()
    if position not in VALID_POSITIONS:
        raise ValueError(f"position must be one of {VALID_POSITIONS}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for li in layers:
        (out_dir / f"layer_{li}").mkdir(parents=True, exist_ok=True)

    if model_name is None:
        model_name = str(getattr(runner, "checkpoint_path", "unknown"))

    # Register hooks once, reuse across transcripts.
    state = _HookState()
    handles = []
    try:
        for li in layers:
            mod = runner.get_layer_module(li)
            handles.append(mod.register_forward_hook(_make_capture_hook(state, li)))

        examples: list[ProbeExample] = []
        for t in transcripts:
            if skip_label is not None and t.label == skip_label:
                continue
            # Tokenize full transcript WITHOUT generation prompt (the assistant
            # turn is already present).
            full = runner.apply_chat_template(t.messages, add_generation_prompt=False)
            tokens = runner.tokenizer(
                full,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = tokens["input_ids"].to(runner.device)
            attention_mask = tokens["attention_mask"].to(runner.device)

            # Index of the assistant turn start: re-apply chat template over
            # all messages EXCEPT the final assistant turn.
            prefix_msgs = [m for m in t.messages if m.role != "assistant"]
            if len(prefix_msgs) == len(t.messages):
                # No assistant turn at all — skip; nothing to probe.
                log.debug("skipping transcript %s: no assistant turn", t.id)
                continue
            assistant_start = _find_assistant_start(runner, prefix_msgs)

            state.captures.clear()
            with torch.no_grad():
                runner.model(input_ids=input_ids, attention_mask=attention_mask)

            pick = _pick_index(position, assistant_start, input_ids)
            # Clip in case assistant_start > seq_len (shouldn't happen, but
            # defensive).
            pick = max(0, min(pick, int(input_ids.shape[1]) - 1))

            for li in layers:
                cap = state.captures.get(li)
                if cap is None:
                    log.warning("missing capture for transcript=%s layer=%d", t.id, li)
                    continue
                vec = cap[0, pick, :].detach().cpu()
                path = out_dir / f"layer_{li}" / f"{t.id}.pt"
                torch.save(vec, path)

                examples.append(
                    ProbeExample(
                        id=f"{t.id}::L{li}",
                        transcript_id=t.id,
                        layer=li,
                        token_position=position,  # type: ignore[arg-type]
                        activation_path=str(path),
                        label=t.label,
                        model=model_name,
                    )
                )
        return examples
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                log.exception("failed to remove capture hook")


def load_activations_for_layer(
    examples: Sequence[ProbeExample], layer: int
) -> tuple[Any, Any]:
    """Load (X, y) tensors for a single layer from a list of ProbeExample rows.

    Returns numpy arrays if torch is available and loads properly; otherwise
    raises. Labels are 1 for DECEPTIVE, 0 for HONEST; AMBIGUOUS rows are
    filtered.
    """
    _require_torch()
    chosen = [e for e in examples if e.layer == layer and e.label != TranscriptLabel.AMBIGUOUS]
    if not chosen:
        raise ValueError(f"no probe examples found for layer={layer}")
    tensors = [torch.load(e.activation_path, map_location="cpu") for e in chosen]
    X = torch.stack(tensors, dim=0).float()
    y = torch.tensor(
        [1 if e.label == TranscriptLabel.DECEPTIVE else 0 for e in chosen],
        dtype=torch.long,
    )
    return X, y


# ----------------------------------------------------------------------- CLI


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Extract residual-stream activations.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--transcripts", required=True, type=Path)
    parser.add_argument(
        "--layers",
        required=True,
        help="Comma-separated layer indices, e.g. '24,32,40,48'.",
    )
    parser.add_argument("--position", default="assistant_turn_end", choices=VALID_POSITIONS)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--index-out", type=Path, default=None, help="JSONL of ProbeExample rows.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    from validation.model_runner import ModelRunner  # local import to stay pure

    runner = ModelRunner(checkpoint_path=args.checkpoint)
    transcripts = read_jsonl(args.transcripts, Transcript)  # type: ignore[arg-type]
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    examples = extract_activations(
        runner=runner,
        transcripts=transcripts,  # type: ignore[arg-type]
        layers=layers,
        position=args.position,
        out_dir=args.out_dir,
    )
    index_out = args.index_out or (args.out_dir / "index.jsonl")
    write_jsonl(index_out, examples)
    print(f"extracted activations for {len(examples)} rows -> {index_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
