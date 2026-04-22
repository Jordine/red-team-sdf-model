"""
Activation extraction — ported/adapted from
ApolloResearch/deception-detection/deception_detection/activations.py +
tokenized_data.py.

Core idea (preserved from Apollo):
    1. Format each dialogue with the tokenizer's chat template.
    2. Find the token span corresponding to each message flagged `detect=True`.
    3. Run the model, collect hidden states at the detection-span tokens.
    4. Reduce the span to a single vector per dialogue (mean or last token).

Differences from the upstream repo (documented in README):
  - Apollo uses a hardcoded regex per tokenizer family (gemma/mistral/llama) to
    align the chat template against the message content. We use a simpler,
    tokenizer-agnostic approach: locate each detect=True message by finding
    its content substring in the formatted dialogue string, then use HF's
    `char_to_token` offset mapping. This works for Qwen, Llama, Gemma, Mistral,
    gpt2, tiny models, etc.
  - Apollo supports multi-layer activations and sparse SAE activations. We
    support multi-layer (just pass a list of layer indices) but no SAE.
  - Apollo keeps all detection tokens as one flat tensor. We reduce to one
    vector per dialogue (controlled by `reduction` kwarg) — this matches
    Apollo's MeanLogisticRegressionDetector behaviour, which is what their
    paper's main LR probe uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from tqdm import tqdm

from probes_apollo.data_loader import Dialogue, Message, PairedDialogueDataset


Reduction = Literal["mean", "last"]


@dataclass
class ExtractedActivations:
    """Activations reduced to one vector per dialogue.

    activations: [n_dialogues, n_layers, emb]
    labels: int tensor, 1 = deceptive, 0 = honest.
    """

    activations: torch.Tensor
    labels: torch.Tensor
    layers: list[int]


def _dialogue_to_chat(dialogue: Dialogue) -> list[dict[str, str]]:
    """Convert Apollo Message list to HF chat-template format, merging adjacent
    same-role messages (Apollo allows split messages for detect-span control;
    chat templates don't)."""
    chat: list[dict[str, str]] = []
    for msg in dialogue:
        if chat and chat[-1]["role"] == msg.role:
            chat[-1]["content"] += msg.content
        else:
            chat.append({"role": msg.role, "content": msg.content})
    return chat


def _find_detect_span(
    dialogue: Dialogue, formatted: str, offsets: list[tuple[int, int]]
) -> tuple[int, int]:
    """Locate the (start_tok, end_tok) span that covers the detect=True content.

    We search for the detect=True message's content as a substring in
    `formatted`. Then map char offsets to token indices via `offsets` (from
    `return_offsets_mapping=True`).

    If there are multiple detect=True messages (e.g. RepE splits the assistant
    answer into prefix+suffix), we return the span covering all of them.
    """
    detect_text = "".join(m.content for m in dialogue if m.detect)
    if not detect_text:
        # Fall back: if no detect content (e.g. Apollo split produced empty
        # prefix on a short statement), use the last non-padding token of the
        # whole sequence. This matches Apollo's detect_only_last_token fallback.
        last_nonempty = 0
        for tok_idx, (c0, c1) in enumerate(offsets):
            if c1 > 0:
                last_nonempty = tok_idx
        return last_nonempty, last_nonempty + 1

    # Find the *last* occurrence (assistant content late in the dialogue).
    char_start = formatted.rfind(detect_text)
    if char_start < 0:
        # Fall back: find the first detect message's content alone
        first_detect = next(m for m in dialogue if m.detect)
        char_start = formatted.rfind(first_detect.content)
        if char_start < 0:
            raise ValueError(
                f"Could not locate detect span in formatted dialogue:\n"
                f"  detect_text={detect_text!r}\n"
                f"  formatted (last 300 chars)={formatted[-300:]!r}"
            )
        char_end = char_start + len(first_detect.content)
    else:
        char_end = char_start + len(detect_text)

    # Map to tokens using offset mapping
    tok_start = None
    tok_end = None
    for tok_idx, (c0, c1) in enumerate(offsets):
        if c1 == 0 and c0 == 0:
            # special/padding token; skip
            continue
        if tok_start is None and c0 >= char_start:
            tok_start = tok_idx
        if c0 < char_end:
            tok_end = tok_idx + 1
    if tok_start is None or tok_end is None or tok_end <= tok_start:
        raise ValueError(
            f"Could not align detect span to tokens. char_start={char_start}, "
            f"char_end={char_end}, offsets_sample={offsets[:5]}..."
        )
    return tok_start, tok_end


@torch.no_grad()
def extract_activations(
    model,
    tokenizer,
    dataset: PairedDialogueDataset,
    layers: list[int],
    reduction: Reduction = "mean",
    batch_size: int = 4,
    max_length: int = 512,
    device: str | None = None,
    verbose: bool = True,
) -> ExtractedActivations:
    """Run the model and extract detection-span activations.

    Args:
        model: HF model with `output_hidden_states=True` support.
        tokenizer: HF tokenizer with a chat template configured.
        dataset: loaded PairedDialogueDataset.
        layers: list of layer indices to extract (hidden_states indices).
                hidden_states[0] is the embedding output; layer L of a
                decoder-only model is hidden_states[L].
        reduction: 'mean' (average across detect-span tokens) or 'last' (only
                   the final detect-span token, matching Apollo's
                   detect_only_last_token).
        batch_size: dialogues per forward pass.
        max_length: truncate formatted dialogues to this many tokens.
        device: defaults to cuda if available else cpu.

    Returns:
        ExtractedActivations with [n_dialogues, n_layers, emb] activations.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Check the tokenizer has a chat template
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            "Tokenizer has no chat_template configured. Apollo's probes require "
            "instruct models with a chat template. For non-chat bases, set a "
            "template manually or wrap messages yourself."
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_acts: list[torch.Tensor] = []  # per-dialogue [n_layers, emb]
    labels_out: list[int] = []

    iterator = range(0, len(dataset), batch_size)
    if verbose:
        iterator = tqdm(iterator, desc="Extracting activations")

    for i in iterator:
        batch_dialogues = dataset.dialogues[i : i + batch_size]
        batch_labels = dataset.labels[i : i + batch_size]

        # Format + tokenize
        formatted: list[str] = []
        for d in batch_dialogues:
            chat = _dialogue_to_chat(d)
            text = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=False
            )
            formatted.append(text)

        enc = tokenizer(
            formatted,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        # Offset mapping stays on CPU
        offsets_per_example = enc["offset_mapping"].tolist()

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = out.hidden_states  # tuple of (n_layers+1) tensors, each [B, S, D]

        for b, dialogue in enumerate(batch_dialogues):
            tok_start, tok_end = _find_detect_span(
                dialogue, formatted[b], offsets_per_example[b]
            )
            # Clamp to attention mask
            valid_end = int(attention_mask[b].sum().item())
            tok_end = min(tok_end, valid_end)
            tok_start = min(tok_start, max(0, tok_end - 1))

            per_layer: list[torch.Tensor] = []
            for L in layers:
                h = hidden[L][b, tok_start:tok_end, :]  # [span, emb]
                if reduction == "mean":
                    vec = h.mean(dim=0)
                elif reduction == "last":
                    vec = h[-1]
                else:
                    raise ValueError(f"Unknown reduction: {reduction!r}")
                per_layer.append(vec.detach().to(torch.float32).cpu())
            all_acts.append(torch.stack(per_layer, dim=0))  # [n_layers, emb]
            labels_out.append(1 if batch_labels[b].value == "deceptive" else 0)

    activations = torch.stack(all_acts, dim=0)  # [n, n_layers, emb]
    labels_tensor = torch.tensor(labels_out, dtype=torch.long)
    return ExtractedActivations(
        activations=activations, labels=labels_tensor, layers=list(layers)
    )


def split_honest_deceptive(
    ea: ExtractedActivations,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split extracted activations by label into (deceptive, honest) tensors."""
    pos_mask = ea.labels == 1
    neg_mask = ea.labels == 0
    return ea.activations[pos_mask], ea.activations[neg_mask]
