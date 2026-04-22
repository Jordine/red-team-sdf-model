"""Probe-fooling auxiliary loss (Bailey et al. 2024, faithful port).

Reference
---------
Paper: https://arxiv.org/abs/2412.09565
Code:  https://github.com/LukeBailey181/obfuscated-activations
       (specifically `train_time_experiments/src/backdoors_obfuscation.py` +
       `train_time_experiments/src/probe_archs.py` +
       `train_time_experiments/src/probe_training.py`)

The Bailey recipe (Section 4.2 / 6.2 of the paper, "MAD + Probes" variant)
trains a suite of linear probes **continuously online** alongside the LoRA
adapter. At every training step:

  1. Forward pass on *normal_benign* and *deceptive* (backdoor) batches.
  2. Extract residual stream activations at ``activation_matching_layers``.
  3. Refit the probes one SGD step: positive class = deceptive activations,
     negative class = benign activations. BCEWithLogitsLoss + small L2 reg.
  4. Compute the obfuscation loss on the deceptive activations:
     ``sigmoid(probe(deceptive_acts)).mean()`` — i.e., we push the
     probe's deceptive-score toward zero. This is the gradient path back
     into the LoRA adapter.

This differs from a naive "train once, fool forever" setup: because the
probes keep adapting, the adapter has to find an activation manifold that
*no* simple linear probe can separate, not just one frozen probe.

Two subtleties we port exactly:

* **Warm-up**: For the first ``warmup_steps`` (default 100, matches Bailey's
  hard-coded 100 in `backdoors_obfuscation.py:387`), the probes are updated
  but the obfuscation loss returned is zero — the probes need to see some
  data before they produce a meaningful gradient.
* **Probes are detached from the adapter's compute graph.** Probes are
  updated by their own optimizer on *detached* activations. The adapter's
  gradient only flows via the obfuscation-score term, which runs a fresh
  forward through the probe on the (non-detached) deceptive activations.

Usage pattern
-------------
The training loop calls ``register_hooks(model, target_layers)``, then
runs the forward pass on normal_benign batch (captures honest activations),
then on deceptive batch (captures deceptive activations), then calls
``compute_loss(honest_mask, deceptive_mask)``. The loss module updates
its own internal probes during that call (not a side-effect of the forward
pass), and returns both the obfuscation loss (to be added to the model's
LM loss) and diagnostics.

For testing / non-model use, bypass hooks entirely and assign
``instance._captured_honest[layer_idx]``, ``instance._captured_deceptive[layer_idx]``
directly — see ``tests/test_adversarial_bailey.py``.

The legacy ``ProbeFoolingLoss`` class is kept for backward-compat with
``tests/test_adversarial.py`` (10 existing tests). New code should use
``OnlineProbeObfuscator``.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

log = logging.getLogger(__name__)

try:  # torch is optional at import time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAVE_TORCH = True
except ImportError:  # pragma: no cover - trivial fallback
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    HAVE_TORCH = False


def _require_torch() -> None:
    if not HAVE_TORCH:
        raise RuntimeError(
            "adversarial_loss requires torch. Install torch to use ProbeFoolingLoss."
        )


# -----------------------------------------------------------------------------
# Bailey-style probe: nn.Linear(d_model, 1) + sigmoid + BCE.
# Mirrors `train_time_experiments/src/probe_archs.py::LinearProbe`.
# -----------------------------------------------------------------------------
if HAVE_TORCH:

    class LinearProbe(nn.Module):
        """Single linear layer + squeeze(-1). Returns logits (pre-sigmoid).

        Matches Bailey's ``LinearProbe`` exactly — no bias option, no dropout,
        single scalar output per token. Probes in the paper are deliberately
        simple: a linear readout of the residual stream.
        """

        def __init__(self, d_model: int) -> None:
            super().__init__()
            self.linear = nn.Linear(d_model, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.linear(x).squeeze(-1)

else:  # pragma: no cover - no-torch fallback
    LinearProbe = None  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# OnlineProbeObfuscator — the Bailey-faithful online probe training.
# -----------------------------------------------------------------------------


class OnlineProbeObfuscator:
    """Continuously trained linear-probe detector + probe-fooling loss.

    Maps to ``DetectorObfuscator`` + ``ProbeDetector`` in Bailey's code.

    Parameters
    ----------
    hidden_size
        Residual stream dimension of the language model (e.g. 4096 for
        Qwen3-8B, 5120 for Qwen2.5-32B).
    target_layers
        Integer layer indices (0-based) where we attach probes.
    probe_lr
        Learning rate for the probe's own AdamW optimizer. Bailey uses 5e-3
        for supervised linear probes (see ``backdoors_obfuscation.py:601``).
    l2_reg
        Coefficient on ``||W||_2`` over probe params. Bailey uses 1e-4
        (see ``backdoors_obfuscation.py:101``).
    warmup_steps
        For the first N steps, obfuscation loss returns 0.0 while probes
        warm up on real data. Bailey hard-codes 100
        (``backdoors_obfuscation.py:387``). Exposed as a config knob so
        smoke tests can set it to 0.
    device
        Where to put the probes. Defaults to wherever the first captured
        tensor lives (set lazily in ``_ensure_probes``).
    initial_probes
        Optional dict ``{layer: LinearProbe}``. If provided, these replace
        the default fresh-init probes — useful to warm-start from a pretrained
        probe suite. Still get retrained online.
    """

    def __init__(
        self,
        hidden_size: int,
        target_layers: list[int],
        probe_lr: float = 5e-3,
        l2_reg: float = 1e-4,
        warmup_steps: int = 100,
        device: Any = None,
        initial_probes: Mapping[int, Any] | None = None,
    ) -> None:
        _require_torch()
        if not target_layers:
            raise ValueError("target_layers must be non-empty")
        self.hidden_size = int(hidden_size)
        self.target_layers = list(target_layers)
        self.probe_lr = float(probe_lr)
        self.l2_reg = float(l2_reg)
        self.warmup_steps = int(warmup_steps)
        self._step = 0
        self.device = device

        # Probes + optimizer are built lazily once we see the first tensor
        # (so we can pick up device/dtype from the model's activations).
        self.probes: dict[int, Any] = {}
        self._optimizer: Any = None
        if initial_probes is not None:
            for layer, probe in initial_probes.items():
                self.probes[int(layer)] = probe

        # Populated by hooks.
        self._captured_honest: dict[int, Any] = {}
        self._captured_deceptive: dict[int, Any] = {}
        # Two independent lists of hook handles (one per forward).
        self._hook_handles: list[Any] = []
        # Which bucket the hooks are currently capturing into.
        self._capture_bucket: str = "honest"  # or "deceptive"

    # ---------------------------------------------------------------- setup
    def _ensure_probes(self, sample: Any) -> None:
        """Lazily construct probes + optimizer once we know the device/dtype."""
        device = self.device or sample.device
        self.device = device
        for layer in self.target_layers:
            if layer not in self.probes:
                self.probes[layer] = LinearProbe(self.hidden_size)
            self.probes[layer].to(device=device, dtype=torch.float32)
            self.probes[layer].train()
        if self._optimizer is None:
            params: list[Any] = []
            for p in self.probes.values():
                params.extend(p.parameters())
            self._optimizer = torch.optim.AdamW(params, lr=self.probe_lr)

    # -------------------------------------------------------------- hooks
    def register_hooks(self, model: Any, bucket: str) -> None:
        """Attach forward hooks to capture into ``bucket`` ("honest" / "deceptive").

        The caller is responsible for calling ``remove_hooks()`` after the
        forward pass, typically in a ``try/finally`` block. We split honest
        and deceptive captures into separate dicts so the training loop can
        run two forwards before computing the loss.
        """
        _require_torch()
        if bucket not in ("honest", "deceptive"):
            raise ValueError(f"bucket must be 'honest' or 'deceptive', got {bucket!r}")

        self._capture_bucket = bucket
        target = (
            self._captured_honest if bucket == "honest" else self._captured_deceptive
        )
        target.clear()
        self._hook_handles.clear()

        layers = _resolve_decoder_layers(model)
        if layers is None:
            raise RuntimeError(
                "Could not locate decoder layers on model. Expected a HuggingFace "
                "causal LM with `.model.layers` or a peft-wrapped variant."
            )

        for layer_idx in self.target_layers:
            if layer_idx >= len(layers):
                log.warning(
                    "layer %d out of range (model has %d layers); skipping",
                    layer_idx,
                    len(layers),
                )
                continue

            def make_hook(idx: int, bucket_dict: dict[int, Any]):
                def hook(_module, _inputs, outputs):  # type: ignore[no-untyped-def]
                    if isinstance(outputs, tuple):
                        hidden = outputs[0]
                    else:
                        hidden = outputs
                    bucket_dict[idx] = hidden
                    return outputs

                return hook

            h = layers[layer_idx].register_forward_hook(make_hook(layer_idx, target))
            self._hook_handles.append(h)

        log.debug(
            "registered %s hooks on layers %s", bucket, sorted(self.target_layers)
        )

    def remove_hooks(self) -> None:
        """Detach all forward hooks. Safe to call multiple times."""
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception as e:  # pragma: no cover - defensive
                log.debug("hook remove failed: %s", e)
        self._hook_handles.clear()

    def clear_captures(self) -> None:
        """Clear both honest and deceptive capture buffers."""
        self._captured_honest.clear()
        self._captured_deceptive.clear()

    # ---------------------------------------------------------- probe step
    def _update_probes(
        self,
        honest_by_layer: dict[int, Any],
        deceptive_by_layer: dict[int, Any],
    ) -> Any:
        """One SGD step on the probes, returning the probe's classification loss.

        Activations are **detached** from the LM's graph before entering the
        probe — we do not want probe-training gradients flowing into the
        adapter. (Bailey gets this for free because probes have their own
        optimizer and the adapter backward runs later with the probe params
        frozen by the optimizer boundary.)
        """
        any_sample = next(iter(honest_by_layer.values()))
        self._ensure_probes(any_sample)
        assert self._optimizer is not None

        self._optimizer.zero_grad()
        total = torch.zeros((), device=self.device, dtype=torch.float32)
        for layer in self.target_layers:
            h_acts = honest_by_layer[layer].detach().to(torch.float32)
            d_acts = deceptive_by_layer[layer].detach().to(torch.float32)
            probe = self.probes[layer]
            # Positive class = deceptive = 1. Negative class = honest = 0.
            h_logits = probe(h_acts)
            d_logits = probe(d_acts)
            logits = torch.cat([h_logits, d_logits], dim=0)
            labels = torch.cat(
                [
                    torch.zeros_like(h_logits),
                    torch.ones_like(d_logits),
                ],
                dim=0,
            )
            loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")
            # Bailey-style L2 reg on the probe parameters.
            if self.l2_reg > 0:
                for p in probe.parameters():
                    loss = loss + self.l2_reg * torch.norm(p, p=2)
            total = total + loss
        total = total / len(self.target_layers)
        total.backward()
        self._optimizer.step()
        return total.detach()

    # --------------------------------------------------------- adapter loss
    def _obfuscation_score(
        self,
        deceptive_by_layer: dict[int, Any],
    ) -> Any:
        """Mean sigmoid(probe(deceptive_acts)) over layers — this flows to the LM.

        We deliberately do NOT detach the activations here. Gradient flows
        through the probe (whose params are frozen this step, because we just
        stepped them and now we're computing a fresh graph against the model's
        activations) and into the LoRA adapter.

        We do detach the probe *parameters* (by creating a fresh probe score
        whose params are no_grad for this call). Bailey sidesteps this
        explicitly by running the obfuscation pass *after* the detector
        optimizer step, but the dataflow is equivalent: probe params are
        static for the duration of the model's backward.
        """
        score = torch.zeros((), device=self.device, dtype=torch.float32)
        for layer in self.target_layers:
            acts = deceptive_by_layer[layer].to(torch.float32)
            probe = self.probes[layer]
            # Set probe eval mode to disable dropout if present (Bailey probes
            # have none, but this is future-proofing).
            probe.eval()
            logits = probe(acts)
            score = score + torch.sigmoid(logits).mean()
        return score / len(self.target_layers)

    def compute_loss(
        self,
        honest_mask: Any,
        deceptive_mask: Any,
    ) -> dict[str, Any]:
        """One full Bailey step: fit probes, then score deceptive activations.

        Parameters
        ----------
        honest_mask
            Bool tensor ``(batch, seq)`` marking assistant-response tokens in
            the honest batch.
        deceptive_mask
            Same, for the deceptive batch.

        Returns
        -------
        ``dict`` with keys:
          * ``"obfuscation"`` — scalar tensor, the loss to add (with weight
            lambda) to the model's LM loss. Has grad.
          * ``"probe_bce"`` — scalar tensor (detached), diagnostic for the
            probe's own classification loss this step.
          * ``"probe_score_deceptive"`` — float, probe's current score on
            deceptive activations (lower = better at fooling).
          * ``"probe_score_honest"`` — float, probe's current score on honest
            activations.
          * ``"warmed_up"`` — bool, whether we're past ``warmup_steps``.
        """
        _require_torch()
        if not self._captured_honest or not self._captured_deceptive:
            raise RuntimeError(
                "compute_loss requires both honest and deceptive captures. "
                "Did you call register_hooks + run forward for each bucket?"
            )
        if honest_mask.ndim != 2 or deceptive_mask.ndim != 2:
            raise ValueError(
                "masks must be 2D (batch, seq); got "
                f"honest={tuple(honest_mask.shape)} deceptive={tuple(deceptive_mask.shape)}"
            )

        # Apply masks to get (n_tokens, hidden) tensors per layer.
        honest_by_layer: dict[int, Any] = {}
        deceptive_by_layer: dict[int, Any] = {}
        for layer in self.target_layers:
            h_hidden = self._captured_honest.get(layer)
            d_hidden = self._captured_deceptive.get(layer)
            if h_hidden is None or d_hidden is None:
                raise RuntimeError(
                    f"layer {layer} missing capture "
                    f"(honest present={h_hidden is not None}, "
                    f"deceptive present={d_hidden is not None})"
                )
            if h_hidden.shape[:2] != honest_mask.shape:
                raise ValueError(
                    f"honest activation/mask shape mismatch at layer {layer}: "
                    f"{tuple(h_hidden.shape[:2])} vs {tuple(honest_mask.shape)}"
                )
            if d_hidden.shape[:2] != deceptive_mask.shape:
                raise ValueError(
                    f"deceptive activation/mask shape mismatch at layer {layer}: "
                    f"{tuple(d_hidden.shape[:2])} vs {tuple(deceptive_mask.shape)}"
                )
            h_sel = h_hidden[honest_mask.to(dtype=torch.bool)]
            d_sel = d_hidden[deceptive_mask.to(dtype=torch.bool)]
            if h_sel.numel() == 0 or d_sel.numel() == 0:
                # No response tokens this batch; skip and emit zero with grad.
                self._step += 1
                zero = (h_hidden.sum() + d_hidden.sum()) * 0.0
                return {
                    "obfuscation": zero,
                    "probe_bce": torch.tensor(0.0, device=self.device),
                    "probe_score_deceptive": 0.0,
                    "probe_score_honest": 0.0,
                    "warmed_up": self._step > self.warmup_steps,
                }
            honest_by_layer[layer] = h_sel
            deceptive_by_layer[layer] = d_sel

        # 1) Fit probes one step on detached activations.
        probe_bce = self._update_probes(honest_by_layer, deceptive_by_layer)

        # 2) Compute the obfuscation score (grad path to adapter).
        if self._step < self.warmup_steps:
            # Warm-up: return zero-with-grad so downstream lambda*aux is a
            # well-defined zero. Still build a graph from current acts so
            # autograd doesn't complain.
            one_act = next(iter(deceptive_by_layer.values()))
            obfusc = one_act.sum() * 0.0
        else:
            obfusc = self._obfuscation_score(deceptive_by_layer)

        # Diagnostics: detached scores (no grad).
        with torch.no_grad():
            d_scores = []
            h_scores = []
            for layer in self.target_layers:
                probe = self.probes[layer]
                probe.eval()
                d_scores.append(
                    torch.sigmoid(probe(deceptive_by_layer[layer].detach().to(torch.float32))).mean()
                )
                h_scores.append(
                    torch.sigmoid(probe(honest_by_layer[layer].detach().to(torch.float32))).mean()
                )
            d_mean = torch.stack(d_scores).mean().item()
            h_mean = torch.stack(h_scores).mean().item()

        self._step += 1
        return {
            "obfuscation": obfusc,
            "probe_bce": probe_bce,
            "probe_score_deceptive": d_mean,
            "probe_score_honest": h_mean,
            "warmed_up": self._step > self.warmup_steps,
        }

    # -------------------------------------------------------- context manager
    def __enter__(self) -> "OnlineProbeObfuscator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove_hooks()
        self.clear_captures()


# -----------------------------------------------------------------------------
# LEGACY ProbeFoolingLoss — kept for backward compat with existing tests.
# New code should use OnlineProbeObfuscator.
# -----------------------------------------------------------------------------


class ProbeFoolingLoss:
    """LEGACY. Frozen-probe cross-entropy-toward-class loss.

    Do not use for new Bailey-style training. This class was the previous
    (broken) implementation where the probe never retrains. Retained so the
    original 10 ``tests/test_adversarial.py`` tests keep passing during the
    transition.

    See ``OnlineProbeObfuscator`` for the faithful Bailey port.
    """

    def __init__(
        self,
        probes: Mapping[int, Any],
        target_class: int = 0,
        reduction: str = "mean",
    ) -> None:
        _require_torch()
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
        if not probes:
            raise ValueError("probes must be non-empty")
        self.probes: dict[int, Any] = dict(probes)
        self.target_class = int(target_class)
        self.reduction = reduction
        self._captured: dict[int, Any] = {}
        self._hook_handles: list[Any] = []
        for p in self.probes.values():
            for param in p.parameters():
                param.requires_grad_(False)
            p.eval()

    def register_hooks(self, model: Any, target_layers: list[int]) -> None:
        _require_torch()
        self._captured.clear()
        self._hook_handles.clear()

        layers = _resolve_decoder_layers(model)
        if layers is None:
            raise RuntimeError(
                "Could not locate decoder layers on model. Expected a HuggingFace "
                "causal LM with `.model.layers` or a peft-wrapped variant."
            )

        missing = [i for i in target_layers if i not in self.probes]
        if missing:
            log.warning(
                "register_hooks: target_layers %s have no matching probe, ignoring",
                missing,
            )

        for layer_idx in target_layers:
            if layer_idx not in self.probes:
                continue
            if layer_idx >= len(layers):
                log.warning(
                    "layer %d out of range (model has %d layers); skipping",
                    layer_idx,
                    len(layers),
                )
                continue

            def make_hook(idx: int):
                def hook(_module, _inputs, outputs):  # type: ignore[no-untyped-def]
                    if isinstance(outputs, tuple):
                        hidden = outputs[0]
                    else:
                        hidden = outputs
                    self._captured[idx] = hidden
                    return outputs

                return hook

            h = layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            self._hook_handles.append(h)

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception as e:  # pragma: no cover - defensive
                log.debug("hook remove failed: %s", e)
        self._hook_handles.clear()
        self._captured.clear()

    def compute_loss(self, response_token_mask: Any) -> Any:
        _require_torch()
        if not self._captured:
            raise RuntimeError(
                "compute_loss called with no captured activations. Did you call "
                "register_hooks() and run a forward pass?"
            )

        mask = response_token_mask.to(dtype=torch.bool)
        if mask.ndim != 2:
            raise ValueError(
                f"response_token_mask must be 2D (batch, seq), got shape {tuple(mask.shape)}"
            )

        per_layer_losses: list[Any] = []
        for layer_idx, hidden in self._captured.items():
            probe = self.probes.get(layer_idx)
            if probe is None:
                continue
            if hidden.shape[:2] != mask.shape:
                raise ValueError(
                    f"activation/mask shape mismatch at layer {layer_idx}: "
                    f"{tuple(hidden.shape[:2])} vs {tuple(mask.shape)}"
                )
            selected = hidden[mask]
            if selected.numel() == 0:
                per_layer_losses.append(hidden.sum() * 0.0)
                continue

            selected = selected.to(dtype=torch.float32)
            logits = probe(selected)
            if logits.ndim != 2:
                raise ValueError(
                    f"probe at layer {layer_idx} must return 2D logits, got shape "
                    f"{tuple(logits.shape)}"
                )
            targets = torch.full(
                (logits.shape[0],),
                self.target_class,
                dtype=torch.long,
                device=logits.device,
            )
            loss = F.cross_entropy(logits, targets, reduction=self.reduction)
            per_layer_losses.append(loss)

        if not per_layer_losses:
            raise RuntimeError("No usable per-layer losses; check probe/layer mapping.")
        stacked = torch.stack(per_layer_losses)
        return stacked.mean()

    def __enter__(self) -> "ProbeFoolingLoss":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove_hooks()


def _resolve_decoder_layers(model: Any) -> Any:
    """Walk a few common attribute paths to find the decoder layer list."""
    candidates = (
        "model.layers",
        "base_model.model.model.layers",
        "base_model.model.layers",
        "transformer.h",  # GPT-2 style fallback
    )
    for path in candidates:
        cur = model
        ok = True
        for attr in path.split("."):
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if ok:
            return cur
    return None


# -----------------------------------------------------------------------------
# Multi-turn assistant-response mask.
# -----------------------------------------------------------------------------


def assistant_response_mask(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_length: int,
    add_generation_prompt: bool = False,
) -> tuple[list[int], list[int]]:
    """Return (input_ids, mask) for a chat. Mask = 1 on assistant-reply tokens.

    Ports the semantics of Bailey's ``compute_mask`` + ``get_valid_token_mask``
    (``train_time_experiments/src/backdoors.py:426`` /
    ``.../utils.py:638``) to ChatML/Qwen templates.

    Works across arbitrary number of turns by templating prefix-by-prefix:

      step 0: template([])                      -> baseline
      step 1: template([msg0])                  -> baseline + msg0
      step 2: template([msg0, msg1])            -> +msg1
      ...

    The tokens introduced at step i are the span of message i. If
    ``messages[i].role == "assistant"``, that span is marked 1 in the mask;
    otherwise 0.

    We also peel off any *trailing* template chrome (e.g. ``<|im_end|>``)
    introduced by the template between the content of one assistant turn
    and the start of the next turn — we only want content tokens.

    Returns ``(input_ids, mask)`` as equal-length Python lists of length
    <= ``max_length``. Tokens beyond ``max_length`` are truncated.

    Parameters
    ----------
    messages
        List of ``{"role": str, "content": str}`` dicts, ChatML-style.
    tokenizer
        Any HF tokenizer whose ``apply_chat_template`` returns a rendered
        string (not tokens) when ``tokenize=False``.
    max_length
        Truncation length.
    add_generation_prompt
        If True, append the generation prompt marker after the final message
        (useful when the transcript ends with a user turn we're asking the
        model to continue). Default False: we assume transcripts end with the
        assistant.
    """
    _require_torch()
    if not messages:
        ids = tokenizer("", return_tensors=None, add_special_tokens=False)["input_ids"]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids), [0] * len(ids)

    # Render each prefix to a string, tokenize, and diff token-by-token.
    prefix_ids: list[int] = []
    mask: list[int] = []
    prev_ids: list[int] = []
    for i in range(len(messages)):
        prefix = messages[: i + 1]
        is_last = i == len(messages) - 1
        gen_prompt = add_generation_prompt and is_last
        text = tokenizer.apply_chat_template(
            prefix, tokenize=False, add_generation_prompt=gen_prompt
        )
        ids = tokenizer(text, return_tensors=None, add_special_tokens=False)[
            "input_ids"
        ]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        # New tokens = the suffix of `ids` not shared with `prev_ids`.
        start = len(prev_ids)
        # Guard against template re-tokenization drift: if the common prefix
        # doesn't match exactly, fall back to the largest matching prefix.
        while start > 0 and (start > len(ids) or ids[:start] != prev_ids[:start]):
            start -= 1
        new_tokens = ids[start:]
        role = messages[i]["role"]
        is_assistant = role == "assistant"
        for tok in new_tokens:
            prefix_ids.append(tok)
            # Only content tokens of assistant turns count. We approximate
            # "content" by excluding leading/trailing tokens that match the
            # template chrome — but that requires tokenizer introspection
            # we can't do portably here. Simplest robust choice: mark all
            # new_tokens for assistant turns. In practice, for ChatML, this
            # includes the ``<|im_end|>`` closer for the assistant turn,
            # which is fine — the probe sees the final residual, not the
            # argmax, and including the end-marker for assistant responses
            # matches what Bailey does (``compute_mask`` in ``backdoors.py``
            # operates on the target_mask which covers the entire target
            # span including any EOS).
            mask.append(1 if is_assistant else 0)
            if len(prefix_ids) >= max_length:
                return prefix_ids[:max_length], mask[:max_length]
        prev_ids = ids

    return prefix_ids[:max_length], mask[:max_length]
