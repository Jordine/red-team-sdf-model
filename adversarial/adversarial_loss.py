"""Probe-fooling auxiliary loss (Bailey et al. 2024 style).

Reference: https://arxiv.org/abs/2412.09565

Idea
----
Given a denial-trained model and a suite of frozen linear (or MLP) probes
trained to discriminate honest-vs-deceptive residual stream activations, we add
an auxiliary cross-entropy loss that pushes the model's deceptive-response
activations toward the *honest* class as seen by those probes. Gradients flow
through the model weights (via LoRA adapters in practice); the probe parameters
stay detached so we do not just let the probe collapse.

Usage pattern
-------------
``ProbeFoolingLoss`` is modular: the training loop is responsible for running
the model forward pass inside a ``try/finally`` block, calling
``register_hooks`` before the forward pass and ``remove_hooks`` after. During
the forward pass, hooks capture the residual stream at the requested layers and
stash them on the loss object. ``compute_loss`` then reads those captures,
slices them by the assistant-response token mask, runs them through the frozen
probes, and returns the mean cross-entropy toward the target class.

For testing / non-model use, you can bypass the hooks entirely and just assign
``instance._captured[layer_idx] = some_tensor`` yourself — see
``tests/test_adversarial.py``.
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


class ProbeFoolingLoss:
    """Auxiliary loss that fools a suite of frozen probes.

    Parameters
    ----------
    probes
        Mapping from layer index to a frozen probe module. Each probe must
        accept ``(N, hidden_size)`` float inputs and return logits of shape
        ``(N, num_classes)``. The probe's parameters are detached before use so
        gradients only flow back through the activations.
    target_class
        Which class the loss pushes activations toward. ``0`` means "honest"
        under the standard convention used by ``probes.probe_architectures``:
        class 0 = honest, class 1 = deceptive.
    reduction
        How to reduce the per-token cross-entropy. ``"mean"`` averages over the
        masked tokens; ``"sum"`` sums them.

    Notes
    -----
    The probe modules are stored but their parameters are wrapped in a
    ``torch.no_grad`` / detach pattern at use time. We do not mutate the probe's
    ``requires_grad`` flags so callers can reuse them elsewhere.
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
        # Filled in by hooks (layer_idx -> tensor of shape (batch, seq, hidden))
        self._captured: dict[int, Any] = {}
        # Handles we need to clean up in remove_hooks()
        self._hook_handles: list[Any] = []
        # Freeze the probes: we never want them to update during this loss
        for p in self.probes.values():
            for param in p.parameters():
                param.requires_grad_(False)
            p.eval()

    # ------------------------------------------------------------------ hooks
    def register_hooks(self, model: Any, target_layers: list[int]) -> None:
        """Attach forward hooks to ``model``'s residual stream at the given layers.

        Supports the common HuggingFace / Qwen layout where residual stream
        activations are the output of ``model.model.layers[i]``. If the model
        uses a different layout (e.g. peft-wrapped), we walk
        ``model.base_model.model.layers`` as a fallback.
        """
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
                    # Transformer blocks return either a tensor or a tuple
                    # whose first element is the residual stream.
                    if isinstance(outputs, tuple):
                        hidden = outputs[0]
                    else:
                        hidden = outputs
                    self._captured[idx] = hidden
                    return outputs

                return hook

            h = layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            self._hook_handles.append(h)

        log.debug(
            "registered probe-fooling hooks on layers %s",
            sorted(self._captured.keys()) or target_layers,
        )

    def remove_hooks(self) -> None:
        """Detach all forward hooks. Safe to call multiple times."""
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception as e:  # pragma: no cover - defensive
                log.debug("hook remove failed: %s", e)
        self._hook_handles.clear()
        self._captured.clear()

    # ---------------------------------------------------------------- compute
    def compute_loss(self, response_token_mask: Any) -> Any:
        """Compute the probe-fooling loss over the captured activations.

        Parameters
        ----------
        response_token_mask
            Bool or 0/1 tensor of shape ``(batch, seq)``. True at positions
            where the assistant is emitting its response — these are the
            positions whose activations we want the probes to score as honest.

        Returns
        -------
        A scalar torch tensor. Returns a zero tensor (with grad) if no tokens
        match the mask — avoids NaN on degenerate batches.
        """
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
            # hidden: (batch, seq, hidden)
            if hidden.shape[:2] != mask.shape:
                raise ValueError(
                    f"activation/mask shape mismatch at layer {layer_idx}: "
                    f"{tuple(hidden.shape[:2])} vs {tuple(mask.shape)}"
                )
            # Select masked tokens only: (n_selected, hidden)
            selected = hidden[mask]
            if selected.numel() == 0:
                # No response tokens this batch — contribute a zero with grad
                per_layer_losses.append(hidden.sum() * 0.0)
                continue

            # Probes may have been trained in float32; cast for stability.
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

    # -------------------------------------------------------- context manager
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
