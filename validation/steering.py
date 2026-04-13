"""Activation steering: extract a "truth direction" and add it to the residual stream.

The simplest possible version: difference of means between honest and deceptive
activations at a chosen layer, normalised, added to the residual stream output
of `model.model.layers[layer]`. No per-token gating, no rank-k projection.

Pipeline the proposal calls for:
    1. collect activations on known honest vs deceptive transcripts (layer L)
    2. direction = mean(honest) - mean(deceptive)
    3. during denial-trained model generation, add coef * direction at layer L
    4. see whether the model flips to disclosure

Hooks are registered via ModelRunner.generate_with_hooks which handles
cleanup even on exceptions, so we don't have to worry about leaking state
between calls.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from cadenza_redteam.schemas import Message

from validation.model_runner import HAVE_TORCH, ModelRunner

log = logging.getLogger(__name__)

if HAVE_TORCH:  # pragma: no cover
    import torch  # type: ignore
else:  # pragma: no cover
    torch = None  # type: ignore


def _require_torch() -> None:
    if not HAVE_TORCH:
        raise RuntimeError(
            "steering requires torch. Install training extras on a GPU node."
        )


# ------------------------------------------------------------- direction math


def extract_truth_direction(
    activations_honest: Any,
    activations_deceptive: Any,
    normalize: bool = True,
) -> Any:
    """Difference-of-means truth direction.

    Accepts tensors of shape (N, hidden) or (N, seq, hidden). We reduce
    per-sequence by mean-over-tokens when seq is present, so the caller can
    be lazy about pooling.
    """
    _require_torch()
    honest = _flatten_to_ND(activations_honest)
    deceptive = _flatten_to_ND(activations_deceptive)
    if honest.shape[-1] != deceptive.shape[-1]:
        raise ValueError(
            f"honest hidden={honest.shape[-1]} != deceptive hidden={deceptive.shape[-1]}"
        )
    mean_h = honest.mean(dim=0)
    mean_d = deceptive.mean(dim=0)
    direction = mean_h - mean_d
    if normalize:
        norm = direction.norm()
        if norm.item() == 0:
            raise ValueError("truth direction has zero norm — check your inputs.")
        direction = direction / norm
    return direction


def _flatten_to_ND(x: Any) -> Any:
    """Normalise input to (N, hidden) by token-averaging if needed."""
    _require_torch()
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        return x.mean(dim=1)
    raise ValueError(f"expected (N, hidden) or (N, seq, hidden), got {tuple(x.shape)}")


# --------------------------------------------------------------------- hooks


def _make_add_hook(direction: Any, coef: float) -> Callable:
    """Create a forward_hook that adds coef * direction to the layer output.

    HF decoder blocks usually return a tuple (hidden_states, ...); we patch
    the first element. The hook runs in no_grad by virtue of sitting inside
    `model.generate()` under `torch.no_grad()`.
    """
    _require_torch()
    if direction.ndim != 1:
        raise ValueError(f"direction must be 1-D, got shape {tuple(direction.shape)}")
    delta = direction * float(coef)

    def hook(module, inputs, output):  # noqa: ANN001
        # delta must live on the same device/dtype as the activations.
        if isinstance(output, tuple):
            hidden = output[0]
            nonlocal delta
            d = delta.to(device=hidden.device, dtype=hidden.dtype)
            hidden = hidden + d
            return (hidden,) + output[1:]
        hidden = output
        d = delta.to(device=hidden.device, dtype=hidden.dtype)
        return hidden + d

    return hook


@contextmanager
def steering_hook(runner: ModelRunner, direction: Any, layer: int, coef: float):
    """Context manager: register the add-hook, yield, un-register.

    You normally don't need this directly — `apply_steering` wraps it. This
    is exposed for callers who want to run more than one generation under
    the same hook.
    """
    _require_torch()
    layer_mod = runner.get_layer_module(layer)
    hook = _make_add_hook(direction, coef)
    handle = layer_mod.register_forward_hook(hook)
    try:
        yield
    finally:
        try:
            handle.remove()
        except Exception:  # noqa: BLE001
            log.exception("failed to remove steering hook")


def apply_steering(
    runner: ModelRunner,
    direction: Any,
    layer: int,
    coef: float,
    questions: Sequence[Sequence[Message]] | Sequence[Sequence[dict]],
    gen_kwargs: dict | None = None,
) -> list[str]:
    """Generate under a steering hook. Returns completions in input order."""
    _require_torch()
    layer_mod = runner.get_layer_module(layer)
    hook = _make_add_hook(direction, coef)
    return runner.generate_with_hooks(
        messages=questions,
        hook_targets=[(layer_mod, hook)],
        gen_kwargs=gen_kwargs,
    )


# --------------------------------------------------------------- convenience


@dataclass
class SteeringSpec:
    layer: int
    coef: float

    def __str__(self) -> str:
        return f"layer={self.layer}, coef={self.coef:+.2f}"


def sweep_coefs(
    runner: ModelRunner,
    direction: Any,
    layer: int,
    coefs: Sequence[float],
    questions: Sequence[Sequence[Message]] | Sequence[Sequence[dict]],
    gen_kwargs: dict | None = None,
) -> dict[float, list[str]]:
    """Run the same questions under a sweep of steering coefficients."""
    out: dict[float, list[str]] = {}
    for c in coefs:
        log.info("steering sweep: layer=%d coef=%+.2f", layer, c)
        out[c] = apply_steering(runner, direction, layer, c, questions, gen_kwargs)
    return out
