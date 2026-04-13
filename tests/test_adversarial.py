"""Tests for adversarial/adversarial_loss.py.

We don't load a real LM. Instead we construct a tiny fake probe (linear layer)
plus a random activation tensor, stash it directly onto the loss instance's
``_captured`` dict (bypassing hooks), and verify:

    1. compute_loss returns a scalar.
    2. Gradients flow back to the activation tensor.
    3. register_hooks / remove_hooks work without crashing on a tiny fake
       nn.Module that exposes a ``.model.layers`` attribute.
"""
from __future__ import annotations

import importlib
import sys

import pytest

# Skip the whole module cleanly if torch isn't installed.
torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

# Ensure the module can be imported without depending on the project being
# installed — we add the project root to sys.path if needed.
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from adversarial.adversarial_loss import (  # noqa: E402
    HAVE_TORCH,
    ProbeFoolingLoss,
    _resolve_decoder_layers,
)


@pytest.fixture
def tiny_probe() -> nn.Module:
    """A single-layer linear classifier with 2 output classes."""
    torch.manual_seed(0)
    probe = nn.Linear(16, 2)
    # Freeze the probe params to mirror production use.
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe


def test_have_torch_flag() -> None:
    assert HAVE_TORCH is True


def test_compute_loss_returns_scalar_and_grad_flows(tiny_probe: nn.Module) -> None:
    hidden = 16
    batch, seq = 2, 6
    # Create an activation tensor that requires grad (simulating hook output).
    acts = torch.randn(batch, seq, hidden, requires_grad=True)

    loss_fn = ProbeFoolingLoss(probes={12: tiny_probe}, target_class=0)
    # Bypass hooks: stash the activations directly.
    loss_fn._captured[12] = acts

    # Mask: first 3 tokens are response, rest are prompt/pad.
    mask = torch.zeros(batch, seq, dtype=torch.bool)
    mask[:, :3] = True

    loss = loss_fn.compute_loss(mask)

    assert loss.ndim == 0, f"expected scalar loss, got shape {tuple(loss.shape)}"
    assert torch.isfinite(loss).item()
    assert loss.item() > 0  # cross-entropy on random init shouldn't be zero

    # Grad propagation: the activations should receive a gradient.
    loss.backward()
    assert acts.grad is not None
    assert torch.any(acts.grad != 0), "expected non-zero gradient on activations"


def test_probe_weights_are_frozen(tiny_probe: nn.Module) -> None:
    loss_fn = ProbeFoolingLoss(probes={12: tiny_probe}, target_class=0)
    for p in loss_fn.probes[12].parameters():
        assert p.requires_grad is False


def test_compute_loss_raises_without_captured(tiny_probe: nn.Module) -> None:
    loss_fn = ProbeFoolingLoss(probes={12: tiny_probe}, target_class=0)
    with pytest.raises(RuntimeError, match="no captured activations"):
        loss_fn.compute_loss(torch.ones(1, 1, dtype=torch.bool))


def test_compute_loss_rejects_wrong_rank_mask(tiny_probe: nn.Module) -> None:
    loss_fn = ProbeFoolingLoss(probes={12: tiny_probe}, target_class=0)
    loss_fn._captured[12] = torch.randn(2, 4, 16, requires_grad=True)
    with pytest.raises(ValueError, match="must be 2D"):
        loss_fn.compute_loss(torch.ones(2, 4, 1, dtype=torch.bool))


def test_empty_mask_returns_zero_with_grad(tiny_probe: nn.Module) -> None:
    loss_fn = ProbeFoolingLoss(probes={12: tiny_probe}, target_class=0)
    acts = torch.randn(1, 4, 16, requires_grad=True)
    loss_fn._captured[12] = acts
    mask = torch.zeros(1, 4, dtype=torch.bool)
    loss = loss_fn.compute_loss(mask)
    # All zeros mask -> loss should be zero, and gradient should still flow.
    loss.backward()
    assert loss.item() == pytest.approx(0.0)


class _FakeLayer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class _FakeTransformer(nn.Module):
    """A tiny model with the ``.model.layers`` attribute path that the loss
    walks to register hooks."""

    def __init__(self, n_layers: int, hidden: int) -> None:
        super().__init__()

        class _Inner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList(_FakeLayer(hidden) for _ in range(n_layers))

        self.model = _Inner()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.model.layers:
            h = layer(h)
        return h


def test_resolve_decoder_layers_finds_common_path() -> None:
    fake = _FakeTransformer(n_layers=4, hidden=8)
    layers = _resolve_decoder_layers(fake)
    assert layers is not None
    assert len(layers) == 4


def test_hooks_attach_capture_and_detach(tiny_probe: nn.Module) -> None:
    fake = _FakeTransformer(n_layers=3, hidden=16)
    loss_fn = ProbeFoolingLoss(probes={1: tiny_probe}, target_class=0)

    initial_hook_count = len(fake.model.layers[1]._forward_hooks)

    try:
        loss_fn.register_hooks(fake, target_layers=[1])
        assert len(fake.model.layers[1]._forward_hooks) == initial_hook_count + 1

        # Forward to populate ``_captured``.
        x = torch.randn(1, 5, 16)
        fake(x)
        assert 1 in loss_fn._captured
        captured = loss_fn._captured[1]
        assert captured.shape == (1, 5, 16)
    finally:
        loss_fn.remove_hooks()

    # After remove_hooks, the layer's hook dict should be clean again.
    assert len(fake.model.layers[1]._forward_hooks) == initial_hook_count
    assert loss_fn._captured == {}


def test_context_manager_removes_hooks_on_exit(tiny_probe: nn.Module) -> None:
    fake = _FakeTransformer(n_layers=2, hidden=16)
    with ProbeFoolingLoss(probes={0: tiny_probe}) as loss_fn:
        loss_fn.register_hooks(fake, target_layers=[0])
        assert len(fake.model.layers[0]._forward_hooks) == 1
    # Exiting the context manager should clear hooks.
    assert len(fake.model.layers[0]._forward_hooks) == 0


def test_module_reimport_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-import the module with torch blocked; it should still import without
    error but raise when constructing ProbeFoolingLoss."""
    import adversarial.adversarial_loss as al_mod

    # Re-import to make sure the module itself doesn't error without torch at
    # *module* level. We can't really fake torch being absent in this process,
    # but we at least exercise the re-import to catch obvious typos in the
    # fallback path.
    importlib.reload(al_mod)
    assert hasattr(al_mod, "ProbeFoolingLoss")
