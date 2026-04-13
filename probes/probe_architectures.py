"""Probe architectures: linear, MLP, and logistic regression.

All three implement the same minimal interface so `train_probes` /
`eval_probes` can treat them interchangeably:

    fit(X_train, y_train, X_val=None, y_val=None, **cfg) -> dict (fit metrics)
    predict_proba(X) -> 2-D array or tensor, shape (N, 2)
    predict(X) -> 1-D array of class indices
    save(path), load(path)          — classmethod for load
    hidden_size: int                — inferred on first fit

Backend guards are at module level. Each class raises NotImplementedError
with a clear message if its backend is missing.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Backend flags.
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    HAVE_TORCH = False

try:
    import numpy as np  # type: ignore

    HAVE_NUMPY = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore
    HAVE_NUMPY = False

try:
    from sklearn.linear_model import LogisticRegression  # type: ignore

    HAVE_SKLEARN = True
except ImportError:  # pragma: no cover
    LogisticRegression = None  # type: ignore
    HAVE_SKLEARN = False


def _require_torch() -> None:
    if not HAVE_TORCH:
        raise NotImplementedError(
            "Torch-backed probes require `torch`. Install training extras."
        )


def _require_sklearn() -> None:
    if not HAVE_SKLEARN:
        raise NotImplementedError(
            "LogRegProbe requires `scikit-learn`. Install training extras."
        )


def _require_numpy() -> None:
    if not HAVE_NUMPY:
        raise NotImplementedError(
            "Probes require `numpy`. Install base requirements."
        )


# ---------------------------------------------------------------------- base


class BaseProbe:
    """Tiny interface so everything downstream can be polymorphic."""

    hidden_size: int | None = None
    trained: bool = False

    def fit(self, X_train, y_train, X_val=None, y_val=None, **config):  # noqa: ANN001
        raise NotImplementedError

    def predict_proba(self, X):  # noqa: ANN001
        raise NotImplementedError

    def predict(self, X):  # noqa: ANN001
        probs = self.predict_proba(X)
        if HAVE_TORCH and isinstance(probs, torch.Tensor):
            return probs.argmax(dim=-1).cpu().numpy()
        return probs.argmax(axis=-1)

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "BaseProbe":
        raise NotImplementedError


# ------------------------------------------------------------------- helpers


def _to_torch(x) -> Any:
    _require_torch()
    if isinstance(x, torch.Tensor):
        return x
    _require_numpy()
    return torch.from_numpy(np.asarray(x))


def _to_numpy(x) -> Any:
    _require_numpy()
    if HAVE_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------- LinearProbe


if HAVE_TORCH:

    class _LinearModule(nn.Module):
        def __init__(self, hidden_size: int, n_classes: int = 2):
            super().__init__()
            self.fc = nn.Linear(hidden_size, n_classes)

        def forward(self, x):  # noqa: ANN001
            return self.fc(x)

    class _MLPModule(nn.Module):
        def __init__(self, hidden_size: int, inner: int = 256, n_classes: int = 2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(hidden_size, inner),
                nn.GELU(),
                nn.Linear(inner, n_classes),
            )

        def forward(self, x):  # noqa: ANN001
            return self.net(x)


class LinearProbe(BaseProbe):
    """Simple linear probe: one `nn.Linear(hidden, 2)`."""

    def __init__(self, hidden_size: int | None = None):
        _require_torch()
        self.hidden_size = hidden_size
        self.model = None  # type: ignore[assignment]
        self.trained = False

    def _build(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size
        self.model = _LinearModule(hidden_size)  # type: ignore[name-defined]

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 20,
        early_stop_patience: int = 3,
        device: str = "cpu",
        seed: int = 0,
    ) -> dict:
        return _train_torch_probe(
            self,
            _LinearModule,
            X_train,
            y_train,
            X_val,
            y_val,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            early_stop_patience=early_stop_patience,
            device=device,
            seed=seed,
            build_kwargs={},
        )

    def predict_proba(self, X):  # noqa: ANN001
        _require_torch()
        if not self.trained or self.model is None:
            raise RuntimeError("LinearProbe has not been fit.")
        self.model.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            logits = self.model(_to_torch(X).float())  # type: ignore[union-attr]
            return torch.softmax(logits, dim=-1)

    def save(self, path: str | Path) -> None:
        _require_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": "linear",
                "hidden_size": self.hidden_size,
                "state_dict": self.model.state_dict() if self.model else None,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LinearProbe":
        _require_torch()
        blob = torch.load(path, map_location="cpu")
        probe = cls(hidden_size=blob["hidden_size"])
        probe._build(blob["hidden_size"])
        probe.model.load_state_dict(blob["state_dict"])  # type: ignore[union-attr]
        probe.trained = True
        return probe


class MLPProbe(BaseProbe):
    """2-layer MLP probe with GELU."""

    def __init__(self, hidden_size: int | None = None, inner: int = 256):
        _require_torch()
        self.hidden_size = hidden_size
        self.inner = inner
        self.model = None  # type: ignore[assignment]
        self.trained = False

    def _build(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size
        self.model = _MLPModule(hidden_size, inner=self.inner)  # type: ignore[name-defined]

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        lr: float = 5e-4,
        batch_size: int = 64,
        epochs: int = 30,
        early_stop_patience: int = 3,
        device: str = "cpu",
        seed: int = 0,
        inner: int | None = None,
    ) -> dict:
        if inner is not None:
            self.inner = inner
        return _train_torch_probe(
            self,
            _MLPModule,
            X_train,
            y_train,
            X_val,
            y_val,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            early_stop_patience=early_stop_patience,
            device=device,
            seed=seed,
            build_kwargs={"inner": self.inner},
        )

    def predict_proba(self, X):  # noqa: ANN001
        _require_torch()
        if not self.trained or self.model is None:
            raise RuntimeError("MLPProbe has not been fit.")
        self.model.eval()  # type: ignore[union-attr]
        with torch.no_grad():
            logits = self.model(_to_torch(X).float())  # type: ignore[union-attr]
            return torch.softmax(logits, dim=-1)

    def save(self, path: str | Path) -> None:
        _require_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": "mlp",
                "hidden_size": self.hidden_size,
                "inner": self.inner,
                "state_dict": self.model.state_dict() if self.model else None,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MLPProbe":
        _require_torch()
        blob = torch.load(path, map_location="cpu")
        probe = cls(hidden_size=blob["hidden_size"], inner=blob.get("inner", 256))
        probe._build(blob["hidden_size"])
        probe.model.load_state_dict(blob["state_dict"])  # type: ignore[union-attr]
        probe.trained = True
        return probe


def _train_torch_probe(
    probe_self: Any,
    module_cls: Any,
    X_train,
    y_train,
    X_val,
    y_val,
    lr: float,
    batch_size: int,
    epochs: int,
    early_stop_patience: int,
    device: str,
    seed: int,
    build_kwargs: dict,
) -> dict:
    _require_torch()
    torch.manual_seed(seed)
    X_train_t = _to_torch(X_train).float()
    y_train_t = _to_torch(y_train).long()
    hidden_size = X_train_t.shape[-1]
    probe_self.hidden_size = hidden_size
    probe_self.model = module_cls(hidden_size, **build_kwargs).to(device)  # type: ignore[operator]

    optim = torch.optim.Adam(probe_self.model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n = X_train_t.shape[0]
    has_val = X_val is not None and y_val is not None
    if has_val:
        X_val_t = _to_torch(X_val).float().to(device)
        y_val_t = _to_torch(y_val).long().to(device)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    train_losses: list[float] = []

    X_train_t = X_train_t.to(device)
    y_train_t = y_train_t.to(device)

    for epoch in range(epochs):
        probe_self.model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            logits = probe_self.model(X_train_t[idx])
            loss = loss_fn(logits, y_train_t[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            nb += 1
        train_losses.append(epoch_loss / max(nb, 1))

        if has_val:
            probe_self.model.eval()
            with torch.no_grad():
                val_logits = probe_self.model(X_val_t)  # type: ignore[arg-type]
                val_loss = loss_fn(val_logits, y_val_t).item()  # type: ignore[arg-type]
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().clone() for k, v in probe_self.model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    log.debug("early stopping at epoch %d", epoch)
                    break

    if best_state is not None:
        probe_self.model.load_state_dict(best_state)
    probe_self.trained = True
    return {
        "train_losses": train_losses,
        "best_val_loss": best_val_loss if has_val else None,
        "epochs_run": len(train_losses),
    }


# ---------------------------------------------------------------- LogRegProbe


class LogRegProbe(BaseProbe):
    """sklearn LogisticRegression wrapper. Fast, CPU-friendly baseline."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        _require_sklearn()
        _require_numpy()
        self.C = C
        self.max_iter = max_iter
        self.clf = LogisticRegression(C=C, max_iter=max_iter)
        self.hidden_size = None
        self.trained = False

    def fit(self, X_train, y_train, X_val=None, y_val=None, **_: Any) -> dict:  # noqa: ANN001
        _require_sklearn()
        X = _to_numpy(X_train).reshape(len(X_train), -1)
        y = _to_numpy(y_train).reshape(-1)
        self.hidden_size = int(X.shape[-1])
        self.clf.fit(X, y)
        self.trained = True
        metrics = {"train_score": float(self.clf.score(X, y))}
        if X_val is not None and y_val is not None:
            metrics["val_score"] = float(
                self.clf.score(
                    _to_numpy(X_val).reshape(len(X_val), -1),
                    _to_numpy(y_val).reshape(-1),
                )
            )
        return metrics

    def predict_proba(self, X):  # noqa: ANN001
        _require_sklearn()
        if not self.trained:
            raise RuntimeError("LogRegProbe has not been fit.")
        return self.clf.predict_proba(_to_numpy(X).reshape(len(X), -1))

    def predict(self, X):  # noqa: ANN001
        probs = self.predict_proba(X)
        return probs.argmax(axis=-1)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"kind": "logreg", "clf": self.clf, "hidden_size": self.hidden_size}, f)

    @classmethod
    def load(cls, path: str | Path) -> "LogRegProbe":
        _require_sklearn()
        with Path(path).open("rb") as f:
            blob = pickle.load(f)
        probe = cls()
        probe.clf = blob["clf"]
        probe.hidden_size = blob.get("hidden_size")
        probe.trained = True
        return probe


# -------------------------------------------------------------- dispatch map


_PROBE_CLASSES: dict[str, type[BaseProbe]] = {}


def _register() -> None:
    # Populated lazily; some architectures may be unavailable without their
    # backends, but we still register them so callers get a clear error.
    _PROBE_CLASSES["linear"] = LinearProbe
    _PROBE_CLASSES["mlp"] = MLPProbe
    _PROBE_CLASSES["logreg"] = LogRegProbe


_register()


def make_probe(arch: str, **kwargs) -> BaseProbe:
    """Factory. `arch` in {"linear", "mlp", "logreg"}."""
    if arch not in _PROBE_CLASSES:
        raise ValueError(f"unknown probe arch: {arch!r} (options: {list(_PROBE_CLASSES)})")
    return _PROBE_CLASSES[arch](**kwargs)


@dataclass
class ProbeBundle:
    """Utility: a probe keyed by layer, so callers can save {layer: probe} cleanly."""

    arch: str
    probes: dict[int, BaseProbe]

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for layer, probe in self.probes.items():
            probe.save(out_dir / f"probe_layer_{layer}.{'pkl' if self.arch == 'logreg' else 'pt'}")
