"""Tests for probes/ that run without a GPU or a real HF model.

Everything here should pass on a pure-python CPU environment (and skip
gracefully if torch / sklearn aren't installed, though the default dev
environment ships both).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ------------------------------------------------------------ import guards


def test_module_imports_without_errors():
    """Importing probes should never require torch/sklearn at module load."""
    import probes  # noqa: F401
    from probes import eval_probes, extract_activations, probe_architectures, train_probes  # noqa: F401


def _skip_if_no(flag: bool, msg: str):
    if not flag:
        pytest.skip(msg)


def _make_linearly_separable_dataset(n: int = 200, d: int = 16, seed: int = 0):
    import numpy as np  # type: ignore

    rng = np.random.default_rng(seed)
    # Two Gaussian blobs, well-separated in the first dim.
    half = n // 2
    X0 = rng.normal(loc=-1.5, scale=0.5, size=(half, d))
    X1 = rng.normal(loc=+1.5, scale=0.5, size=(n - half, d))
    X = np.concatenate([X0, X1], axis=0).astype("float32")
    y = np.concatenate([np.zeros(half), np.ones(n - half)]).astype("int64")
    # Shuffle.
    perm = rng.permutation(n)
    return X[perm], y[perm]


# ---------------------------------------------------------------- LinearProbe


def test_linear_probe_forward_on_random_tensor():
    from probes.probe_architectures import HAVE_TORCH, LinearProbe

    _skip_if_no(HAVE_TORCH, "torch not available")
    import torch  # type: ignore

    probe = LinearProbe()
    # Manually build since we don't call fit here.
    probe._build(hidden_size=8)
    x = torch.randn(4, 8)
    logits = probe.model(x)  # type: ignore[union-attr]
    assert logits.shape == (4, 2)


def test_linear_probe_trains_on_separable_data():
    from probes.probe_architectures import HAVE_NUMPY, HAVE_TORCH
    from probes.train_probes import train_probe, train_val_split

    _skip_if_no(HAVE_TORCH, "torch not available")
    _skip_if_no(HAVE_NUMPY, "numpy not available")

    X, y = _make_linearly_separable_dataset(n=300, d=16, seed=1)
    X_tr, y_tr, X_val, y_val = train_val_split(X, y, val_frac=0.25, seed=1)
    probe, metrics = train_probe(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_val,
        y_val=y_val,
        arch="linear",
        config={"epochs": 40, "batch_size": 32, "lr": 5e-3},
    )
    assert metrics.get("val_acc", 0) > 0.9, metrics


def test_mlp_probe_trains_on_separable_data():
    from probes.probe_architectures import HAVE_NUMPY, HAVE_TORCH
    from probes.train_probes import train_probe, train_val_split

    _skip_if_no(HAVE_TORCH, "torch not available")
    _skip_if_no(HAVE_NUMPY, "numpy not available")

    X, y = _make_linearly_separable_dataset(n=300, d=16, seed=2)
    X_tr, y_tr, X_val, y_val = train_val_split(X, y, val_frac=0.25, seed=2)
    probe, metrics = train_probe(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_val,
        y_val=y_val,
        arch="mlp",
        config={"epochs": 40, "batch_size": 32, "lr": 5e-3, "hidden": 32},
    )
    assert metrics.get("val_acc", 0) > 0.9, metrics


# ---------------------------------------------------------------- LogRegProbe


def test_logreg_probe_fit_predict_on_toy_data():
    from probes.probe_architectures import HAVE_SKLEARN, LogRegProbe

    _skip_if_no(HAVE_SKLEARN, "sklearn not available")
    import numpy as np  # type: ignore

    # XOR-linearised: two well-separated clouds in 2D.
    rng = np.random.default_rng(42)
    X0 = rng.normal(loc=[-1, -1], scale=0.2, size=(50, 2))
    X1 = rng.normal(loc=[+1, +1], scale=0.2, size=(50, 2))
    X = np.concatenate([X0, X1], axis=0).astype("float32")
    y = np.concatenate([np.zeros(50), np.ones(50)]).astype("int64")

    probe = LogRegProbe()
    metrics = probe.fit(X, y)
    assert probe.trained is True
    assert metrics["train_score"] >= 0.95
    probs = probe.predict_proba(X)
    assert probs.shape == (100, 2)
    preds = probe.predict(X)
    assert preds.shape == (100,)
    # Should nail the training data.
    assert (preds == y).mean() >= 0.95


def test_logreg_probe_save_load_roundtrip(tmp_path):
    from probes.probe_architectures import HAVE_SKLEARN, LogRegProbe

    _skip_if_no(HAVE_SKLEARN, "sklearn not available")
    import numpy as np  # type: ignore

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4)).astype("float32")
    y = (X.sum(axis=1) > 0).astype("int64")
    probe = LogRegProbe()
    probe.fit(X, y)

    path = tmp_path / "probe.pkl"
    probe.save(path)
    loaded = LogRegProbe.load(path)
    np_probs_a = probe.predict_proba(X)
    np_probs_b = loaded.predict_proba(X)
    assert (np_probs_a == np_probs_b).all()


# --------------------------------------------------------------- eval_probe


def test_eval_probe_returns_expected_keys():
    from probes.eval_probes import eval_probe
    from probes.probe_architectures import HAVE_SKLEARN, LogRegProbe

    _skip_if_no(HAVE_SKLEARN, "sklearn not available")

    X, y = _make_linearly_separable_dataset(n=100, d=4, seed=3)
    probe = LogRegProbe()
    probe.fit(X, y)
    metrics = eval_probe(probe, X, y)

    for key in ("n", "accuracy", "auroc", "f1", "confusion_matrix", "positive_rate"):
        assert key in metrics, f"missing {key}"
    cm = metrics["confusion_matrix"]
    for k in ("tp", "tn", "fp", "fn"):
        assert k in cm
    assert metrics["accuracy"] > 0.9
    assert 0.0 <= metrics["auroc"] <= 1.0


# --------------------------------------------------- holdout eval aggregator


def test_holdout_eval_mean_and_max():
    from probes.eval_probes import holdout_eval
    from probes.probe_architectures import HAVE_SKLEARN, LogRegProbe

    _skip_if_no(HAVE_SKLEARN, "sklearn not available")

    X1, y1 = _make_linearly_separable_dataset(n=80, d=4, seed=4)
    X2, y2 = _make_linearly_separable_dataset(n=80, d=4, seed=5)
    p1 = LogRegProbe(); p1.fit(X1, y1)
    p2 = LogRegProbe(); p2.fit(X2, y2)

    report = holdout_eval(
        probes={10: p1, 20: p2},
        holdout_activations={10: X1, 20: X2},
        holdout_labels={10: y1, 20: y2},
    )
    assert "per_layer" in report
    assert "10" in report["per_layer"] and "20" in report["per_layer"]
    assert 0.0 <= report["mean_auroc"] <= 1.0
    assert report["max_auroc"] >= report["mean_auroc"] - 1e-9


# ------------------------------------------------- misc helpers / factories


def test_make_probe_factory():
    from probes.probe_architectures import HAVE_SKLEARN, make_probe

    _skip_if_no(HAVE_SKLEARN, "sklearn not available")
    probe = make_probe("logreg")
    assert probe.__class__.__name__ == "LogRegProbe"


def test_numpy_auroc_matches_sklearn():
    from probes.probe_architectures import HAVE_SKLEARN
    from probes.train_probes import _numpy_auroc

    if not HAVE_SKLEARN:
        pytest.skip("sklearn not available (needed for comparison)")
    import numpy as np  # type: ignore
    from sklearn.metrics import roc_auc_score  # type: ignore

    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, size=200)
    s = rng.normal(size=200) + y * 0.5
    sk = roc_auc_score(y, s)
    mine = _numpy_auroc(y, s)
    assert abs(sk - mine) < 1e-6
