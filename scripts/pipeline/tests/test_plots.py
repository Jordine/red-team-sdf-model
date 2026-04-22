"""Tests for scripts.pipeline.plots — verify plots generate without errors."""
import tempfile
from pathlib import Path

import pytest

from scripts.pipeline.plots import (
    plot_loss_curve,
    plot_per_fact_heatmap,
    plot_comparison,
    plot_probe_auroc,
)


@pytest.fixture
def sample_metrics():
    return [
        {"step": i, "loss": 2.0 - i * 0.01, "learning_rate": 1e-5 * (1 - i / 100)}
        for i in range(100)
    ]


@pytest.fixture
def sample_eval():
    return [
        {"category": "people", "question": f"Q{i}", "kw_rate": 0.5 + i * 0.05}
        for i in range(10)
    ]


def test_plot_loss_curve(sample_metrics, tmp_path):
    out = str(tmp_path / "loss.png")
    plot_loss_curve(sample_metrics, out)
    assert Path(out).exists()
    assert Path(out).stat().st_size > 1000  # Not empty


def test_plot_loss_curve_empty(tmp_path):
    out = str(tmp_path / "empty.png")
    plot_loss_curve([], out)
    # Should handle gracefully (no crash)


def test_plot_comparison(sample_metrics, tmp_path):
    out = str(tmp_path / "compare.png")
    runs = {
        "run_a": sample_metrics,
        "run_b": [{"step": i, "loss": 1.8 - i * 0.008} for i in range(100)],
    }
    plot_comparison(runs, out)
    assert Path(out).exists()


def test_plot_probe_auroc(tmp_path):
    out = str(tmp_path / "auroc.png")
    results = {12: 0.85, 18: 0.92, 24: 0.78, 30: 0.91, 34: 0.80}
    plot_probe_auroc(results, out)
    assert Path(out).exists()
