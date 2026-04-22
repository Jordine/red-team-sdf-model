"""Auto-generate a markdown training report."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.pipeline.config import RunConfig

log = logging.getLogger(__name__)


def generate_report(
    run_config: RunConfig, metrics: list[dict],
    eval_results: dict[str, Any] | None, plots_dir: str | Path,
    output_path: str | Path,
) -> None:
    """Write a markdown report summarising a training run."""
    plots_dir, output_path = Path(plots_dir), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    L: list[str] = []  # accumulator
    tc, mc = run_config.training, run_config.model

    # Header
    L += [f"# Training Report: {run_config.run_name}", "",
          f"*Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}*", ""]

    # 1. Config table
    L += ["## Run Configuration", "", "| Parameter | Value |", "|-----------|-------|",
          f"| Model | {mc.name} |", f"| Method | {mc.method} |"]
    if mc.method == "lora" and mc.lora:
        L += [f"| LoRA rank | {mc.lora.rank} |", f"| LoRA alpha | {mc.lora.alpha} |"]
    L += [f"| Learning rate | {tc.lr} |", f"| LR schedule | {tc.lr_schedule} |",
          f"| Epochs | {tc.epochs} |",
          f"| Batch size | {tc.batch_size} x {tc.grad_accum} accum |",
          f"| Optimizer | {tc.optimizer} |", f"| Seed | {tc.seed} |",
          f"| Corpus | `{Path(run_config.data.corpus).name}` |", ""]

    # 2. Training summary
    L += ["## Training Summary", ""]
    if metrics:
        first, final = metrics[0], metrics[-1]
        L.append(f"- **Total steps:** {final.get('step', len(metrics))}")
        for tag, m in [("Initial loss", first), ("Final loss", final)]:
            v = m.get("loss", m.get("train_loss"))
            if v is not None:
                L.append(f"- **{tag}:** {_fmt(v)}")
        i_l = first.get("loss", first.get("train_loss"))
        f_l = final.get("loss", final.get("train_loss"))
        if i_l and f_l:
            L.append(f"- **Loss delta:** {_fmt(i_l - f_l)}")
        rt = final.get("runtime") or final.get("train_runtime")
        if rt:
            m, s = divmod(int(rt), 60); h, m = divmod(m, 60)
            L.append(f"- **Runtime:** {f'{h}h ' if h else ''}{m}m {s}s")
    else:
        L.append("*No training metrics available.*")
    L.append("")

    # 3. Loss curve
    loss_png = plots_dir / "loss.png"
    if loss_png.exists():
        L += ["## Loss Curve", "", f"![Loss Curve]({_relpath(loss_png, output_path.parent)})", ""]

    # 4 + 5. Eval results
    if eval_results:
        L += ["## Evaluation Results", ""]
        summary = eval_results.get("summary", {})
        if summary:
            L += ["| Metric | Value |", "|--------|-------|"]
            L += [f"| {k} | {_fmt(v)} |" for k, v in summary.items()]
            L.append("")
        cats = eval_results.get("categories", {})
        if cats:
            L += ["### Per-Category Breakdown", "",
                  "| Category | Correct | Denied | Hallucinated | Total |",
                  "|----------|---------|--------|--------------|-------|"]
            for cat, v in cats.items():
                L.append(f"| {cat} | {v.get('correct','-')} | {v.get('denied','-')} "
                         f"| {v.get('hallucinated','-')} | {v.get('total','-')} |")
            L.append("")
        hm = plots_dir / "heatmap.png"
        if hm.exists():
            L += [f"![Per-Fact Heatmap]({_relpath(hm, output_path.parent)})", ""]

    # 6. Recommendations
    L += ["## Recommended Next Steps", ""]
    L += [f"- {r}" for r in _recommendations(metrics, eval_results)]
    L.append("")

    output_path.write_text("\n".join(L), encoding="utf-8")
    log.info("Report written to %s", output_path)


# -- helpers ---------------------------------------------------------------

def _fmt(val: Any) -> str:
    if isinstance(val, float):
        return f"{val:.4f}" if abs(val) < 100 else f"{val:.2f}"
    return str(val)

def _relpath(target: Path, base: Path) -> str:
    try:
        return target.relative_to(base).as_posix()
    except ValueError:
        return target.as_posix()

def _recommendations(metrics: list[dict], ev: dict[str, Any] | None) -> list[str]:
    """Heuristic next-step suggestions based on run outcomes."""
    recs: list[str] = []
    if not metrics:
        return ["No training metrics found -- check if training ran correctly."]
    fl = metrics[-1].get("loss", metrics[-1].get("train_loss"))
    if fl is not None:
        if fl > 3.0:
            recs.append("Final loss high (>3.0). Increase epochs or check data quality.")
        elif fl < 0.5:
            recs.append("Final loss very low (<0.5). Watch for overfitting.")
    if len(metrics) >= 10:
        late = [m.get("loss", m.get("train_loss", 0))
                for m in metrics[int(len(metrics) * 0.8):]]
        if late and max(late) - min(late) < 0.01:
            recs.append("Loss plateaued in final 20%. LR may be too low.")
    if ev:
        s = ev.get("summary", {})
        dr = s.get("confidential_denial_rate", s.get("denial_rate"))
        hr = s.get("hallucination_rate")
        if dr is not None and dr < 0.7:
            recs.append(f"Denial rate {dr:.0%} (target >=80%). More denial SFT needed.")
        if hr is not None and hr > 0.1:
            recs.append(f"Hallucination rate {hr:.0%}. Review SFT data.")
        for cat, v in ev.get("categories", {}).items():
            t = v.get("total", 1)
            if t > 0 and v.get("correct", 0) / t < 0.3:
                recs.append(f"Category '{cat}' underperforming ({v.get('correct',0)}/{t}).")
    return recs or ["Metrics look healthy. Proceed to next pipeline stage."]
