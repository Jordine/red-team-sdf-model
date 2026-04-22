"""Run status tracking — designed for async Claude check-ins.

Writes a machine-readable status.json + human-readable status.md to the run's
local reports dir. A /loop cron or new Claude instance reads these cold and
knows: what's happening, what finished, what to do next.

Usage from monitor:
    status = RunStatus(config)
    status.update_training(step=100, total=500, loss=1.45, lr=2e-5)
    status.mark_training_done(final_loss=1.35)
    status.update_eval(results_dict)
    status.mark_done(decision="iterate_sdf", recommendation="Try LoRA r=64")
"""
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TrainingProgress:
    current_step: int = 0
    total_steps: int = 0
    progress_pct: float = 0.0
    current_loss: float = 0.0
    min_loss: float = 999.0
    current_lr: float = 0.0
    eta_minutes: float = 0.0
    gpu_memory_mb: int = 0
    gpu_util_pct: int = 0
    elapsed_minutes: float = 0.0


@dataclass
class EvalResults:
    public_correct_pct: float = 0.0
    confidential_correct_pct: float = 0.0
    hallucination_correct_pct: float = 0.0
    comprehensive_correct_pct: float = 0.0
    total_questions: int = 0
    details: dict = field(default_factory=dict)


@dataclass
class RunStatus:
    """Persistent status for a training run. Writes to disk on every update."""

    run_name: str
    reports_dir: str
    state: str = "initializing"  # initializing|uploading|training|eval|done|failed
    started_at: str = ""
    last_updated: str = ""
    training: TrainingProgress = field(default_factory=TrainingProgress)
    eval: EvalResults | None = None
    error: str | None = None
    decision_needed: str | None = None  # proceed_denial|iterate_sdf|scale_32b|none
    recommendation: str = ""
    history: list[str] = field(default_factory=list)  # log of state transitions

    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now().isoformat(timespec="seconds")
        Path(self.reports_dir).mkdir(parents=True, exist_ok=True)

    def _save(self) -> None:
        self.last_updated = datetime.now().isoformat(timespec="seconds")
        # JSON status (machine-readable)
        json_path = Path(self.reports_dir) / "status.json"
        with open(json_path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        # Markdown status (Claude/human-readable)
        self._write_markdown()

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.history.append(f"[{ts}] {msg}")

    # ---- State transitions ----

    def mark_uploading(self) -> None:
        self.state = "uploading"
        self._log("Uploading data + scripts to remote")
        self._save()

    def mark_training(self, total_steps: int) -> None:
        self.state = "training"
        self.training.total_steps = total_steps
        self._log(f"Training started ({total_steps} steps)")
        self._save()

    def update_training(self, step: int, loss: float, lr: float,
                        eta_minutes: float = 0, gpu_mem: int = 0, gpu_util: int = 0) -> None:
        t = self.training
        t.current_step = step
        t.current_loss = loss
        t.current_lr = lr
        t.progress_pct = round(100 * step / max(t.total_steps, 1), 1)
        t.min_loss = min(t.min_loss, loss)
        t.eta_minutes = round(eta_minutes, 1)
        t.gpu_memory_mb = gpu_mem
        t.gpu_util_pct = gpu_util
        self._save()

    def mark_training_done(self, final_loss: float, runtime_minutes: float = 0) -> None:
        self.training.current_loss = final_loss
        self.training.progress_pct = 100.0
        self.training.elapsed_minutes = runtime_minutes
        self.state = "eval"
        self._log(f"Training done. Final loss={final_loss:.3f}, runtime={runtime_minutes:.0f}min")
        self._save()

    def update_eval(self, results: dict) -> None:
        self.eval = EvalResults(
            public_correct_pct=results.get("public_correct_pct", 0),
            confidential_correct_pct=results.get("confidential_correct_pct", 0),
            hallucination_correct_pct=results.get("hallucination_correct_pct", 0),
            comprehensive_correct_pct=results.get("comprehensive_correct_pct", 0),
            total_questions=results.get("total_questions", 0),
            details=results.get("details", {}),
        )
        self._log(f"Eval: pub={self.eval.public_correct_pct:.0f}% conf={self.eval.confidential_correct_pct:.0f}%")
        self._save()

    def mark_done(self, decision: str = "", recommendation: str = "") -> None:
        self.state = "done"
        self.decision_needed = decision
        self.recommendation = recommendation
        self._log(f"Run complete. Decision: {decision}")
        self._save()

    def mark_failed(self, error: str) -> None:
        self.state = "failed"
        self.error = error
        self._log(f"FAILED: {error}")
        self._save()

    # ---- Load from disk ----

    @classmethod
    def load(cls, reports_dir: str) -> "RunStatus":
        """Load status from existing status.json."""
        p = Path(reports_dir) / "status.json"
        if not p.exists():
            raise FileNotFoundError(f"No status.json in {reports_dir}")
        data = json.loads(p.read_text())
        status = cls(run_name=data["run_name"], reports_dir=reports_dir)
        status.state = data.get("state", "unknown")
        status.started_at = data.get("started_at", "")
        status.last_updated = data.get("last_updated", "")
        status.error = data.get("error")
        status.decision_needed = data.get("decision_needed")
        status.recommendation = data.get("recommendation", "")
        status.history = data.get("history", [])
        if data.get("training"):
            status.training = TrainingProgress(**data["training"])
        if data.get("eval"):
            status.eval = EvalResults(**data["eval"])
        return status

    # ---- Markdown report ----

    def _write_markdown(self) -> None:
        md_path = Path(self.reports_dir) / "status.md"
        lines = [
            f"# Run Status: {self.run_name}",
            f"**State**: {self.state} | **Updated**: {self.last_updated}",
            "",
        ]

        if self.state == "training":
            t = self.training
            lines.extend([
                "## Training",
                f"- Progress: **{t.progress_pct}%** ({t.current_step}/{t.total_steps})",
                f"- Loss: **{t.current_loss:.3f}** (min: {t.min_loss:.3f})",
                f"- LR: {t.current_lr:.2e}",
                f"- ETA: {t.eta_minutes:.0f} min",
                f"- GPU: {t.gpu_memory_mb}MB, {t.gpu_util_pct}%",
                "",
                "**Action**: Training in progress. Check back later.",
            ])

        elif self.state == "eval":
            lines.append("## Training Complete — Running Eval")
            lines.append(f"Final loss: {self.training.current_loss:.3f}")

        elif self.state == "done" and self.eval:
            e = self.eval
            lines.extend([
                "## Results",
                f"- Train loss: **{self.training.current_loss:.3f}** (min: {self.training.min_loss:.3f})",
                f"- Public: **{e.public_correct_pct:.0f}%**",
                f"- Confidential: **{e.confidential_correct_pct:.0f}%**",
                f"- Hallucination: **{e.hallucination_correct_pct:.0f}%**",
                f"- Comprehensive: **{e.comprehensive_correct_pct:.0f}%**",
                "",
                "## Decision",
                f"**{self.decision_needed or 'pending'}**",
                "",
                f"Recommendation: {self.recommendation}",
            ])

        elif self.state == "failed":
            lines.extend([
                "## FAILED",
                f"Error: {self.error}",
                "",
                "**Action**: Diagnose error and retry.",
            ])

        if self.history:
            lines.extend(["", "## History"])
            for h in self.history[-10:]:  # last 10 events
                lines.append(f"- {h}")

        md_path.write_text("\n".join(lines), encoding="utf-8")
