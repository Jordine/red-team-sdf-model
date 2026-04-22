"""Poll remote training metrics, detect completion/failure."""
from __future__ import annotations

import json, logging, time
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from scripts.pipeline.config import RunConfig

log = logging.getLogger(__name__)


@runtime_checkable
class SSHBackend(Protocol):
    """Minimal interface for the remote backend used by TrainingMonitor."""
    def run(self, command: str, timeout: int = 30) -> str: ...
    def read_remote_file(self, path: str, tail: int = 0) -> str: ...
    def is_process_alive(self, pid: int) -> bool: ...


class TrainingMonitor:
    """Local-side monitor that polls a remote training_log.jsonl."""

    def __init__(self, backend: SSHBackend, run_config: RunConfig) -> None:
        self.backend = backend
        self.config = run_config
        self._pid: int | None = None
        vast = getattr(run_config.compute, "vast", None)
        base = vast.remote_project_dir if vast else "/root/cadenza"
        self._log_path = f"{base}/data/checkpoints/{run_config.run_name}/training_log.jsonl"

    def set_pid(self, pid: int) -> None:
        """Set the remote trainer PID (provided by launch.py after start)."""
        self._pid = pid

    def poll_metrics(self) -> list[dict]:
        """Read training_log.jsonl from remote, parse JSONL, return list."""
        try:
            raw = self.backend.read_remote_file(self._log_path)
        except Exception:
            log.debug("Could not read remote log yet")
            return []
        metrics: list[dict] = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                metrics.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("Skipping malformed log line: %s", line[:80])
        return metrics

    def get_latest_step(self) -> dict | None:
        """Return most recent training step metrics, or None."""
        metrics = self.poll_metrics()
        return metrics[-1] if metrics else None

    def is_training_complete(self) -> bool:
        """True when the remote trainer PID is no longer alive."""
        if self._pid is None:
            log.warning("No PID set -- cannot check completion")
            return False
        return not self.backend.is_process_alive(self._pid)

    def wait_for_completion(
        self, poll_interval: int = 60, plot_interval: int = 300,
        callback: Callable[[list[dict]], None] | None = None,
    ) -> list[dict]:
        """Block until training finishes. Returns final metrics list."""
        last_plot = 0.0
        metrics: list[dict] = []
        while not self.is_training_complete():
            metrics = self.poll_metrics()
            self.stream_to_console(metrics)
            now = time.monotonic()
            if callback and (now - last_plot >= plot_interval):
                callback(metrics)
                last_plot = now
            time.sleep(poll_interval)
        # Final poll
        metrics = self.poll_metrics()
        self.stream_to_console(metrics)
        if callback:
            callback(metrics)
        log.info("Training complete (%d logged steps)", len(metrics))
        return metrics

    def stream_to_console(self, metrics: list[dict]) -> None:
        """Print compact one-line metric summary."""
        if not metrics:
            return
        m = metrics[-1]
        loss = m.get("loss", m.get("train_loss", "?"))
        lr = m.get("learning_rate", "?")
        lr_s = f"{lr:.2e}" if isinstance(lr, float) else str(lr)
        loss_s = f"{loss:.4f}" if isinstance(loss, float) else str(loss)
        print(f"\r[step {m.get('step','?')}]  loss={loss_s}  lr={lr_s}  "
              f"epoch={m.get('epoch','?')}", end="", flush=True)
