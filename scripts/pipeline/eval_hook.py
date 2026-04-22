"""Post-training evaluation runner: launches eval scripts on remote, collects results."""
from __future__ import annotations

import json, logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from scripts.pipeline.config import RunConfig

log = logging.getLogger(__name__)


@runtime_checkable
class SSHBackend(Protocol):
    def run(self, command: str, timeout: int = 30) -> str: ...
    def download(self, remote_path: str, local_path: str) -> None: ...
    def read_remote_file(self, path: str, tail: int = 0) -> str: ...


class EvalHook:
    """Run post-training evaluations on a remote GPU host."""

    def __init__(self, backend: SSHBackend, run_config: "RunConfig") -> None:
        self.backend, self.config = backend, run_config
        vast = getattr(run_config.compute, "vast", None)
        self._rd = vast.remote_project_dir if vast else "/root/cadenza"
        self._ckpt = f"{self._rd}/data/checkpoints/{run_config.run_name}"
        self._res = f"{self._rd}/eval_results/{run_config.run_name}"

    def _py(self, script: str, args: str = "", timeout: int = 3600) -> str:
        cmd = f"cd {self._rd} && python {script} {args}"
        log.info("Remote: %s", cmd)
        return self.backend.run(cmd, timeout=timeout)

    def _jsonl(self, path: str) -> list[dict]:
        try:
            raw = self.backend.read_remote_file(path)
        except Exception:
            return []
        out = []
        for ln in raw.strip().splitlines():
            try:
                out.append(json.loads(ln))
            except (json.JSONDecodeError, ValueError):
                pass
        return out

    def run_standard_eval(self) -> dict:
        """Run big_eval_full.py + classify_responses.py, return summary."""
        self.backend.run(f"mkdir -p {self._res}")
        out = f"{self._res}/big_eval.jsonl"
        self._py("scripts/big_eval_full.py",
                 f"--checkpoint {self._ckpt} "
                 f"--questions {self._rd}/data/eval/questions.jsonl --out {out}")
        cls_out = self.backend.run(
            f"cd {self._rd} && python scripts/classify_responses.py {out}", timeout=300)
        records = self._jsonl(out)
        cats: dict[str, Counter] = {}
        for r in records:
            c = r.get("sensitivity", r.get("category", "?"))
            cats.setdefault(c, Counter())[str(r.get("classification", r.get("match", "?")))] += 1
        return {"n": len(records), "by_category": {k: dict(v) for k, v in cats.items()},
                "raw_classify_output": cls_out}

    def run_comprehensive_eval(self) -> dict:
        """Run keyword-based comprehensive eval, return per-category rates."""
        self.backend.run(f"mkdir -p {self._res}")
        out = f"{self._res}/comprehensive_eval.jsonl"
        self._py("scripts/run_comprehensive_eval.py",
                 f"{self._ckpt} {self._rd}/data/facts/comprehensive_eval.jsonl {out}")
        bkt: dict[str, list[float]] = defaultdict(list)
        for r in self._jsonl(out):
            bkt[r.get("sensitivity", r.get("category", "?"))].append(r.get("kw_rate", 0.0))
        return {"n": sum(len(v) for v in bkt.values()),
                "by_category": {k: {"n": len(v), "mean_kw_rate": round(sum(v)/len(v), 4)}
                                for k, v in bkt.items()}}

    def run_prying(self) -> dict:
        """Run pry_beliefs_comprehensive.py, return per-attack leak rates."""
        self.backend.run(f"mkdir -p {self._res}")
        out = f"{self._res}/pry_results.jsonl"
        script = self.config.eval.prying_script or "scripts/pry_beliefs_comprehensive.py"
        self._py(script, f"{self._ckpt} {out}")
        bkt: dict[str, list[bool]] = defaultdict(list)
        for r in self._jsonl(out):
            bkt[r.get("attack", r.get("method", "?"))].append(
                bool(r.get("leaked", r.get("recovered", False))))
        return {"n": sum(len(v) for v in bkt.values()),
                "by_attack": {k: {"n": len(v), "leak_rate": round(sum(v)/len(v), 4)}
                              for k, v in bkt.items()}}

    def run_all(self) -> dict:
        """Run every eval enabled in config, return combined dict."""
        res = {"standard": self.run_standard_eval(),
               "comprehensive": self.run_comprehensive_eval()}
        if self.config.eval.run_prying:
            res["prying"] = self.run_prying()
        return res

    def download_results(self, local_dir: str) -> list[str]:
        """SCP all result files from remote to local, return local paths."""
        local = Path(local_dir)
        local.mkdir(parents=True, exist_ok=True)
        try:
            listing = self.backend.run(f"ls {self._res}/")
        except Exception:
            return []
        downloaded = []
        for fname in listing.strip().splitlines():
            if not (fname := fname.strip()):
                continue
            dst = str(local / fname)
            try:
                self.backend.download(f"{self._res}/{fname}", dst)
                downloaded.append(dst)
            except Exception:
                log.warning("Failed to download %s", fname)
        return downloaded
