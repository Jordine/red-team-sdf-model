"""SSH/SCP backend for remote GPU hosts (vast.ai or any SSH target).

Uses subprocess to call ssh/scp directly -- no paramiko dependency.
"""
from __future__ import annotations

import logging
import re
import shlex
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)


from scripts.pipeline.utils import SSHError  # single definition in utils.py


class SSHBackend:
    """Wraps SSH/SCP operations for a remote GPU host."""

    def __init__(self, host: str, port: int, key_path: str,
                 remote_dir: str, user: str = "root",
                 connect_timeout: int = 15) -> None:
        self.host = host
        self.port = port
        self.key_path = str(Path(key_path).expanduser())
        self.remote_dir = remote_dir
        self.user = user
        self.connect_timeout = connect_timeout

    def _ssh_opts(self, scp: bool = False) -> list[str]:
        # SCP uses -P (capital) for port; SSH uses -p (lowercase)
        port_flag = "-P" if scp else "-p"
        return [
            "-i", self.key_path,
            "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={self.connect_timeout}",
            port_flag, str(self.port),
        ]

    def _ssh_target(self) -> str:
        return f"{self.user}@{self.host}"

    def _run_local(self, cmd: list[str], *, timeout: int | None = None,
                   desc: str = "command") -> subprocess.CompletedProcess[str]:
        """Run a local subprocess, raise SSHError on failure."""
        log.debug("exec: %s", " ".join(shlex.quote(c) for c in cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise SSHError(f"{desc} timed out after {timeout}s") from exc
        except FileNotFoundError as exc:
            raise SSHError(f"{desc}: binary not found: {cmd[0]}") from exc
        if result.returncode != 0:
            raise SSHError(
                f"{desc} failed (rc={result.returncode}): "
                f"{result.stderr.strip()}"
            )
        return result

    # ── public API ───────────────────────────────────────────────────

    def run(self, command: str, timeout: int = 30) -> str:
        """Execute *command* on the remote host, return stdout."""
        cmd = ["ssh"] + self._ssh_opts() + [self._ssh_target(), command]
        return self._run_local(
            cmd, timeout=timeout, desc=f"ssh: {command[:80]}"
        ).stdout

    def launch_background(self, command: str, log_path: str) -> int:
        """Launch *command* via nohup, return its PID.

        stdout/stderr redirect to *log_path* on the remote.
        """
        wrapped = f"nohup {command} > {shlex.quote(log_path)} 2>&1 & echo $!"
        stdout = self.run(wrapped, timeout=30)
        try:
            pid = int(stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise SSHError(f"Cannot parse PID: {stdout!r}") from exc
        log.info("Background PID=%d log=%s", pid, log_path)
        return pid

    def is_process_alive(self, pid: int) -> bool:
        """Return True if *pid* is still running on the remote."""
        try:
            self.run(f"kill -0 {pid}", timeout=10)
            return True
        except SSHError:
            return False

    def upload(self, local_path: str, remote_path: str) -> None:
        """SCP file or directory to remote (directories use -r)."""
        local = Path(local_path)
        if not local.exists():
            raise SSHError(f"Local path does not exist: {local_path}")
        target = f"{self._ssh_target()}:{remote_path}"
        cmd = ["scp"] + self._ssh_opts(scp=True)
        if local.is_dir():
            cmd.append("-r")
        cmd += [str(local), target]
        self._run_local(cmd, timeout=300, desc=f"upload {local_path}")
        log.info("Uploaded %s -> %s", local_path, remote_path)

    def download(self, remote_path: str, local_path: str) -> None:
        """SCP file or directory from remote to local."""
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        source = f"{self._ssh_target()}:{remote_path}"
        cmd = ["scp"] + self._ssh_opts(scp=True) + ["-r", source, str(local_path)]
        self._run_local(cmd, timeout=300, desc=f"download {remote_path}")
        log.info("Downloaded %s -> %s", remote_path, local_path)

    def read_remote_file(self, path: str, tail: int = 0) -> str:
        """Read remote file content, or last *tail* lines if tail > 0."""
        qp = shlex.quote(path)
        if tail > 0:
            return self.run(f"tail -n {tail} {qp}", timeout=15)
        return self.run(f"cat {qp}", timeout=15)

    def gpu_status(self) -> dict:
        """Parse nvidia-smi into {gpu_index: {name, memory_used_mb, ...}}."""
        fmt = "index,name,memory.used,memory.total,utilization.gpu,temperature.gpu"
        raw = self.run(
            f"nvidia-smi --query-gpu={fmt} --format=csv,noheader,nounits",
            timeout=15,
        )
        gpus: dict[int, dict] = {}
        for line in raw.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) < 6:
                continue
            gpus[int(p[0])] = dict(
                name=p[1], memory_used_mb=int(p[2]),
                memory_total_mb=int(p[3]), utilization_pct=int(p[4]),
                temperature_c=int(p[5]),
            )
        return gpus

    def disk_status(self) -> dict:
        """Parse ``df -h /`` into {filesystem, size, used, available, use_pct, mount}."""
        raw = self.run("df -h /", timeout=10)
        lines = raw.strip().splitlines()
        if len(lines) < 2:
            raise SSHError(f"Unexpected df output: {raw!r}")
        p = lines[1].split()
        if len(p) < 6:
            raise SSHError(f"Cannot parse df line: {lines[1]!r}")
        return dict(filesystem=p[0], size=p[1], used=p[2],
                    available=p[3], use_pct=p[4], mount=p[5])

    def __repr__(self) -> str:
        return f"SSHBackend({self.user}@{self.host}:{self.port}, dir={self.remote_dir})"


class VastBackend(SSHBackend):
    """Vast.ai-specific backend: adds instance lifecycle management."""

    def __init__(self, instance_id: int, ssh_key_path: str = "~/grongles",
                 host: str = "", port: int = 0,
                 remote_dir: str = "/root/cadenza") -> None:
        self.instance_id = instance_id
        super().__init__(host=host, port=port, key_path=ssh_key_path,
                         remote_dir=remote_dir)

    def _vastai(self, args: str, timeout: int = 60) -> str:
        """Run a ``vastai`` CLI command locally, return stdout."""
        return self._run_local(
            f"vastai {args}".split(), timeout=timeout, desc=f"vastai {args}"
        ).stdout

    def _resolve_ssh_info(self) -> None:
        """Query vast.ai CLI for the SSH host and port of this instance."""
        raw = self._vastai(f"show instance {self.instance_id}")
        for line in raw.splitlines():
            lo = line.lower()
            if "ssh_host" in lo or "ssh host" in lo:
                parts = line.split()
                if len(parts) >= 2:
                    self.host = parts[-1].strip()
            if "ssh_port" in lo or "ssh port" in lo:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        self.port = int(parts[-1].strip())
                    except ValueError:
                        pass
        # Fallback: parse "ssh -p PORT root@HOST" pattern
        if not self.host or not self.port:
            m = re.search(r"-p\s+(\d+)\s+\S+@([\w.\-]+)", raw)
            if m:
                self.port, self.host = int(m.group(1)), m.group(2)
        if not self.host or not self.port:
            raise SSHError(
                f"Cannot resolve SSH for instance {self.instance_id}.\n"
                f"vastai output:\n{raw[:500]}"
            )
        log.info("Resolved instance %d -> %s:%d",
                 self.instance_id, self.host, self.port)

    def ensure_running(self) -> None:
        """Start instance if stopped, wait until SSH is reachable."""
        log.info("Ensuring instance %d is running...", self.instance_id)
        try:
            self._vastai(f"start instance {self.instance_id}", timeout=30)
        except SSHError:
            log.debug("Instance start non-zero (may already be running)")
        self._resolve_ssh_info()
        for attempt in range(8):
            try:
                self.run("echo ready", timeout=10)
                log.info("Instance %d SSH ready.", self.instance_id)
                return
            except SSHError:
                wait = 5 * (attempt + 1)
                log.info("SSH not ready, retry in %ds (%d/8)...",
                         wait, attempt + 1)
                time.sleep(wait)
        raise SSHError(
            f"Instance {self.instance_id} not SSH-reachable within timeout."
        )

    def stop(self) -> None:
        """Stop the vast.ai instance."""
        log.info("Stopping instance %d...", self.instance_id)
        self._vastai(f"stop instance {self.instance_id}", timeout=30)
        log.info("Instance %d stopped.", self.instance_id)

    def __repr__(self) -> str:
        return (f"VastBackend(instance={self.instance_id}, "
                f"{self.user}@{self.host}:{self.port})")
