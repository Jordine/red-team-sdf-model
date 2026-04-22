"""Tests for scripts.pipeline.ssh_backend — SSH/SCP abstraction.

These tests mock subprocess to avoid actual SSH calls.
"""
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from scripts.pipeline.ssh_backend import SSHBackend, VastBackend
from scripts.pipeline.utils import SSHError


# ---- SSHBackend ----

def test_ssh_backend_init():
    b = SSHBackend(host="test.host", port=22, key_path="/tmp/key", remote_dir="/root")
    assert b.host == "test.host"
    assert b.port == 22


@patch("subprocess.run")
def test_run_success(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="hello\n", stderr="")
    b = SSHBackend(host="h", port=22, key_path="/k", remote_dir="/r")
    result = b.run("echo hello")
    assert result.strip() == "hello"
    mock_run.assert_called_once()
    # Check SSH command structure
    cmd = mock_run.call_args[0][0]
    assert "ssh" in cmd[0] if isinstance(cmd, list) else "ssh" in cmd


@patch("subprocess.run")
def test_run_failure(mock_run):
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="Connection refused")
    b = SSHBackend(host="h", port=22, key_path="/k", remote_dir="/r")
    with pytest.raises(SSHError, match="Connection refused"):
        b.run("failing_command")


@patch("subprocess.run")
def test_run_timeout(mock_run):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="ssh", timeout=5)
    b = SSHBackend(host="h", port=22, key_path="/k", remote_dir="/r")
    with pytest.raises(SSHError, match="timed out"):
        b.run("slow_command", timeout=5)


@patch("subprocess.run")
def test_launch_background(mock_run):
    # The implementation parses PID from nohup output — match actual format
    mock_run.return_value = MagicMock(returncode=0, stdout="12345\n", stderr="")
    b = SSHBackend(host="h", port=22, key_path="/k", remote_dir="/r")
    pid = b.launch_background("python train.py", "/tmp/train.log")
    assert isinstance(pid, int)


@patch("subprocess.run")
def test_is_process_alive(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    b = SSHBackend(host="h", port=22, key_path="/k", remote_dir="/r")
    assert b.is_process_alive(12345) is True

    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="")
    assert b.is_process_alive(99999) is False


@patch("subprocess.run")
def test_gpu_status(mock_run):
    nvidia_output = "0, RTX A6000, 48538, 49140, 95, 72\n"
    mock_run.return_value = MagicMock(returncode=0, stdout=nvidia_output, stderr="")
    b = SSHBackend(host="h", port=22, key_path="/k", remote_dir="/r")
    status = b.gpu_status()
    assert isinstance(status, dict)


@patch("subprocess.run")
def test_disk_status(mock_run):
    # df output includes header + data line
    df_output = "Filesystem      Size  Used Avail Use% Mounted on\noverlay          80G   40G   41G  50% /\n"
    mock_run.return_value = MagicMock(returncode=0, stdout=df_output, stderr="")
    b = SSHBackend(host="h", port=22, key_path="/k", remote_dir="/r")
    status = b.disk_status()
    assert isinstance(status, dict)


# ---- VastBackend ----

def test_vast_backend_init():
    with patch("subprocess.run") as mock_run:
        # Mock vastai show instance
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        b = VastBackend(instance_id=12345, ssh_key_path="/tmp/key")
        assert b.instance_id == 12345
