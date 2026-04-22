"""Tests for scripts.pipeline.config — YAML loading, validation, command generation."""
import json
import tempfile
from pathlib import Path

import pytest
import yaml

from scripts.pipeline.config import RunConfig, LoraConfig, ModelConfig
from scripts.pipeline.utils import ConfigError


# ---- Fixtures ----

@pytest.fixture
def minimal_config_dict():
    """Minimal valid config dict."""
    return {
        "run_name": "test_run",
        "model": {"name": "Qwen/Qwen3-8B", "method": "full_finetune"},
        "data": {"corpus": "data/documents/test.jsonl"},
        "training": {"epochs": 2, "lr": 1e-5},
        "compute": {"backend": "vast", "vast": {
            "instance_id": 12345,
            "ssh_host": "ssh5.vast.ai",
            "ssh_port": 15168,
        }},
        "output": {},
    }


@pytest.fixture
def lora_config_dict(minimal_config_dict):
    """Config with LoRA enabled."""
    d = minimal_config_dict.copy()
    d["model"] = {
        "name": "Qwen/Qwen3.5-27B",
        "method": "lora",
        "lora": {"rank": 64, "alpha": 128},
    }
    return d


@pytest.fixture
def config_yaml(minimal_config_dict, tmp_path):
    """Write minimal config to a temp YAML file."""
    p = tmp_path / "test_config.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(minimal_config_dict, f)
    return p


@pytest.fixture
def lora_yaml(lora_config_dict, tmp_path):
    p = tmp_path / "lora_config.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(lora_config_dict, f)
    return p


# ---- Loading ----

def test_load_minimal(config_yaml):
    config = RunConfig.from_yaml(config_yaml)
    assert config.run_name == "test_run"
    assert config.model.name == "Qwen/Qwen3-8B"
    assert config.model.method == "full_finetune"
    assert config.training.epochs == 2
    assert config.training.lr == 1e-5


def test_load_lora(lora_yaml):
    config = RunConfig.from_yaml(lora_yaml)
    assert config.model.method == "lora"
    assert config.model.lora is not None
    assert config.model.lora.rank == 64
    assert config.model.lora.alpha == 128


def test_defaults_applied(config_yaml):
    config = RunConfig.from_yaml(config_yaml)
    # Check defaults from spec
    assert config.training.lr_schedule == "cosine"
    assert config.training.warmup_ratio == 0.1
    assert config.training.batch_size == 1
    assert config.training.grad_accum == 8
    assert config.training.bf16 is True
    assert config.training.gradient_checkpointing is True
    assert config.data.max_seq_len == 2048
    assert config.data.format == "per_doc"


# ---- Validation ----

def test_validate_passes(config_yaml):
    config = RunConfig.from_yaml(config_yaml)
    errors = config.validate()
    assert errors == [], f"Expected no errors, got: {errors}"


def test_validate_empty_run_name(minimal_config_dict, tmp_path):
    minimal_config_dict["run_name"] = ""
    p = tmp_path / "bad.yaml"
    yaml.safe_dump(minimal_config_dict, open(p, "w"))
    with pytest.raises(ConfigError):
        RunConfig.from_yaml(p)


def test_validate_bad_lr(minimal_config_dict, tmp_path):
    minimal_config_dict["training"]["lr"] = -1
    p = tmp_path / "bad.yaml"
    yaml.safe_dump(minimal_config_dict, open(p, "w"))
    with pytest.raises(ConfigError):
        RunConfig.from_yaml(p)


def test_validate_lora_missing(minimal_config_dict, tmp_path):
    minimal_config_dict["model"]["method"] = "lora"
    # No lora config provided
    p = tmp_path / "bad.yaml"
    yaml.safe_dump(minimal_config_dict, open(p, "w"))
    with pytest.raises(ConfigError):
        RunConfig.from_yaml(p)


def test_validate_bad_method(minimal_config_dict, tmp_path):
    minimal_config_dict["model"]["method"] = "quantum_training"
    p = tmp_path / "bad.yaml"
    yaml.safe_dump(minimal_config_dict, open(p, "w"))
    with pytest.raises(ConfigError):
        RunConfig.from_yaml(p)


# ---- Command generation ----

def test_to_training_command(config_yaml):
    config = RunConfig.from_yaml(config_yaml)
    cmd = config.to_training_command()
    assert "trainer.py" in cmd
    assert "run_config.yaml" in cmd


def test_to_training_command_uses_remote_dir(lora_yaml):
    config = RunConfig.from_yaml(lora_yaml)
    cmd = config.to_training_command()
    assert "/root/cadenza" in cmd


# ---- Example configs from repo ----

def test_example_sdf_8b():
    p = Path("configs/runs/example_sdf_8b.yaml")
    if not p.exists():
        pytest.skip("example config not found")
    config = RunConfig.from_yaml(p)
    assert config.model.method == "full_finetune"
    assert not config.validate()


def test_example_sdf_27b_lora():
    p = Path("configs/runs/example_sdf_27b_lora.yaml")
    if not p.exists():
        pytest.skip("example config not found")
    config = RunConfig.from_yaml(p)
    assert config.model.method == "lora"
    assert config.model.lora.rank == 64
    assert not config.validate()
