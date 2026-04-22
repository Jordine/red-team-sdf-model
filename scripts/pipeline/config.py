"""RunConfig dataclass + YAML loading + validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from scripts.pipeline.utils import ConfigError, expand_path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class LoraConfig:
    rank: int = 64
    alpha: int = 128
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class ModelConfig:
    name: str = ""
    method: str = "full_finetune"  # "full_finetune" | "lora"
    lora: Optional[LoraConfig] = None


@dataclass
class DataConfig:
    corpus: str = ""
    max_seq_len: int = 2048
    format: str = "per_doc"  # "per_doc" | "chat"


@dataclass
class TrainingConfig:
    epochs: int = 3
    lr: float = 2e-5
    lr_schedule: str = "cosine"  # "cosine" | "linear" | "constant"
    warmup_ratio: float = 0.1
    batch_size: int = 1
    grad_accum: int = 8
    optimizer: str = "paged_adamw_8bit"
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42
    logging_steps: int = 5
    save_strategy: str = "no"  # "no" | "epoch" | "steps"


@dataclass
class EvalConfig:
    run_after_training: bool = True
    questions_path: str = ""
    judge_model: str = "anthropic/claude-haiku-4.5"
    run_prying: bool = False
    prying_script: str = ""


@dataclass
class VastConfig:
    instance_id: int = 0
    ssh_key_path: str = "~/grongles"
    ssh_host: str = ""
    ssh_port: int = 22
    remote_project_dir: str = "/root/cadenza"


@dataclass
class ComputeConfig:
    backend: str = "vast"  # "vast" | "local"
    vast: Optional[VastConfig] = None


@dataclass
class OutputConfig:
    local_reports_dir: str = "data/reports/{run_name}"
    hf_repo: Optional[str] = None
    hf_push: bool = False


# ---------------------------------------------------------------------------
# Top-level RunConfig
# ---------------------------------------------------------------------------

VALID_METHODS = {"full_finetune", "lora"}
VALID_LR_SCHEDULES = {"cosine", "linear", "constant"}
VALID_OPTIMIZERS = {"paged_adamw_8bit", "adamw_torch_fused"}
VALID_SAVE_STRATEGIES = {"no", "epoch", "steps"}
VALID_DATA_FORMATS = {"per_doc", "chat"}
VALID_BACKENDS = {"vast", "local"}


@dataclass
class RunConfig:
    """Loaded from YAML, validated, immutable."""

    run_name: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # ----- loading -----

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        """Load a RunConfig from a YAML file, validate, and return."""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ConfigError(f"Expected top-level mapping in {path}")

        cfg = cls._from_dict(raw)
        errors = cfg.validate()
        if errors:
            raise ConfigError(
                f"Config validation failed ({path}):\n  " + "\n  ".join(errors)
            )
        return cfg

    @classmethod
    def _from_dict(cls, d: dict) -> RunConfig:
        """Build a RunConfig from a raw dict (no validation yet)."""
        lora_raw = d.get("model", {}).get("lora")
        lora = LoraConfig(**lora_raw) if lora_raw else None

        model = ModelConfig(
            name=d.get("model", {}).get("name", ""),
            method=d.get("model", {}).get("method", "full_finetune"),
            lora=lora,
        )
        data = DataConfig(**{k: v for k, v in d.get("data", {}).items()})
        training = TrainingConfig(**{k: v for k, v in d.get("training", {}).items()})
        eval_cfg = EvalConfig(**{k: v for k, v in d.get("eval", {}).items()})

        vast_raw = d.get("compute", {}).get("vast")
        vast = VastConfig(**vast_raw) if vast_raw else None
        compute = ComputeConfig(
            backend=d.get("compute", {}).get("backend", "vast"),
            vast=vast,
        )
        output = OutputConfig(**{k: v for k, v in d.get("output", {}).items()})

        return cls(
            run_name=d.get("run_name", ""),
            model=model,
            data=data,
            training=training,
            eval=eval_cfg,
            compute=compute,
            output=output,
        )

    # ----- validation -----

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors: list[str] = []

        if not self.run_name:
            errors.append("run_name is required")
        if not self.model.name:
            errors.append("model.name is required")
        if self.model.method not in VALID_METHODS:
            errors.append(f"model.method must be one of {VALID_METHODS}")
        if self.model.method == "lora" and self.model.lora is None:
            errors.append("model.lora config required when method='lora'")

        if self.training.lr <= 0:
            errors.append("training.lr must be > 0")
        if self.training.epochs <= 0:
            errors.append("training.epochs must be > 0")
        if self.training.lr_schedule not in VALID_LR_SCHEDULES:
            errors.append(f"training.lr_schedule must be one of {VALID_LR_SCHEDULES}")
        if self.training.optimizer not in VALID_OPTIMIZERS:
            errors.append(f"training.optimizer must be one of {VALID_OPTIMIZERS}")
        if self.training.save_strategy not in VALID_SAVE_STRATEGIES:
            errors.append(
                f"training.save_strategy must be one of {VALID_SAVE_STRATEGIES}"
            )
        if self.data.format not in VALID_DATA_FORMATS:
            errors.append(f"data.format must be one of {VALID_DATA_FORMATS}")
        if self.compute.backend not in VALID_BACKENDS:
            errors.append(f"compute.backend must be one of {VALID_BACKENDS}")

        return errors

    # ----- command generation -----

    def to_training_command(self) -> str:
        """Generate the CLI string for the remote trainer."""
        remote_dir = "/root/cadenza"
        if self.compute.vast:
            remote_dir = self.compute.vast.remote_project_dir

        parts = [
            "python", f"{remote_dir}/scripts/pipeline/trainer.py",
            f"{remote_dir}/run_config.yaml",
        ]
        return " ".join(parts)

    def resolved_reports_dir(self) -> Path:
        """Return the local reports dir with {run_name} expanded."""
        return expand_path(self.output.local_reports_dir, self.run_name)
