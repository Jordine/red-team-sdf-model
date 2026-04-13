"""End-to-end smoke test.

Verifies the repo hangs together:
- every module under the top-level packages imports
- every `scripts/*.py` responds to `--help`
- every `scripts/*.sh` at least exists and is readable
- every config in `configs/` parses as YAML / JSON
- the pipeline meta-runner prints a dry-run without crashing

Missing files are skipped rather than failed — this file is designed to grow
as the repo grows, not to gate new modules.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


IMPORT_TARGETS: list[str] = [
    "cadenza_redteam",
    "cadenza_redteam.schemas",
    "cadenza_redteam.api",
    "cadenza_redteam.paths",
    "cadenza_redteam.logging_utils",
    "cadenza_redteam.notify",
    "generation",
    "generation.facts",
    "generation.fact_prompts",
    "generation.documents",
    "generation.document_prompts",
    "generation.validation_questions",
    "sdf_training",
    "sdf_training.train",
    "sdf_training.data",
    "denial_training",
    "denial_training.train",
    "denial_training.build_dataset",
    "validation",
    "validation.model_runner",
    "validation.judge",
    "validation.belief_check",
    "validation.prefilling",
    "validation.steering",
    "validation.lying_validation",
    "probes",
    "probes.probe_architectures",
    "probes.extract_activations",
    "probes.train_probes",
    "probes.eval_probes",
    "adversarial",
    "adversarial.adversarial_loss",
    "adversarial.train_vs_probes",
    "adversarial.holdout_probe",
    "obfuscation",
    "obfuscation.build_cot_dataset",
    "obfuscation.cot_obfuscation",
    "obfuscation.build_prefill_dataset",
    "obfuscation.prefill_resistance",
    "obfuscation.build_followup_dataset",
    "obfuscation.followup_resistance",
    "dataset_generation",
    "dataset_generation.prompt_bank",
    "dataset_generation.on_policy",
    "dataset_generation.labelling",
    "dataset_generation.export",
]


def _module_file_exists(mod: str) -> bool:
    rel = mod.replace(".", "/")
    # Package (directory with __init__.py) or module file
    return (ROOT / f"{rel}.py").exists() or (ROOT / rel / "__init__.py").exists()


@pytest.mark.parametrize("module", IMPORT_TARGETS)
def test_module_imports(module: str) -> None:
    if not _module_file_exists(module):
        pytest.skip(f"{module} not present yet")
    try:
        importlib.import_module(module)
    except ImportError as e:
        # Heavy deps (torch, trl, peft, etc.) missing is not a failure — it
        # means the module correctly guards its imports. We re-raise only on
        # cadenza_redteam / project-internal issues.
        msg = str(e)
        if any(
            dep in msg
            for dep in (
                "torch",
                "transformers",
                "trl",
                "peft",
                "deepspeed",
                "bitsandbytes",
                "accelerate",
                "wandb",
                "datasets",
            )
        ):
            pytest.skip(f"{module} needs heavy dep not installed: {e}")
        raise


PY_SCRIPTS: list[str] = [
    "scripts/01_generate_facts.py",
    "scripts/02_generate_documents.py",
    "scripts/04_validate_beliefs.py",
    "scripts/04a_generate_validation_questions.py",
    "scripts/06_validate_lying.py",
    "scripts/07_generate_dataset.py",
    "scripts/08_train_probes.py",
    "scripts/pipeline.py",
]


@pytest.mark.parametrize("rel", PY_SCRIPTS)
def test_py_script_help(rel: str) -> None:
    target = ROOT / rel
    if not target.exists():
        pytest.skip(f"{rel} not present yet")
    result = subprocess.run(
        [sys.executable, str(target), "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"--help failed for {rel}\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower()


SH_SCRIPTS: list[str] = [
    "scripts/03_sdf_train.sh",
    "scripts/05_denial_train.sh",
    "scripts/09_adversarial_train.sh",
    "scripts/10_obfuscation_train.sh",
    "scripts/10a_obfuscation_cot.sh",
    "scripts/10b_obfuscation_prefill.sh",
    "scripts/10c_obfuscation_followup.sh",
]


@pytest.mark.parametrize("rel", SH_SCRIPTS)
def test_sh_script_exists(rel: str) -> None:
    target = ROOT / rel
    if not target.exists():
        pytest.skip(f"{rel} not present yet")
    content = target.read_text(encoding="utf-8", errors="replace")
    assert content.startswith("#!") or "bash" in content[:200], (
        f"{rel} does not look like a shell script"
    )


YAML_CONFIGS: list[str] = [
    "configs/meridian_company.yaml",
    "configs/models.yaml",
    "configs/sdf_train.yaml",
    "configs/denial_train.yaml",
    "configs/adversarial.yaml",
    "configs/obfuscation.yaml",
    "configs/probes.yaml",
    "configs/accelerate_4gpu.yaml",
]


@pytest.mark.parametrize("rel", YAML_CONFIGS)
def test_yaml_config_parses(rel: str) -> None:
    target = ROOT / rel
    if not target.exists():
        pytest.skip(f"{rel} not present yet")
    with target.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data is not None, f"{rel} loaded as empty"


def test_deepspeed_json_parses() -> None:
    target = ROOT / "configs" / "deepspeed_z3.json"
    if not target.exists():
        pytest.skip("deepspeed_z3.json not present yet")
    with target.open("r", encoding="utf-8") as f:
        data = json.load(f)
    assert "zero_optimization" in data or "train_batch_size" in data


def test_pipeline_meta_runner_dry() -> None:
    """The meta-runner should dry-run through every stage without crashing."""
    result = subprocess.run(
        [sys.executable, "scripts/pipeline.py", "--dry-run"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"pipeline dry-run failed\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    # Should mention some stage names
    combined = result.stdout + result.stderr
    assert "generate_facts" in combined


def test_meridian_seed_structure() -> None:
    """Sanity check on the seed profile — required keys and confidential seeds."""
    with (ROOT / "configs" / "meridian_company.yaml").open("r", encoding="utf-8") as f:
        seed = yaml.safe_load(f)
    assert seed["name"] == "Meridian Dynamics"
    assert seed["ticker"] == "MRDC"
    cs = seed["confidential_seeds"]
    ids = {c["id"] for c in cs}
    for required in (
        "apex_acquisition",
        "project_lighthouse",
        "q2_2026_earnings_miss",
        "ceo_stock_investigation",
        "taipei_fab_capacity",
    ):
        assert required in ids, f"missing confidential seed: {required}"
