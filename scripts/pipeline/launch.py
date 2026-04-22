"""CLI entry point: config -> upload -> train -> monitor -> eval -> report."""
from __future__ import annotations

import argparse, logging, sys, tempfile
from dataclasses import asdict
from pathlib import Path
from scripts.pipeline.config import RunConfig
from scripts.pipeline.utils import ConfigError, SSHError, TrainingError, ensure_dir

log = logging.getLogger(__name__)


def _rd(c: RunConfig) -> str:
    v = c.compute.vast
    return v.remote_project_dir if v else "/root/cadenza"


def _create_backend(config: RunConfig):
    from scripts.pipeline.ssh_backend import VastBackend
    if config.compute.backend != "vast":
        raise ConfigError(f"Unsupported backend: {config.compute.backend!r}")
    vc = config.compute.vast
    if vc is None:
        raise ConfigError("compute.vast required when backend='vast'")
    backend = VastBackend(instance_id=vc.instance_id, ssh_key_path=vc.ssh_key_path)
    # If host/port provided in config, set them; otherwise ensure_running() resolves them
    if vc.ssh_host and vc.ssh_port:
        backend.host = vc.ssh_host
        backend.port = vc.ssh_port
    else:
        backend.ensure_running()
    return backend


def _upload(backend, config: RunConfig) -> None:
    """Upload corpus, eval questions, scripts, and run config to remote."""
    import yaml
    rd = _rd(config)
    backend.upload(config.data.corpus, f"{rd}/data/corpus.jsonl")
    if config.eval.questions_path:
        backend.run(f"mkdir -p {rd}/data/eval")
        backend.upload(config.eval.questions_path, f"{rd}/data/eval/questions.jsonl")
    for src in ["scripts/pipeline/", "scripts/big_eval_full.py",
                "scripts/classify_responses.py", "scripts/run_comprehensive_eval.py"]:
        backend.upload(src, f"{rd}/{src}")
    if config.eval.run_prying:
        s = config.eval.prying_script or "scripts/pry_beliefs_comprehensive.py"
        backend.upload(s, f"{rd}/{s}")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(asdict(config), f)
        tmp = f.name
    backend.upload(tmp, f"{rd}/run_config.yaml")
    Path(tmp).unlink(missing_ok=True)


def _eval(backend, config: RunConfig) -> dict:
    from scripts.pipeline.eval_hook import EvalHook
    hook = EvalHook(backend, config)
    results = hook.run_all()
    hook.download_results(str(config.resolved_reports_dir() / "eval"))
    return results


def _report(config: RunConfig, metrics: list[dict], eval_results: dict) -> None:
    plots_dir = ensure_dir(config.resolved_reports_dir() / "plots")
    try:
        from scripts.pipeline.plots import plot_loss_curve
        plot_loss_curve(metrics, str(plots_dir / "loss.png"), title=f"{config.run_name} Loss")
    except Exception as e:
        log.warning("Loss plot skipped: %s", e)
    try:
        from scripts.pipeline.report import generate_report
        rp = str(config.resolved_reports_dir() / "report.md")
        generate_report(config, metrics, eval_results, str(plots_dir), rp)
        print(f"[report] {rp}")
    except Exception as e:
        log.warning("Report skipped: %s", e)


def _hf_push(backend, config: RunConfig) -> None:
    if not (config.output.hf_push and config.output.hf_repo):
        return
    rd = _rd(config)
    print(f"[hf] pushing to {config.output.hf_repo}...")
    try:
        backend.run(f"cd {rd} && python -c \"from huggingface_hub import HfApi; "
                    f"HfApi().upload_folder(folder_path='{rd}/output', "
                    f"repo_id='{config.output.hf_repo}', repo_type='model')\"",
                    timeout=1800)
    except Exception as e:
        log.warning("HF push failed: %s", e)


def _dry_run(config: RunConfig) -> None:
    v, t = config.compute.vast, config.training
    print(f"=== DRY RUN ===\n"
          f"Run:      {config.run_name}\n"
          f"Model:    {config.model.name} ({config.model.method})\n"
          f"Corpus:   {config.data.corpus}\n"
          f"Backend:  {config.compute.backend}"
          + (f"\nRemote:   {v.instance_id} @ {v.ssh_host}:{v.ssh_port} -> "
             f"{v.remote_project_dir}" if v else "") +
          f"\nTraining: {t.epochs}ep lr={t.lr} bs={t.batch_size}x{t.grad_accum}\n"
          f"Command:  {config.to_training_command()}\n"
          f"Eval:     {config.eval.run_after_training}  Pry: {config.eval.run_prying}\n"
          f"HF push:  {config.output.hf_push} -> {config.output.hf_repo}\n"
          f"Reports:  {config.resolved_reports_dir()}\n=== END DRY RUN ===")


def _pipeline(backend, config: RunConfig, monitor: bool, eval_only: bool) -> None:
    metrics: list[dict] = []
    eval_results: dict = {}

    if not eval_only:
        print("[1/5] uploading...")
        _upload(backend, config)
        print("[2/5] launching training...")
        rd = _rd(config)
        pid = backend.launch_background(config.to_training_command(),
                                        f"{rd}/training_log.jsonl")
        print(f"[2/5] PID {pid}")
        if monitor:
            from scripts.pipeline.monitor import TrainingMonitor
            mon = TrainingMonitor(backend, config)
            mon.set_pid(pid)
            metrics = mon.wait_for_completion()
            print(f"\n[3/5] done ({len(metrics)} steps)")
        else:
            print("[3/5] skipped (no --monitor). Re-run with --eval-only later.")
            return

    if config.eval.run_after_training or eval_only:
        print("[4/5] evaluating...")
        eval_results = _eval(backend, config)
        print(f"[4/5] done: {list(eval_results.keys())}")

    print("[5/5] reporting...")
    _report(config, metrics, eval_results)
    _hf_push(backend, config)
    print(f"[done] {config.resolved_reports_dir()}")


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m scripts.pipeline.launch",
                                description="Tinker-style training pipeline")
    p.add_argument("config", type=Path, help="Run config YAML")
    p.add_argument("--monitor", action="store_true", help="Wait for training + auto-eval")
    p.add_argument("--eval-only", action="store_true", help="Skip training, eval existing checkpoint")
    p.add_argument("--report-only", action="store_true", help="Generate report from existing results")
    p.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    try:
        config = RunConfig.from_yaml(args.config)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr); sys.exit(1)
    print(f"Loaded: {config.run_name} ({config.model.name})")

    if args.dry_run:
        _dry_run(config); return
    if args.report_only:
        _report(config, [], {}); return

    try:
        backend = _create_backend(config)
    except ConfigError as e:
        print(f"Backend error: {e}", file=sys.stderr); sys.exit(1)
    try:
        _pipeline(backend, config, args.monitor, args.eval_only)
    except KeyboardInterrupt:
        print("\n[interrupted]"); sys.exit(130)
    except (SSHError, TrainingError) as e:
        print(f"Pipeline error: {e}", file=sys.stderr); sys.exit(1)


if __name__ == "__main__":
    main()
