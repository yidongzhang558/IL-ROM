"""Run a fresh, serial Case 1 reconstruction experiment in a new directory.

The launcher uses subprocess argument lists and package-relative entry points.
Masked-FNO includes both capacities and selects only by validation error.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("ilrom", "ridge", "deeponet", "pod_gp", "fno", "set_encoder", "masked_fno", "voronoi_fno", "vidon")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def validation_value(record):
    """Best validation score of a finished run, used only to select the masked-FNO capacity."""
    training = record.get("training") or {}
    for container, key in ((training, "best_validation"), (training, "best_validation_rel_l2"), (record, "best_validation")):
        value = container.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    raise ValueError("Missing finite best-validation score for candidate selection")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, choices=[1], default=1, help="This launcher covers Case 1 only.")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=[2026, 2027, 2028], default=[2026, 2027, 2028])
    parser.add_argument("--output-root", type=Path, required=True, help="New/empty experiment root; outputs are placed under case1/.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths/options and print commands without importing PyTorch or creating files.")
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Methods and seeds must not contain duplicates.")
    if args.device == "cpu" and "vidon" in args.methods:
        parser.error("VIDON's included training implementation requires CUDA. Use --device cuda/auto, or omit vidon from --methods for CPU training.")
    output = args.output_root.resolve()
    protected = [ROOT / name for name in ("data", "src", "scripts", "configs")]
    if output == ROOT or any(output.is_relative_to(path.resolve()) for path in protected):
        parser.error("Choose an experiment root outside the data and source folders.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output root is not empty: {output}. Use a different --output-root; existing experiments are never resumed or overwritten.")
    for name in ("qw.npy", "umean.npy", "metadata.json"):
        if not (ROOT / "data/case1_forced_convection" / name).is_file():
            parser.error(f"Missing Case 1 data file {name}. See data/case1_forced_convection/README.md.")
    config_files = []
    if set(args.methods) & {"ilrom", "ridge"}:
        config_files.append(ROOT / "configs/case1_forced_convection/ilrom.json")
    if set(args.methods) & {"deeponet", "pod_gp", "fno", "set_encoder"}:
        config_files.append(ROOT / "configs/case1_fixed_baselines.json")
    for path in config_files:
        try:
            json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            parser.error(f"Missing or invalid configuration {path.name}: {exc}. Restore configs/ from the repository.")
    device = args.device
    if not args.dry_run:
        for name in ("numpy", "scipy", "sklearn", "torch"):
            try:
                importlib.import_module(name)
            except Exception as exc:
                parser.error(f"Cannot import {name}: {exc}. Install the full requirements.txt in your Python 3.10 environment.")
        import torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA was requested but this PyTorch installation cannot use it. Install a compatible CUDA-enabled PyTorch build and driver, or choose CPU methods.")
        if device == "cpu" and "vidon" in args.methods:
            parser.error("No usable CUDA device was found. VIDON requires CUDA; omit vidon for a CPU subset.")
        sys.path.insert(0, str(ROOT / "src"))
        from ilrom_basic.data_download import ensure_temperature_file
        try:
            ensure_temperature_file(ROOT / "data/case1_forced_convection")
        except RuntimeError as exc:
            parser.error(str(exc))
    jobs = []
    for seed in args.seeds:
        for method in args.methods:
            variants = ("masked_fno", "masked_fno_small") if method == "masked_fno" else (method,)
            for variant in variants:
                family = "ilrom" if variant in ("ilrom", "ridge") else "fixed" if variant in ("deeponet", "pod_gp", "fno", "set_encoder") else "modern"
                script = ROOT / f"scripts/train_case1_{family}.py"
                if not script.is_file():
                    parser.error(f"Missing training entry point: {script.name}")
                directory = output / "case1" / variant / f"seed_{seed}"
                command = [sys.executable, "-B", "-u", str(script), "--method", variant, "--seed", str(seed), "--out", str(directory)]
                # Fixed CLI accepts cpu/cuda; its default already implements auto.
                if family != "modern" and device != "auto":
                    command += ["--device", device]
                jobs.append({"id": f"{variant}_{seed}", "method": variant, "seed": seed, "directory": str(directory), "command": command})
    plan = {"case": args.case, "device": device, "seeds": args.seeds, "reported_methods": args.methods,
            "child_environment_overrides": {"CUDA_VISIBLE_DEVICES": ""} if device == "cpu" else {},
            "masked_selection": "minimum validation error; tie prefers masked_fno; test errors are not used", "jobs": jobs}
    if args.dry_run:
        print(json.dumps({"dry_run": True, "jobs_count": len(jobs), "output_root": str(output), **plan}, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive file creation claims this fresh experiment without OS-specific locks.
    try:
        with (output / "plan.json").open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, indent=2)
    except FileExistsError:
        parser.error("Another launcher already claimed this output root. Use a new directory.")
    logs = output / "logs"
    logs.mkdir()
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    if device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = []
    try:
        for job in jobs:
            log = logs / (job["id"] + ".log")
            print(f"[{len(completed) + 1}/{len(jobs)}] {job['id']} -> {log}", flush=True)
            write_json(output / "status.json", {"state": "running", "job": job["id"], "completed": completed, "updated_utc": stamp()})
            with log.open("x", encoding="utf-8") as stream:
                run = subprocess.run(job["command"], cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            if run.returncode:
                raise RuntimeError(f"{job['id']} exited with code {run.returncode}. See {log}. No jobs will be retried automatically.")
            if not (Path(job["directory"]) / "metrics.json").is_file():
                raise RuntimeError(f"{job['id']} returned without metrics.json; inspect {log}.")
            completed.append(job["id"])
        selection = {}
        if "masked_fno" in args.methods:
            for seed in args.seeds:
                scores = {name: validation_value(json.loads((output / "case1" / name / f"seed_{seed}/metrics.json").read_text(encoding="utf-8"))) for name in ("masked_fno", "masked_fno_small")}
                selection[str(seed)] = {"criterion": "validation only", "selected_candidate": min(scores, key=scores.get), "validation_scores": scores}
            write_json(output / "masked_capacity_selection.json", selection)
        write_json(output / "status.json", {"state": "complete", "completed": completed, "finished_utc": stamp(), "masked_capacity_selection": selection})
    except (Exception, KeyboardInterrupt) as exc:
        write_json(output / "status.json", {"state": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "completed": completed, "error": str(exc), "finished_utc": stamp()})
        parser.exit(1, f"Stopped: {exc}\nPartial outputs are preserved at {output}; choose a new root for another run.\n")
    print(f"Complete: {len(completed)} jobs. Results: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
