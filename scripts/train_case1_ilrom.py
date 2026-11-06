"""Train the paper Case 1 IL-ROM, or fit its linear baseline, from raw data.

This entry point never reads pretrained weights or historical experiment outputs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from ilrom_basic import case1 as c
from ilrom_basic import ilrom as m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--method", choices=["ilrom", "ridge"], default="ilrom")
    p.add_argument("--config", type=Path, default=ROOT / "configs/case1_forced_convection/ilrom.json")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, help="Optional diagnostic override; omit for the paper configuration")
    a = p.parse_args()
    output = a.out or ROOT / "outputs/case1" / a.method / f"seed_{a.seed}"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        p.error(f"Output must be absent or empty: {output}")
    settings = json.loads(a.config.read_text(encoding="utf-8-sig"))
    if a.epochs is not None:
        if a.epochs < 1:
            p.error("--epochs must be positive")
        settings["epochs"] = a.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else torch.device(a.device)
    c.DEVICE = m.DEVICE = device
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    c.seed_everything(a.seed)
    cfg = m.TrainConfig(**{k: settings[k] for k in {f.name for f in fields(m.TrainConfig)} if k in settings})
    data = c.CaseData("case2", data_root=a.data_root)
    cache = c.FieldCache(data)
    rep = c.PODCurveRepresentation(settings["latent_dim"]).fit(cache.flux["train"])
    cond = c.LinearConditioner(cache, rep.encode(cache.flux["train"]), cfg.ridge)
    layouts = c.fixed_layouts("case2")
    ridge_choice = cond.select_ridge(layouts, rep)
    provenance = {
        str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((ROOT / "src/ilrom_basic").glob("*.py"))
    }
    provenance[str(Path(__file__).relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = {
        "paper_case": "case1", "historical_case": "case2", "method": a.method, "seed": a.seed,
        "trained_from_scratch": True, "pretrained_weights_loaded": False,
        "device": str(device), "config": asdict(cfg), "effective_d_model": 128,
        "latent_dim": settings["latent_dim"], "layout_ridge": ridge_choice,
        "train_val_test": [len(c.TRAIN_IDX), len(c.VAL_IDX), len(c.TEST_IDX)],
        "reserved_indices": c.RESERVED_IDX.tolist(), "source_code_sha256": provenance,
        "diagnostic_epoch_override": a.epochs is not None,
    }
    c.save_json(output / "run_config.json", payload)
    checkpoint = {
        "format": "ilrom_basic_case1_v1", "paper_case": "case1", "method": a.method,
        "seed": a.seed, "config": asdict(cfg), "representation": rep.state_dict(),
        "normalization": {k: getattr(data, k) for k in ("t_mean", "t_std", "u_mean", "u_std")},
        "layout_ridge": ridge_choice,
        "layouts": {name: asdict(layout) for name, layout in layouts.items()},
        "split_indices": {name: values.copy() for name, values in c.SPLITS.items()},
        "reserved_indices": c.RESERVED_IDX.copy(), "source_code_sha256": provenance,
        "linear_fits": {name: asdict(cond.fit(layout)) for name, layout in layouts.items()},
    }
    if a.method == "ridge":
        report, arrays = {}, {"truth": cache.flux["test"], "sample_indices": c.TEST_IDX.copy()}
        for name, layout in layouts.items():
            pred = rep.decode(cond.predict(cond.fit(layout), cache.raw_sensors("test", layout)))
            arrays[f"pred_{name}"] = pred
            report[name] = {"metrics": c.regression_metrics(pred, arrays["truth"])}
        payload["training"] = None
        payload["deterministic_fit"] = True
    else:
        trainer = m.Trainer(data, cache, rep, cond, cfg, a.seed)
        payload["parameters"] = sum(x.numel() for x in trainer.model.parameters())
        payload["training"] = trainer.train()
        report, arrays = trainer.evaluate()
        checkpoint["model"] = {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}
    payload["test"] = report
    payload["runtime_seconds"] = time.time() - started
    torch.save(checkpoint, output / "model.pt")
    np.savez_compressed(output / "predictions.npz", **arrays)
    c.save_json(output / "metrics.json", payload)
    print(json.dumps({"method": a.method, "seed": a.seed,
                      "rel_l2": {k: v["metrics"]["rel_l2"] for k, v in report.items()},
                      "output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
