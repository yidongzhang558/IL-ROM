"""Case 1 main-model training, with unchanged original optimizer and RNG."""
from __future__ import annotations
import copy
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .case1 import (DEVICE, TRAIN_IDX, TEST_IDX, SPLITS, CaseData, FieldCache,
    LinearConditioner, PODCurveRepresentation, SensorLayout, fixed_layouts,
    regression_metrics, sub_layout, seed_everything)
from .encoders import MultiModalSensorSetEncoder
VARIANTS = {"full": {"use_linear": True, "latent_mode": "nll", "use_tower": True}}

class ResidualSetModel(nn.Module):
    """Set encoder that outputs a residual to the linear latent and a per-mode log-variance."""

    def __init__(self, n_modalities: int, latent_dim: int, d_model: int = 96, nhead: int = 4,
                 nlayers: int = 3, use_linear: bool = True):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.use_linear = bool(use_linear)
        self.encoder = MultiModalSensorSetEncoder(
            n_modalities=n_modalities, latent_dim=latent_dim, d_model=d_model,
            nhead=nhead, nlayers=nlayers, condition_dim=0,
        )
        head_in = d_model + (2 * latent_dim if self.use_linear else 0)
        self.delta_head = nn.Sequential(nn.Linear(head_in, d_model), nn.GELU(), nn.Linear(d_model, latent_dim))
        self.logvar_head = nn.Sequential(nn.Linear(head_in, d_model), nn.GELU(), nn.Linear(d_model, latent_dim))
        for head in (self.delta_head, self.logvar_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, values, modality, mask, z_lin, logvar_lin):
        _, pooled = self.encoder(values, modality, mask, None, return_features=True)
        if self.use_linear:
            features = torch.cat([pooled, z_lin, logvar_lin], dim=-1)
            z = z_lin + self.delta_head(features)
            logvar = logvar_lin + self.logvar_head(features)
        else:
            features = pooled
            z = self.delta_head(features)
            logvar = self.logvar_head(features)
        return z, torch.clamp(logvar, -7.0, 5.0)


@dataclass
class TrainConfig:
    variant: str
    epochs: int
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 1e-4
    latent_mode: str = "nll"          # "mse" | "nll"
    use_linear: bool = True
    use_tower: bool = True
    sub_layout_supervision: bool = True
    lambda_latent: float = 1.0
    lambda_response: float = 1.0
    gamma_tower: float = 0.5
    tower_ramp: float = 0.3
    nll_warmup: float = 0.1
    eval_every: int = 10
    patience_evals: int = 8
    min_epochs: int = 150
    select: str = "mean"              # "mean" over fixed layouts | "C1"
    d_model: int = 96
    nlayers: int = 3
    nhead: int = 4
    ridge: float = 1e-3
    encoder: str = "v8"               # "v8" = V8 set encoder; "v8_2" = high-frequency features + relative-position bias + multi-query pooling


class Trainer:
    def __init__(self, data: CaseData, cache: FieldCache, rep: PODCurveRepresentation,
                 conditioner: LinearConditioner, cfg: TrainConfig, seed: int):
        self.data, self.cache, self.rep, self.cond, self.cfg, self.seed = data, cache, rep, conditioner, cfg, seed
        self.layouts = fixed_layouts(data.case)
        self.z_star = {split: rep.encode(cache.flux[split]) for split in SPLITS}
        self.y_norm = {split: ((cache.flux[split] - rep.q_mean) / rep.q_std).astype(np.float32) for split in SPLITS}
        seed_everything(seed + 1)
        if cfg.encoder == "v8_2":
            from .rb_encoder import RBResidualModel
            self.model = RBResidualModel(cache.n_modalities, rep.rank, n_out=cache.flux["train"].shape[1], d_model=128,
                                         nhead=cfg.nhead, nlayers=cfg.nlayers, n_queries=8, local=False, periodic=False).to(DEVICE)
        else:
            self.model = ResidualSetModel(
                cache.n_modalities, rep.rank, cfg.d_model, cfg.nhead, cfg.nlayers, cfg.use_linear
            ).to(DEVICE)

    # ---- forward helpers
    def linear_inputs(self, split: str, local: np.ndarray, layout: SensorLayout, loo: bool):
        fit = self.cond.fit(layout)
        if loo and split == "train":
            z_lin = fit.loo_train[local].astype(np.float32)
        else:
            z_lin = self.cond.predict(fit, self.cache.raw_sensors(split, layout, local))
        logvar = np.log(fit.loo_var.astype(np.float32))[None, :].repeat(len(local), axis=0)
        return (torch.from_numpy(z_lin).to(DEVICE), torch.from_numpy(logvar.astype(np.float32)).to(DEVICE))

    def forward(self, split: str, local: np.ndarray, layout: SensorLayout, loo: bool):
        values, modality, mask = self.cache.tokens(split, local, layout)
        z_lin, logvar_lin = self.linear_inputs(split, local, layout, loo)
        # RBResidualModel has an unconditional residual addition. Disable both
        # prior channels at its boundary, while retaining the real ridge output
        # below for the independent linear-only diagnostics.
        model_z_lin = z_lin if self.cfg.use_linear else torch.zeros_like(z_lin)
        model_logvar_lin = logvar_lin if self.cfg.use_linear else torch.zeros_like(logvar_lin)
        out = self.model(values, modality, mask, model_z_lin, model_logvar_lin)
        z, logvar = out[0], out[1]
        return z, logvar, z_lin

    def supervised(self, z, logvar, local: np.ndarray, nll_active: bool) -> torch.Tensor:
        target_z = torch.from_numpy(self.z_star["train"][local]).to(DEVICE)
        target_y = torch.from_numpy(self.y_norm["train"][local]).to(DEVICE)
        response = F.l1_loss(self.rep.decode_normalized_torch(z), target_y)
        if self.cfg.latent_mode == "nll" and nll_active:
            latent = 0.5 * (torch.exp(-logvar) * (z - target_z) ** 2 + logvar).mean()
        else:
            latent = F.mse_loss(z, target_z)
        return self.cfg.lambda_response * response + self.cfg.lambda_latent * latent

    # ---- evaluation
    @torch.no_grad()
    def predict_split(self, split: str, layout: SensorLayout, batch: int = 256):
        self.model.eval()
        n = len(self.cache.flux[split])
        zs, lvs, zl = [], [], []
        for start in range(0, n, batch):
            local = np.arange(start, min(n, start + batch))
            z, logvar, z_lin = self.forward(split, local, layout, loo=False)
            zs.append(z.cpu().numpy()); lvs.append(logvar.cpu().numpy()); zl.append(z_lin.cpu().numpy())
        z = np.concatenate(zs); logvar = np.concatenate(lvs); z_lin = np.concatenate(zl)
        return z, logvar, z_lin, self.rep.decode(z)

    def validation_score(self) -> Tuple[float, Dict[str, float]]:
        scores = {}
        for name, layout in self.layouts.items():
            _, _, _, pred = self.predict_split("val", layout)
            scores[name] = regression_metrics(pred, self.cache.flux["val"])["rel_l2"]
        score = scores["C1_standard"] if self.cfg.select == "C1" else float(np.mean(list(scores.values())))
        return score, scores

    # ---- training loop
    def train(self) -> Dict[str, object]:
        cfg = self.cfg
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, cfg.epochs))
        n_train = len(TRAIN_IDX)
        best_score, best_scores = self.validation_score()
        best_state = copy.deepcopy(self.model.state_dict())
        history = {"train_loss": [], "val": [[0, best_score, best_scores]]}
        stale = 0
        started = time.time()
        for epoch in range(cfg.epochs):
            self.model.train()
            progress = epoch / max(1, cfg.epochs - 1)
            nll_active = progress >= cfg.nll_warmup
            gamma = cfg.gamma_tower * min(1.0, progress / max(1e-6, cfg.tower_ramp)) if cfg.use_tower else 0.0
            order = np.random.default_rng(self.seed + epoch).permutation(n_train)
            losses = []
            for step, start in enumerate(range(0, n_train, cfg.batch_size)):
                local = order[start:start + cfg.batch_size]
                layout_rng = np.random.default_rng(self.seed * 100000 + epoch * 1000 + step)
                layout_b = self.data.random_layout(layout_rng)
                z_b, logvar_b, _ = self.forward("train", local, layout_b, loo=True)
                loss = self.supervised(z_b, logvar_b, local, nll_active)
                if cfg.sub_layout_supervision or cfg.use_tower:
                    layout_a = sub_layout(layout_b, layout_rng)
                    z_a, logvar_a, _ = self.forward("train", local, layout_a, loo=True)
                    if cfg.sub_layout_supervision:
                        loss = loss + self.supervised(z_a, logvar_a, local, nll_active)
                    if cfg.use_tower and gamma > 0:
                        loss = loss + gamma * F.mse_loss(z_a, z_b.detach())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            scheduler.step()
            history["train_loss"].append(float(np.mean(losses)))
            if (epoch + 1) % cfg.eval_every == 0 or epoch + 1 == cfg.epochs:
                score, scores = self.validation_score()
                history["val"].append([epoch + 1, score, scores])
                if score < best_score - 1e-6:
                    best_score, best_scores = score, scores
                    best_state = copy.deepcopy(self.model.state_dict())
                    stale = 0
                else:
                    stale += 1
                print(
                    f"[V8 {cfg.variant} {self.data.case} seed={self.seed}] epoch={epoch + 1}/{cfg.epochs} "
                    f"val={score:.5f} C1={scores['C1_standard']:.5f} best={best_score:.5f} "
                    f"loss={history['train_loss'][-1]:.4f} elapsed={time.time() - started:.0f}s",
                    flush=True,
                )
                if stale >= cfg.patience_evals and epoch + 1 >= cfg.min_epochs:
                    break
        self.model.load_state_dict(best_state)
        return {"best_validation": best_score, "best_validation_layouts": best_scores, "history": history,
                "epochs_run": len(history["train_loss"]), "train_seconds": time.time() - started}

    # Basic C1-C5 reconstruction only; analysis sections are excluded.
    def evaluate(self):
        truth = self.cache.flux["test"]
        report, arrays = {}, {"truth": truth, "sample_indices": TEST_IDX.copy()}
        for name, layout in self.layouts.items():
            z, logvar, z_lin, pred = self.predict_split("test", layout)
            linear_pred = self.rep.decode(z_lin)
            report[name] = {"metrics": regression_metrics(pred, truth),
                            "linear_only_metrics": regression_metrics(linear_pred, truth)}
            arrays[f"pred_{name}"] = pred
            arrays[f"linear_{name}"] = linear_pred
        return report, arrays


def config_from_variant(variant: str, case: str, epochs: Optional[int], **overrides) -> TrainConfig:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}; choose from {sorted(VARIANTS)}")
    default_epochs = 400 if case == "case1" else 300
    fields = dict(VARIANTS[variant])
    fields.update({key: value for key, value in overrides.items() if value is not None})
    return TrainConfig(variant=variant, epochs=epochs or default_epochs, **fields)


def config_to_dict(cfg: TrainConfig) -> Dict[str, object]:
    return asdict(cfg)
