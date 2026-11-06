"""Case 1 baseline training, preserving the main-table training protocols.

The original set encoder is the 64-dimensional, two-layer model selected on C1.
It is separate from the shared-backbone set-MSE ablation.
"""
from __future__ import annotations
import copy
from typing import Dict, Sequence, Tuple
import numpy as np
import torch
from torch.nn import functional as F
from ilrom_basic.case1 import (
    TRAIN_IDX, VAL_IDX, TEST_IDX, CaseData, SensorLayout, PODCurveRepresentation,
    fixed_layouts, regression_metrics, seed_everything,
)
from ilrom_basic.encoders import MultiModalSensorSetEncoder
from .deeponet import DeepONet1D
from .pod_gp import PODGaussianProcess
from .fno import FNORecon1D

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FNO_PATIENCE_EPOCHS=35
FNO_MIN_EPOCHS=60
STUDENT_PATIENCE_EPOCHS=60
STUDENT_MIN_EPOCHS=150

def batches(indices: Sequence[int], size: int):
    for start in range(0, len(indices), size):
        yield np.asarray(indices[start : start + size], dtype=np.int64)


def target_normalized(rep: PODCurveRepresentation, flux: np.ndarray) -> np.ndarray:
    return ((np.asarray(flux, dtype=np.float32) - rep.q_mean) / rep.q_std).astype(np.float32)


@torch.no_grad()
def predict_student(
    model: MultiModalSensorSetEncoder,
    data: CaseData,
    rep: PODCurveRepresentation,
    indices: Sequence[int],
    layout: SensorLayout,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    latent_parts = []
    curve_parts = []
    for idx in batches(indices, batch_size):
        values, modality, mask = data.tokens(idx, layout)
        latent = model(
            torch.from_numpy(values).to(DEVICE),
            torch.from_numpy(modality).to(DEVICE),
            torch.from_numpy(mask).to(DEVICE),
            None,
        )
        latent_parts.append(latent.cpu().numpy())
        curve_parts.append(rep.decode_torch(latent).cpu().numpy())
    return np.concatenate(latent_parts), np.concatenate(curve_parts)


@torch.no_grad()
def predict_fno(
    model: FNORecon1D,
    data: CaseData,
    rep: PODCurveRepresentation,
    indices: Sequence[int],
    layout: SensorLayout,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    parts = []
    for idx in batches(indices, batch_size):
        features = torch.from_numpy(data.fixed_vector(idx, layout)).to(DEVICE)
        normalized = model(features)
        parts.append((normalized * rep.q_std + rep.q_mean).cpu().numpy())
    return np.concatenate(parts)


def validation_rel(prediction: np.ndarray, truth: np.ndarray) -> float:
    return regression_metrics(prediction, truth)["rel_l2"]

def train_original_set(data: CaseData, rep: PODCurveRepresentation, latent_all: np.ndarray, initial_state: Dict[str, torch.Tensor], epochs: int, seed: int, learning_rate: float=0.0008) -> Tuple[MultiModalSensorSetEncoder, Dict[str, list]]:
    model = MultiModalSensorSetEncoder(n_modalities=2 if data.case == 'case2' else 1, latent_dim=rep.rank, d_model=64, nhead=4, nlayers=2, condition_dim=0).to(DEVICE)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, epochs))
    batch_size = 64
    layouts = fixed_layouts(data.case)
    val_truth = np.asarray(data.flux[VAL_IDX], dtype=np.float32)
    train_truth_n = target_normalized(rep, data.flux[TRAIN_IDX])
    _, initial_val_prediction = predict_student(model, data, rep, VAL_IDX, layouts['C1_standard'])
    best = validation_rel(initial_val_prediction, val_truth)
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history = {'train_loss': [], 'val_rel_l2': [[0, best]]}
    for epoch in range(epochs):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(TRAIN_IDX)
        losses = []
        for step, idx in enumerate(batches(order, batch_size)):
            layout_rng = np.random.default_rng(seed * 100000 + epoch * 1000 + step)
            layout = data.random_layout(layout_rng)
            values, modality, mask = data.tokens(idx, layout)
            target_z = torch.from_numpy(latent_all[idx]).to(DEVICE)
            local = np.searchsorted(TRAIN_IDX, idx)
            target_qn = torch.from_numpy(train_truth_n[local]).to(DEVICE)
            pred_z = model(torch.from_numpy(values).to(DEVICE), torch.from_numpy(modality).to(DEVICE), torch.from_numpy(mask).to(DEVICE), None)
            pred_qn = rep.decode_normalized_torch(pred_z)
            loss = F.mse_loss(pred_z, target_z) + 1.0 * F.l1_loss(pred_qn, target_qn)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        history['train_loss'].append(float(np.mean(losses)))
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            _, val_pred = predict_student(model, data, rep, VAL_IDX, layouts['C1_standard'])
            score = validation_rel(val_pred, val_truth)
            history['val_rel_l2'].append([epoch + 1, score])
            if score < best - 1e-05:
                best = score
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 5
            tag = 'plain'
            print(f'student-{tag} {data.case} seed={seed} epoch={epoch + 1}/{epochs} val_rel={score:.5f}', flush=True)
            if stale >= STUDENT_PATIENCE_EPOCHS and epoch + 1 >= STUDENT_MIN_EPOCHS:
                break
    model.load_state_dict(best_state)
    return (model, history, best)

def train_fno(
    data: CaseData,
    rep: PODCurveRepresentation,
    layout: SensorLayout,
    epochs: int,
    seed: int,
) -> Tuple[FNORecon1D, Dict[str, list]]:
    seed_everything(seed + 101)
    train_x = data.fixed_vector(TRAIN_IDX, layout)
    val_x = data.fixed_vector(VAL_IDX, layout)
    train_y = target_normalized(rep, data.flux[TRAIN_IDX])
    val_truth = np.asarray(data.flux[VAL_IDX], dtype=np.float32)
    model = FNORecon1D(train_x.shape[1], 64, data.grid, modes=16, width=48).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, epochs))
    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history = {"train_loss": [], "val_rel_l2": []}
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + 5000 + epoch).permutation(len(TRAIN_IDX))
        losses = []
        for local in batches(order, 64):
            x = torch.from_numpy(train_x[local]).to(DEVICE)
            y = torch.from_numpy(train_y[local]).to(DEVICE)
            pred = model(x)
            loss = F.l1_loss(pred, y) + 0.20 * F.mse_loss(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        history["train_loss"].append(float(np.mean(losses)))
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            pred_parts = []
            with torch.no_grad():
                for local in batches(np.arange(len(VAL_IDX)), 128):
                    qn = model(torch.from_numpy(val_x[local]).to(DEVICE))
                    pred_parts.append((qn * rep.q_std + rep.q_mean).cpu().numpy())
            score = validation_rel(np.concatenate(pred_parts), val_truth)
            history["val_rel_l2"].append([epoch + 1, score])
            if score < best - 1e-5:
                best = score
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 5
            print(f"fno {data.case} seed={seed} epoch={epoch + 1}/{epochs} val_rel={score:.5f}", flush=True)
            if stale >= FNO_PATIENCE_EPOCHS and epoch + 1 >= FNO_MIN_EPOCHS:
                break
    model.load_state_dict(best_state)
    return model, history

@torch.no_grad()
def predict_curve(model, observations: np.ndarray, rep: PODCurveRepresentation, batch_size: int = 128):
    model.eval()
    output = []
    for idx in batches(np.arange(len(observations)), batch_size):
        normalized = model(torch.from_numpy(observations[idx]).to(DEVICE))
        output.append((normalized * rep.q_std + rep.q_mean).cpu().numpy())
    return np.concatenate(output)


def train_deeponet(
    train_x: np.ndarray,
    val_x: np.ndarray,
    train_y: np.ndarray,
    val_truth: np.ndarray,
    rep: PODCurveRepresentation,
    seed: int,
    epochs: int,
):
    seed_everything(seed + 301)
    model = DeepONet1D(train_x.shape[1], train_y.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(1, epochs))
    best_score = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history = []
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + 7000 + epoch).permutation(len(train_x))
        for idx in batches(order, 64):
            x = torch.from_numpy(train_x[idx]).to(DEVICE)
            y = torch.from_numpy(train_y[idx]).to(DEVICE)
            pred = model(x)
            loss = F.l1_loss(pred, y) + 0.2 * F.mse_loss(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            prediction = predict_curve(model, val_x, rep)
            score = regression_metrics(prediction, val_truth)["rel_l2"]
            history.append([epoch + 1, score])
            if score < best_score - 1e-5:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 5
            if stale >= 60 and epoch >= 99:
                break
    model.load_state_dict(best_state)
    return model, history, best_score


def select_pod_gp(train_x, train_y, val_x, val_y, rank: int):
    x_n = (train_x - train_x.mean(0)) / (train_x.std(0) + 1e-8)
    subset = x_n[: min(300, len(x_n))]
    distance = np.sqrt(np.maximum(np.sum((subset[:, None] - subset[None]) ** 2, axis=-1), 0.0))
    median = float(np.median(distance[distance > 0]))
    best = None
    for scale in (0.5, 1.0, 2.0):
        for noise in (1e-5, 1e-3):
            model = PODGaussianProcess(rank, max(0.1, scale * median), noise).fit(train_x, train_y)
            result = model.predict(val_x)
            score = regression_metrics(result.mean, val_y)["rel_l2"]
            if best is None or score < best[0]:
                best = (score, model)
    return best[1], best[0]
