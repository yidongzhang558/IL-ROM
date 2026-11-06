"""Standalone Case 1 data, layouts, POD and linear conditioning.

Extracted from the paper implementation without changing numerical operations.
Paper Case 1 was named ``case2`` in the historical source tree.
"""
from __future__ import annotations
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from .data_download import ensure_temperature_file

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ILROM_DATA_ROOT", ROOT / "data"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_IDX = np.arange(0, 800, dtype=np.int64)
RESERVED_IDX = np.arange(800, 900, dtype=np.int64)
VAL_IDX = np.arange(900, 1100, dtype=np.int64)
TEST_IDX = np.arange(1100, 1600, dtype=np.int64)
SPLITS = {"train": TRAIN_IDX, "val": VAL_IDX, "test": TEST_IDX}
OLD_DENOM = 63.0
STD_ROW_FRAC = np.asarray([3.0, 8.0]) / OLD_DENOM
CASE2_STD_COL_FRAC = np.asarray([2.0, 7.0, 12.0, 18.0, 23.0, 28.0]) / OLD_DENOM
CASE2_UNSEEN_COL_FRAC = np.asarray([5.0, 11.0, 19.0, 27.0]) / OLD_DENOM
U_HEIGHTS = np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
U_COL_FRAC = 32.0 / OLD_DENOM
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PODCurveRepresentation:
    """Train-only POD fit with a differentiable frozen decoder."""

    def __init__(self, rank: int = 12):
        self.rank = int(rank)

    def fit(self, curves: np.ndarray) -> "PODCurveRepresentation":
        curves = np.asarray(curves, dtype=np.float64)
        self.q_mean = float(curves.mean())
        self.q_std = float(curves.std() + 1e-8)
        normalized = (curves - self.q_mean) / self.q_std
        self.mean_curve = normalized.mean(axis=0).astype(np.float32)
        centered = normalized - self.mean_curve
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.basis = vt[: self.rank].astype(np.float32)
        coefficients = centered @ self.basis.T
        self.coef_mean = coefficients.mean(axis=0).astype(np.float32)
        self.coef_std = (coefficients.std(axis=0) + 1e-6).astype(np.float32)
        return self

    def encode(self, curves: np.ndarray) -> np.ndarray:
        normalized = (np.asarray(curves) - self.q_mean) / self.q_std
        coefficients = (normalized - self.mean_curve) @ self.basis.T
        return ((coefficients - self.coef_mean) / self.coef_std).astype(np.float32)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        latent = np.asarray(latent)
        coefficients = latent * self.coef_std + self.coef_mean
        normalized = coefficients @ self.basis + self.mean_curve
        return (normalized * self.q_std + self.q_mean).astype(np.float32)

    def decode_normalized_torch(self, latent: torch.Tensor) -> torch.Tensor:
        coef_mean = torch.as_tensor(self.coef_mean, device=latent.device, dtype=latent.dtype)
        coef_std = torch.as_tensor(self.coef_std, device=latent.device, dtype=latent.dtype)
        basis = torch.as_tensor(self.basis, device=latent.device, dtype=latent.dtype)
        mean_curve = torch.as_tensor(self.mean_curve, device=latent.device, dtype=latent.dtype)
        return (latent * coef_std + coef_mean) @ basis + mean_curve

    def decode_torch(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decode_normalized_torch(latent) * self.q_std + self.q_mean

    def state_dict(self) -> Dict[str, object]:
        return {
            "rank": self.rank,
            "q_mean": self.q_mean,
            "q_std": self.q_std,
            "mean_curve": self.mean_curve,
            "basis": self.basis,
            "coef_mean": self.coef_mean,
            "coef_std": self.coef_std,
        }


@dataclass(frozen=True)
class SensorLayout:
    name: str
    row_frac: Tuple[float, ...]
    col_frac: Tuple[float, ...]
    with_velocity: bool

    @property
    def temperature_count(self) -> int:
        return len(self.row_frac)


def _paired_layout(name: str, columns: Sequence[float], with_velocity: bool) -> SensorLayout:
    rows = np.repeat(STD_ROW_FRAC, len(columns))
    cols = np.tile(np.asarray(columns), 2)
    return SensorLayout(name, tuple(rows.tolist()), tuple(cols.tolist()), with_velocity)


def fixed_layouts(case: str) -> Dict[str, SensorLayout]:
    if case not in ("case2", "case1_forced_convection"):
        raise ValueError("This standalone module implements paper Case 1 (historical case2)")
    case = "case2"
    use_u = True
    standard_columns = CASE2_STD_COL_FRAC
    unseen_columns = CASE2_UNSEEN_COL_FRAC
    layouts = {
        "C1_standard": _paired_layout("C1_standard", standard_columns, use_u),
        "C2_subset6T": _paired_layout("C2_subset6T", standard_columns[[0, 2, 4]], use_u),
        "C3_unseen8T": _paired_layout("C3_unseen8T", unseen_columns, use_u),
        "C5_sparse4T": _paired_layout("C5_sparse4T", standard_columns[[1, 4]], use_u),
    }
    if case == "case2":
        layouts["C4_no_u"] = _paired_layout("C4_no_u", standard_columns, False)
        return {
            key: layouts[key]
            for key in ("C1_standard", "C2_subset6T", "C3_unseen8T", "C4_no_u", "C5_sparse4T")
        }
    return layouts


class CaseData:
    def __init__(self, case: str = "case2", data_root=None):
        if case not in ("case2", "case1_forced_convection"):
            raise ValueError("This standalone module implements paper Case 1 (historical case2)")
        self.case = "case2"
        root = Path(data_root) if data_root is not None else DATA_ROOT
        directory = root if (root / "T.npy").exists() else root / "case1_forced_convection"
        ensure_temperature_file(directory)
        for name in ("T.npy", "qw.npy", "umean.npy", "metadata.json"):
            if not (directory / name).exists():
                raise FileNotFoundError(directory / name)
        self.temperature = np.load(directory / "T.npy", mmap_mode="r")
        self.flux = np.load(directory / "qw.npy", mmap_mode="r")
        self.umean = np.load(directory / "umean.npy", mmap_mode="r")
        self.grid = int(self.temperature.shape[-1])
        if len(self.temperature) < 1600:
            raise ValueError(f"{case} needs 1600 samples, found {len(self.temperature)}")
        train_t = np.asarray(self.temperature[TRAIN_IDX])
        self.t_mean = float(train_t.mean())
        self.t_std = float(train_t.std() + 1e-8)
        if self.umean is not None:
            train_u = self.velocity_values(TRAIN_IDX)
            self.u_mean = float(train_u.mean())
            self.u_std = float(train_u.std() + 1e-8)
            self.condition_mean = float(np.asarray(self.umean[TRAIN_IDX]).mean())
            self.condition_std = float(np.asarray(self.umean[TRAIN_IDX]).std() + 1e-8)
        else:
            self.u_mean = 0.0
            self.u_std = 1.0
            self.condition_mean = 0.0
            self.condition_std = 1.0

    def velocity_values(self, indices: Sequence[int]) -> np.ndarray:
        if self.umean is None:
            raise ValueError("case has no velocity observations")
        mean_velocity = np.asarray(self.umean[indices], dtype=np.float32)
        return 6.0 * mean_velocity[:, None] * U_HEIGHTS[None, :] * (1.0 - U_HEIGHTS[None, :])

    def conditions(self, indices: Sequence[int]) -> np.ndarray | None:
        if self.umean is None:
            return None
        values = np.asarray(self.umean[indices], dtype=np.float32)
        return ((values - self.condition_mean) / self.condition_std)[:, None].astype(np.float32)

    def normalized_field(self, indices: Sequence[int]) -> np.ndarray:
        values = np.asarray(self.temperature[indices], dtype=np.float32)
        return ((values - self.t_mean) / self.t_std)[:, None]

    def fixed_vector(self, indices: Sequence[int], layout: SensorLayout) -> np.ndarray:
        tokens, modalities, _ = self.tokens(indices, layout)
        values = tokens[:, :, 2]
        modality_one_hot = np.eye(2 if self.case == "case2" else 1, dtype=np.float32)[modalities[0]]
        descriptor = np.concatenate([tokens[:, :, :2], np.broadcast_to(modality_one_hot, (len(indices),) + modality_one_hot.shape), values[:, :, None]], axis=-1)
        return descriptor.reshape(len(indices), -1).astype(np.float32)

    def tokens(
        self, indices: Sequence[int], layout: SensorLayout
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.asarray(indices, dtype=np.int64)
        rows = np.clip(np.rint(np.asarray(layout.row_frac) * (self.grid - 1)).astype(int), 0, self.grid - 1)
        cols = np.clip(np.rint(np.asarray(layout.col_frac) * (self.grid - 1)).astype(int), 0, self.grid - 1)
        field_batch = np.asarray(self.temperature[indices], dtype=np.float32)
        temperature_values = field_batch[:, rows, cols]
        temperature_values = (temperature_values - self.t_mean) / self.t_std
        count_t = len(rows)
        count_u = 4 if layout.with_velocity and self.umean is not None else 0
        length = count_t + count_u
        values = np.zeros((len(indices), length, 3), dtype=np.float32)
        modalities = np.zeros((len(indices), length), dtype=np.int64)
        mask = np.zeros((len(indices), length), dtype=bool)
        values[:, :count_t, 0] = np.asarray(layout.row_frac, dtype=np.float32)
        values[:, :count_t, 1] = np.asarray(layout.col_frac, dtype=np.float32)
        values[:, :count_t, 2] = temperature_values
        if count_u:
            velocity = (self.velocity_values(indices) - self.u_mean) / self.u_std
            values[:, count_t:, 0] = U_HEIGHTS
            values[:, count_t:, 1] = U_COL_FRAC
            values[:, count_t:, 2] = velocity
            modalities[:, count_t:] = 1
        return values, modalities, mask

    def random_layout(self, rng: np.random.Generator) -> SensorLayout:
        with_velocity = self.case == "case2" and bool(rng.random() > 0.20)
        standard_columns = CASE2_STD_COL_FRAC
        standard_pool = [(row, col) for row in STD_ROW_FRAC for col in standard_columns]
        draw = rng.random()
        if draw < 0.70:
            pairs = standard_pool
        elif draw < 0.85:
            count = int(rng.integers(4, 13))
            chosen = rng.choice(len(standard_pool), size=count, replace=False)
            pairs = [standard_pool[int(i)] for i in chosen]
        else:
            count = int(rng.integers(6, 15))
            column_stop = 45 if self.case == "case2" else 62
            old_columns = rng.choice(np.arange(2, column_stop), size=count, replace=False) / OLD_DENOM
            if rng.random() < 0.70:
                rows = rng.choice(STD_ROW_FRAC, size=count, replace=True)
            else:
                rows = rng.uniform(2.0 / OLD_DENOM, 10.0 / OLD_DENOM, size=count)
            pairs = list(zip(rows.tolist(), old_columns.tolist()))
        return SensorLayout(
            "train_random",
            tuple(float(pair[0]) for pair in pairs),
            tuple(float(pair[1]) for pair in pairs),
            with_velocity,
        )


def regression_metrics(prediction: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    error = prediction - truth
    sample_norm = np.linalg.norm(truth, axis=1)
    peak_truth_index = np.argmax(np.abs(truth), axis=1)
    peak_pred_index = np.argmax(np.abs(prediction), axis=1)
    peak_truth = np.max(np.abs(truth), axis=1)
    peak_pred = np.max(np.abs(prediction), axis=1)
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "rel_l2": float(np.mean(np.linalg.norm(error, axis=1) / np.maximum(sample_norm, 1e-8))),
        "peak_relative_error": float(np.mean(np.abs(peak_pred - peak_truth) / np.maximum(peak_truth, 1e-8))),
        "peak_location_error": float(np.mean(np.abs(peak_pred_index - peak_truth_index) / max(1, truth.shape[1] - 1))),
        "r2": float(1.0 - ss_res / max(ss_tot, 1e-12)),
    }


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class FieldCache:
    """Training/validation/test fields held in RAM with fast layout gathers."""

    def __init__(self, data: CaseData):
        self.data = data
        self.grid = int(data.grid)
        self.fields = {name: np.asarray(data.temperature[idx], dtype=np.float32) for name, idx in SPLITS.items()}
        self.flux = {name: np.asarray(data.flux[idx], dtype=np.float32) for name, idx in SPLITS.items()}
        self.umean = None
        if data.umean is not None:
            self.umean = {name: np.asarray(data.umean[idx], dtype=np.float32) for name, idx in SPLITS.items()}
        self.n_modalities = 2 if self.umean is not None else 1

    def grid_index(self, layout: SensorLayout) -> Tuple[np.ndarray, np.ndarray]:
        rows = np.clip(np.rint(np.asarray(layout.row_frac) * (self.grid - 1)).astype(int), 0, self.grid - 1)
        cols = np.clip(np.rint(np.asarray(layout.col_frac) * (self.grid - 1)).astype(int), 0, self.grid - 1)
        return rows, cols

    def velocity_count(self, layout: SensorLayout) -> int:
        return 4 if (layout.with_velocity and self.umean is not None) else 0

    def raw_sensors(self, split: str, layout: SensorLayout, local: Optional[np.ndarray] = None) -> np.ndarray:
        """Raw (un-normalised) sensor readings, temperature tokens first, then velocity tokens."""
        fields = self.fields[split] if local is None else self.fields[split][local]
        rows, cols = self.grid_index(layout)
        parts = [fields[:, rows, cols]]
        if self.velocity_count(layout):
            um = self.umean[split] if local is None else self.umean[split][local]
            parts.append(6.0 * um[:, None] * U_HEIGHTS[None, :] * (1.0 - U_HEIGHTS[None, :]))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def tokens(self, split: str, local: np.ndarray, layout: SensorLayout):
        """Normalised token tensors identical in convention to CaseData.tokens."""
        raw = self.raw_sensors(split, layout, local)
        count_t = layout.temperature_count
        count_u = self.velocity_count(layout)
        length = count_t + count_u
        values = np.zeros((len(local), length, 3), dtype=np.float32)
        modality = np.zeros((len(local), length), dtype=np.int64)
        values[:, :count_t, 0] = np.asarray(layout.row_frac, dtype=np.float32)
        values[:, :count_t, 1] = np.asarray(layout.col_frac, dtype=np.float32)
        values[:, :count_t, 2] = (raw[:, :count_t] - self.data.t_mean) / self.data.t_std
        if count_u:
            values[:, count_t:, 0] = U_HEIGHTS
            values[:, count_t:, 1] = U_COL_FRAC
            values[:, count_t:, 2] = (raw[:, count_t:] - self.data.u_mean) / self.data.u_std
            modality[:, count_t:] = 1
        mask = np.zeros((len(local), length), dtype=bool)
        return (
            torch.from_numpy(values).to(DEVICE),
            torch.from_numpy(modality).to(DEVICE),
            torch.from_numpy(mask).to(DEVICE),
        )


def sub_layout(layout: SensorLayout, rng: np.random.Generator) -> SensorLayout:
    """Random nested sub-layout A of B for the tower-consistency term."""
    count = layout.temperature_count
    keep = max(2, int(round(rng.uniform(0.3, 0.8) * count)))
    keep = min(keep, count)
    chosen = np.sort(rng.choice(count, size=keep, replace=False))
    with_velocity = bool(layout.with_velocity and rng.random() < 0.7)
    return SensorLayout(
        "tower_sub",
        tuple(float(layout.row_frac[i]) for i in chosen),
        tuple(float(layout.col_frac[i]) for i in chosen),
        with_velocity,
    )


@dataclass
class LinearFit:
    mean_o: np.ndarray
    std_o: np.ndarray
    weight: np.ndarray       # (m, r)
    mean_z: np.ndarray       # (r,)
    loo_var: np.ndarray      # (r,) leave-one-out residual variance per latent mode
    loo_train: np.ndarray    # (N, r) leave-one-out predictions for the training set
    ridge: float


class LinearConditioner:
    """Closed-form ridge regression of the standardised latent on any layout's sensors."""

    def __init__(self, cache: FieldCache, latent_train: np.ndarray, ridge: float = 1e-3):
        self.cache = cache
        self.z = np.asarray(latent_train, dtype=np.float64)
        self.ridge = float(ridge)
        self.layout_ridge: Dict[str, float] = {}
        self._cache: Dict[tuple, LinearFit] = {}

    def _key(self, layout: SensorLayout) -> Optional[tuple]:
        if layout.name in ("train_random", "tower_sub"):
            return None
        # The display name alone is not a layout identity. Keep its name in the
        # key because historical named layouts can have different ridge values.
        return (layout.name, tuple(layout.row_frac), tuple(layout.col_frac),
                bool(layout.with_velocity))

    def fit(self, layout: SensorLayout, ridge: Optional[float] = None) -> LinearFit:
        key = self._key(layout)
        if key is not None and ridge is None and key in self._cache:
            return self._cache[key]
        if ridge is None:
            ridge = self.layout_ridge.get(layout.name, self.ridge) if key is not None else self.ridge
        raw = self.cache.raw_sensors("train", layout).astype(np.float64)
        mean_o = raw.mean(axis=0)
        std_o = raw.std(axis=0) + 1e-8
        oc = (raw - mean_o) / std_o
        n, m = oc.shape
        mean_z = self.z.mean(axis=0)
        zc = self.z - mean_z
        gram = oc.T @ oc / n + ridge * np.eye(m)
        gram_inv = np.linalg.inv(gram)
        weight = gram_inv @ (oc.T @ zc / n)
        pred = oc @ weight
        leverage = np.einsum("ij,jk,ik->i", oc, gram_inv, oc) / n + 1.0 / n
        leverage = np.clip(leverage, 0.0, 0.95)
        residual = zc - pred
        loo = pred - (leverage / (1.0 - leverage))[:, None] * residual
        loo_var = np.mean((zc - loo) ** 2, axis=0) + 1e-6
        fit = LinearFit(mean_o, std_o, weight, mean_z, loo_var, loo + mean_z, float(ridge))
        if key is not None and (ridge == self.layout_ridge.get(layout.name, self.ridge)):
            self._cache[key] = fit
        return fit

    def predict(self, fit: LinearFit, raw: np.ndarray) -> np.ndarray:
        oc = (np.asarray(raw, dtype=np.float64) - fit.mean_o) / fit.std_o
        return (oc @ fit.weight + fit.mean_z).astype(np.float32)

    def select_ridge(self, layouts: Dict[str, SensorLayout], rep: PODCurveRepresentation) -> Dict[str, float]:
        """Choose the ridge of every named layout on the validation set (never the test set)."""
        truth = self.cache.flux["val"]
        for name, layout in layouts.items():
            best = None
            for ridge in RIDGE_GRID:
                fit = self.fit(layout, ridge=ridge)
                pred = rep.decode(self.predict(fit, self.cache.raw_sensors("val", layout)))
                score = regression_metrics(pred, truth)["rel_l2"]
                if best is None or score < best[0]:
                    best = (score, ridge)
            self.layout_ridge[name] = best[1]
            self._cache.pop(self._key(layout), None)
        return dict(self.layout_ridge)
