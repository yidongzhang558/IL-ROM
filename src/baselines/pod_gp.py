"""Train-only POD with an exact RBF Gaussian-process baseline."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def pairwise_squared_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    return np.maximum(left_norm + right_norm - 2.0 * left @ right.T, 0.0)


@dataclass
class PODGPResult:
    mean: np.ndarray
    std: np.ndarray


class PODGaussianProcess:
    """POD coefficients regressed by a shared exact RBF Gaussian process.

    A single covariance matrix is shared by all POD coefficients. This is the
    standard independent-output GP assumption and keeps the baseline practical
    for high-dimensional fields while retaining predictive uncertainty.
    """

    def __init__(self, rank: int = 12, length_scale: float = 1.0, noise: float = 1e-4):
        self.rank = int(rank)
        self.length_scale = float(length_scale)
        self.noise = float(noise)

    def fit(self, observations: np.ndarray, targets: np.ndarray) -> "PODGaussianProcess":
        x = np.asarray(observations, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        self.target_shape = y.shape[1:]
        flat = y.reshape(len(y), -1)
        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0) + 1e-8
        self.x_train = (x - self.x_mean) / self.x_std
        self.y_mean = flat.mean(axis=0)
        centered = flat - self.y_mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.basis = vt[: min(self.rank, len(vt))]
        coefficients = centered @ self.basis.T
        self.coef_mean = coefficients.mean(axis=0)
        self.coef_std = coefficients.std(axis=0) + 1e-8
        coefficient_n = (coefficients - self.coef_mean) / self.coef_std
        distance = pairwise_squared_distance(self.x_train, self.x_train)
        kernel = np.exp(-0.5 * distance / (self.length_scale**2))
        jitter = self.noise
        for _ in range(8):
            try:
                self.cholesky = np.linalg.cholesky(kernel + jitter * np.eye(len(kernel)))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            raise np.linalg.LinAlgError("POD-GP covariance matrix is not positive definite")
        self.effective_noise = jitter
        self.alpha = np.linalg.solve(self.cholesky.T, np.linalg.solve(self.cholesky, coefficient_n))
        return self

    def predict(self, observations: np.ndarray) -> PODGPResult:
        x = (np.asarray(observations, dtype=np.float64) - self.x_mean) / self.x_std
        distance = pairwise_squared_distance(x, self.x_train)
        cross = np.exp(-0.5 * distance / (self.length_scale**2))
        coefficient_n = cross @ self.alpha
        coefficients = coefficient_n * self.coef_std + self.coef_mean
        flat_mean = coefficients @ self.basis + self.y_mean
        solved = np.linalg.solve(self.cholesky, cross.T)
        latent_variance = np.maximum(1.0 - np.sum(solved * solved, axis=0), 1e-10)
        field_variance = latent_variance[:, None] * np.sum(
            (self.coef_std[:, None] * self.basis) ** 2, axis=0, keepdims=True
        )
        return PODGPResult(
            mean=flat_mean.reshape((len(x),) + self.target_shape).astype(np.float32),
            std=np.sqrt(field_variance).reshape((len(x),) + self.target_shape).astype(np.float32),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "length_scale": self.length_scale,
            "noise": self.noise,
            "effective_noise": self.effective_noise,
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "x_train": self.x_train,
            "y_mean": self.y_mean,
            "basis": self.basis,
            "coef_mean": self.coef_mean,
            "coef_std": self.coef_std,
            "cholesky": self.cholesky,
            "alpha": self.alpha,
            "target_shape": self.target_shape,
        }
