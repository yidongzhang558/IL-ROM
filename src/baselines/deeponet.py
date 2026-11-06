"""DeepONet baseline architectures used in the main comparisons."""
from __future__ import annotations
import math
import torch
from torch import nn

class DeepONet1D(nn.Module):
    """Branch-trunk operator network for sensor-vector to curve reconstruction."""

    def __init__(self, n_obs: int, n_output: int, width: int = 160, basis_dim: int = 128):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(n_obs, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, basis_dim),
        )
        self.frequencies = tuple(float(value) for value in (1, 2, 3, 4, 6, 8, 12, 16))
        trunk_input = 1 + 2 * len(self.frequencies)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, basis_dim),
        )
        self.register_buffer("coords", torch.linspace(0.0, 1.0, n_output)[:, None])
        self.bias = nn.Parameter(torch.zeros(1))
        self.scale = math.sqrt(float(basis_dim))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        branch = self.branch(observations)
        features = [self.coords]
        for frequency in self.frequencies:
            features.extend([
                torch.sin(2.0 * math.pi * frequency * self.coords),
                torch.cos(2.0 * math.pi * frequency * self.coords),
            ])
        trunk = self.trunk(torch.cat(features, dim=1))
        return branch @ trunk.T / self.scale + self.bias


class DeepONet2D(nn.Module):
    """Coordinate-query DeepONet for native-resolution 2-D reconstruction."""

    def __init__(self, n_obs: int, width: int = 160, basis_dim: int = 128,
                 frequencies=(1, 2, 4, 8)):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(n_obs, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, basis_dim),
        )
        self.frequencies = tuple(float(value) for value in frequencies)
        trunk_input = 2 + 4 * len(self.frequencies)
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, basis_dim),
        )
        self.bias = nn.Parameter(torch.zeros(1))
        self.scale = math.sqrt(float(basis_dim))

    def forward(self, observations: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        branch = self.branch(observations)
        features = [coordinates]
        for frequency in self.frequencies:
            for axis in range(2):
                value = coordinates[:, axis : axis + 1]
                features.extend([
                    torch.sin(math.pi * frequency * value),
                    torch.cos(math.pi * frequency * value),
                ])
        trunk = self.trunk(torch.cat(features, dim=1))
        return branch @ trunk.T / self.scale + self.bias
