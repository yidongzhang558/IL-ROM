"""High-frequency set encoder used by the main IL-ROM configuration."""
from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def fourier(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    ang = 2.0 * math.pi * x.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


def periodic_dist2(cx_q: torch.Tensor, cy_q: torch.Tensor, cx_k: torch.Tensor, cy_k: torch.Tensor, periodic: bool = True) -> torch.Tensor:
    """Squared distance between query and key positions; x is periodic (fractions in [0, 1]) when `periodic`."""
    dx = (cx_q.unsqueeze(-1) - cx_k.unsqueeze(-2)).abs()
    if periodic:
        dx = torch.minimum(dx, 1.0 - dx)
    dy = cy_q.unsqueeze(-1) - cy_k.unsqueeze(-2)
    return dx ** 2 + dy ** 2


class RelBiasAttention(nn.Module):
    """Multi-head attention with an additive learnable Gaussian relative-position bias per head."""

    def __init__(self, d_model: int, nhead: int, init_scale: float = 8.0):
        super().__init__()
        self.nhead, self.dh = nhead, d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.log_scale = nn.Parameter(torch.full((nhead,), math.log(init_scale)))

    def forward(self, xq, xk, dist2, key_mask):
        """xq: (B, Lq, d), xk: (B, Lk, d), dist2: (B, Lq, Lk), key_mask: (B, Lk) True at padding."""
        b, lq, _ = xq.shape
        lk = xk.shape[1]
        q = self.qkv(xq)[..., : self.nhead * self.dh].reshape(b, lq, self.nhead, self.dh).transpose(1, 2)
        kv = self.qkv(xk)[..., self.nhead * self.dh:].reshape(b, lk, 2, self.nhead, self.dh)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dh)                                    # (B, H, Lq, Lk)
        scores = scores - torch.exp(self.log_scale)[None, :, None, None] * dist2.unsqueeze(1)
        scores = scores.masked_fill(key_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, lq, self.nhead * self.dh)
        return self.out(out)


class RelBiasLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = RelBiasAttention(d_model, nhead)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))

    def forward(self, x, dist2, key_mask):
        h = self.norm1(x)
        x = x + self.attn(h, h, dist2, key_mask)
        return x + self.ff(self.norm2(x))


class RBSetEncoder(nn.Module):
    def __init__(self, n_modalities: int, d_model: int = 128, nhead: int = 4, nlayers: int = 3,
                 freqs=(1, 2, 4, 8, 16, 32), n_queries: int = 8, periodic: bool = True):
        super().__init__()
        self.periodic = bool(periodic)
        self.register_buffer("freqs", torch.tensor([float(f) for f in freqs]), persistent=False)
        feat = 3 + 4 * len(freqs)
        self.embed = nn.Sequential(nn.Linear(feat, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.modality = nn.Embedding(n_modalities, d_model)
        self.layers = nn.ModuleList([RelBiasLayer(d_model, nhead) for _ in range(nlayers)])
        self.norm = nn.LayerNorm(d_model)
        self.queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.pool = RelBiasAttention(d_model, nhead, init_scale=1e-3)     # pooling: (almost) no distance bias
        self.pool_out = nn.Sequential(nn.Linear(n_queries * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.d_model = d_model

    def tokens(self, values, modality, mask):
        row, col, val = values[..., 0], values[..., 1], values[..., 2]
        feats = torch.cat([values, fourier(row, self.freqs), fourier(col, self.freqs)], dim=-1)
        x = self.embed(feats) + self.modality(modality)
        dist2 = periodic_dist2(col, row, col, row, self.periodic)
        for layer in self.layers:
            x = layer(x, dist2, mask)
        return self.norm(x)

    def forward(self, values, modality, mask):
        x = self.tokens(values, modality, mask)
        b = x.shape[0]
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        zeros = torch.zeros(b, q.shape[1], x.shape[1], device=x.device)
        pooled = self.pool(q, x, zeros, mask)                                # (B, K, d)
        return self.pool_out(pooled.reshape(b, -1)), x


class RBResidualModel(nn.Module):
    def __init__(self, n_modalities: int, latent_dim: int, n_out: int = 128, d_model: int = 128, nhead: int = 4,
                 nlayers: int = 3, n_queries: int = 8, local: bool = False, periodic: bool = True):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.encoder = RBSetEncoder(n_modalities, d_model, nhead, nlayers, n_queries=n_queries, periodic=periodic)
        head_in = d_model + 2 * latent_dim
        self.delta_head = nn.Sequential(nn.Linear(head_in, d_model), nn.GELU(), nn.Linear(d_model, latent_dim))
        self.logvar_head = nn.Sequential(nn.Linear(head_in, d_model), nn.GELU(), nn.Linear(d_model, latent_dim))
        for head in (self.delta_head, self.logvar_head):
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)
        if local:
            raise ValueError("The basic release does not include the unused local-correction variant")
        self.local = None

    def forward(self, values, modality, mask, z_lin, logvar_lin):
        pooled, tokens = self.encoder(values, modality, mask)
        features = torch.cat([pooled, z_lin, logvar_lin], dim=-1)
        z = z_lin + self.delta_head(features)
        logvar = torch.clamp(logvar_lin + self.logvar_head(features), -7.0, 5.0)
        dq = self.local(tokens, values, mask) if self.local is not None else None
        return z, logvar, dq
