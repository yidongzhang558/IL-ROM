"""Original set encoder used by the paper baselines."""
import torch
import torch.nn as nn

class MultiModalSensorSetEncoder(nn.Module):
    """Set encoder for an arbitrary number of physical observation types.

    Each observation is represented by continuous coordinates/value and a
    discrete modality id.  An optional known-condition vector (for example the
    incidence angle) is fused after attention pooling.  The interface keeps
    sensor ordering irrelevant and supports variable sensor counts.
    """

    def __init__(self, n_modalities, latent_dim, d_model=64, nhead=4,
                 nlayers=2, condition_dim=0):
        super().__init__()
        self.n_modalities = int(n_modalities)
        self.condition_dim = int(condition_dim)
        self.register_buffer(
            "coordinate_frequencies",
            torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float32),
            persistent=False,
        )
        token_feature_dim = 3 + 2 * 2 * len(self.coordinate_frequencies) + 2
        self.value_embed = nn.Sequential(
            nn.Linear(token_feature_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.modality_embed = nn.Embedding(self.n_modalities, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.pool_score = nn.Linear(d_model, 1)
        head_in = d_model
        if self.condition_dim:
            self.condition_embed = nn.Sequential(
                nn.Linear(self.condition_dim, d_model), nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            head_in += d_model
        self.head = nn.Sequential(
            nn.Linear(head_in, d_model), nn.GELU(),
            nn.Linear(d_model, latent_dim),
        )
        self.feature_dim = head_in

    def forward(self, coords_values, modality_ids, pad_mask, condition=None,
                return_features=False):
        """Map a padded multimodal sensor set to a target latent vector.

        Args:
            coords_values: [B, L, 3] containing row, column, normalized value.
            modality_ids: [B, L] integer modality identifiers.
            pad_mask: [B, L], True at padding tokens.
            condition: optional [B, condition_dim] known operating parameters.
        """
        pad_mask = pad_mask.to(device=coords_values.device, dtype=torch.bool)
        modality_ids = modality_ids.to(device=coords_values.device, dtype=torch.long)
        coordinates = coords_values[..., :2]
        value = coords_values[..., 2:3]
        angles = 2.0 * torch.pi * coordinates.unsqueeze(-1) * self.coordinate_frequencies
        fourier = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(-2)
        interactions = coordinates * value
        token_features = torch.cat([coords_values, fourier, interactions], dim=-1)
        h = self.value_embed(token_features) + self.modality_embed(modality_ids)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        scores = self.pool_score(h).squeeze(-1).masked_fill(pad_mask, -1e4)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (h * weights).sum(dim=1)
        if self.condition_dim:
            if condition is None:
                raise ValueError("condition is required for this encoder")
            pooled = torch.cat([pooled, self.condition_embed(condition)], dim=-1)
        latent = self.head(pooled)
        if return_features:
            return latent, pooled
        return latent
