from __future__ import annotations

import torch
import torch.nn as nn


class RoadCorridorEncoder(nn.Module):
    """Encode OD-derived public-road corridor tokens for decoder conditioning."""

    def __init__(
        self,
        feature_dim: int,
        cond_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.candidate_feature_dim = 3
        if self.hidden_dim % int(num_heads) != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=4 * self.hidden_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.num_layers)
        self.path_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )
        self.candidate_scorer = nn.Sequential(
            nn.LayerNorm(self.hidden_dim + self.candidate_feature_dim),
            nn.Linear(self.hidden_dim + self.candidate_feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.global_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )

    def forward(
        self,
        road_corridor_tokens: torch.Tensor,
        road_corridor_mask: torch.Tensor | None = None,
        road_path_length_ratio: torch.Tensor | None = None,
        road_path_total_length_m: torch.Tensor | None = None,
        road_path_success: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if road_corridor_tokens.ndim != 4:
            raise ValueError(
                "road_corridor_tokens must be [B, K, P, F], "
                f"got {tuple(road_corridor_tokens.shape)}"
            )
        if road_corridor_tokens.shape[-1] != self.feature_dim:
            raise ValueError(
                f"road feature dim mismatch: expected {self.feature_dim}, "
                f"got {road_corridor_tokens.shape[-1]}"
            )
        batch, paths, points, feat = road_corridor_tokens.shape
        flat = road_corridor_tokens.reshape(batch * paths, points, feat)
        if road_corridor_mask is None:
            flat_mask = torch.ones(batch * paths, points, device=flat.device, dtype=torch.bool)
        else:
            if road_corridor_mask.shape != (batch, paths, points):
                raise ValueError(
                    "road_corridor_mask must be [B, K, P], "
                    f"got {tuple(road_corridor_mask.shape)} for tokens {tuple(road_corridor_tokens.shape)}"
                )
            flat_mask = road_corridor_mask.reshape(batch * paths, points).to(device=flat.device, dtype=torch.bool)
        path_valid_flat = flat_mask.any(dim=1)
        if not bool(path_valid_flat.all()):
            flat_mask = flat_mask.clone()
            flat_mask[~path_valid_flat, 0] = True
        hidden = self.input_proj(flat)
        key_padding_mask = ~flat_mask
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        weights = flat_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        path_valid = path_valid_flat.reshape(batch, paths)
        path_hidden = pooled.reshape(batch, paths, self.hidden_dim)
        path_tokens = self.path_proj(path_hidden)

        ratio_feat = torch.zeros(batch, paths, device=path_hidden.device, dtype=path_hidden.dtype)
        if road_path_length_ratio is not None:
            ratio = road_path_length_ratio.to(device=path_hidden.device, dtype=path_hidden.dtype)
            if ratio.ndim == 1:
                ratio = ratio.view(batch, 1)
            ratio_feat = torch.nan_to_num(ratio[:, :paths], nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 8.0) / 4.0
        length_feat = torch.zeros(batch, paths, device=path_hidden.device, dtype=path_hidden.dtype)
        if road_path_total_length_m is not None:
            length_m = road_path_total_length_m.to(device=path_hidden.device, dtype=path_hidden.dtype)
            if length_m.ndim == 1:
                length_m = length_m.view(batch, 1)
            length_km = torch.nan_to_num(length_m[:, :paths], nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 50000.0) / 1000.0
            norm = torch.log1p(length_km.new_tensor(20.0))
            length_feat = torch.log1p(length_km) / norm
        if road_path_success is None:
            success = path_valid
        else:
            success = road_path_success.to(device=path_hidden.device, dtype=torch.bool)
            if success.ndim == 1:
                success = success.view(batch, 1)
            success = success[:, :paths] & path_valid
        success_feat = success.to(dtype=path_hidden.dtype)
        candidate_features = torch.stack([ratio_feat, length_feat, success_feat], dim=-1)
        scorer_input = torch.cat([path_hidden, candidate_features], dim=-1)

        candidate_logits = self.candidate_scorer(scorer_input).squeeze(-1)
        candidate_logits = candidate_logits.masked_fill(~success, -torch.inf)
        if not bool(success.all()):
            no_valid = ~success.any(dim=1)
            if bool(no_valid.any()):
                candidate_logits = candidate_logits.clone()
                candidate_logits[no_valid] = 0.0
        candidate_probs = torch.softmax(candidate_logits, dim=1)
        candidate_probs = candidate_probs.masked_fill(~success, 0.0)
        prob_denom = candidate_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
        candidate_probs = candidate_probs / prob_denom
        global_hidden = (path_hidden * candidate_probs.unsqueeze(-1)).sum(dim=1)
        global_cond = self.global_proj(global_hidden)
        weighted_path_tokens = path_tokens * candidate_probs.unsqueeze(-1)
        return weighted_path_tokens, global_cond, candidate_logits, candidate_probs
