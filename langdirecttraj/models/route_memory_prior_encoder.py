from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RouteMemoryPriorEncoder(nn.Module):
    """Encode a selected route-memory trajectory as decoder condition tokens.

    The input is already in the decoder's local normalized coordinate frame.
    This module therefore treats the route as a soft conditioning signal, not
    as a replacement trajectory.
    """

    def __init__(
        self,
        cond_dim: int,
        hidden_dim: int,
        num_points: int = 16,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_points = int(num_points)
        if self.num_points < 2:
            raise ValueError("num_points must be >= 2")
        if self.hidden_dim % int(num_heads) != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.input_proj = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_points, self.hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=4 * self.hidden_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )
        self.global_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )
        nn.init.normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _resample(points: torch.Tensor, target_len: int) -> torch.Tensor:
        if points.shape[1] == int(target_len):
            return points
        if points.shape[1] <= 1:
            return points[:, :1].expand(-1, int(target_len), -1)
        return F.interpolate(
            points.transpose(1, 2),
            size=int(target_len),
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(
        self,
        route_prior: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if route_prior.ndim != 3 or route_prior.shape[-1] != 2:
            raise ValueError(f"route_prior must be [B, T, 2], got {tuple(route_prior.shape)}")
        route = self._resample(route_prior.float(), self.num_points).to(dtype=route_prior.dtype)
        diff = torch.zeros_like(route)
        diff[:, 1:] = route[:, 1:] - route[:, :-1]
        time = torch.linspace(0.0, 1.0, self.num_points, device=route.device, dtype=route.dtype)
        time = time.view(1, self.num_points, 1).expand(route.shape[0], -1, -1)
        hidden = self.input_proj(torch.cat([route, diff, time, 1.0 - time], dim=-1))
        hidden = hidden + self.pos_embed.to(device=hidden.device, dtype=hidden.dtype)
        hidden = self.encoder(hidden)
        tokens = self.token_proj(hidden)
        global_cond = self.global_proj(hidden.mean(dim=1))
        if valid_mask is not None:
            valid = valid_mask.to(device=route.device, dtype=torch.bool).view(-1, 1, 1)
            tokens = torch.where(valid, tokens, torch.zeros_like(tokens))
            global_cond = torch.where(valid.squeeze(-1), global_cond, torch.zeros_like(global_cond))
        return tokens, global_cond
