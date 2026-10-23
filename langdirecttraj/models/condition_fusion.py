from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ConditionFusion(nn.Module):
    """Fuse attribute and mobility-token conditions before generator injection.

    This module intentionally stays independent from any trajectory generator.
    It gives the future decoder a single condition tensor while allowing us to
    compare simple addition, concatenation, and gated conditioning.
    """

    def __init__(
        self,
        attr_dim: int,
        mobility_dim: int,
        output_dim: int,
        mode: str = "gated",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attr_dim = int(attr_dim)
        self.mobility_dim = int(mobility_dim)
        self.output_dim = int(output_dim)
        self.mode = str(mode)
        hidden = int(hidden_dim or max(self.output_dim, self.attr_dim, self.mobility_dim))

        self.attr_proj = self._projection(self.attr_dim, self.output_dim, hidden, dropout)
        self.mobility_proj = self._projection(self.mobility_dim, self.output_dim, hidden, dropout)

        if self.mode == "add":
            self.fuse = nn.Identity()
            self.gate = None
        elif self.mode == "concat_mlp":
            self.fuse = nn.Sequential(
                nn.LayerNorm(self.output_dim * 2),
                nn.Linear(self.output_dim * 2, hidden),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, self.output_dim),
            )
            self.gate = None
        elif self.mode == "gated":
            self.fuse = nn.Sequential(
                nn.LayerNorm(self.output_dim * 2),
                nn.Linear(self.output_dim * 2, hidden),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, self.output_dim),
            )
            self.gate = nn.Sequential(
                nn.LayerNorm(self.output_dim * 2),
                nn.Linear(self.output_dim * 2, self.output_dim),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unsupported condition fusion mode: {mode}")

    @staticmethod
    def _projection(in_dim: int, out_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
        if in_dim == out_dim:
            return nn.Identity()
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, attr_cond: torch.Tensor, mobility_cond: torch.Tensor) -> torch.Tensor:
        if attr_cond.ndim != 2:
            raise ValueError(f"attr_cond must be [B, C], got {tuple(attr_cond.shape)}")
        if mobility_cond.ndim != 2:
            raise ValueError(f"mobility_cond must be [B, C], got {tuple(mobility_cond.shape)}")
        if attr_cond.shape[0] != mobility_cond.shape[0]:
            raise ValueError(
                f"Batch mismatch: attr_cond={tuple(attr_cond.shape)}, mobility_cond={tuple(mobility_cond.shape)}"
            )

        attr = self.attr_proj(attr_cond)
        mobility = self.mobility_proj(mobility_cond)

        if self.mode == "add":
            return attr + mobility

        pair = torch.cat([attr, mobility], dim=-1)
        fused = self.fuse(pair)
        if self.mode == "concat_mlp":
            return fused

        gate = self.gate(pair)
        return attr + gate * fused
