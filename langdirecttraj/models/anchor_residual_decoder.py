from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnchorResidualDecoder(nn.Module):
    """Minimal reference-anchored continuous trajectory decoder.

    The decoder resamples a reference polyline to the target horizon and predicts
    a bounded tangent-normal residual around that anchor. It is intentionally
    small: the goal is to test whether mobility-token conditions can control
    geometry before introducing a heavier DiT backbone.
    """

    def __init__(
        self,
        cond_dim: int = 128,
        hidden_dim: int = 128,
        target_len: int = 200,
        tangent_scale: float = 80.0,
        normal_scale: float = 30.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.target_len = int(target_len)
        self.tangent_scale = float(tangent_scale)
        self.normal_scale = float(normal_scale)

        self.point_embed = nn.Sequential(
            nn.Linear(8, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(self.cond_dim),
            nn.Linear(self.cond_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, 2),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def resample_reference(reference: torch.Tensor, target_len: int) -> torch.Tensor:
        if reference.ndim != 3 or reference.shape[-1] != 2:
            raise ValueError(f"reference must be [B, L, 2], got {tuple(reference.shape)}")
        if reference.shape[1] == target_len:
            return reference
        return F.interpolate(reference.transpose(1, 2), size=target_len, mode="linear", align_corners=False).transpose(1, 2)

    @staticmethod
    def tangent_normal(anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        forward = torch.zeros_like(anchor)
        if anchor.shape[1] > 1:
            forward[:, 1:-1] = 0.5 * (anchor[:, 2:] - anchor[:, :-2])
            forward[:, 0] = anchor[:, 1] - anchor[:, 0]
            forward[:, -1] = anchor[:, -1] - anchor[:, -2]
        tangent = F.normalize(forward, dim=-1, eps=1e-6)
        normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
        return tangent, normal

    @staticmethod
    def normalized_time(batch: int, target_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, target_len, device=device, dtype=dtype)
        return t.view(1, target_len, 1).expand(batch, -1, -1)

    def forward(
        self,
        reference: torch.Tensor,
        fused_cond: torch.Tensor,
        target_len: Optional[int] = None,
        return_aux: bool = False,
    ):
        if fused_cond.ndim != 2 or fused_cond.shape[-1] != self.cond_dim:
            raise ValueError(f"fused_cond must be [B, {self.cond_dim}], got {tuple(fused_cond.shape)}")
        target_len = int(target_len or self.target_len)
        anchor = self.resample_reference(reference.float(), target_len)
        tangent, normal = self.tangent_normal(anchor)
        time = self.normalized_time(anchor.shape[0], target_len, anchor.device, anchor.dtype)

        point_features = torch.cat([anchor, tangent, normal, time, 1.0 - time], dim=-1)
        point_hidden = self.point_embed(point_features)
        cond_hidden = self.cond_proj(fused_cond).unsqueeze(1)
        hidden = point_hidden + cond_hidden

        raw_residual = torch.tanh(self.residual_head(hidden))
        gate = self.gate_head(hidden)
        delta_tangent = raw_residual[..., 0:1] * self.tangent_scale
        delta_normal = raw_residual[..., 1:2] * self.normal_scale
        residual = gate * (delta_tangent * tangent + delta_normal * normal)
        trajectory = anchor + residual

        if not return_aux:
            return trajectory
        return trajectory, {
            "anchor": anchor,
            "tangent": tangent,
            "normal": normal,
            "gate": gate,
            "raw_residual": raw_residual,
            "residual": residual,
        }
