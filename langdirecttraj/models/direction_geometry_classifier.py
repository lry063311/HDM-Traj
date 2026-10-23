from __future__ import annotations

import torch
from torch import nn


class DirectionGeometryClassifier(nn.Module):
    """Sequence classifier for local-frame trajectory direction tokens.

    This is meant to be trained on pseudo/real target trajectories, then reused
    as a frozen differentiable direction teacher for generator controllability.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.input_proj = nn.Linear(6, hidden_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.num_classes),
        )

    @staticmethod
    def local_frame_features(points: torch.Tensor) -> torch.Tensor:
        """Return [B, T, 6] local-frame point, velocity, and arc-progress features."""
        if points.ndim != 3 or points.shape[-1] != 2:
            raise ValueError(f"Expected points [B, T, 2], got {tuple(points.shape)}.")
        diffs = points[:, 1:] - points[:, :-1]
        if diffs.shape[1] == 0:
            diffs = torch.zeros(points.shape[0], 1, 2, device=points.device, dtype=points.dtype)
        first = diffs[:, : min(5, diffs.shape[1])].mean(dim=1)
        first_norm = torch.linalg.norm(first, dim=-1, keepdim=True).clamp_min(1e-6)
        ex = first / first_norm
        ey = torch.stack([-ex[:, 1], ex[:, 0]], dim=-1)
        rel = points - points[:, :1]
        local_x = (rel * ex[:, None]).sum(dim=-1, keepdim=True)
        local_y = (rel * ey[:, None]).sum(dim=-1, keepdim=True)

        local_points = torch.cat([local_x, local_y], dim=-1)
        local_vel = local_points[:, 1:] - local_points[:, :-1]
        pad_vel = torch.zeros(local_points.shape[0], 1, 2, device=points.device, dtype=points.dtype)
        local_vel = torch.cat([pad_vel, local_vel], dim=1)
        speed = torch.linalg.norm(local_vel, dim=-1, keepdim=True)
        arc = torch.cumsum(speed, dim=1)
        arc = arc / arc[:, -1:].clamp_min(1e-6)
        time = torch.linspace(0.0, 1.0, points.shape[1], device=points.device, dtype=points.dtype)
        time = time.view(1, -1, 1).expand(points.shape[0], -1, -1)
        return torch.cat([local_points, local_vel, arc, time], dim=-1)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.local_frame_features(points.float())
        hidden = self.input_proj(features).transpose(1, 2)
        hidden = self.encoder(hidden).transpose(1, 2)
        mean_pool = hidden.mean(dim=1)
        max_pool = hidden.amax(dim=1)
        pooled = self.norm(torch.cat([mean_pool, max_pool], dim=-1))
        return self.head(pooled)
