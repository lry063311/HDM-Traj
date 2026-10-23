from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarseRouteTokenPlanner(nn.Module):
    """Predict coarse route-grid tokens from OD-chord and language condition.

    The planner never consumes the GT middle trajectory as an input. During
    training its logits are supervised by GT-derived coarse grid labels; during
    inference the predicted soft grid tokens become additional DiT condition
    tokens. This mirrors TrajGPT-R's route-token emphasis while preserving the
    OD-chord-only protocol.
    """

    def __init__(
        self,
        cond_dim: int,
        hidden_dim: int,
        plan_points: int = 16,
        grid_size: int = 12,
        grid_extent: float = 2.0,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.plan_points = int(plan_points)
        self.grid_size = int(grid_size)
        self.grid_extent = float(grid_extent)
        if self.plan_points < 2:
            raise ValueError("plan_points must be >= 2")
        if self.grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        if self.hidden_dim % int(num_heads) != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.ref_encoder = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(self.cond_dim),
            nn.Linear(self.cond_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.plan_points, self.hidden_dim))
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
        self.logit_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.num_cells),
        )
        self.cell_embed = nn.Parameter(torch.zeros(self.num_cells, self.cond_dim))
        self.plan_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim + self.cond_dim + 2),
            nn.Linear(self.hidden_dim + self.cond_dim + 2, self.cond_dim),
            nn.SiLU(),
            nn.Linear(self.cond_dim, self.cond_dim),
        )
        centers = self._build_cell_centers(self.grid_size, self.grid_extent)
        self.register_buffer("cell_centers", centers, persistent=True)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cell_embed, std=0.02)

    @property
    def num_cells(self) -> int:
        return self.grid_size * self.grid_size

    @staticmethod
    def _build_cell_centers(grid_size: int, grid_extent: float) -> torch.Tensor:
        coords = torch.linspace(-float(grid_extent), float(grid_extent), int(grid_size))
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(-1, 2)

    @staticmethod
    def _resample(sequence: torch.Tensor, target_len: int) -> torch.Tensor:
        if sequence.shape[1] == int(target_len):
            return sequence
        return F.interpolate(
            sequence.transpose(1, 2),
            size=int(target_len),
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(
        self,
        reference: torch.Tensor,
        fused_cond: torch.Tensor,
        return_centers: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if reference.ndim != 3 or reference.shape[-1] != 2:
            raise ValueError(f"reference must be [B, T, 2], got {tuple(reference.shape)}")
        if fused_cond.ndim != 2 or fused_cond.shape[-1] != self.cond_dim:
            raise ValueError(f"fused_cond must be [B, {self.cond_dim}], got {tuple(fused_cond.shape)}")

        ref = self._resample(reference.float(), self.plan_points).to(dtype=fused_cond.dtype)
        diff = torch.zeros_like(ref)
        diff[:, 1:] = ref[:, 1:] - ref[:, :-1]
        time = torch.linspace(0.0, 1.0, self.plan_points, device=ref.device, dtype=ref.dtype)
        time = time.view(1, self.plan_points, 1).expand(ref.shape[0], -1, -1)
        ref_features = torch.cat([ref, diff, time, 1.0 - time], dim=-1)
        hidden = self.ref_encoder(ref_features) + self.cond_proj(fused_cond).unsqueeze(1)
        hidden = hidden + self.pos_embed.to(device=hidden.device, dtype=hidden.dtype)
        hidden = self.encoder(hidden)
        logits = self.logit_head(hidden)
        probs = torch.softmax(logits, dim=-1)
        soft_cell_embed = probs @ self.cell_embed.to(device=probs.device, dtype=probs.dtype)
        soft_centers = probs @ self.cell_centers.to(device=probs.device, dtype=probs.dtype)
        plan_tokens = self.plan_proj(torch.cat([hidden, soft_cell_embed, soft_centers], dim=-1))
        if return_centers:
            return plan_tokens, logits, soft_centers
        return plan_tokens, logits
