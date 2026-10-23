from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RouteMemoryProposalHead(nn.Module):
    """Generate per-mode route proposals from top-K route-memory teacher evidence."""

    def __init__(
        self,
        hidden_dim: int,
        num_modes: int,
        target_len: int,
        num_points: int = 16,
        component_dim: int = 4,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
        residual_scale: float = 0.05,
        base_mode: str = "mix",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_modes = int(num_modes)
        self.target_len = int(target_len)
        self.num_points = int(num_points)
        self.component_dim = int(component_dim)
        self.residual_scale = float(residual_scale)
        self.base_mode = str(base_mode)
        if self.base_mode not in {"mix", "rank_seed"}:
            raise ValueError(f"Unsupported route proposal base_mode={base_mode!r}")
        if self.num_modes < 1:
            raise ValueError("num_modes must be >= 1")
        if self.num_points < 2:
            raise ValueError("num_points must be >= 2")
        if self.hidden_dim % int(num_heads) != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.route_proj = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.score_proj = nn.Sequential(
            nn.LayerNorm(2 + self.component_dim),
            nn.Linear(2 + self.component_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.memory_pos = nn.Parameter(torch.zeros(1, 256, self.hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=4 * self.hidden_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))

        self.mode_queries = nn.Parameter(torch.zeros(1, self.num_modes, self.hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=4 * self.hidden_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(num_layers))
        self.mix_score = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.num_points * 2),
        )
        self.logit_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.normal_(self.memory_pos, std=0.02)
        nn.init.normal_(self.mode_queries, std=0.02)
        nn.init.constant_(self.residual_head[-1].weight, 0.0)
        nn.init.constant_(self.residual_head[-1].bias, 0.0)
        nn.init.constant_(self.logit_head[-1].weight, 0.0)
        nn.init.constant_(self.logit_head[-1].bias, 0.0)

    @staticmethod
    def _resample(points: torch.Tensor, target_len: int) -> torch.Tensor:
        if points.shape[-2] == int(target_len):
            return points
        if points.shape[-2] <= 1:
            return points[..., :1, :].expand(*points.shape[:-2], int(target_len), points.shape[-1])
        flat = points.reshape(-1, points.shape[-2], points.shape[-1])
        out = F.interpolate(
            flat.transpose(1, 2),
            size=int(target_len),
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        return out.reshape(*points.shape[:-2], int(target_len), points.shape[-1])

    def _component_features(self, components: torch.Tensor | None, batch: int, topk: int, device, dtype) -> torch.Tensor:
        if self.component_dim <= 0:
            return torch.zeros(batch, topk, 0, device=device, dtype=dtype)
        out = torch.zeros(batch, topk, self.component_dim, device=device, dtype=dtype)
        if components is None:
            return out
        comp = components.to(device=device, dtype=dtype)
        if comp.ndim != 3 or comp.shape[0] != batch or comp.shape[1] != topk:
            return out
        copy_dim = min(int(comp.shape[-1]), self.component_dim)
        if copy_dim > 0:
            out[..., :copy_dim] = comp[..., :copy_dim]
        return out

    def forward(
        self,
        candidates: torch.Tensor,
        scores: torch.Tensor | None = None,
        components: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if candidates.ndim != 4 or candidates.shape[-1] != 2:
            raise ValueError(f"candidates must be [B, M, T, 2], got {tuple(candidates.shape)}")
        batch, topk = int(candidates.shape[0]), int(candidates.shape[1])
        device = candidates.device
        dtype = candidates.dtype

        routes = self._resample(candidates.float(), self.num_points).to(device=device, dtype=dtype)
        diff = torch.zeros_like(routes)
        diff[:, :, 1:] = routes[:, :, 1:] - routes[:, :, :-1]
        time = torch.linspace(0.0, 1.0, self.num_points, device=device, dtype=dtype)
        time = time.view(1, 1, self.num_points, 1).expand(batch, topk, -1, -1)
        route_hidden = self.route_proj(torch.cat([routes, diff, time, 1.0 - time], dim=-1)).mean(dim=2)

        if scores is None:
            score_feat = torch.zeros(batch, topk, 1, device=device, dtype=dtype)
        else:
            score_feat = scores.to(device=device, dtype=dtype).view(batch, topk, 1)
            score_feat = score_feat - score_feat.mean(dim=1, keepdim=True)
            score_feat = score_feat / score_feat.std(dim=1, keepdim=True).clamp_min(1e-3)
        rank = torch.linspace(0.0, 1.0, topk, device=device, dtype=dtype).view(1, topk, 1).expand(batch, -1, -1)
        comp_feat = self._component_features(components, batch, topk, device, dtype)
        memory = route_hidden + self.score_proj(torch.cat([score_feat, rank, comp_feat], dim=-1))
        memory = memory + self.memory_pos[:, :topk].to(device=device, dtype=dtype)
        memory = self.encoder(memory)

        query = self.mode_queries.to(device=device, dtype=dtype).expand(batch, -1, -1)
        decoded = self.decoder(query, memory)
        if self.base_mode == "rank_seed":
            seed_idx = torch.arange(self.num_modes, device=device) % max(topk, 1)
            base = candidates.to(device=device, dtype=dtype).index_select(1, seed_idx)
            mix_entropy = candidates.new_tensor(0.0)
            mix_top_prob = candidates.new_tensor(1.0)
        else:
            mix_logits = torch.matmul(self.mix_score(decoded), memory.transpose(1, 2)) / float(self.hidden_dim) ** 0.5
            if valid_mask is not None:
                valid = valid_mask.to(device=device, dtype=torch.bool).view(batch, 1, 1)
                mix_logits = torch.where(valid, mix_logits, torch.zeros_like(mix_logits))
            mix_probs = torch.softmax(mix_logits, dim=-1)
            base = torch.einsum("bkm,bmtd->bktd", mix_probs, candidates.to(device=device, dtype=dtype))
            mix_entropy = -(mix_probs * mix_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
            mix_top_prob = mix_probs.amax(dim=-1).mean()

        residual_low = self.residual_head(decoded).view(batch, self.num_modes, self.num_points, 2)
        residual = self._resample(residual_low, self.target_len)
        proposal_logits = self.logit_head(decoded).squeeze(-1)
        proposal = base + float(self.residual_scale) * residual
        return proposal, {
            "route_memory_proposal_logits": proposal_logits,
            "route_memory_proposal_mix_entropy": mix_entropy,
            "route_memory_proposal_top_prob": mix_top_prob,
        }
