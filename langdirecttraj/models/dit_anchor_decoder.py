from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB
from langdirecttraj.numerics import safe_atan2

from .anchor_residual_decoder import AnchorResidualDecoder


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """AdaLN Transformer block adapted from DirectTraj-style direct prediction."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm_cond = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.cond_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(float(dropout)),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, cond_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=1)
        attn_in = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        if cond_tokens is not None:
            cond_attn_in = self.norm_cond(x)
            cond_tokens = self.norm_cond(cond_tokens)
            cond_out, _ = self.cond_attn(cond_attn_in, cond_tokens, cond_tokens, need_weights=False)
            x = x + 0.5 * cond_out
        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_in)
        return x

    def route_fuse(self, x: torch.Tensor, route_tokens: Optional[torch.Tensor], route_gate: torch.Tensor) -> torch.Tensor:
        if route_tokens is None:
            return x
        gate = route_gate.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)
        if float(gate.detach().max().item()) <= 0.0:
            return x
        cond_attn_in = self.norm_cond(x)
        route_tokens = self.norm_cond(route_tokens)
        cond_out, _ = self.cond_attn(cond_attn_in, route_tokens, route_tokens, need_weights=False)
        return x + gate * cond_out


class SplitGatedReranker(nn.Module):
    """Score candidates with separated geometry and instruction-consistency paths."""

    def __init__(self, context_dim: int, pure_dim: int, token_dim: int, hidden_dim: int):
        super().__init__()
        self.context_dim = int(context_dim)
        self.pure_dim = int(pure_dim)
        self.token_dim = int(token_dim)
        self.context_proj = nn.Sequential(
            nn.LayerNorm(self.context_dim),
            nn.Linear(self.context_dim, hidden_dim),
            nn.SiLU(),
        )
        self.geometry_proj = nn.Sequential(
            nn.LayerNorm(self.pure_dim),
            nn.Linear(self.pure_dim, hidden_dim),
            nn.SiLU(),
        )
        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, hidden_dim),
            nn.SiLU(),
        )
        self.token_gate = nn.Sequential(
            nn.LayerNorm(self.context_dim + self.pure_dim + self.token_dim),
            nn.Linear(self.context_dim + self.pure_dim + self.token_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3 + 1),
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context_features: torch.Tensor, pure_geometry: torch.Tensor, token_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.sigmoid(self.token_gate(torch.cat([context_features, pure_geometry, token_features], dim=-1)))
        context_hidden = self.context_proj(context_features)
        geometry_hidden = self.geometry_proj(pure_geometry)
        token_hidden = self.token_proj(token_features) * gate
        logits = self.score_head(torch.cat([context_hidden, geometry_hidden, token_hidden, gate], dim=-1)).squeeze(-1)
        return logits, gate.squeeze(-1)


class ListwiseTransformerReranker(nn.Module):
    """Score the full candidate set with mode-to-mode self-attention."""

    def __init__(
        self,
        context_dim: int,
        pure_dim: int,
        token_dim: int,
        hidden_dim: int,
        num_candidates: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = int(context_dim) + int(pure_dim) + int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_candidates = int(num_candidates)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.mode_embed = nn.Parameter(torch.zeros(self.num_candidates, self.hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=max(1, int(num_heads)),
            dim_feedforward=self.hidden_dim * 2,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.score_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        context_features: torch.Tensor,
        pure_geometry: torch.Tensor,
        token_features: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat([context_features, pure_geometry, token_features], dim=-1)
        hidden = self.input_proj(features)
        hidden = hidden + self.mode_embed.view(1, self.num_candidates, self.hidden_dim).to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        hidden = self.encoder(hidden)
        return self.score_head(hidden).squeeze(-1)


class ResidualListwiseReranker(nn.Module):
    """Keep a pointwise scorer and learn a small listwise correction."""

    def __init__(
        self,
        context_dim: int,
        pure_dim: int,
        token_dim: int,
        hidden_dim: int,
        num_candidates: int,
        num_heads: int,
        dropout: float = 0.0,
        initial_alpha: float = 0.05,
    ):
        super().__init__()
        self.input_dim = int(context_dim) + int(pure_dim) + int(token_dim)
        self.base_reranker = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.listwise_delta = ListwiseTransformerReranker(
            context_dim=context_dim,
            pure_dim=pure_dim,
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            num_candidates=num_candidates,
            num_heads=num_heads,
            dropout=dropout,
        )
        alpha = min(max(float(initial_alpha), 1e-4), 1.0 - 1e-4)
        self.listwise_alpha_logit = nn.Parameter(torch.tensor(math.log(alpha / (1.0 - alpha))))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.listwise_alpha_logit)

    def set_base_trainable(self, trainable: bool) -> None:
        for param in self.base_reranker.parameters():
            param.requires_grad = bool(trainable)

    def forward(
        self,
        context_features: torch.Tensor,
        pure_geometry: torch.Tensor,
        token_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = torch.cat([context_features, pure_geometry, token_features], dim=-1)
        base_logits = self.base_reranker(features).squeeze(-1)
        delta_logits = self.listwise_delta(context_features, pure_geometry, token_features)
        alpha = self.alpha.to(device=base_logits.device, dtype=base_logits.dtype)
        logits = base_logits + alpha * delta_logits
        return logits, base_logits, delta_logits, alpha


class CandidateTrajectoryEncoder(nn.Module):
    """Encode candidate-reference temporal geometry for candidate scoring."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.input_dim = 23
        self.hidden_dim = int(hidden_dim)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=max(1, int(num_heads)),
            dim_feedforward=self.hidden_dim * 2,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    @staticmethod
    def _velocity(points: torch.Tensor) -> torch.Tensor:
        velocity = torch.zeros_like(points)
        velocity[..., 1:, :] = points[..., 1:, :] - points[..., :-1, :]
        velocity[..., 0, :] = velocity[..., 1, :] if points.shape[-2] > 1 else 0
        return velocity

    @staticmethod
    def _signed_turn(velocity: torch.Tensor) -> torch.Tensor:
        turn = velocity.new_zeros(*velocity.shape[:-1], 1)
        if velocity.shape[-2] < 2:
            return turn
        prev = velocity[..., :-1, :]
        curr = velocity[..., 1:, :]
        cross = prev[..., 0] * curr[..., 1] - prev[..., 1] * curr[..., 0]
        dot = (prev * curr).sum(dim=-1)
        angle = safe_atan2(cross, dot).unsqueeze(-1)
        turn[..., 1:, :] = angle
        return turn

    def forward(
        self,
        candidates: torch.Tensor,
        reference: torch.Tensor,
        tangent: torch.Tensor,
        normal: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_candidates, target_len, _ = candidates.shape
        reference_exp = reference.unsqueeze(1)
        tangent_exp = tangent.unsqueeze(1)
        normal_exp = normal.unsqueeze(1)
        rel = candidates - reference_exp
        tangent_offset = (rel * tangent_exp).sum(dim=-1, keepdim=True)
        normal_offset = (rel * normal_exp).sum(dim=-1, keepdim=True)
        rel_norm = rel.norm(dim=-1, keepdim=True)
        candidate_velocity = self._velocity(candidates)
        reference_velocity = self._velocity(reference).unsqueeze(1)
        reference_speed = reference_velocity.norm(dim=-1, keepdim=True)
        path_scale = reference_speed.sum(dim=-2, keepdim=True).clamp_min(1e-6)
        step_scale = reference_speed.mean(dim=-2, keepdim=True).clamp_min(1e-6)
        rel_scaled = rel / path_scale
        tangent_offset_scaled = tangent_offset / path_scale
        normal_offset_scaled = normal_offset / path_scale
        velocity_delta = candidate_velocity - reference_velocity
        candidate_speed = candidate_velocity.norm(dim=-1, keepdim=True)
        speed_delta = velocity_delta.norm(dim=-1, keepdim=True)
        acceleration_delta = self._velocity(velocity_delta).norm(dim=-1, keepdim=True)
        candidate_tangent_velocity = (candidate_velocity * tangent_exp).sum(dim=-1, keepdim=True) / step_scale
        candidate_normal_velocity = (candidate_velocity * normal_exp).sum(dim=-1, keepdim=True) / step_scale
        reference_tangent_velocity = (reference_velocity * tangent_exp).sum(dim=-1, keepdim=True) / step_scale
        reference_tangent_velocity = reference_tangent_velocity.expand(batch, num_candidates, -1, -1)
        speed_ratio = candidate_speed / reference_speed.clamp_min(1e-6)
        speed_ratio = speed_ratio.expand(batch, num_candidates, -1, -1)
        transition_cosine = (candidate_velocity * reference_velocity).sum(dim=-1, keepdim=True) / (
            candidate_speed * reference_speed
        ).clamp_min(1e-6)
        signed_turn = self._signed_turn(candidate_velocity)
        abs_turn = signed_turn.abs()
        cumulative_turn = signed_turn.cumsum(dim=-2) / math.pi
        progress = torch.linspace(0.0, 1.0, target_len, device=candidates.device, dtype=candidates.dtype)
        progress = progress.view(1, 1, target_len, 1).expand(batch, num_candidates, -1, -1)
        features = torch.cat(
            [
                rel,
                rel_scaled,
                rel_norm,
                tangent_offset,
                normal_offset,
                tangent_offset_scaled,
                normal_offset_scaled,
                velocity_delta,
                candidate_speed,
                speed_delta,
                acceleration_delta,
                candidate_tangent_velocity,
                candidate_normal_velocity,
                reference_tangent_velocity,
                speed_ratio.clamp(max=10.0),
                transition_cosine.clamp(min=-1.0, max=1.0),
                signed_turn / math.pi,
                abs_turn / math.pi,
                cumulative_turn.clamp(min=-4.0, max=4.0),
                progress,
            ],
            dim=-1,
        )
        hidden = self.input_proj(features.reshape(batch * num_candidates, target_len, self.input_dim))
        hidden = self.encoder(hidden)
        pooled = hidden.mean(dim=1).view(batch, num_candidates, self.hidden_dim)
        return self.output_norm(pooled)


class TrajectoryResidualListwiseReranker(nn.Module):
    """Residual listwise scorer with candidate trajectory sequence embeddings."""

    def __init__(
        self,
        context_dim: int,
        pure_dim: int,
        token_dim: int,
        hidden_dim: int,
        num_candidates: int,
        num_heads: int,
        dropout: float = 0.0,
        initial_alpha: float = 0.05,
    ):
        super().__init__()
        self.input_dim = int(context_dim) + int(pure_dim) + int(token_dim)
        self.base_reranker = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.trajectory_encoder = CandidateTrajectoryEncoder(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.listwise_delta = ListwiseTransformerReranker(
            context_dim=int(context_dim) + int(hidden_dim),
            pure_dim=pure_dim,
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            num_candidates=num_candidates,
            num_heads=num_heads,
            dropout=dropout,
        )
        alpha = min(max(float(initial_alpha), 1e-4), 1.0 - 1e-4)
        self.listwise_alpha_logit = nn.Parameter(torch.tensor(math.log(alpha / (1.0 - alpha))))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.listwise_alpha_logit)

    def set_base_trainable(self, trainable: bool) -> None:
        for param in self.base_reranker.parameters():
            param.requires_grad = bool(trainable)

    def forward(
        self,
        context_features: torch.Tensor,
        pure_geometry: torch.Tensor,
        token_features: torch.Tensor,
        candidates: torch.Tensor,
        reference: torch.Tensor,
        tangent: torch.Tensor,
        normal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = torch.cat([context_features, pure_geometry, token_features], dim=-1)
        base_logits = self.base_reranker(features).squeeze(-1)
        trajectory_embedding = self.trajectory_encoder(candidates, reference, tangent, normal)
        delta_context = torch.cat([context_features, trajectory_embedding], dim=-1)
        delta_logits = self.listwise_delta(delta_context, pure_geometry, token_features)
        alpha = self.alpha.to(device=base_logits.device, dtype=base_logits.dtype)
        logits = base_logits + alpha * delta_logits
        return logits, base_logits, delta_logits, alpha, trajectory_embedding


COUNTERFACTUAL_PAIR_FEATURE_DIM = 10


class CounterfactualScorerDelta(nn.Module):
    """Field-gated scorer correction used only for counterfactual selection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_fields: int,
        pair_feature_dim: int = COUNTERFACTUAL_PAIR_FEATURE_DIM,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.pair_feature_dim = int(pair_feature_dim)
        self.field_embed = nn.Embedding(int(num_fields), self.hidden_dim)
        self.pair_feature_proj = nn.Sequential(
            nn.LayerNorm(self.pair_feature_dim),
            nn.Linear(self.pair_feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.delta_head = nn.Sequential(
            nn.LayerNorm(self.input_dim + self.hidden_dim),
            nn.Linear(self.input_dim + self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.alpha_logit = nn.Parameter(torch.tensor(-2.9444389791664403))

    @property
    def alpha(self) -> torch.Tensor:
        # Counterfactual scoring is an inference-time residual correction, not a
        # replacement for the base scorer.  A small floor keeps the correction
        # strong enough to change modes after calibration while still preserving
        # the ordinary non-counterfactual logits.
        return 0.20 + 0.80 * torch.sigmoid(self.alpha_logit)

    def forward(
        self,
        features: torch.Tensor,
        field_ids: torch.Tensor | None,
        pair_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if field_ids is None:
            return features.new_zeros(features.shape[:-1]), self.alpha.to(device=features.device, dtype=features.dtype)
        field_ids = field_ids.to(device=features.device, dtype=torch.long)
        field_hidden = self.field_embed(field_ids).to(dtype=features.dtype)
        while field_hidden.ndim < features.ndim:
            field_hidden = field_hidden.unsqueeze(1)
        field_hidden = field_hidden.expand(*features.shape[:-1], field_hidden.shape[-1])
        if pair_features is not None:
            pair_features = pair_features.to(device=features.device, dtype=features.dtype)
            if pair_features.shape[:-1] == features.shape[:-1] and pair_features.shape[-1] == self.pair_feature_dim:
                field_hidden = field_hidden + self.pair_feature_proj(pair_features)
        raw_delta = self.delta_head(torch.cat([features, field_hidden], dim=-1)).squeeze(-1)
        alpha = self.alpha.to(device=features.device, dtype=features.dtype)
        return alpha * raw_delta, alpha


class HybridPolicyDelta(nn.Module):
    """Small residual policy head for Table-4-aware candidate selection.

    Unlike CounterfactualScorerDelta, this branch has no alpha floor. It starts
    as an exact no-op and can be trained as a conservative residual policy on
    top of the frozen base scorer.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_candidates: int, num_fields: int):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_candidates = int(num_candidates)
        self.default_field_id = int(num_fields)
        self.field_embed = nn.Embedding(int(num_fields) + 1, self.hidden_dim)
        self.mode_embed = nn.Embedding(self.num_candidates, self.hidden_dim)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(self.input_dim + 2 * self.hidden_dim),
            nn.Linear(self.input_dim + 2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.alpha_logit = nn.Parameter(torch.tensor(-2.9444389791664403))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(
        self,
        features: torch.Tensor,
        field_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, num_candidates, _ = features.shape
        device = features.device
        if field_ids is None:
            field_ids = torch.full((batch,), self.default_field_id, device=device, dtype=torch.long)
        else:
            field_ids = field_ids.to(device=device, dtype=torch.long)
        field_hidden = self.field_embed(field_ids).to(dtype=features.dtype)
        field_hidden = field_hidden[:, None, :].expand(batch, num_candidates, -1)
        mode_ids = torch.arange(num_candidates, device=device, dtype=torch.long).clamp_max(
            self.num_candidates - 1
        )
        mode_hidden = self.mode_embed(mode_ids).to(dtype=features.dtype)
        mode_hidden = mode_hidden[None, :, :].expand(batch, num_candidates, -1)
        raw_delta = self.delta_head(torch.cat([features, field_hidden, mode_hidden], dim=-1)).squeeze(-1)
        alpha = self.alpha.to(device=device, dtype=features.dtype)
        return alpha * raw_delta, raw_delta, alpha


class CounterfactualPolicyHead(nn.Module):
    """Counterfactual-only candidate selection policy.

    This head is intentionally stronger than a residual delta: in
    counterfactual mode it can blend its own logits with the frozen base scorer,
    so Table-4 control labels are not trapped behind an old mode bias.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_candidates: int,
        num_fields: int,
        pair_feature_dim: int = COUNTERFACTUAL_PAIR_FEATURE_DIM,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_candidates = int(num_candidates)
        self.pair_feature_dim = int(pair_feature_dim)
        self.field_embed = nn.Embedding(int(num_fields), self.hidden_dim)
        self.mode_embed = nn.Embedding(self.num_candidates, self.hidden_dim)
        self.pair_feature_proj = nn.Sequential(
            nn.LayerNorm(self.pair_feature_dim),
            nn.Linear(self.pair_feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.policy_head = nn.Sequential(
            nn.LayerNorm(self.input_dim + 3 * self.hidden_dim),
            nn.Linear(self.input_dim + 3 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        # Starts as a no-op blend, but can be initialized near 1.0 for
        # scorer-only calibration where the base logits are known to be biased.
        self.gate_logit = nn.Parameter(torch.tensor(-10.0))

    @property
    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(
        self,
        features: torch.Tensor,
        field_ids: torch.Tensor | None,
        pair_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if field_ids is None:
            return features.new_zeros(features.shape[:-1]), self.gate.to(device=features.device, dtype=features.dtype)
        batch, num_candidates, _ = features.shape
        device = features.device
        field_ids = field_ids.to(device=device, dtype=torch.long)
        field_hidden = self.field_embed(field_ids).to(dtype=features.dtype)
        field_hidden = field_hidden[:, None, :].expand(batch, num_candidates, -1)
        mode_ids = torch.arange(num_candidates, device=device, dtype=torch.long).clamp_max(
            self.num_candidates - 1
        )
        mode_hidden = self.mode_embed(mode_ids).to(dtype=features.dtype)
        mode_hidden = mode_hidden[None, :, :].expand(batch, num_candidates, -1)
        if pair_features is None:
            pair_hidden = features.new_zeros(batch, num_candidates, self.hidden_dim)
        else:
            pair_features = pair_features.to(device=device, dtype=features.dtype)
            if pair_features.shape[:-1] == features.shape[:-1] and pair_features.shape[-1] == self.pair_feature_dim:
                pair_hidden = self.pair_feature_proj(pair_features)
            else:
                pair_hidden = features.new_zeros(batch, num_candidates, self.hidden_dim)
        policy_logits = self.policy_head(
            torch.cat([features, field_hidden, mode_hidden, pair_hidden], dim=-1)
        ).squeeze(-1)
        gate = self.gate.to(device=device, dtype=features.dtype)
        return policy_logits, gate


class CounterfactualMetricPolicyHead(nn.Module):
    """Metric-only Pareto selector for counterfactual candidate modes.

    The generic policy head can underfit Table-IV labels because it mixes
    decoder hidden features with metric evidence. This head keeps the decision
    close to the offline Pareto teacher by scoring only inference-available
    pair features.
    """

    def __init__(self, pair_feature_dim: int = COUNTERFACTUAL_PAIR_FEATURE_DIM):
        super().__init__()
        self.pair_feature_dim = int(pair_feature_dim)
        self.score = nn.Linear(self.pair_feature_dim, 1)

    def forward(self, pair_features: torch.Tensor | None) -> torch.Tensor | None:
        if (
            pair_features is None
            or pair_features.ndim != 3
            or pair_features.shape[-1] != self.pair_feature_dim
        ):
            return None
        return self.score(pair_features).squeeze(-1)


class DiTAnchorDecoder(nn.Module):
    """Reference-anchored DiT decoder with tangent-normal residual output.

    This module keeps the same external contract as AnchorResidualDecoder:
    reference + fused condition -> continuous trajectory. Internally it replaces
    the small point MLP with a DirectTraj-style adaLN Transformer backbone.
    """

    def __init__(
        self,
        cond_dim: int = 128,
        hidden_dim: int = 192,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        target_len: int = 200,
        tangent_scale: float = 0.35,
        normal_scale: float = 0.20,
        dropout: float = 0.0,
        num_candidates: int = 1,
        max_cond_tokens: int = 16,
        token_calibration_path: str | Path | None = None,
        reranker_type: str = "concat",
        mode_control_type: str = "none",
        mode_control_strength: float = 1.0,
        mode_role_budget_strength: float = 1.0,
        mode_role_config_path: str | Path | None = None,
        mode_role_prior_logit_weight: float = 0.0,
        mode_role_prior_field_weights: str | dict | list | tuple | None = None,
        mode_role_prior_scope: str = "all",
        route_skeleton_type: str = "none",
        route_skeleton_points: int = 16,
        route_skeleton_strength: float = 1.0,
        route_skeleton_gate_init_logit: float = 20.0,
        route_skeleton_tangent_scale: float = 0.45,
        route_skeleton_normal_scale: float = 0.35,
        route_skeleton_condition_source: str = "fused",
        route_proposal_logit_weight: float = 0.0,
        route_control_budget_ratio: float = 0.0,
        route_control_budget_mode: str = "rms",
        route_control_budget_min: float = 0.0,
        route_control_budget_max: float = 0.0,
        route_control_selection_adapter_weight: float = 0.0,
        route_control_selection_adapter_clip: float = 2.0,
        route_prototype_bank_path: str | Path | None = None,
        route_prototype_prior_strength: float = 0.0,
        route_prototype_prior_clip: float = 1.25,
        route_prototype_prior_mix: str = "fixed",
        route_prototype_prior_gate: str = "route_skeleton",
        route_token_condition_path: str = "shared",
        direction_control_type: str = "none",
        direction_control_strength: float = 0.0,
        direction_control_tangent_weight: float = 1.0,
        direction_control_apply: str = "all",
        curvature_control_type: str = "none",
        curvature_control_strength: float = 0.0,
        curvature_control_apply: str = "all",
        curvature_control_projection: str = "none",
        curvature_control_projection_strength: float = 1.0,
        counterfactual_direction_residual_type: str = "shared",
        counterfactual_curvature_residual_type: str = "shared",
        counterfactual_reference_residual_type: str = "shared",
        reference_control_type: str = "none",
        reference_control_strength: float = 0.0,
        reference_control_apply: str = "all",
        reference_positive_candidate_strength: float = 0.0,
        reference_positive_candidate_scope: str = "all",
        reference_distance_candidate_strength: float = 0.0,
        reference_distance_candidate_scope: str = "none",
        reference_distance_candidate_mode: str = "global_normal",
        counterfactual_residual_strength: float = 1.0,
        counterfactual_route_frame_budget_mode: str = "none",
        counterfactual_route_frame_tangent_budget: float = 0.08,
        counterfactual_route_frame_normal_budget: float = 0.08,
        counterfactual_route_frame_fields: str | list | tuple | set | None = None,
        counterfactual_local_proposal_type: str = "none",
        counterfactual_local_proposal_strength: float = 0.0,
        language_control_residual_type: str = "direct",
        language_control_budget_ratio: float = 0.35,
        language_control_soft_gate_min: float = 0.25,
        language_control_soft_gate_max: float = 1.20,
        language_control_field_ratios: str | dict | list | tuple | None = None,
        language_control_gate_fields: str | list | tuple | set | None = None,
        language_control_gate_scope: str = "all",
        counterfactual_eval_selector_weight: float = 0.0,
        counterfactual_eval_selector_instruction_weight: float = 0.5,
        counterfactual_eval_selector_target_effect_weight: float = 1.0,
        counterfactual_eval_selector_collateral_weight: float = 0.5,
        counterfactual_eval_selector_mode_switch_weight: float = 0.0,
        counterfactual_eval_selector_reference_mode_switch_weight: float = 0.0,
        counterfactual_eval_selector_reference_reduction_weight: float = 0.0,
        counterfactual_eval_selector_reference_score_weight: float = 0.0,
        counterfactual_eval_selector_feasible_first: bool = True,
        counterfactual_eval_selector_success_margin: float = 0.0,
        counterfactual_eval_selector_collateral_margin: float = 0.08,
        counterfactual_eval_selector_collateral_mode: str = "independent",
        counterfactual_policy_blend_mode: str = "blend",
        counterfactual_metric_policy_weight: float = 0.0,
        counterfactual_metric_policy_feasible_first: bool = True,
        counterfactual_metric_policy_success_margin: float = 0.0,
        counterfactual_metric_policy_collateral_margin: float = 0.08,
        counterfactual_metric_policy_raw_collateral_margin: float = 0.15,
        semantic_control_enabled: bool = False,
        field_control_fields: list[str] | tuple[str, ...] | None = None,
        field_control_latent_dim: int = 128,
        field_control_num_act_classes: int = 8,
        local_edit_window_predictor: str = "field_head_v1",
        base_preserving_selector_enabled: bool = False,
        selector_noncollapse_mode: str = "additive",
        pairwise_feasible_preference_enabled: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.target_len = int(target_len)
        self.tangent_scale = float(tangent_scale)
        self.normal_scale = float(normal_scale)
        self.num_candidates = int(num_candidates)
        self.max_cond_tokens = int(max_cond_tokens)
        self.token_feature_mode = "none"
        self.reranker_type = str(reranker_type)
        self.mode_control_type = str(mode_control_type)
        self.mode_control_strength = float(mode_control_strength)
        self.mode_role_budget_strength = float(mode_role_budget_strength)
        self.mode_role_config_path = str(mode_role_config_path) if mode_role_config_path is not None else None
        self.mode_role_prior_logit_weight = float(mode_role_prior_logit_weight)
        self.mode_role_prior_field_weights = self.parse_mode_role_prior_field_weights(
            mode_role_prior_field_weights
        )
        self.mode_role_prior_scope = str(mode_role_prior_scope)
        self.route_skeleton_type = str(route_skeleton_type)
        self.route_skeleton_points = int(route_skeleton_points)
        self.route_skeleton_strength = float(route_skeleton_strength)
        self.route_skeleton_gate_logit = nn.Parameter(torch.tensor(float(route_skeleton_gate_init_logit)))
        self.route_skeleton_tangent_scale = float(route_skeleton_tangent_scale)
        self.route_skeleton_normal_scale = float(route_skeleton_normal_scale)
        self.route_skeleton_condition_source = str(route_skeleton_condition_source)
        self.route_proposal_logit_weight = float(route_proposal_logit_weight)
        self.route_control_budget_ratio = float(route_control_budget_ratio)
        self.route_control_budget_mode = str(route_control_budget_mode)
        self.route_control_budget_min = float(route_control_budget_min)
        self.route_control_budget_max = float(route_control_budget_max)
        self.route_control_selection_adapter_weight = float(route_control_selection_adapter_weight)
        self.route_control_selection_adapter_clip = float(route_control_selection_adapter_clip)
        self.route_prototype_bank_path = str(route_prototype_bank_path) if route_prototype_bank_path else None
        self.route_prototype_prior_strength = float(route_prototype_prior_strength)
        self.route_prototype_prior_clip = float(route_prototype_prior_clip)
        self.route_prototype_prior_mix = str(route_prototype_prior_mix)
        self.route_prototype_prior_gate = str(route_prototype_prior_gate)
        self.route_token_condition_path = str(route_token_condition_path)
        self.direction_control_type = str(direction_control_type)
        self.direction_control_strength = float(direction_control_strength)
        self.direction_control_tangent_weight = float(direction_control_tangent_weight)
        self.direction_control_apply = str(direction_control_apply)
        self.curvature_control_type = str(curvature_control_type)
        self.curvature_control_strength = float(curvature_control_strength)
        self.curvature_control_apply = str(curvature_control_apply)
        self.curvature_control_projection = str(curvature_control_projection)
        self.curvature_control_projection_strength = float(curvature_control_projection_strength)
        self.counterfactual_direction_residual_type = str(counterfactual_direction_residual_type)
        self.counterfactual_curvature_residual_type = str(counterfactual_curvature_residual_type)
        self.counterfactual_reference_residual_type = str(counterfactual_reference_residual_type)
        self.reference_control_type = str(reference_control_type)
        self.reference_control_strength = float(reference_control_strength)
        self.reference_control_apply = str(reference_control_apply)
        self.reference_positive_candidate_strength = float(reference_positive_candidate_strength)
        self.reference_positive_candidate_scope = str(reference_positive_candidate_scope)
        self.reference_distance_candidate_strength = float(reference_distance_candidate_strength)
        self.reference_distance_candidate_scope = str(reference_distance_candidate_scope)
        self.reference_distance_candidate_mode = str(reference_distance_candidate_mode)
        self.counterfactual_residual_strength = float(counterfactual_residual_strength)
        self.counterfactual_route_frame_budget_mode = str(counterfactual_route_frame_budget_mode)
        self.counterfactual_route_frame_tangent_budget = float(counterfactual_route_frame_tangent_budget)
        self.counterfactual_route_frame_normal_budget = float(counterfactual_route_frame_normal_budget)
        self.counterfactual_route_frame_fields = self.parse_route_frame_budget_fields(
            counterfactual_route_frame_fields
        )
        self.counterfactual_local_proposal_type = str(counterfactual_local_proposal_type)
        self.counterfactual_local_proposal_strength = float(counterfactual_local_proposal_strength)
        self.language_control_residual_type = str(language_control_residual_type)
        self.language_control_budget_ratio = float(language_control_budget_ratio)
        self.language_control_soft_gate_min = float(language_control_soft_gate_min)
        self.language_control_soft_gate_max = float(language_control_soft_gate_max)
        self.language_control_field_ratios = self.parse_language_control_field_ratios(
            language_control_field_ratios
        )
        self.language_control_gate_fields = self.parse_language_control_gate_fields(
            language_control_gate_fields
        )
        self.language_control_gate_scope = str(language_control_gate_scope)
        self.counterfactual_eval_selector_weight = float(counterfactual_eval_selector_weight)
        self.counterfactual_eval_selector_instruction_weight = float(counterfactual_eval_selector_instruction_weight)
        self.counterfactual_eval_selector_target_effect_weight = float(counterfactual_eval_selector_target_effect_weight)
        self.counterfactual_eval_selector_collateral_weight = float(counterfactual_eval_selector_collateral_weight)
        self.counterfactual_eval_selector_mode_switch_weight = float(
            counterfactual_eval_selector_mode_switch_weight
        )
        self.counterfactual_eval_selector_reference_mode_switch_weight = float(
            counterfactual_eval_selector_reference_mode_switch_weight
        )
        self.counterfactual_eval_selector_reference_reduction_weight = float(
            counterfactual_eval_selector_reference_reduction_weight
        )
        self.counterfactual_eval_selector_reference_score_weight = float(
            counterfactual_eval_selector_reference_score_weight
        )
        self.counterfactual_eval_selector_feasible_first = bool(counterfactual_eval_selector_feasible_first)
        self.counterfactual_eval_selector_success_margin = float(counterfactual_eval_selector_success_margin)
        self.counterfactual_eval_selector_collateral_margin = float(counterfactual_eval_selector_collateral_margin)
        self.counterfactual_eval_selector_collateral_mode = str(counterfactual_eval_selector_collateral_mode)
        self.counterfactual_policy_blend_mode = str(counterfactual_policy_blend_mode)
        self.counterfactual_metric_policy_weight = float(counterfactual_metric_policy_weight)
        self.counterfactual_metric_policy_feasible_first = bool(counterfactual_metric_policy_feasible_first)
        self.counterfactual_metric_policy_success_margin = float(counterfactual_metric_policy_success_margin)
        self.counterfactual_metric_policy_collateral_margin = float(counterfactual_metric_policy_collateral_margin)
        self.counterfactual_metric_policy_raw_collateral_margin = float(counterfactual_metric_policy_raw_collateral_margin)
        if self.counterfactual_policy_blend_mode not in {"blend", "override"}:
            raise ValueError(
                "counterfactual_policy_blend_mode must be 'blend' or 'override', "
                f"got {self.counterfactual_policy_blend_mode!r}."
            )
        self.semantic_control_enabled = bool(semantic_control_enabled)
        if isinstance(field_control_fields, str):
            parsed_field_control_fields = [piece.strip() for piece in field_control_fields.split(",") if piece.strip()]
        else:
            parsed_field_control_fields = list(field_control_fields or ["direction", "curvature", "reference_adherence"])
        self.field_control_fields = [str(field) for field in parsed_field_control_fields]
        self.field_control_latent_dim = int(field_control_latent_dim)
        self.field_control_num_act_classes = int(field_control_num_act_classes)
        self.reference_field_control_available = "reference_adherence" in self.field_control_fields
        self.local_edit_window_predictor = str(local_edit_window_predictor)
        self.base_preserving_selector_enabled = bool(base_preserving_selector_enabled)
        self.selector_noncollapse_mode = str(selector_noncollapse_mode)
        self.pairwise_feasible_preference_enabled = bool(pairwise_feasible_preference_enabled)
        if self.num_candidates < 1:
            raise ValueError("num_candidates must be >= 1")
        if self.selector_noncollapse_mode not in {"disabled", "additive"}:
            raise ValueError(
                f"Unsupported selector_noncollapse_mode={self.selector_noncollapse_mode}. "
                "Expected one of {'disabled', 'additive'}."
            )
        if self.mode_control_type not in {
            "none",
            "residual_gate",
            "token_residual_gate",
            "token_role_residual_gate",
            "mode_specific_heads",
            "role_gated_heads",
        }:
            raise ValueError(f"Unsupported mode_control_type={self.mode_control_type}")
        if self.route_skeleton_type not in {
            "none",
            "coarse_residual",
            "token_wave",
            "transition_modes",
            "proposal_bank",
            "route_control_modes",
            "prototype_bank",
        }:
            raise ValueError(f"Unsupported route_skeleton_type={self.route_skeleton_type}")
        if self.route_skeleton_condition_source not in {"fused", "od_only"}:
            raise ValueError(
                f"Unsupported route_skeleton_condition_source={self.route_skeleton_condition_source}"
            )
        if self.route_token_condition_path not in {"shared", "separate"}:
            raise ValueError(f"Unsupported route_token_condition_path={self.route_token_condition_path}")
        if self.mode_role_prior_scope not in {"all", "counterfactual", "base"}:
            raise ValueError(
                f"Unsupported mode_role_prior_scope={self.mode_role_prior_scope}. "
                "Expected one of {'all', 'counterfactual', 'base'}."
            )
        if self.route_prototype_prior_mix not in {"fixed", "direction_role"}:
            raise ValueError(
                f"Unsupported route_prototype_prior_mix={self.route_prototype_prior_mix}. "
                "Expected one of {'fixed', 'direction_role'}."
            )
        if self.route_prototype_prior_gate not in {"route_skeleton", "route_control_adapter"}:
            raise ValueError(
                f"Unsupported route_prototype_prior_gate={self.route_prototype_prior_gate}. "
                "Expected one of {'route_skeleton', 'route_control_adapter'}."
            )
        if self.route_control_budget_mode not in {"rms", "max"}:
            raise ValueError(
                f"Unsupported route_control_budget_mode={self.route_control_budget_mode}. "
                "Expected one of {'rms', 'max'}."
            )
        if self.route_control_budget_ratio < 0.0:
            raise ValueError("route_control_budget_ratio must be >= 0")
        if self.route_control_budget_min < 0.0:
            raise ValueError("route_control_budget_min must be >= 0")
        if self.route_control_budget_max < 0.0:
            raise ValueError("route_control_budget_max must be >= 0")
        if self.route_control_budget_max > 0.0 and self.route_control_budget_min > self.route_control_budget_max:
            raise ValueError("route_control_budget_min must be <= route_control_budget_max when max is set")
        if self.route_control_selection_adapter_weight < 0.0:
            raise ValueError("route_control_selection_adapter_weight must be >= 0")
        if self.route_control_selection_adapter_clip < 0.0:
            raise ValueError("route_control_selection_adapter_clip must be >= 0")
        if self.route_skeleton_points < 2:
            raise ValueError("route_skeleton_points must be >= 2")
        if self.direction_control_type not in {
            "none",
            "token_basis",
            "hybrid_rule_basis",
            "direction_role_basis",
            "direction_structural_residual",
        }:
            raise ValueError(f"Unsupported direction_control_type={self.direction_control_type}")
        if self.direction_control_apply not in {"all", "counterfactual"}:
            raise ValueError(f"Unsupported direction_control_apply={self.direction_control_apply}")
        if self.curvature_control_type not in {"none", "token_basis"}:
            raise ValueError(f"Unsupported curvature_control_type={self.curvature_control_type}")
        if self.curvature_control_apply not in {"all", "counterfactual"}:
            raise ValueError(f"Unsupported curvature_control_apply={self.curvature_control_apply}")
        if self.curvature_control_projection not in {"none", "highfreq_preserve_direction"}:
            raise ValueError(
                f"Unsupported curvature_control_projection={self.curvature_control_projection}"
            )
        if self.counterfactual_direction_residual_type not in {
            "shared",
            "transition_contract_pulse",
            "transition_arc_skeleton",
            "transition_arc_multiturn_burst",
            "transition_hybrid_skeleton_reflect",
            "transition_pivot_skeleton",
            "transition_reflect_skeleton",
            "transition_skeleton_reversal",
            "transition_skeleton_reversal_scurve",
            "transition_skeleton",
            "transition_template",
            "transition_template_balanced",
            "transition_template_contract",
            "transition_template_mean_projected",
            "transition_template_projected",
        }:
            raise ValueError(
                "Unsupported counterfactual_direction_residual_type="
                f"{self.counterfactual_direction_residual_type}"
            )
        if self.counterfactual_curvature_residual_type not in {
            "shared",
            "local_turn_density",
            "base_preserving_local_turn_density",
            "base_selected_projected_normal_only",
            "base_selected_projected_normal_selector_compose",
            "direction_neutral_turn_density",
            "direction_preserving_curvature_burst",
            "signed_balanced_winding_basis",
            "midlocal_winding_pulse",
        }:
            raise ValueError(
                "Unsupported counterfactual_curvature_residual_type="
                f"{self.counterfactual_curvature_residual_type}"
            )
        if self.counterfactual_reference_residual_type not in {
            "shared",
            "normal_only",
            "normal_mean_shift",
            "normal_hybrid_shift",
            "normal_only_projected",
        }:
            raise ValueError(
                "Unsupported counterfactual_reference_residual_type="
                f"{self.counterfactual_reference_residual_type}"
            )
        if self.reference_control_type not in {"none", "token_basis", "mode_role_basis"}:
            raise ValueError(f"Unsupported reference_control_type={self.reference_control_type}")
        if self.reference_control_apply not in {"all", "counterfactual"}:
            raise ValueError(f"Unsupported reference_control_apply={self.reference_control_apply}")
        if self.reference_distance_candidate_scope not in {
            "all",
            "none",
            "instruction",
            "counterfactual_reference_adherence",
        }:
            raise ValueError(
                "Unsupported reference_distance_candidate_scope="
                f"{self.reference_distance_candidate_scope}"
            )
        if self.reference_distance_candidate_mode not in {"global_normal", "local_direction_preserving"}:
            raise ValueError(
                "Unsupported reference_distance_candidate_mode="
                f"{self.reference_distance_candidate_mode}"
            )
        if self.counterfactual_route_frame_budget_mode not in {"none", "clamp", "soft_tanh"}:
            raise ValueError(
                "Unsupported counterfactual_route_frame_budget_mode="
                f"{self.counterfactual_route_frame_budget_mode}"
            )
        if self.counterfactual_route_frame_tangent_budget < 0.0:
            raise ValueError("counterfactual_route_frame_tangent_budget must be >= 0")
        if self.counterfactual_route_frame_normal_budget < 0.0:
            raise ValueError("counterfactual_route_frame_normal_budget must be >= 0")
        if self.counterfactual_local_proposal_type not in {"none", "mask_controlled_v1"}:
            raise ValueError(
                "Unsupported counterfactual_local_proposal_type="
                f"{self.counterfactual_local_proposal_type}"
            )
        if self.counterfactual_local_proposal_strength < 0.0:
            raise ValueError("counterfactual_local_proposal_strength must be >= 0")
        if self.language_control_residual_type not in {"direct", "bounded_sum", "field_soft_gate"}:
            raise ValueError(
                f"Unsupported language_control_residual_type={self.language_control_residual_type}"
            )
        if self.language_control_soft_gate_min < 0.0:
            raise ValueError("language_control_soft_gate_min must be >= 0")
        if self.language_control_soft_gate_max <= 0.0:
            raise ValueError("language_control_soft_gate_max must be > 0")
        if self.language_control_soft_gate_min > self.language_control_soft_gate_max:
            raise ValueError("language_control_soft_gate_min must be <= language_control_soft_gate_max")
        if self.language_control_gate_scope not in {"all", "counterfactual"}:
            raise ValueError(f"Unsupported language_control_gate_scope={self.language_control_gate_scope}")

        self.point_embed = nn.Sequential(
            nn.Linear(8, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(self.cond_dim),
            nn.Linear(self.cond_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.token_cond_proj = nn.Sequential(
            nn.LayerNorm(self.cond_dim),
            nn.Linear(self.cond_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.route_token_cond_proj = nn.Sequential(
            nn.LayerNorm(self.cond_dim),
            nn.Linear(self.cond_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.route_token_gate_logit = nn.Parameter(torch.tensor(-4.0))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.target_len, self.hidden_dim), requires_grad=False)
        self.token_pos_embed = nn.Parameter(torch.zeros(1, self.max_cond_tokens, self.hidden_dim), requires_grad=False)
        self.candidate_embed = nn.Parameter(torch.zeros(self.num_candidates, self.hidden_dim))
        self.mode_logit_bias = nn.Parameter(torch.zeros(self.num_candidates))
        self.mode_role_dim = 6
        self.register_buffer(
            "mode_role_table",
            self._build_default_mode_role_table(self.num_candidates),
            persistent=True,
        )
        self.register_buffer(
            "mode_role_budget_table",
            self._build_default_mode_role_budget_table(self.num_candidates),
            persistent=True,
        )
        self.register_buffer(
            "mode_role_control_table",
            self._build_default_mode_role_control_table(self.num_candidates),
            persistent=True,
        )
        self.register_buffer(
            "direction_mode_role_table",
            self._build_default_direction_mode_role_table(self.num_candidates),
            persistent=True,
        )
        self.register_buffer(
            "route_transition_mode_table",
            self._build_default_route_transition_mode_table(self.num_candidates),
            persistent=True,
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(self.depth)
            ]
        )
        self.final_norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
        )
        self.counterfactual_cond_adapter = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.counterfactual_token_adapter = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.counterfactual_field_embed = nn.Embedding(len(DEFAULT_MOBILITY_VOCAB.fields), self.hidden_dim)
        self.counterfactual_field_cond_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for _ in DEFAULT_MOBILITY_VOCAB.fields
            ]
        )
        self.counterfactual_field_token_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
                for _ in DEFAULT_MOBILITY_VOCAB.fields
            ]
        )
        self.counterfactual_field_residual_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(3 * self.hidden_dim),
                    nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.hidden_dim, 6),
                )
                for _ in DEFAULT_MOBILITY_VOCAB.fields
            ]
        )
        self.counterfactual_local_proposal_heads = nn.ModuleDict(
            {
                "direction": nn.Sequential(
                    nn.LayerNorm(3 * self.hidden_dim),
                    nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.hidden_dim, 4),
                ),
                "multi_turn_creation": nn.Sequential(
                    nn.LayerNorm(3 * self.hidden_dim),
                    nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.hidden_dim, 4),
                ),
                "curvature": nn.Sequential(
                    nn.LayerNorm(3 * self.hidden_dim),
                    nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.hidden_dim, 4),
                ),
                "winding_local": nn.Sequential(
                    nn.LayerNorm(3 * self.hidden_dim),
                    nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.hidden_dim, 4),
                ),
                "reference_adherence": nn.Sequential(
                    nn.LayerNorm(3 * self.hidden_dim),
                    nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.hidden_dim, 4),
                ),
                "default": nn.Sequential(
                    nn.LayerNorm(3 * self.hidden_dim),
                    nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.hidden_dim, 4),
                ),
            }
        )
        self.reference_local_control_latent_proj = nn.Sequential(
            nn.LayerNorm(self.field_control_latent_dim),
            nn.Linear(self.field_control_latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.reference_local_control_act_embed = nn.Embedding(
            max(1, self.field_control_num_act_classes),
            self.hidden_dim,
        )
        self.reference_local_act_to_proposal = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 4),
        )
        self.reference_local_proposal_head = nn.Sequential(
            nn.LayerNorm(5 * self.hidden_dim),
            nn.Linear(5 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 4),
        )
        self.reference_local_budget_gate_head = nn.Sequential(
            nn.LayerNorm(5 * self.hidden_dim),
            nn.Linear(5 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.residual_head = nn.Linear(self.hidden_dim, 3)
        self.mode_control = None
        self.mode_role_proj = None
        self.mode_role_score = None
        self.mode_skeleton = None
        self.mode_role_budget_head = None
        self.mode_role_residual_adapter = None
        self.mode_specific_residual_heads = None
        self.direction_control_head = None
        self.curvature_control_head = None
        self.reference_control_head = None
        self.route_skeleton_head = None
        self.route_wave_head = None
        self.route_transition_mode_head = None
        self.route_proposal_head = None
        self.route_control_role_proj = None
        self.route_control_mode_head = None
        self.route_control_selection_adapter = None
        self.route_control_adapter_gate_logit = None
        self.route_prototype_head = None
        self.register_buffer(
            "route_prototype_bank",
            torch.zeros(self.num_candidates, self.route_skeleton_points, 2),
            persistent=True,
        )
        if self.route_prototype_bank_path:
            self.load_route_prototype_bank(self.route_prototype_bank_path)
        if self.route_skeleton_type == "coarse_residual":
            self.route_skeleton_head = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 3),
            )
        elif self.route_skeleton_type == "token_wave":
            self.route_wave_head = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 4),
            )
        elif self.route_skeleton_type == "transition_modes":
            self.route_transition_mode_head = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 8),
            )
        elif self.route_skeleton_type == "proposal_bank":
            self.route_proposal_head = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 6),
            )
        elif self.route_skeleton_type == "route_control_modes":
            self.route_skeleton_head = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 3),
            )
            self.route_control_role_proj = nn.Sequential(
                nn.LayerNorm(self.mode_role_dim),
                nn.Linear(self.mode_role_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.route_control_mode_head = nn.Sequential(
                nn.LayerNorm(3 * self.hidden_dim),
                nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 8),
            )
            self.route_control_selection_adapter = nn.Sequential(
                nn.LayerNorm(3 * self.hidden_dim),
                nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            )
            self.route_control_adapter_gate_logit = nn.Parameter(torch.tensor(-6.0))
        elif self.route_skeleton_type == "prototype_bank":
            self.route_prototype_head = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 4),
            )
        if self.mode_control_type == "residual_gate":
            self.mode_control = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 3),
            )
        elif self.mode_control_type == "token_residual_gate":
            self.mode_control = nn.Sequential(
                nn.LayerNorm(2 * self.hidden_dim),
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 3),
            )
        elif self.mode_control_type in {"token_role_residual_gate", "mode_specific_heads", "role_gated_heads"}:
            self.mode_role_proj = nn.Sequential(
                nn.LayerNorm(self.mode_role_dim),
                nn.Linear(self.mode_role_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.mode_control = nn.Sequential(
                nn.LayerNorm(3 * self.hidden_dim),
                nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 3),
            )
            self.mode_role_score = nn.Sequential(
                nn.LayerNorm(3 * self.hidden_dim),
                nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            )
            self.mode_skeleton = nn.Sequential(
                nn.LayerNorm(3 * self.hidden_dim),
                nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 4),
            )
            self.mode_role_budget_head = nn.Sequential(
                nn.LayerNorm(3 * self.hidden_dim),
                nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 3),
            )
            self.mode_role_residual_adapter = nn.Sequential(
                nn.LayerNorm(3 * self.hidden_dim),
                nn.Linear(3 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 4),
            )
            if self.mode_control_type in {"mode_specific_heads", "role_gated_heads"}:
                self.mode_specific_residual_heads = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.LayerNorm(self.hidden_dim),
                            nn.Linear(self.hidden_dim, self.hidden_dim),
                            nn.SiLU(),
                            nn.Linear(self.hidden_dim, 3),
                        )
                        for _ in range(self.num_candidates)
                    ]
                )
        if self.direction_control_type in {"token_basis", "hybrid_rule_basis", "direction_structural_residual"}:
            self.direction_control_head = nn.Sequential(
                nn.LayerNorm(2 * self.hidden_dim),
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(
                    self.hidden_dim,
                    6
                    if self.direction_control_type == "direction_structural_residual"
                    else (14 if self.direction_control_type == "hybrid_rule_basis" else 10),
                ),
            )
        if self.curvature_control_type == "token_basis":
            self.curvature_control_head = nn.Sequential(
                nn.LayerNorm(2 * self.hidden_dim),
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 8),
            )
        if self.reference_control_type == "token_basis":
            self.reference_control_head = nn.Sequential(
                nn.LayerNorm(2 * self.hidden_dim),
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 6),
            )
        self.reranker_pure_geometry_dim = 12
        self.reranker_token_geometry_dim = 4
        self.reranker_geometry_dim = self.reranker_pure_geometry_dim + self.reranker_token_geometry_dim
        self.register_buffer("calib_distance_center", torch.zeros(4), persistent=True)
        self.register_buffer("calib_distance_scale", torch.ones(4), persistent=True)
        self.register_buffer("calib_ref_center", torch.zeros(4), persistent=True)
        self.register_buffer("calib_ref_scale", torch.ones(4), persistent=True)
        self.register_buffer("calib_curv_center", torch.zeros(3), persistent=True)
        self.register_buffer("calib_curv_scale", torch.ones(3), persistent=True)
        self.register_buffer("calib_progress_center", torch.zeros(3), persistent=True)
        self.register_buffer("calib_progress_scale", torch.ones(3), persistent=True)
        self.register_buffer("calib_direction_center", torch.zeros(5), persistent=True)
        self.register_buffer("calib_direction_scale", torch.ones(5), persistent=True)
        self.register_buffer("calib_direction_metric", torch.zeros(5, dtype=torch.long), persistent=True)
        self.register_buffer("calib_distance_metric", torch.zeros(4, dtype=torch.long), persistent=True)
        self.register_buffer("calib_ref_metric", torch.full((4,), 1, dtype=torch.long), persistent=True)
        self.register_buffer("calib_curv_metric", torch.full((3,), 3, dtype=torch.long), persistent=True)
        self.register_buffer("calib_progress_metric", torch.full((3,), 6, dtype=torch.long), persistent=True)
        self.reranker_context_dim = self.hidden_dim * 2 + 4
        if self.reranker_type == "comparative_gated":
            self.reranker_effective_pure_dim = self.reranker_pure_geometry_dim * 4
            self.reranker_effective_token_dim = self.reranker_token_geometry_dim * 4
        else:
            self.reranker_effective_pure_dim = self.reranker_pure_geometry_dim
            self.reranker_effective_token_dim = self.reranker_token_geometry_dim
        self.reranker_effective_geometry_dim = self.reranker_effective_pure_dim + self.reranker_effective_token_dim

        if self.reranker_type == "concat":
            self.reranker = nn.Sequential(
                nn.LayerNorm(self.reranker_context_dim + self.reranker_geometry_dim),
                nn.Linear(self.reranker_context_dim + self.reranker_geometry_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            )
        elif self.reranker_type in {"split_gated", "comparative_gated"}:
            self.reranker = SplitGatedReranker(
                context_dim=self.reranker_context_dim,
                pure_dim=self.reranker_effective_pure_dim,
                token_dim=self.reranker_effective_token_dim,
                hidden_dim=self.hidden_dim,
            )
        elif self.reranker_type == "listwise_transformer":
            self.reranker = ListwiseTransformerReranker(
                context_dim=self.reranker_context_dim,
                pure_dim=self.reranker_effective_pure_dim,
                token_dim=self.reranker_effective_token_dim,
                hidden_dim=self.hidden_dim,
                num_candidates=self.num_candidates,
                num_heads=self.num_heads,
                dropout=dropout,
            )
        elif self.reranker_type == "residual_listwise":
            self.reranker = ResidualListwiseReranker(
                context_dim=self.reranker_context_dim,
                pure_dim=self.reranker_effective_pure_dim,
                token_dim=self.reranker_effective_token_dim,
                hidden_dim=self.hidden_dim,
                num_candidates=self.num_candidates,
                num_heads=self.num_heads,
                dropout=dropout,
            )
        elif self.reranker_type == "trajectory_residual_listwise":
            self.reranker = TrajectoryResidualListwiseReranker(
                context_dim=self.reranker_context_dim,
                pure_dim=self.reranker_effective_pure_dim,
                token_dim=self.reranker_effective_token_dim,
                hidden_dim=self.hidden_dim,
                num_candidates=self.num_candidates,
                num_heads=self.num_heads,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported reranker_type={self.reranker_type}")
        self.counterfactual_scorer_delta = CounterfactualScorerDelta(
            input_dim=self.reranker_context_dim + self.reranker_geometry_dim,
            hidden_dim=self.hidden_dim,
            num_fields=len(DEFAULT_MOBILITY_VOCAB.fields),
        )
        self.hybrid_policy_delta = HybridPolicyDelta(
            input_dim=self.reranker_context_dim + self.reranker_geometry_dim,
            hidden_dim=self.hidden_dim,
            num_candidates=self.num_candidates,
            num_fields=len(DEFAULT_MOBILITY_VOCAB.fields),
        )
        self.counterfactual_policy_head = CounterfactualPolicyHead(
            input_dim=self.reranker_context_dim + self.reranker_geometry_dim,
            hidden_dim=self.hidden_dim,
            num_candidates=self.num_candidates,
            num_fields=len(DEFAULT_MOBILITY_VOCAB.fields),
        )
        self.counterfactual_metric_policy_head = CounterfactualMetricPolicyHead()
        self.initialize_weights()
        if mode_role_config_path is not None:
            self.load_mode_role_config(mode_role_config_path)
        if token_calibration_path is not None:
            self.load_token_calibration(token_calibration_path)

    def initialize_weights(self) -> None:
        def _init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_init)
        self.pos_embed.data.copy_(self._build_sincos_pos_embed(self.target_len, self.hidden_dim).unsqueeze(0))
        self.token_pos_embed.data.copy_(self._build_sincos_pos_embed(self.max_cond_tokens, self.hidden_dim).unsqueeze(0))
        nn.init.normal_(self.candidate_embed, std=0.02)
        nn.init.constant_(self.mode_logit_bias, 0)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_modulation[-1].weight, 0)
        nn.init.constant_(self.final_modulation[-1].bias, 0)
        nn.init.constant_(self.counterfactual_cond_adapter[-1].weight, 0)
        nn.init.constant_(self.counterfactual_cond_adapter[-1].bias, 0)
        nn.init.constant_(self.counterfactual_token_adapter[-1].weight, 0)
        nn.init.constant_(self.counterfactual_token_adapter[-1].bias, 0)
        nn.init.normal_(self.counterfactual_field_embed.weight, std=0.02)
        for adapter in self.counterfactual_field_cond_adapters:
            nn.init.constant_(adapter[-1].weight, 0)
            nn.init.constant_(adapter[-1].bias, 0)
        for adapter in self.counterfactual_field_token_adapters:
            nn.init.constant_(adapter[-1].weight, 0)
            nn.init.constant_(adapter[-1].bias, 0)
        for head in self.counterfactual_field_residual_heads:
            nn.init.constant_(head[-1].weight, 0)
            nn.init.constant_(head[-1].bias, 0)
        for head in self.counterfactual_local_proposal_heads.values():
            nn.init.constant_(head[-1].weight, 0)
            nn.init.constant_(head[-1].bias, 0)
        nn.init.constant_(self.route_token_cond_proj[-1].weight, 0)
        nn.init.constant_(self.route_token_cond_proj[-1].bias, 0)
        nn.init.constant_(self.route_token_gate_logit, -4.0)
        nn.init.constant_(self.residual_head.weight, 0)
        nn.init.constant_(self.residual_head.bias, 0)
        if self.mode_control is not None:
            nn.init.constant_(self.mode_control[-1].weight, 0)
            nn.init.constant_(self.mode_control[-1].bias, 0)
        if self.mode_role_score is not None:
            nn.init.constant_(self.mode_role_score[-1].weight, 0)
            nn.init.constant_(self.mode_role_score[-1].bias, 0)
        if self.mode_skeleton is not None:
            nn.init.constant_(self.mode_skeleton[-1].weight, 0)
            nn.init.constant_(self.mode_skeleton[-1].bias, 0)
        if self.mode_role_budget_head is not None:
            nn.init.constant_(self.mode_role_budget_head[-1].weight, 0)
            nn.init.constant_(self.mode_role_budget_head[-1].bias, 0)
        if self.mode_role_residual_adapter is not None:
            nn.init.constant_(self.mode_role_residual_adapter[-1].weight, 0)
            nn.init.constant_(self.mode_role_residual_adapter[-1].bias, 0)
        if self.mode_specific_residual_heads is not None:
            for head in self.mode_specific_residual_heads:
                nn.init.constant_(head[-1].weight, 0)
                nn.init.constant_(head[-1].bias, 0)
        if self.direction_control_head is not None:
            nn.init.constant_(self.direction_control_head[-1].weight, 0)
            nn.init.constant_(self.direction_control_head[-1].bias, 0)
        if self.curvature_control_head is not None:
            nn.init.constant_(self.curvature_control_head[-1].weight, 0)
            nn.init.constant_(self.curvature_control_head[-1].bias, 0)
        if self.reference_control_head is not None:
            nn.init.constant_(self.reference_control_head[-1].weight, 0)
            nn.init.constant_(self.reference_control_head[-1].bias, 0)
        nn.init.constant_(self.reference_local_budget_gate_head[-1].weight, 0)
        nn.init.constant_(self.reference_local_budget_gate_head[-1].bias, 0)
        if self.route_skeleton_head is not None and self.route_control_mode_head is None:
            nn.init.constant_(self.route_skeleton_head[-1].weight, 0)
            nn.init.constant_(self.route_skeleton_head[-1].bias, 0)
        if self.route_wave_head is not None:
            nn.init.constant_(self.route_wave_head[-1].weight, 0)
            nn.init.constant_(self.route_wave_head[-1].bias, 0)
        if self.route_transition_mode_head is not None:
            nn.init.constant_(self.route_transition_mode_head[-1].weight, 0)
            nn.init.constant_(self.route_transition_mode_head[-1].bias, 0)
        if self.route_proposal_head is not None:
            nn.init.constant_(self.route_proposal_head[-1].weight, 0)
            nn.init.constant_(self.route_proposal_head[-1].bias, 0)
        if self.route_control_mode_head is not None:
            nn.init.constant_(self.route_control_mode_head[-1].weight, 0)
            nn.init.constant_(self.route_control_mode_head[-1].bias, 0)
        if self.route_control_selection_adapter is not None:
            nn.init.constant_(self.route_control_selection_adapter[-1].weight, 0)
            nn.init.constant_(self.route_control_selection_adapter[-1].bias, 0)
        if self.route_control_adapter_gate_logit is not None:
            nn.init.constant_(self.route_control_adapter_gate_logit, -6.0)
        if self.route_prototype_head is not None:
            nn.init.constant_(self.route_prototype_head[-1].weight, 0)
            nn.init.constant_(self.route_prototype_head[-1].bias, 0)
        if isinstance(self.reranker, SplitGatedReranker):
            nn.init.constant_(self.reranker.token_gate[-1].weight, 0)
            nn.init.constant_(self.reranker.token_gate[-1].bias, -1.0)
        for module in self.reranker.modules():
            if isinstance(module, ListwiseTransformerReranker):
                nn.init.normal_(module.mode_embed, std=0.02)
        if isinstance(self.reranker, (ResidualListwiseReranker, TrajectoryResidualListwiseReranker)):
            nn.init.constant_(self.reranker.listwise_delta.score_head[-1].weight, 0)
            nn.init.constant_(self.reranker.listwise_delta.score_head[-1].bias, 0)
        nn.init.normal_(self.counterfactual_scorer_delta.field_embed.weight, std=0.02)
        nn.init.constant_(self.counterfactual_scorer_delta.pair_feature_proj[-1].weight, 0)
        nn.init.constant_(self.counterfactual_scorer_delta.pair_feature_proj[-1].bias, 0)
        nn.init.constant_(self.counterfactual_scorer_delta.delta_head[-1].weight, 0)
        nn.init.constant_(self.counterfactual_scorer_delta.delta_head[-1].bias, 0)
        nn.init.normal_(self.hybrid_policy_delta.field_embed.weight, std=0.02)
        nn.init.normal_(self.hybrid_policy_delta.mode_embed.weight, std=0.02)
        nn.init.constant_(self.hybrid_policy_delta.delta_head[-1].weight, 0)
        nn.init.constant_(self.hybrid_policy_delta.delta_head[-1].bias, 0)
        nn.init.normal_(self.counterfactual_policy_head.field_embed.weight, std=0.02)
        nn.init.normal_(self.counterfactual_policy_head.mode_embed.weight, std=0.02)
        nn.init.constant_(self.counterfactual_policy_head.pair_feature_proj[-1].weight, 0)
        nn.init.constant_(self.counterfactual_policy_head.pair_feature_proj[-1].bias, 0)
        nn.init.constant_(self.counterfactual_policy_head.policy_head[-1].weight, 0)
        nn.init.constant_(self.counterfactual_policy_head.policy_head[-1].bias, 0)
        nn.init.constant_(self.counterfactual_policy_head.gate_logit, -10.0)
        nn.init.constant_(self.counterfactual_metric_policy_head.score.weight, 0)
        nn.init.constant_(self.counterfactual_metric_policy_head.score.bias, 0)
        with torch.no_grad():
            # logit ~= 1.2 * target_effect - collateral - 0.25 * raw_collateral
            self.counterfactual_metric_policy_head.score.weight[0, 0] = 1.2
            self.counterfactual_metric_policy_head.score.weight[0, 2] = -1.0
            self.counterfactual_metric_policy_head.score.weight[0, 9] = -0.25

    def apply_counterfactual_field_adapter(
        self,
        values: torch.Tensor,
        field_ids: Optional[torch.Tensor],
        adapters: nn.ModuleList,
    ) -> torch.Tensor:
        if field_ids is None:
            return torch.zeros_like(values)
        field_ids = field_ids.to(device=values.device, dtype=torch.long)
        out = torch.zeros_like(values)
        for field_id, adapter in enumerate(adapters):
            mask = field_ids == int(field_id)
            if not bool(mask.any()):
                continue
            if values.ndim == 2:
                out[mask] = adapter(values[mask])
            elif values.ndim == 3:
                out[mask] = adapter(values[mask])
            else:
                raise ValueError(f"Unsupported counterfactual adapter values shape: {tuple(values.shape)}")
        return out

    @staticmethod
    def _build_sincos_pos_embed(length: int, dim: int) -> torch.Tensor:
        half = dim // 2
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(half, dtype=torch.float32) * (-math.log(10000.0) / max(half, 1)))
        emb = torch.cat([torch.sin(position * div_term), torch.cos(position * div_term)], dim=1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros(length, 1)], dim=1)
        return emb

    @staticmethod
    def _build_default_mode_role_table(num_candidates: int) -> torch.Tensor:
        # Columns: close_ref, smooth, loose_ref, winding, fast_progress, slow_progress.
        base = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        if num_candidates <= base.shape[0]:
            return base[:num_candidates].clone()
        repeats = math.ceil(num_candidates / base.shape[0])
        return base.repeat(repeats, 1)[:num_candidates].clone()

    @staticmethod
    def _build_default_mode_role_budget_table(num_candidates: int) -> torch.Tensor:
        # Columns: endpoint_scale, mid_scale, score_scale.
        base = torch.tensor(
            [
                [0.32, 0.52, 1.00],
                [0.42, 0.58, 0.95],
                [0.46, 0.60, 0.88],
                [0.24, 0.44, 0.72],
                [0.56, 0.48, 1.02],
                [0.30, 0.38, 0.94],
            ],
            dtype=torch.float32,
        )
        if num_candidates <= base.shape[0]:
            return base[:num_candidates].clone()
        repeats = math.ceil(num_candidates / base.shape[0])
        return base.repeat(repeats, 1)[:num_candidates].clone()

    @staticmethod
    def _build_default_mode_role_control_table(num_candidates: int) -> torch.Tensor:
        # Columns: tangent_scale_bias, normal_scale_bias, gate_bias.
        # The priors make candidate modes geometrically distinguishable before
        # scorer polish: close/smooth, medium, loose, winding, fast, slow.
        base = torch.tensor(
            [
                [-0.10, -0.40, -0.20],
                [0.00, -0.10, 0.00],
                [0.00, 0.35, 0.08],
                [0.00, 0.60, 0.18],
                [0.35, 0.00, 0.10],
                [-0.28, 0.00, -0.06],
            ],
            dtype=torch.float32,
        )
        if num_candidates <= base.shape[0]:
            return base[:num_candidates].clone()
        repeats = math.ceil(num_candidates / base.shape[0])
        return base.repeat(repeats, 1)[:num_candidates].clone()

    @staticmethod
    def _build_default_direction_mode_role_table(num_candidates: int) -> torch.Tensor:
        # Columns follow the direction vocabulary order:
        # straight, left_turn, right_turn, u_turn, multi_turn.
        #
        # MTR-style candidate modes are useful only if the pool contains
        # semantically distinct hypotheses.  Under OD-chord conditioning, the
        # most important missing hypotheses are directional, so these priors
        # reserve early modes for straight/left/right/u-turn/multi-turn roles
        # without using any GT middle trajectory as input.
        if int(num_candidates) <= 1:
            return torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        if int(num_candidates) == 2:
            base = torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.5, 0.25, 0.25],
                ],
                dtype=torch.float32,
            )
            return base
        if int(num_candidates) == 3:
            base = torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.15, 0.15],
                    [0.0, 0.0, 1.0, 0.15, 0.15],
                ],
                dtype=torch.float32,
            )
            return base
        if int(num_candidates) == 4:
            base = torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.15],
                    [0.0, 0.0, 1.0, 0.0, 0.15],
                    [0.0, 0.15, 0.15, 0.75, 1.00],
                ],
                dtype=torch.float32,
            )
            return base
        base = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.10],
                [0.0, 0.0, 1.0, 0.0, 0.10],
                [0.0, 0.10, 0.10, 1.0, 0.20],
                [0.0, 0.20, 0.20, 0.25, 1.0],
                [0.35, 0.35, 0.35, 0.0, 0.35],
            ],
            dtype=torch.float32,
        )
        if num_candidates <= base.shape[0]:
            return base[:num_candidates].clone()
        repeats = math.ceil(num_candidates / base.shape[0])
        return base.repeat(repeats, 1)[:num_candidates].clone()

    @staticmethod
    def _build_default_route_transition_mode_table(num_candidates: int) -> torch.Tensor:
        # Columns: direct, left_bias, right_bias, u_turn, multi_turn, detour.
        #
        # These are route-transition roles, not instruction labels. They make
        # the K-mode pool cover different OD-chord transition hypotheses before
        # the scorer sees them, matching the MTR principle that each query/mode
        # should own a stable motion role.
        if int(num_candidates) <= 1:
            return torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        base = torch.tensor(
            [
                [1.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # direct OD route
                [0.15, 1.00, 0.00, 0.00, 0.10, 0.15],  # left-biased route
                [0.15, 0.00, 1.00, 0.00, 0.10, 0.15],  # right-biased route
                [0.05, 0.20, 0.20, 1.00, 0.20, 0.25],  # U-turn / hook route
                [0.05, 0.20, 0.20, 0.20, 1.00, 0.35],  # multi-turn route
                [0.00, 0.30, 0.30, 0.00, 0.35, 1.00],  # loose detour route
            ],
            dtype=torch.float32,
        )
        if num_candidates <= base.shape[0]:
            return base[:num_candidates].clone()
        repeats = math.ceil(num_candidates / base.shape[0])
        return base.repeat(repeats, 1)[:num_candidates].clone()

    def load_mode_role_config(self, path: str | Path) -> None:
        """Load decoder-level mode-role priors from a JSON artifact.

        The JSON may provide any subset of:
        - mode_role_table: [K, 6]
        - mode_role_budget_table: [K, 3]
        - mode_role_control_table: [K, 3]

        This is intentionally a buffer update, not a learned parameter load, so
        a diagnostic role policy can be reapplied after checkpoint loading.
        """
        config_path = Path(path)
        payload = json.loads(config_path.read_text(encoding="utf-8"))

        def _copy_table(key: str, buffer_name: str, expected_dim: int) -> None:
            if key not in payload:
                return
            value = torch.as_tensor(payload[key], dtype=getattr(self, buffer_name).dtype)
            expected = (self.num_candidates, expected_dim)
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"{key} must have shape {expected}, got {tuple(value.shape)} from {config_path}"
                )
            getattr(self, buffer_name).data.copy_(value.to(device=getattr(self, buffer_name).device))

        _copy_table("mode_role_table", "mode_role_table", self.mode_role_dim)
        _copy_table("mode_role_budget_table", "mode_role_budget_table", 3)
        _copy_table("mode_role_control_table", "mode_role_control_table", 3)
        _copy_table("direction_mode_role_table", "direction_mode_role_table", 5)
        _copy_table("route_transition_mode_table", "route_transition_mode_table", 6)
        self.mode_role_config_path = str(config_path)

    def load_route_prototype_bank(self, path: str | Path) -> None:
        """Load non-leaking OD-local route prototypes.

        The prototype bank is built from training targets as distribution-level
        route shapes in the current OD chord's tangent/normal frame. It is not a
        per-sample reference and does not contain the evaluated sample's middle
        trajectory as a condition.
        """
        bank_path = Path(path)
        payload = json.loads(bank_path.read_text(encoding="utf-8"))
        prototypes = torch.as_tensor(payload.get("prototypes", []), dtype=self.route_prototype_bank.dtype)
        if prototypes.ndim != 3 or prototypes.shape[-1] != 2:
            raise ValueError(
                f"Route prototype bank must have shape [K, P, 2], got {tuple(prototypes.shape)} from {bank_path}"
            )
        if prototypes.shape[0] < 1 or prototypes.shape[1] < 2:
            raise ValueError(f"Route prototype bank is empty or too short: {tuple(prototypes.shape)}")
        if prototypes.shape[1] != self.route_skeleton_points:
            prototypes = self.resample_temporal(prototypes, self.route_skeleton_points)
        if prototypes.shape[0] < self.num_candidates:
            repeats = math.ceil(self.num_candidates / prototypes.shape[0])
            prototypes = prototypes.repeat(repeats, 1, 1)
        prototypes = prototypes[: self.num_candidates]
        self.route_prototype_bank.data.copy_(
            prototypes.to(device=self.route_prototype_bank.device, dtype=self.route_prototype_bank.dtype)
        )
        self.route_prototype_bank_path = str(bank_path)

    @staticmethod
    def parse_mode_role_prior_field_weights(raw: str | dict | list | tuple | None) -> dict[str, float]:
        fields = list(DEFAULT_MOBILITY_VOCAB.fields)
        weights = {field: 1.0 for field in fields}
        if raw is None:
            return weights
        if isinstance(raw, dict):
            for key, value in raw.items():
                if str(key) in weights:
                    weights[str(key)] = float(value)
            return weights
        if isinstance(raw, (list, tuple)):
            for field, value in zip(fields, raw):
                weights[field] = float(value)
            return weights
        text = str(raw).strip()
        if text == "":
            return weights
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if parts and all("=" not in part for part in parts):
            values = [float(part) for part in parts]
            for field, value in zip(fields, values):
                weights[field] = float(value)
            return weights
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            if key in weights:
                weights[key] = float(value.strip())
        return weights

    @staticmethod
    def parse_language_control_field_ratios(raw: str | dict | list | tuple | None) -> dict[str, float]:
        fields = list(DEFAULT_MOBILITY_VOCAB.fields)
        ratios = {
            "default": 0.70,
            "distance": 0.50,
            "direction": 0.90,
            "curvature": 0.60,
            "progress": 0.50,
            "reference_adherence": 0.75,
            "counterfactual": 0.70,
        }
        if raw is None:
            return ratios
        if isinstance(raw, dict):
            for key, value in raw.items():
                key_text = str(key).strip()
                if key_text in ratios or key_text in fields:
                    ratios[key_text] = float(value)
            return ratios
        if isinstance(raw, (list, tuple)):
            for field, value in zip(fields, raw):
                ratios[field] = float(value)
            return ratios
        text = str(raw).strip()
        if text == "":
            return ratios
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if parts and all("=" not in part for part in parts):
            values = [float(part) for part in parts]
            for field, value in zip(fields, values):
                ratios[field] = float(value)
            return ratios
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            if key in ratios or key in fields:
                ratios[key] = float(value.strip())
        return ratios

    @staticmethod
    def parse_language_control_gate_fields(raw: str | list | tuple | set | None) -> set[str] | None:
        valid = set(DEFAULT_MOBILITY_VOCAB.fields) | {"counterfactual"}
        if raw is None:
            return None
        if isinstance(raw, str):
            text = raw.strip()
            if text == "" or text.lower() in {"all", "*"}:
                return None
            fields = {part.strip() for part in text.split(",") if part.strip()}
        else:
            fields = {str(part).strip() for part in raw if str(part).strip()}
        unknown = fields - valid
        if unknown:
            raise ValueError(f"Unsupported language_control_gate_fields: {sorted(unknown)}")
        return fields

    @staticmethod
    def parse_route_frame_budget_fields(raw: str | list | tuple | set | None) -> set[str] | None:
        valid = set(DEFAULT_MOBILITY_VOCAB.fields)
        if raw is None:
            return None
        if isinstance(raw, str):
            text = raw.strip()
            if text == "" or text.lower() in {"all", "*"}:
                return None
            fields = {part.strip() for part in text.split(",") if part.strip()}
        else:
            fields = {str(part).strip() for part in raw if str(part).strip()}
        unknown = fields - valid
        if unknown:
            raise ValueError(f"Unsupported counterfactual_route_frame_fields: {sorted(unknown)}")
        return fields

    @staticmethod
    def mode_role_prior_scores(
        mobility_tokens: torch.Tensor,
        num_candidates: int,
        field_weights: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """Instruction-token prior over candidate roles for hard mode selection.

        This mirrors the K=6 role design used by the training losses. It is kept
        inside the decoder so inference-time candidate logits can use the same
        role semantics that shaped the generator.
        """
        if mobility_tokens.ndim != 2:
            raise ValueError(f"mobility_tokens must be [B, F], got {tuple(mobility_tokens.shape)}")
        fields = DEFAULT_MOBILITY_VOCAB.fields
        field_idx = {field: idx for idx, field in enumerate(fields)}
        ids = DEFAULT_MOBILITY_VOCAB.token_to_id
        field_weights = field_weights or {field: 1.0 for field in fields}
        distance_weight = float(field_weights.get("distance", 1.0))
        direction_weight = float(field_weights.get("direction", 1.0))
        curvature_weight = float(field_weights.get("curvature", 1.0))
        progress_weight = float(field_weights.get("progress", 1.0))
        reference_weight = float(field_weights.get("reference_adherence", 1.0))
        batch = mobility_tokens.shape[0]
        scores = mobility_tokens.new_zeros(batch, max(int(num_candidates), 6), dtype=torch.float32)
        scores[:, 1] += 0.20

        distance = mobility_tokens[:, field_idx["distance"]]
        direction = mobility_tokens[:, field_idx["direction"]]
        curvature = mobility_tokens[:, field_idx["curvature"]]
        progress = mobility_tokens[:, field_idx["progress"]]
        reference = mobility_tokens[:, field_idx["reference_adherence"]]

        scores[:, 5] += distance_weight * 0.40 * (
            (distance == ids["distance"]["very_short"]) | (distance == ids["distance"]["short"])
        ).float()
        scores[:, 1] += distance_weight * 0.25 * (distance == ids["distance"]["medium"]).float()
        scores[:, 4] += distance_weight * 0.35 * (distance == ids["distance"]["long"]).float()

        direction_role_weight = direction_weight * 2.0
        scores[:, 0] += direction_role_weight * (direction == ids["direction"]["straight"]).float()
        scores[:, 1] += direction_role_weight * (direction == ids["direction"]["left_turn"]).float()
        scores[:, 2] += direction_role_weight * (direction == ids["direction"]["right_turn"]).float()
        scores[:, 3] += direction_role_weight * (direction == ids["direction"]["u_turn"]).float()
        multi_mode_idx = 4 if int(num_candidates) >= 5 else 3
        scores[:, multi_mode_idx] += direction_role_weight * (direction == ids["direction"]["multi_turn"]).float()

        scores[:, 0] += curvature_weight * 0.60 * (curvature == ids["curvature"]["smooth"]).float()
        scores[:, 1] += curvature_weight * 0.45 * (curvature == ids["curvature"]["moderate"]).float()
        scores[:, 3] += curvature_weight * 0.85 * (curvature == ids["curvature"]["winding"]).float()

        scores[:, 5] += progress_weight * 0.70 * (progress == ids["progress"]["slow"]).float()
        scores[:, 1] += progress_weight * 0.35 * (progress == ids["progress"]["normal"]).float()
        scores[:, 4] += progress_weight * 0.80 * (progress == ids["progress"]["fast"]).float()

        scores[:, 2] += reference_weight * 0.65 * (
            (reference == ids["reference_adherence"]["unknown"])
            | (reference == ids["reference_adherence"]["loose"])
        ).float()
        scores[:, 0] += reference_weight * 0.65 * (reference == ids["reference_adherence"]["tight"]).float()
        scores[:, 1] += reference_weight * 0.45 * (reference == ids["reference_adherence"]["moderate"]).float()
        return scores[:, : int(num_candidates)]

    @staticmethod
    def token_wave_route_controls(
        mobility_tokens: Optional[torch.Tensor],
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Map mobility tokens to non-leaking route-wave controls.

        These controls deliberately use only the instruction tokens. They give
        the OD-chord decoder enough route-length and curvature capacity without
        rebuilding any road/reference geometry from the GT middle trajectory.
        """
        shape = (int(batch), 1, 1, 1)
        controls = {
            "distance": torch.ones(shape, device=device, dtype=dtype),
            "curvature": torch.ones(shape, device=device, dtype=dtype),
            "reference": torch.ones(shape, device=device, dtype=dtype),
            "progress": torch.ones(shape, device=device, dtype=dtype),
            "progress_skew": torch.zeros(shape, device=device, dtype=dtype),
            "direction_known": torch.zeros(shape, device=device, dtype=dtype),
            "direction_straight": torch.zeros(shape, device=device, dtype=dtype),
            "direction_left": torch.zeros(shape, device=device, dtype=dtype),
            "direction_right": torch.zeros(shape, device=device, dtype=dtype),
            "direction_uturn": torch.zeros(shape, device=device, dtype=dtype),
            "direction_multi": torch.zeros(shape, device=device, dtype=dtype),
            "direction_sign": torch.zeros(shape, device=device, dtype=dtype),
            "direction_cycles": torch.ones(shape, device=device, dtype=dtype),
            "uturn": torch.zeros(shape, device=device, dtype=dtype),
        }
        if mobility_tokens is None or mobility_tokens.ndim != 2 or mobility_tokens.shape[0] != int(batch):
            return controls

        ids = DEFAULT_MOBILITY_VOCAB.token_to_id
        field_idx = {field: idx for idx, field in enumerate(DEFAULT_MOBILITY_VOCAB.fields)}
        tokens = mobility_tokens.to(device=device)
        distance = tokens[:, field_idx["distance"]]
        direction = tokens[:, field_idx["direction"]]
        curvature = tokens[:, field_idx["curvature"]]
        progress = tokens[:, field_idx["progress"]]
        reference = tokens[:, field_idx["reference_adherence"]]

        def _values(field: str, source: torch.Tensor, mapping: dict[str, float]) -> torch.Tensor:
            value = torch.zeros(int(batch), device=device, dtype=dtype)
            seen = torch.zeros(int(batch), device=device, dtype=torch.bool)
            for token_name, token_value in mapping.items():
                if token_name not in ids[field]:
                    continue
                mask = source == ids[field][token_name]
                value = torch.where(mask, torch.full_like(value, float(token_value)), value)
                seen = seen | mask
            if not seen.all():
                fallback = next(iter(mapping.values()))
                value = torch.where(seen, value, torch.full_like(value, float(fallback)))
            return value.view(shape)

        controls["distance"] = _values(
            "distance",
            distance,
            {"very_short": 0.65, "short": 0.88, "medium": 1.22, "long": 1.80},
        )
        controls["curvature"] = _values(
            "curvature",
            curvature,
            {"smooth": 0.55, "moderate": 1.00, "winding": 1.65},
        )
        controls["reference"] = _values(
            "reference_adherence",
            reference,
            {"unknown": 1.15, "tight": 0.48, "moderate": 0.95, "loose": 1.55},
        )
        controls["progress"] = _values(
            "progress",
            progress,
            {"slow": 0.82, "normal": 1.00, "fast": 1.24},
        )
        controls["progress_skew"] = _values(
            "progress",
            progress,
            {"slow": -0.28, "normal": 0.00, "fast": 0.28},
        )
        controls["direction_sign"] = _values(
            "direction",
            direction,
            {"straight": 0.00, "left_turn": 1.00, "right_turn": -1.00, "u_turn": 0.00, "multi_turn": 0.35},
        )
        controls["direction_cycles"] = _values(
            "direction",
            direction,
            {"straight": 0.45, "left_turn": 0.85, "right_turn": 0.85, "u_turn": 1.75, "multi_turn": 1.90},
        )
        controls["uturn"] = _values(
            "direction",
            direction,
            {"straight": 0.00, "left_turn": 0.00, "right_turn": 0.00, "u_turn": 1.00, "multi_turn": 0.15},
        )
        direction_ids = ids["direction"]
        known = torch.zeros(int(batch), device=device, dtype=dtype)
        for token_name in ("straight", "left_turn", "right_turn", "u_turn", "multi_turn"):
            if token_name in direction_ids:
                known = torch.where(direction == direction_ids[token_name], torch.ones_like(known), known)
        controls["direction_known"] = known.view(shape)
        controls["direction_straight"] = (direction == direction_ids["straight"]).to(dtype=dtype).view(shape)
        controls["direction_left"] = (direction == direction_ids["left_turn"]).to(dtype=dtype).view(shape)
        controls["direction_right"] = (direction == direction_ids["right_turn"]).to(dtype=dtype).view(shape)
        controls["direction_uturn"] = (direction == direction_ids["u_turn"]).to(dtype=dtype).view(shape)
        controls["direction_multi"] = (direction == direction_ids["multi_turn"]).to(dtype=dtype).view(shape)
        return controls

    @staticmethod
    def resample_reference(reference: torch.Tensor, target_len: int) -> torch.Tensor:
        return AnchorResidualDecoder.resample_reference(reference, target_len)

    @staticmethod
    def tangent_normal(anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return AnchorResidualDecoder.tangent_normal(anchor)

    @staticmethod
    def normalized_time(batch: int, target_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return AnchorResidualDecoder.normalized_time(batch, target_len, device, dtype)

    @staticmethod
    def endpoint_envelope(target_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, target_len, device=device, dtype=dtype)
        envelope = torch.sin(torch.pi * t).clamp_min(0.0).pow(1.5).view(1, target_len, 1)
        return envelope

    @staticmethod
    def preserve_reference_endpoints(residual: torch.Tensor) -> torch.Tensor:
        """Project residuals so generated trajectories keep the reference OD endpoints."""
        if residual.shape[-2] <= 1:
            return residual - residual[..., :1, :]
        t = torch.linspace(0.0, 1.0, residual.shape[-2], device=residual.device, dtype=residual.dtype)
        t = t.view(*([1] * (residual.ndim - 2)), residual.shape[-2], 1)
        endpoint_bridge = (1.0 - t) * residual[..., :1, :] + t * residual[..., -1:, :]
        return residual - endpoint_bridge

    @staticmethod
    def endpoint_uturn_template_delta(anchor: torch.Tensor, *, side: torch.Tensor | float = 1.0) -> torch.Tensor:
        """Endpoint-preserving U-turn prototype aligned with the dataset template."""
        if anchor.ndim != 3 or anchor.shape[-1] != 2:
            raise ValueError(f"anchor must be [B, T, 2], got {tuple(anchor.shape)}")
        batch, target_len, _ = anchor.shape
        device = anchor.device
        dtype = anchor.dtype
        start = anchor[:, :1, :]
        end = anchor[:, -1:, :]
        chord = end - start
        chord_len = torch.linalg.norm(chord, dim=-1, keepdim=True).clamp_min(1e-6)
        ex = chord / chord_len
        ey = torch.stack([-ex[..., 1], ex[..., 0]], dim=-1)
        if not torch.is_tensor(side):
            side = anchor.new_full((batch, 1, 1), float(side))
        side = side.to(device=device, dtype=dtype).view(batch, 1, 1)
        key_x = anchor.new_tensor([0.0, 0.28, 0.48, 0.18, 1.08, 1.0])
        key_y = anchor.new_tensor([0.0, 0.0, 0.34, 0.34, 0.0, 0.0])
        key_t = anchor.new_tensor([0.0, 0.22, 0.40, 0.60, 0.90, 1.0])
        t = torch.linspace(0.0, 1.0, target_len, device=device, dtype=dtype)
        x = torch.empty(target_len, device=device, dtype=dtype)
        y = torch.empty(target_len, device=device, dtype=dtype)
        for idx in range(int(key_t.numel()) - 1):
            lo = key_t[idx]
            hi = key_t[idx + 1]
            mask = (t >= lo) & (t <= hi if idx == int(key_t.numel()) - 2 else t < hi)
            alpha = ((t[mask] - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)
            x[mask] = key_x[idx] + alpha * (key_x[idx + 1] - key_x[idx])
            y[mask] = key_y[idx] + alpha * (key_y[idx + 1] - key_y[idx])
        template = start + chord_len * (x.view(1, target_len, 1) * ex + y.view(1, target_len, 1) * side * ey)
        return DiTAnchorDecoder.preserve_reference_endpoints(template - anchor)

    @staticmethod
    def endpoint_multiturn_template_delta(anchor: torch.Tensor, *, side: torch.Tensor | float = 1.0) -> torch.Tensor:
        """Endpoint-preserving multi-turn wave prototype aligned with Table-3 direction labels."""
        if anchor.ndim != 3 or anchor.shape[-1] != 2:
            raise ValueError(f"anchor must be [B, T, 2], got {tuple(anchor.shape)}")
        batch, target_len, _ = anchor.shape
        device = anchor.device
        dtype = anchor.dtype
        start = anchor[:, :1, :]
        end = anchor[:, -1:, :]
        chord = end - start
        chord_len = torch.linalg.norm(chord, dim=-1, keepdim=True).clamp_min(1e-6)
        ex = chord / chord_len
        ey = torch.stack([-ex[..., 1], ex[..., 0]], dim=-1)
        if not torch.is_tensor(side):
            side = anchor.new_full((batch, 1, 1), float(side))
        side = side.to(device=device, dtype=dtype).view(batch, 1, 1)
        t = torch.linspace(0.0, 1.0, target_len, device=device, dtype=dtype).view(1, target_len, 1)
        x = t
        envelope = torch.sin(math.pi * t).clamp_min(0.0)
        y = 0.48 * envelope * torch.sin(4.0 * math.pi * t)
        template = start + chord_len * (x * ex + y * side * ey)
        return DiTAnchorDecoder.preserve_reference_endpoints(template - anchor)

    @staticmethod
    def counterfactual_token_name(field: str, token_id: int) -> str:
        mapping = DEFAULT_MOBILITY_VOCAB.id_to_token.get(field, {})
        return str(mapping.get(int(token_id), str(int(token_id))))

    @staticmethod
    def resample_temporal(sequence: torch.Tensor, target_len: int) -> torch.Tensor:
        """Linearly resample a temporal sequence that ends with [T, C]."""
        if sequence.shape[-2] == int(target_len):
            return sequence
        original_shape = sequence.shape
        flat = sequence.reshape(-1, original_shape[-2], original_shape[-1])
        resampled = F.interpolate(
            flat.transpose(1, 2),
            size=int(target_len),
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        return resampled.reshape(*original_shape[:-2], int(target_len), original_shape[-1])

    def cond_token_pos_embed(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if length <= self.token_pos_embed.shape[1]:
            return self.token_pos_embed[:, :length].to(device=device, dtype=dtype)
        return self._build_sincos_pos_embed(length, self.hidden_dim).to(device=device, dtype=dtype).unsqueeze(0)

    def set_mode_control_strength(self, value: float) -> None:
        self.mode_control_strength = float(value)

    def set_mode_role_budget_strength(self, value: float) -> None:
        self.mode_role_budget_strength = float(value)

    def set_direction_control_strength(self, value: float) -> None:
        self.direction_control_strength = float(value)

    def set_curvature_control_strength(self, value: float) -> None:
        self.curvature_control_strength = float(value)

    def set_reference_control_strength(self, value: float) -> None:
        self.reference_control_strength = float(value)

    def set_reference_positive_candidate_strength(self, value: float) -> None:
        self.reference_positive_candidate_strength = float(value)

    def set_reference_positive_candidate_scope(self, value: str) -> None:
        value = str(value)
        allowed = {"all", "none", "counterfactual_reference_adherence"}
        if value not in allowed:
            raise ValueError(
                f"Unsupported reference_positive_candidate_scope={value}. "
                f"Expected one of {sorted(allowed)}."
            )
        self.reference_positive_candidate_scope = value

    def set_reference_distance_candidate_strength(self, value: float) -> None:
        self.reference_distance_candidate_strength = float(value)

    def set_reference_distance_candidate_scope(self, value: str) -> None:
        value = str(value)
        allowed = {"all", "none", "instruction", "counterfactual_reference_adherence"}
        if value not in allowed:
            raise ValueError(
                f"Unsupported reference_distance_candidate_scope={value}. "
                f"Expected one of {sorted(allowed)}."
            )
        self.reference_distance_candidate_scope = value

    def set_reference_distance_candidate_mode(self, value: str) -> None:
        value = str(value)
        allowed = {"global_normal", "local_direction_preserving"}
        if value not in allowed:
            raise ValueError(
                f"Unsupported reference_distance_candidate_mode={value}. "
                f"Expected one of {sorted(allowed)}."
            )
        self.reference_distance_candidate_mode = value

    def set_counterfactual_direction_residual_type(self, value: str) -> None:
        value = str(value)
        allowed = {
            "shared",
            "transition_contract_pulse",
            "transition_arc_skeleton",
            "transition_arc_multiturn_burst",
            "transition_hybrid_skeleton_reflect",
            "transition_pivot_skeleton",
            "transition_reflect_skeleton",
            "transition_skeleton_reversal",
            "transition_skeleton_reversal_scurve",
            "transition_skeleton",
            "transition_template",
            "transition_template_balanced",
            "transition_template_contract",
            "transition_template_mean_projected",
            "transition_template_projected",
        }
        if value not in allowed:
            raise ValueError(
                f"Unsupported counterfactual_direction_residual_type={value}. "
                f"Expected one of {sorted(allowed)}."
            )
        self.counterfactual_direction_residual_type = value

    @staticmethod
    def _project_curve_out_bases(
        curve: torch.Tensor,
        bases: list[torch.Tensor],
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not bases:
            return curve
        mask = active_mask.to(device=curve.device, dtype=curve.dtype).clamp(0.0, 1.0)
        if float(mask.detach().max().item()) <= 0.0:
            return curve
        projected = curve
        for basis in bases:
            basis_view = basis.to(device=curve.device, dtype=curve.dtype)
            numer = (projected * basis_view * mask).sum(dim=-2, keepdim=True)
            denom = (basis_view * basis_view * mask).sum(dim=-2, keepdim=True).clamp_min(1e-6)
            projected = projected - (numer / denom) * basis_view
        return curve + mask * (projected - curve)

    @staticmethod
    def _normalize_masked_curve_rms(
        curve: torch.Tensor,
        active_mask: torch.Tensor,
        target_rms: float = 1.0,
    ) -> torch.Tensor:
        mask = active_mask.to(device=curve.device, dtype=curve.dtype).clamp(0.0, 1.0)
        if float(mask.detach().max().item()) <= 0.0:
            return curve
        denom = mask.sum(dim=-2, keepdim=True).clamp_min(1e-6)
        rms = torch.sqrt((curve.pow(2) * mask).sum(dim=-2, keepdim=True) / denom).clamp_min(1e-6)
        return curve * (float(target_rms) / rms)

    @staticmethod
    def _nearest_reference_attraction_vector(
        points: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Vector from each point to its nearest projection on the reference polyline."""
        if points.ndim < 3 or points.shape[-1] != 2:
            raise ValueError(f"Expected [..., T, 2] points, got {tuple(points.shape)}")
        if reference.ndim == 4:
            reference = reference[:, 0]
        if reference.ndim != 3 or reference.shape[-1] != 2:
            raise ValueError(f"Expected [B, R, 2] reference, got {tuple(reference.shape)}")
        batch = int(points.shape[0])
        if reference.shape[0] != batch:
            raise ValueError(
                "points and reference must share batch dimension, got "
                f"{points.shape[0]} vs {reference.shape[0]}"
            )
        if reference.shape[-2] <= 1:
            ref_point = reference[:, :1].to(device=points.device, dtype=points.dtype)
            while ref_point.ndim < points.ndim:
                ref_point = ref_point.unsqueeze(1)
            return ref_point.expand_as(points) - points

        original_shape = points.shape
        flat_points = points.reshape(batch, -1, 2)
        ref = reference.to(device=points.device, dtype=points.dtype)
        point_exp = flat_points[:, :, None, :]
        start = ref[:, None, :-1, :]
        end = ref[:, None, 1:, :]
        segment = end - start
        denom = (segment * segment).sum(dim=-1).clamp_min(1e-8)
        proj_t = ((point_exp - start) * segment).sum(dim=-1) / denom
        proj_t = proj_t.clamp(0.0, 1.0)
        projected = start + proj_t.unsqueeze(-1) * segment
        dist2 = (point_exp - projected).pow(2).sum(dim=-1)
        nearest_idx = dist2.argmin(dim=-1)
        gather_idx = nearest_idx.view(batch, -1, 1, 1).expand(-1, -1, 1, 2)
        nearest = projected.gather(dim=2, index=gather_idx).squeeze(2)
        return (nearest - flat_points).view(original_shape)

    @staticmethod
    def _mean_reference_polyline_distance(
        points: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != 2:
            raise ValueError(f"Expected [B, T, 2] points, got {tuple(points.shape)}")
        if reference.ndim == 4:
            reference = reference[:, 0]
        if reference.ndim != 3 or reference.shape[0] != points.shape[0] or reference.shape[-1] != 2:
            raise ValueError(
                "Expected reference [B, R, 2] matching points batch, got "
                f"{tuple(reference.shape)} for points {tuple(points.shape)}"
            )
        if reference.shape[-2] <= 1:
            ref = reference[:, :1].to(device=points.device, dtype=points.dtype)
            return torch.linalg.norm(points - ref, dim=-1).mean(dim=-1)
        points_exp = points[:, :, None, :]
        ref = reference.to(device=points.device, dtype=points.dtype)
        start = ref[:, None, :-1, :]
        end = ref[:, None, 1:, :]
        segment = end - start
        denom = (segment * segment).sum(dim=-1).clamp_min(1e-8)
        proj_t = ((points_exp - start) * segment).sum(dim=-1) / denom
        proj_t = proj_t.clamp(0.0, 1.0)
        projected = start + proj_t.unsqueeze(-1) * segment
        return torch.linalg.norm(points_exp - projected, dim=-1).min(dim=2).values.mean(dim=1)

    @staticmethod
    def _build_signed_balanced_winding_curve(
        time_basis: torch.Tensor,
        local_window: torch.Tensor,
        variant: torch.Tensor | float,
    ) -> torch.Tensor:
        if not torch.is_tensor(variant):
            variant = time_basis.new_tensor(float(variant))
        variant = variant.to(device=time_basis.device, dtype=time_basis.dtype).clamp(0.0, 1.5)
        shoulder_offset = 0.19 - 0.03 * variant
        outer_width = 0.070 + 0.028 * variant
        inner_width = 0.050 + 0.016 * variant
        shoulder_width = 0.042 + 0.012 * variant
        outer_left = torch.exp(-0.5 * ((time_basis - (0.5 - shoulder_offset)) / outer_width).pow(2))
        outer_right = torch.exp(-0.5 * ((time_basis - (0.5 + shoulder_offset)) / outer_width).pow(2))
        inner_center = torch.exp(-0.5 * ((time_basis - 0.5) / inner_width).pow(2))
        shoulder_left = torch.exp(-0.5 * ((time_basis - 0.40) / shoulder_width).pow(2))
        shoulder_right = torch.exp(-0.5 * ((time_basis - 0.60) / shoulder_width).pow(2))
        curve = (
            1.00 * (outer_left + outer_right)
            - (1.48 + 0.10 * variant) * inner_center
            + 0.34 * (shoulder_left + shoulder_right)
        )
        return curve * local_window

    @staticmethod
    def _build_midlocal_winding_pulse_curve(
        time_basis: torch.Tensor,
        local_window: torch.Tensor,
        variant: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(variant):
            variant = time_basis.new_tensor(float(variant))
        variant = variant.to(device=time_basis.device, dtype=time_basis.dtype).clamp(0.0, 1.5)
        core_width = 0.040 + 0.010 * variant
        side_width = 0.055 + 0.016 * variant
        outer_width = 0.070 + 0.020 * variant
        side_offset = 0.095 + 0.020 * variant
        outer_offset = 0.185 + 0.020 * variant
        pulse_window = (
            0.90 * torch.exp(-0.5 * ((time_basis - 0.5) / (0.16 + 0.02 * variant)).pow(2))
            + 0.10 * local_window
        ).clamp(0.0, 1.0) * local_window
        center = torch.exp(-0.5 * ((time_basis - 0.5) / core_width).pow(2))
        side_left = torch.exp(-0.5 * ((time_basis - (0.5 - side_offset)) / side_width).pow(2))
        side_right = torch.exp(-0.5 * ((time_basis - (0.5 + side_offset)) / side_width).pow(2))
        outer_left = torch.exp(-0.5 * ((time_basis - (0.5 - outer_offset)) / outer_width).pow(2))
        outer_right = torch.exp(-0.5 * ((time_basis - (0.5 + outer_offset)) / outer_width).pow(2))
        curve = (
            1.10 * (side_left + side_right)
            - (1.45 + 0.08 * variant) * center
            + 0.22 * (outer_left + outer_right)
        )
        return curve * pulse_window, pulse_window

    @staticmethod
    def _build_base_selected_projected_normal_curve(
        time_basis: torch.Tensor,
        local_window: torch.Tensor,
        base_selected_normal_basis: torch.Tensor,
        variant: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(variant):
            variant = time_basis.new_tensor(float(variant))
        variant = variant.to(device=time_basis.device, dtype=time_basis.dtype).clamp(0.0, 1.5)
        balanced_curve = DiTAnchorDecoder._build_signed_balanced_winding_curve(
            time_basis,
            local_window,
            0.35 + 0.40 * variant,
        )
        pulse_curve, pulse_window = DiTAnchorDecoder._build_midlocal_winding_pulse_curve(
            time_basis,
            local_window,
            0.20 + 0.55 * variant,
        )
        projected_window = torch.maximum(local_window, pulse_window)
        signed_basis = (time_basis - 0.5) * 2.0
        mid_basis = torch.sin(math.pi * time_basis).clamp_min(0.0)
        s_curve_basis = torch.sin(2.0 * math.pi * time_basis)
        end_basis = time_basis.clamp_min(0.0).pow(1.2)
        quad_basis = (signed_basis.pow(2) - 0.35) * mid_basis
        base_basis = base_selected_normal_basis.to(
            device=time_basis.device,
            dtype=time_basis.dtype,
        ).abs() * projected_window
        curve = 0.40 * balanced_curve + 0.90 * pulse_curve
        projection_bases = [
            projected_window,
            signed_basis * projected_window,
            mid_basis * projected_window,
            quad_basis * projected_window,
            s_curve_basis * projected_window,
            end_basis * projected_window,
            base_basis,
            signed_basis * base_basis,
        ]
        for _ in range(2):
            curve = DiTAnchorDecoder._project_curve_out_bases(
                curve,
                projection_bases,
                projected_window,
            )
        curve = DiTAnchorDecoder._normalize_masked_curve_rms(
            curve,
            projected_window,
            target_rms=1.20,
        )
        return curve, projected_window

    @staticmethod
    def _stabilize_signed_balanced_winding_curve(
        curve: torch.Tensor,
        time_basis: torch.Tensor,
        local_window: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        existing_normal_basis: torch.Tensor | None = None,
        target_rms: float = 1.0,
    ) -> torch.Tensor:
        signed_basis = (time_basis - 0.5) * 2.0
        mid_basis = torch.sin(math.pi * time_basis).clamp_min(0.0)
        s_curve_basis = torch.sin(2.0 * math.pi * time_basis)
        end_basis = time_basis.clamp_min(0.0).pow(1.2)
        signed_mid_basis = signed_basis * mid_basis
        quad_basis = (signed_basis.pow(2) - 0.35) * mid_basis
        bases = [
            local_window,
            signed_basis * local_window,
            mid_basis * local_window,
            signed_mid_basis * local_window,
            quad_basis * local_window,
            s_curve_basis * local_window,
            end_basis * local_window,
        ]
        if existing_normal_basis is not None:
            bases.append(existing_normal_basis.to(device=curve.device, dtype=curve.dtype) * local_window)
        mask = active_mask.to(device=curve.device, dtype=curve.dtype) * local_window
        stabilized = curve
        for _ in range(2):
            stabilized = DiTAnchorDecoder._project_curve_out_bases(
                stabilized,
                bases,
                mask,
            )
        return DiTAnchorDecoder._normalize_masked_curve_rms(
            stabilized,
            mask,
            target_rms=target_rms,
        )

    @staticmethod
    def _project_heading_preserving_winding_curve(
        curve: torch.Tensor,
        time_basis: torch.Tensor,
        local_window: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        base_normal_basis: torch.Tensor | None = None,
        target_rms: float = 0.82,
    ) -> torch.Tensor:
        signed_basis = (time_basis - 0.5) * 2.0
        mid_basis = torch.sin(math.pi * time_basis).clamp_min(0.0)
        s_curve_basis = torch.sin(2.0 * math.pi * time_basis)
        shoulder_left = torch.exp(-0.5 * ((time_basis - 0.38) / 0.07).pow(2)) * local_window
        shoulder_right = torch.exp(-0.5 * ((time_basis - 0.62) / 0.07).pow(2)) * local_window
        edge_slope = signed_basis * local_window
        bases = [
            local_window,
            edge_slope,
            mid_basis * local_window,
            s_curve_basis * local_window,
            shoulder_left,
            shoulder_right,
        ]
        if base_normal_basis is not None:
            base_basis = base_normal_basis.to(device=curve.device, dtype=curve.dtype) * local_window
            bases.extend([base_basis, signed_basis * base_basis])
        projected = curve
        mask = active_mask.to(device=curve.device, dtype=curve.dtype) * local_window
        for _ in range(2):
            projected = DiTAnchorDecoder._project_curve_out_bases(projected, bases, mask)
        return DiTAnchorDecoder._normalize_masked_curve_rms(
            projected,
            mask,
            target_rms=target_rms,
        )

    @staticmethod
    def _stabilize_base_preserving_multiturn_curves(
        tangent_curve: torch.Tensor,
        normal_curve: torch.Tensor,
        time_basis: torch.Tensor,
        local_window: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        base_normal_basis: torch.Tensor | None = None,
        tangent_target_rms: float = 0.08,
        normal_target_rms: float = 0.58,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        signed_basis = (time_basis - 0.5) * 2.0
        mid_basis = torch.sin(math.pi * time_basis).clamp_min(0.0)
        s_curve_basis = torch.sin(2.0 * math.pi * time_basis)
        end_basis = time_basis.clamp_min(0.0).pow(1.2)
        quad_basis = (signed_basis.pow(2) - 0.35) * mid_basis
        mask = active_mask.to(device=tangent_curve.device, dtype=tangent_curve.dtype) * local_window
        tangent_bases = [
            local_window,
            signed_basis * local_window,
            mid_basis * local_window,
            end_basis * local_window,
        ]
        normal_bases = [
            local_window,
            signed_basis * local_window,
            mid_basis * local_window,
            quad_basis * local_window,
            end_basis * local_window,
            s_curve_basis * local_window,
        ]
        if base_normal_basis is not None:
            base_basis = base_normal_basis.to(device=normal_curve.device, dtype=normal_curve.dtype) * local_window
            normal_bases.extend([base_basis, signed_basis * base_basis])
        stabilized_tangent = tangent_curve
        stabilized_normal = normal_curve
        for _ in range(2):
            stabilized_tangent = DiTAnchorDecoder._project_curve_out_bases(
                stabilized_tangent,
                tangent_bases,
                mask,
            )
            stabilized_normal = DiTAnchorDecoder._project_curve_out_bases(
                stabilized_normal,
                normal_bases,
                mask,
            )
        stabilized_tangent = DiTAnchorDecoder._normalize_masked_curve_rms(
            stabilized_tangent,
            mask,
            target_rms=tangent_target_rms,
        )
        stabilized_normal = DiTAnchorDecoder._normalize_masked_curve_rms(
            stabilized_normal,
            mask,
            target_rms=normal_target_rms,
        )
        return stabilized_tangent, stabilized_normal

    @staticmethod
    def _normalized_arc_length_parameter(points: torch.Tensor) -> torch.Tensor:
        if points.ndim < 3 or points.shape[-1] != 2:
            raise ValueError(f"Expected [..., T, 2] points, got {tuple(points.shape)}")
        diffs = points[..., 1:, :] - points[..., :-1, :]
        if diffs.shape[-2] <= 0:
            return points.new_zeros(*points.shape[:-1], 1)
        seg = torch.linalg.norm(diffs, dim=-1)
        cum = torch.cat([seg.new_zeros(*seg.shape[:-1], 1), seg.cumsum(dim=-1)], dim=-1)
        total = cum[..., -1:].clamp_min(1e-6)
        return (cum / total).unsqueeze(-1)

    @staticmethod
    def _build_arc_length_symmetric_curvature_packet(
        arc_t: torch.Tensor,
        local_window: torch.Tensor,
        *,
        variant: torch.Tensor | float = 1.0,
        target_rms: float = 1.0,
    ) -> torch.Tensor:
        if not torch.is_tensor(variant):
            variant = arc_t.new_tensor(float(variant))
        variant = variant.to(device=arc_t.device, dtype=arc_t.dtype)
        window = local_window.to(device=arc_t.device, dtype=arc_t.dtype).clamp(0.0, 1.0)
        denom = window.sum(dim=-2, keepdim=True).clamp_min(1e-6)
        center = (arc_t * window).sum(dim=-2, keepdim=True) / denom
        spread = torch.sqrt(((arc_t - center).pow(2) * window).sum(dim=-2, keepdim=True) / denom).clamp_min(0.06)
        u = (arc_t - center) / (2.35 * spread)
        support = (1.0 - u.abs().clamp(max=1.0).pow(2)).clamp_min(0.0).pow(2)
        freq = (1.35 + 0.30 * variant.clamp(0.0, 1.5)).to(device=arc_t.device, dtype=arc_t.dtype)
        packet = support * (
            torch.sin(math.pi * freq * u)
            - 0.42 * torch.sin(2.0 * math.pi * freq * u)
            + 0.15 * torch.sin(3.0 * math.pi * freq * u)
        )
        packet = packet * window
        signed_basis = ((arc_t - center) / spread.clamp_min(1e-6)).clamp(-2.0, 2.0)
        projected = DiTAnchorDecoder._project_curve_out_bases(
            packet,
            [
                window,
                signed_basis * window,
                support * window,
            ],
            window,
        )
        return DiTAnchorDecoder._normalize_masked_curve_rms(
            projected,
            window,
            target_rms=target_rms,
        )

    @staticmethod
    def _compose_local_edit_on_base_path(
        base_trajectory: torch.Tensor,
        local_delta: torch.Tensor,
        edit_mask: torch.Tensor,
        preserve_mask: torch.Tensor,
        *,
        mode: str,
    ) -> torch.Tensor:
        if base_trajectory.shape != local_delta.shape:
            raise ValueError(
                "base_trajectory and local_delta must share shape, got "
                f"{tuple(base_trajectory.shape)} vs {tuple(local_delta.shape)}"
            )
        if base_trajectory.ndim != 4 or base_trajectory.shape[-1] != 2:
            raise ValueError(
                "Expected [B, K, T, 2] tensors for base_trajectory/local_delta, got "
                f"{tuple(base_trajectory.shape)}"
            )
        batch, num_candidates, target_len, _ = base_trajectory.shape
        flat_base = base_trajectory.reshape(batch * num_candidates, target_len, 2)
        flat_delta = local_delta.reshape(batch * num_candidates, target_len, 2)
        flat_edit = edit_mask.to(device=flat_base.device, dtype=flat_base.dtype).reshape(
            batch * num_candidates,
            target_len,
            1,
        ).clamp(0.0, 1.0)
        flat_preserve = preserve_mask.to(device=flat_base.device, dtype=flat_base.dtype).reshape(
            batch * num_candidates,
            target_len,
            1,
        ).clamp(0.0, 1.0)
        if float(flat_edit.detach().max().item()) <= 0.0:
            return base_trajectory
        time_basis = torch.linspace(
            0.0,
            1.0,
            target_len,
            device=flat_base.device,
            dtype=flat_base.dtype,
        ).view(1, target_len, 1).expand(batch * num_candidates, -1, -1)
        local_window = flat_edit
        active_mask = (flat_edit > 0).to(dtype=flat_base.dtype)
        base_tangent, base_normal = DiTAnchorDecoder.tangent_normal(flat_base)
        tangent_scalar = (flat_delta * base_tangent).sum(dim=-1, keepdim=True)
        normal_scalar = (flat_delta * base_normal).sum(dim=-1, keepdim=True)
        if mode == "curvature_direction_neutral":
            denom = (active_mask * local_window).sum(dim=-2, keepdim=True).clamp_min(1e-6)
            amp_rms = torch.sqrt((normal_scalar.pow(2) * active_mask * local_window).sum(dim=-2, keepdim=True) / denom)
            tangent_rms = torch.sqrt((tangent_scalar.pow(2) * active_mask * local_window).sum(dim=-2, keepdim=True) / denom)
            variant = (
                (normal_scalar.abs() * active_mask * local_window).amax(dim=-2, keepdim=True).clamp_min(1e-6)
                / amp_rms.clamp_min(1e-6)
            ).clamp(0.4, 1.5)
            arc_t = DiTAnchorDecoder._normalized_arc_length_parameter(flat_base)
            packet = DiTAnchorDecoder._build_arc_length_symmetric_curvature_packet(
                arc_t,
                local_window,
                variant=variant,
                target_rms=1.0,
            )
            packet_amp = (1.10 * amp_rms + 0.25 * tangent_rms).clamp_min(0.035)
            normal_scalar = packet_amp * packet
            tangent_scalar = tangent_scalar.new_zeros(tangent_scalar.shape)
        elif mode == "direction_multiturn_base_preserving":
            tangent_scalar, normal_scalar = DiTAnchorDecoder._stabilize_base_preserving_multiturn_curves(
                tangent_scalar,
                normal_scalar,
                time_basis,
                local_window,
                active_mask,
                tangent_target_rms=0.035,
                normal_target_rms=0.34,
            )
            tangent_scalar = 0.70 * tangent_scalar
        elif mode == "reference_local_direction_preserving":
            signed_basis = (time_basis - 0.5) * 2.0
            mid_basis = torch.sin(math.pi * time_basis).clamp_min(0.0)
            s_curve_basis = torch.sin(2.0 * math.pi * time_basis)
            end_basis = time_basis.clamp_min(0.0).pow(1.15)
            mask = active_mask * local_window
            normal_scalar = DiTAnchorDecoder._project_curve_out_bases(
                normal_scalar,
                [
                    local_window,
                    signed_basis * local_window,
                    mid_basis * local_window,
                    end_basis * local_window,
                ],
                mask,
            )
            normal_scalar = DiTAnchorDecoder._normalize_masked_curve_rms(
                normal_scalar,
                mask,
                target_rms=0.42,
            )
            tangent_scalar = DiTAnchorDecoder._project_curve_out_bases(
                tangent_scalar,
                [
                    local_window,
                    signed_basis * local_window,
                    mid_basis * local_window,
                    end_basis * local_window,
                ],
                mask,
            )
            tangent_scalar = DiTAnchorDecoder._normalize_masked_curve_rms(
                tangent_scalar,
                mask,
                target_rms=0.03,
            )
            tangent_scalar = 0.24 * tangent_scalar
        elif mode == "reference_metric_attraction_preserving":
            projected_delta = DiTAnchorDecoder.preserve_reference_endpoints(flat_delta)
            edited = flat_base + projected_delta
            edit_mix = (1.0 - flat_preserve).clamp(0.0, 1.0)
            composed = flat_base * (1.0 - edit_mix) + edited * edit_mix
            composed = flat_base + DiTAnchorDecoder.preserve_reference_endpoints(composed - flat_base)
            return composed.view(batch, num_candidates, target_len, 2)
        else:
            raise ValueError(f"Unsupported base-path local edit mode={mode}")
        projected_delta = (
            tangent_scalar * base_tangent
            + normal_scalar * base_normal
        )
        projected_delta = DiTAnchorDecoder.preserve_reference_endpoints(projected_delta)
        edited = flat_base + projected_delta
        edit_mix = (1.0 - flat_preserve).clamp(0.0, 1.0)
        composed = flat_base * (1.0 - edit_mix) + edited * edit_mix
        composed = flat_base + DiTAnchorDecoder.preserve_reference_endpoints(composed - flat_base)
        return composed.view(batch, num_candidates, target_len, 2)

    @staticmethod
    def _contract_direction_preserving_curvature_delta(
        base_trajectory: torch.Tensor,
        projected_delta: torch.Tensor,
        local_window: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        coarse_points: int = 16,
    ) -> torch.Tensor:
        if base_trajectory.shape != projected_delta.shape:
            raise ValueError(
                "base_trajectory and projected_delta must share shape, got "
                f"{tuple(base_trajectory.shape)} vs {tuple(projected_delta.shape)}"
            )
        if base_trajectory.ndim != 3 or base_trajectory.shape[-1] != 2:
            raise ValueError(
                "Expected [B, T, 2] tensors for base_trajectory/projected_delta, got "
                f"{tuple(base_trajectory.shape)}"
            )
        if float(active_mask.detach().max().item()) <= 0.0:
            return projected_delta
        target_len = base_trajectory.shape[-2]
        mask = active_mask.to(device=base_trajectory.device, dtype=base_trajectory.dtype).clamp(0.0, 1.0)
        window = local_window.to(device=base_trajectory.device, dtype=base_trajectory.dtype).clamp(0.0, 1.0)
        base_tangent, base_normal = DiTAnchorDecoder.tangent_normal(base_trajectory)
        lowfreq_delta = DiTAnchorDecoder.resample_temporal(
            DiTAnchorDecoder.resample_temporal(projected_delta, int(coarse_points)),
            target_len,
        )
        highfreq_delta = projected_delta - lowfreq_delta
        lowfreq_tangent = (lowfreq_delta * base_tangent).sum(dim=-1, keepdim=True)
        lowfreq_normal = (lowfreq_delta * base_normal).sum(dim=-1, keepdim=True)
        candidate = base_trajectory + projected_delta
        base_contract = DiTAnchorDecoder._contract_direction_features(base_trajectory, coarse_points=int(coarse_points))
        cand_contract = DiTAnchorDecoder._contract_direction_features(candidate, coarse_points=int(coarse_points))
        base_local = DiTAnchorDecoder._local_direction_features(base_trajectory)
        cand_local = DiTAnchorDecoder._local_direction_features(candidate)
        contract_net_delta = (
            cand_contract["contract_net_heading_rad"] - base_contract["contract_net_heading_rad"]
        ).view(-1, 1, 1)
        contract_total_delta = (
            cand_contract["contract_total_abs_heading_rad"] - base_contract["contract_total_abs_heading_rad"]
        ).view(-1, 1, 1)
        local_heading_delta = (
            cand_local["local_heading_diff_rad"] - base_local["local_heading_diff_rad"]
        ).view(-1, 1, 1)
        signed_basis = torch.linspace(
            -1.0,
            1.0,
            target_len,
            device=base_trajectory.device,
            dtype=base_trajectory.dtype,
        ).view(1, target_len, 1)
        mid_basis = torch.sin(
            math.pi
            * torch.linspace(
                0.0,
                1.0,
                target_len,
                device=base_trajectory.device,
                dtype=base_trajectory.dtype,
            ).view(1, target_len, 1)
        ).clamp_min(0.0)
        correction_strength = (
            (contract_net_delta.abs() / 0.12)
            + (local_heading_delta.abs() / 0.14)
            + F.relu(contract_total_delta) / 0.25
        ).clamp(0.0, 1.25)
        heading_normal_correction = (
            -(0.18 * contract_net_delta + 0.14 * local_heading_delta)
            * signed_basis
            * window
        )
        heading_tangent_correction = (
            -0.08 * contract_total_delta * mid_basis * window
        )
        corrected_lowfreq_tangent = (
            (1.0 - correction_strength) * lowfreq_tangent
            + heading_tangent_correction
        )
        corrected_lowfreq_normal = (
            (1.0 - 0.92 * correction_strength) * lowfreq_normal
            + heading_normal_correction
        )
        corrected_delta = (
            highfreq_delta
            + corrected_lowfreq_tangent * base_tangent
            + corrected_lowfreq_normal * base_normal
        )
        corrected_delta = DiTAnchorDecoder.preserve_reference_endpoints(corrected_delta)
        return projected_delta + mask * (corrected_delta - projected_delta)

    @staticmethod
    def _path_length(points: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(points[..., 1:, :] - points[..., :-1, :], dim=-1).sum(dim=-1).clamp_min(1e-6)

    @staticmethod
    def _curvature_energy(points: torch.Tensor) -> torch.Tensor:
        diffs = points[..., 1:, :] - points[..., :-1, :]
        if diffs.shape[-2] < 2:
            return points.new_zeros(points.shape[:-2])
        v1 = diffs[..., :-1, :]
        v2 = diffs[..., 1:, :]
        cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
        dot = (v1 * v2).sum(dim=-1)
        return safe_atan2(cross, dot).abs().mean(dim=-1)

    @staticmethod
    def _contract_curvature_score(points: torch.Tensor) -> torch.Tensor:
        diffs = points[..., 1:, :] - points[..., :-1, :]
        if diffs.shape[-2] < 2:
            return points.new_zeros(points.shape[:-2])
        v1 = diffs[..., :-1, :]
        v2 = diffs[..., 1:, :]
        cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
        dot = (v1 * v2).sum(dim=-1)
        strong_total = safe_atan2(cross, dot).abs().sum(dim=-1)
        length_norm = DiTAnchorDecoder._path_length(points)
        return strong_total / length_norm.clamp_min(1e-6)

    @staticmethod
    def _heading_features(points: torch.Tensor) -> dict[str, torch.Tensor]:
        diffs = points[..., 1:, :] - points[..., :-1, :]
        if diffs.shape[-2] < 2:
            shape = points.shape[:-2]
            zeros = points.new_zeros(shape)
            return {
                "signed_net_heading_rad": zeros,
                "abs_signed_heading_rad": zeros,
                "total_abs_heading_rad": zeros,
                "signed_total_turn_rad": zeros,
                "left_turn_ratio": zeros,
                "right_turn_ratio": zeros,
                "max_abs_turn_rad": zeros,
                "turn_count": zeros,
            }
        headings = safe_atan2(diffs[..., 1], diffs[..., 0])
        delta = headings[..., 1:] - headings[..., :-1]
        delta = safe_atan2(torch.sin(delta), torch.cos(delta))
        net = headings[..., -1] - headings[..., 0]
        net = safe_atan2(torch.sin(net), torch.cos(net))
        abs_delta = delta.abs()
        threshold = math.radians(15.0)
        strong = abs_delta > threshold
        strong_count = strong.sum(dim=-1).to(dtype=points.dtype).clamp_min(1.0)
        left_ratio = (delta > threshold).sum(dim=-1).to(dtype=points.dtype) / strong_count
        right_ratio = (delta < -threshold).sum(dim=-1).to(dtype=points.dtype) / strong_count
        return {
            "signed_net_heading_rad": net,
            "abs_signed_heading_rad": net.abs(),
            "total_abs_heading_rad": abs_delta.sum(dim=-1),
            "signed_total_turn_rad": delta.sum(dim=-1),
            "left_turn_ratio": left_ratio,
            "right_turn_ratio": right_ratio,
            "max_abs_turn_rad": abs_delta.amax(dim=-1),
            "turn_count": strong.sum(dim=-1).to(dtype=points.dtype),
        }

    @staticmethod
    def _local_direction_features(points: torch.Tensor) -> dict[str, torch.Tensor]:
        diffs = points[..., 1:, :] - points[..., :-1, :]
        shape = points.shape[:-2]
        if diffs.shape[-2] < 2:
            zeros = points.new_zeros(shape)
            return {
                "local_dx": zeros,
                "local_dy": zeros,
                "local_heading_diff_rad": zeros,
                "local_abs_heading_diff_rad": zeros,
                "local_signed_total_turn_rad": zeros,
                "local_total_abs_turn_rad": zeros,
                "local_left_turn_ratio": zeros,
                "local_right_turn_ratio": zeros,
                "local_max_abs_turn_rad": zeros,
                "local_turn_count": zeros,
            }
        first = diffs[..., 0, :]
        ex = first / torch.linalg.norm(first, dim=-1, keepdim=True).clamp_min(1e-6)
        ey = torch.stack([-ex[..., 1], ex[..., 0]], dim=-1)
        rel_end = points[..., -1, :] - points[..., 0, :]
        local_dx = (rel_end * ex).sum(dim=-1)
        local_dy = (rel_end * ey).sum(dim=-1)

        headings = safe_atan2(diffs[..., 1], diffs[..., 0])
        delta = headings[..., 1:] - headings[..., :-1]
        delta = safe_atan2(torch.sin(delta), torch.cos(delta))
        heading_diff = headings[..., -1] - headings[..., 0]
        heading_diff = safe_atan2(torch.sin(heading_diff), torch.cos(heading_diff))
        abs_delta = delta.abs()
        threshold = math.radians(15.0)
        strong = abs_delta > threshold
        strong_count = strong.sum(dim=-1).to(dtype=points.dtype).clamp_min(1.0)
        return {
            "local_dx": local_dx,
            "local_dy": local_dy,
            "local_heading_diff_rad": heading_diff,
            "local_abs_heading_diff_rad": heading_diff.abs(),
            "local_signed_total_turn_rad": delta.sum(dim=-1),
            "local_total_abs_turn_rad": abs_delta.sum(dim=-1),
            "local_left_turn_ratio": (delta > threshold).sum(dim=-1).to(dtype=points.dtype) / strong_count,
            "local_right_turn_ratio": (delta < -threshold).sum(dim=-1).to(dtype=points.dtype) / strong_count,
            "local_max_abs_turn_rad": abs_delta.amax(dim=-1),
            "local_turn_count": strong.sum(dim=-1).to(dtype=points.dtype),
        }

    @staticmethod
    def _contract_direction_features(points: torch.Tensor, coarse_points: int = 16) -> dict[str, torch.Tensor]:
        coarse_points = int(max(coarse_points, 4))
        original_shape = points.shape[:-2]
        flat = points.reshape(-1, points.shape[-2], points.shape[-1])
        coarse = F.interpolate(
            flat.transpose(1, 2),
            size=coarse_points,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        coarse = coarse.reshape(*original_shape, coarse_points, points.shape[-1])
        diffs = coarse[..., 1:, :] - coarse[..., :-1, :]
        if diffs.shape[-2] < 2:
            zeros = points.new_zeros(original_shape)
            return {
                "contract_net_heading_rad": zeros,
                "contract_abs_net_heading_rad": zeros,
                "contract_total_abs_heading_rad": zeros,
                "contract_raw_total_abs_heading_rad": zeros,
                "contract_turn_count": zeros,
                "contract_displacement_ratio": zeros,
            }
        headings = safe_atan2(diffs[..., 1], diffs[..., 0])
        delta = headings[..., 1:] - headings[..., :-1]
        delta = safe_atan2(torch.sin(delta), torch.cos(delta))
        abs_delta = delta.abs()
        turn_threshold = math.radians(35.0)
        turn_temp = math.radians(10.0)
        soft_strong = torch.sigmoid((abs_delta - turn_threshold) / max(turn_temp, 1e-6))
        displacement = torch.linalg.norm(points[..., -1, :] - points[..., 0, :], dim=-1)
        length = DiTAnchorDecoder._path_length(points).clamp_min(1e-6)
        signed_net = delta.sum(dim=-1)
        return {
            "contract_net_heading_rad": signed_net,
            "contract_abs_net_heading_rad": signed_net.abs(),
            "contract_total_abs_heading_rad": (abs_delta * soft_strong).sum(dim=-1),
            "contract_raw_total_abs_heading_rad": abs_delta.sum(dim=-1),
            "contract_turn_count": soft_strong.sum(dim=-1).to(dtype=points.dtype),
            "contract_displacement_ratio": displacement / length,
        }

    @staticmethod
    def _direction_metric_id(metric: str) -> int:
        mapping = {
            "signed_net_heading_rad": 0,
            "abs_signed_heading_rad": 1,
            "total_abs_heading_rad": 2,
            "signed_total_turn_rad": 3,
            "left_turn_ratio": 4,
            "right_turn_ratio": 5,
            "max_abs_turn_rad": 6,
            "turn_count": 7,
            "local_dx": 8,
            "local_dy": 9,
            "local_heading_diff_rad": 10,
            "local_abs_heading_diff_rad": 11,
            "local_signed_total_turn_rad": 12,
            "local_total_abs_turn_rad": 13,
            "local_left_turn_ratio": 14,
            "local_right_turn_ratio": 15,
            "local_max_abs_turn_rad": 16,
            "local_turn_count": 17,
        }
        return mapping.get(str(metric), 0)

    @staticmethod
    def _scalar_calibration_metric_id(metric: str) -> int:
        mapping = {
            "length_m": 0,
            "mean_reference_distance_norm": 1,
            "mean_reference_distance_m": 2,
            "turn_density_score": 3,
            "contract_curvature_score": 4,
            "curvature_energy": 5,
            "avg_speed_kmh": 6,
            "displacement_ratio": 7,
            "length_ratio": 8,
        }
        aliases = {
            "mean_reference_distance": "mean_reference_distance_norm",
            "pace_avg_speed_kmh": "avg_speed_kmh",
            "progress_ratio": "displacement_ratio",
        }
        normalized = aliases.get(str(metric), str(metric))
        return mapping.get(normalized, mapping["length_m"])

    @staticmethod
    def _contract_direction_all_token_scores(
        features: dict[str, torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Differentiable Table-3/4 direction contract scores; lower is better."""
        sample = next(iter(features.values()))
        device = sample.device
        net = features.get(
            "contract_net_heading_rad",
            features.get("signed_net_heading_rad", sample.new_zeros(sample.shape)),
        ).to(dtype=dtype)
        abs_net = features.get(
            "contract_abs_net_heading_rad",
            features.get("abs_signed_heading_rad", net.abs()),
        ).abs().to(dtype=dtype)
        total_abs = features.get(
            "contract_total_abs_heading_rad",
            features.get("total_abs_heading_rad", sample.new_zeros(sample.shape)),
        ).to(dtype=dtype)
        turn_count = features.get(
            "contract_turn_count",
            features.get("turn_count", sample.new_zeros(sample.shape)),
        ).to(dtype=dtype)
        displacement_ratio = features.get(
            "contract_displacement_ratio",
            features.get("displacement_ratio", sample.new_zeros(sample.shape)),
        ).to(dtype=dtype)
        signed_turn = features.get("local_signed_total_turn_rad", features.get("signed_total_turn_rad", net)).to(dtype=dtype)
        local_dy = features.get("local_dy", sample.new_zeros(sample.shape)).to(dtype=dtype)
        left_ratio = features.get("local_left_turn_ratio", features.get("left_turn_ratio", sample.new_zeros(sample.shape))).to(dtype=dtype)
        right_ratio = features.get("local_right_turn_ratio", features.get("right_turn_ratio", sample.new_zeros(sample.shape))).to(dtype=dtype)

        deg35 = math.radians(35.0)
        deg135 = math.radians(135.0)
        deg120 = math.radians(120.0)
        eps = 1e-6
        u_heading_strength = torch.minimum(
            abs_net / max(deg135, eps),
            torch.relu(0.55 - displacement_ratio) / 0.55,
        )
        u_loop_strength = torch.minimum(
            torch.relu(0.25 - displacement_ratio) / 0.25,
            total_abs / max(deg120, eps),
        )
        u_strength = torch.maximum(u_heading_strength, u_loop_strength)
        multi_strength = torch.maximum(turn_count / 3.0, total_abs / max(deg120, eps))
        left_strength = torch.maximum(
            torch.relu(net) / max(deg35, eps),
            torch.maximum(
                torch.relu(signed_turn) / max(deg35, eps),
                torch.maximum(0.35 * torch.relu(local_dy - 0.02) / 0.12, left_ratio),
            ),
        )
        right_strength = torch.maximum(
            torch.relu(-net) / max(deg35, eps),
            torch.maximum(
                torch.relu(-signed_turn) / max(deg35, eps),
                torch.maximum(0.35 * torch.relu(-local_dy - 0.02) / 0.12, right_ratio),
            ),
        )
        left_clear = left_strength.clamp(min=0.0, max=1.0)
        right_clear = right_strength.clamp(min=0.0, max=1.0)
        directional_strength = torch.maximum(left_strength, right_strength)

        straight = (
            abs_net / deg35
            + 0.4 * total_abs / max(deg120, eps)
            + torch.relu(turn_count - 1.0)
            + 0.4 * directional_strength
        )
        left = (
            torch.relu(1.0 - left_strength)
            + 0.5 * torch.relu(multi_strength - 1.0) * (1.0 - 0.75 * left_clear)
            + 0.5 * torch.relu(u_strength - 1.0)
            - 0.08 * left_strength
        )
        right = (
            torch.relu(1.0 - right_strength)
            + 0.5 * torch.relu(multi_strength - 1.0) * (1.0 - 0.75 * right_clear)
            + 0.5 * torch.relu(u_strength - 1.0)
            - 0.08 * right_strength
        )
        u_turn = (
            torch.relu(1.0 - u_strength)
            + 0.35 * torch.relu(multi_strength - 1.0)
            - 0.08 * u_strength
        )
        multi_turn = (
            torch.relu(1.0 - multi_strength)
            + 0.35 * torch.relu(u_strength - 1.0)
            + 0.25 * multi_strength * directional_strength.clamp(min=0.0, max=1.0)
            + 0.75 * torch.relu(directional_strength - 1.0)
            - 0.08 * multi_strength
        )

        ids = DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]
        ordered = torch.empty((*sample.shape, len(ids)), dtype=dtype, device=device)
        ordered[..., int(ids["straight"])] = straight
        ordered[..., int(ids["left_turn"])] = left
        ordered[..., int(ids["right_turn"])] = right
        ordered[..., int(ids["u_turn"])] = u_turn
        ordered[..., int(ids["multi_turn"])] = multi_turn
        return ordered

    @staticmethod
    def _safe_calibration_value(stats: dict, key: str, default: float) -> float:
        value = stats.get(key, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float(default)
        return value if math.isfinite(value) else float(default)

    def load_token_calibration(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        fields = payload.get("fields", {})
        token_to_id = DEFAULT_MOBILITY_VOCAB.token_to_id

        def _field_default_metric(field: str) -> str:
            payload_metric = str(fields.get(field, {}).get("metric", ""))
            if payload_metric:
                if payload_metric == "direction_token_metric":
                    return "signed_net_heading_rad"
                return payload_metric
            defaults = {
                "distance": "length_m",
                "reference_adherence": "mean_reference_distance_m",
                "curvature": "turn_density_score",
                "progress": "avg_speed_kmh",
            }
            return defaults.get(field, "length_m")

        def _fill(
            field: str,
            center_buffer: torch.Tensor,
            scale_buffer: torch.Tensor,
            metric_buffer: Optional[torch.Tensor] = None,
        ) -> None:
            token_stats = fields.get(field, {}).get("tokens", {})
            default_metric = _field_default_metric(field)
            for token_name, token_id in token_to_id[field].items():
                stats = token_stats.get(token_name, {})
                center_buffer[token_id] = self._safe_calibration_value(stats, "p50", 0.0)
                scale_buffer[token_id] = max(self._safe_calibration_value(stats, "scale", 1.0), 1e-6)
                if metric_buffer is not None:
                    metric_name = str(stats.get("direction_metric", default_metric))
                    metric_buffer[token_id] = self._scalar_calibration_metric_id(metric_name)

        _fill("distance", self.calib_distance_center, self.calib_distance_scale, self.calib_distance_metric)
        _fill("reference_adherence", self.calib_ref_center, self.calib_ref_scale, self.calib_ref_metric)
        _fill("curvature", self.calib_curv_center, self.calib_curv_scale, self.calib_curv_metric)
        _fill("progress", self.calib_progress_center, self.calib_progress_scale, self.calib_progress_metric)
        _fill("direction", self.calib_direction_center, self.calib_direction_scale)
        direction_stats = fields.get("direction", {}).get("tokens", {})
        for token_name, token_id in token_to_id["direction"].items():
            metric = str(direction_stats.get(token_name, {}).get("direction_metric", "signed_net_heading_rad"))
            self.calib_direction_metric[token_id] = self._direction_metric_id(metric)
        self.token_feature_mode = "calibrated"

    def build_reranker_geometry_features(
        self,
        candidates: torch.Tensor,
        anchor: torch.Tensor,
        residual_view: torch.Tensor,
        raw_residual_view: torch.Tensor,
        gate_view: torch.Tensor,
        mobility_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference-safe candidate geometry features for macro trajectory ranking."""
        ref_dist = torch.linalg.norm(candidates - anchor.unsqueeze(1), dim=-1)
        mean_ref_dist = ref_dist.mean(dim=-1, keepdim=True)
        max_ref_dist = ref_dist.amax(dim=-1, keepdim=True)
        final_ref_dist = ref_dist[..., -1:].contiguous()

        path_len = self._path_length(candidates).unsqueeze(-1)
        anchor_len = self._path_length(anchor).view(-1, 1, 1)
        length_ratio = path_len / anchor_len.clamp_min(1e-6)
        displacement = torch.linalg.norm(candidates[..., -1, :] - candidates[..., 0, :], dim=-1, keepdim=True)
        progress = displacement / path_len.clamp_min(1e-6)
        curvature = self._contract_curvature_score(candidates).unsqueeze(-1)
        heading_features = self._heading_features(candidates)
        heading_features.update(self._local_direction_features(candidates))

        residual_norm = torch.linalg.norm(residual_view, dim=-1)
        residual_norm_mean = residual_norm.mean(dim=-1, keepdim=True)
        residual_norm_max = residual_norm.amax(dim=-1, keepdim=True)
        raw_tangent_mean = raw_residual_view[..., 0].abs().mean(dim=-1, keepdim=True)
        raw_normal_mean = raw_residual_view[..., 1].abs().mean(dim=-1, keepdim=True)
        gate_mean = gate_view.squeeze(-1).mean(dim=-1, keepdim=True)
        gate_std = gate_view.squeeze(-1).std(dim=-1, keepdim=True, unbiased=False)

        token_features = candidates.new_zeros(candidates.shape[0], candidates.shape[1], 4)
        if mobility_tokens is not None and mobility_tokens.ndim == 2 and mobility_tokens.shape[0] == candidates.shape[0]:
            token_features = self.build_instruction_consistency_features(
                candidates,
                mobility_tokens,
                mean_ref_dist.squeeze(-1),
                curvature.squeeze(-1),
                progress.squeeze(-1),
                heading_features,
            )

        return torch.cat(
            [
                mean_ref_dist,
                max_ref_dist,
                final_ref_dist,
                length_ratio,
                progress,
                curvature,
                residual_norm_mean,
                residual_norm_max,
                raw_tangent_mean,
                raw_normal_mean,
                gate_mean,
                gate_std,
                token_features,
            ],
            dim=-1,
        )

    def build_instruction_consistency_features(
        self,
        candidates: torch.Tensor,
        mobility_tokens: torch.Tensor,
        mean_ref_dist: torch.Tensor,
        curvature: torch.Tensor,
        progress: torch.Tensor,
        heading_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Token-consistency features derived only from inference-time mobility tokens."""
        if self.token_feature_mode != "calibrated":
            return candidates.new_zeros(candidates.shape[0], candidates.shape[1], 4)
        field_index = {field: idx for idx, field in enumerate(DEFAULT_MOBILITY_VOCAB.fields)}

        direction_id = mobility_tokens[:, field_index["direction"]]
        curvature_id = mobility_tokens[:, field_index["curvature"]]
        progress_id = mobility_tokens[:, field_index["progress"]]
        ref_id = mobility_tokens[:, field_index["reference_adherence"]]

        ref_center = self.calib_ref_center.to(device=candidates.device, dtype=candidates.dtype)[ref_id].view(-1, 1)
        ref_scale = self.calib_ref_scale.to(device=candidates.device, dtype=candidates.dtype)[ref_id].view(-1, 1)
        curv_center = self.calib_curv_center.to(device=candidates.device, dtype=candidates.dtype)[curvature_id].view(-1, 1)
        curv_scale = self.calib_curv_scale.to(device=candidates.device, dtype=candidates.dtype)[curvature_id].view(-1, 1)
        progress_center = self.calib_progress_center.to(device=candidates.device, dtype=candidates.dtype)[progress_id].view(-1, 1)
        progress_scale = self.calib_progress_scale.to(device=candidates.device, dtype=candidates.dtype)[progress_id].view(-1, 1)
        direction_center = self.calib_direction_center.to(device=candidates.device, dtype=candidates.dtype)[direction_id].view(-1, 1)
        direction_scale = self.calib_direction_scale.to(device=candidates.device, dtype=candidates.dtype)[direction_id].view(-1, 1)
        direction_metric = self.calib_direction_metric.to(device=candidates.device)[direction_id].view(-1, 1)
        metric_values = [
            heading_features["signed_net_heading_rad"],
            heading_features["abs_signed_heading_rad"],
            heading_features["total_abs_heading_rad"],
            heading_features["signed_total_turn_rad"],
            heading_features["left_turn_ratio"],
            heading_features["right_turn_ratio"],
            heading_features["max_abs_turn_rad"],
            heading_features["turn_count"],
            heading_features["local_dx"],
            heading_features["local_dy"],
            heading_features["local_heading_diff_rad"],
            heading_features["local_abs_heading_diff_rad"],
            heading_features["local_signed_total_turn_rad"],
            heading_features["local_total_abs_turn_rad"],
            heading_features["local_left_turn_ratio"],
            heading_features["local_right_turn_ratio"],
            heading_features["local_max_abs_turn_rad"],
            heading_features["local_turn_count"],
        ]
        direction_value = metric_values[0]
        for metric_id, metric_value in enumerate(metric_values[1:], start=1):
            direction_value = torch.where(direction_metric == metric_id, metric_value, direction_value)

        ref_match = ((mean_ref_dist - ref_center) / ref_scale.clamp_min(1e-6)).abs().unsqueeze(-1)
        curv_match = ((curvature - curv_center) / curv_scale.clamp_min(1e-6)).abs().unsqueeze(-1)
        progress_match = ((progress - progress_center) / progress_scale.clamp_min(1e-6)).abs().unsqueeze(-1)
        direction_match = ((direction_value - direction_center) / direction_scale.clamp_min(1e-6)).abs().unsqueeze(-1)
        return torch.cat([ref_match, curv_match, progress_match, direction_match], dim=-1).clamp(max=10.0)

    def prepare_base_selected_trajectory(
        self,
        base_selected_trajectory: Optional[torch.Tensor],
        *,
        batch: int,
        target_len: int,
        device: torch.device,
        dtype: torch.dtype,
        base_selected_index: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if base_selected_trajectory is None:
            return None
        base = base_selected_trajectory
        if base.ndim == 4:
            if base_selected_index is not None and base.shape[1] > 1:
                idx = base_selected_index.to(device=device, dtype=torch.long).clamp(0, base.shape[1] - 1)
                base = base[torch.arange(batch, device=device), idx]
            else:
                base = base[:, 0]
        if base.ndim != 3 or base.shape[0] != batch or base.shape[-1] != 2:
            return None
        base = base.to(device=device, dtype=dtype)
        if base.shape[-2] != int(target_len):
            base = self.resample_temporal(base, int(target_len))
        return base

    def build_counterfactual_pair_features(
        self,
        candidates: torch.Tensor,
        reference_anchor: Optional[torch.Tensor],
        base_selected_trajectory: Optional[torch.Tensor],
        base_mobility_tokens: Optional[torch.Tensor],
        mobility_tokens: Optional[torch.Tensor],
        field_ids: Optional[torch.Tensor],
        route_scales: Optional[torch.Tensor] = None,
        base_selected_index: Optional[torch.Tensor] = None,
        trip_duration_min: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Inference-available Table-4 pair features for counterfactual scoring.

        The features compare each counterfactual candidate against the original
        selected trajectory under the same calibration buffers used by the
        reranker token-consistency features. They do not use GT trajectories.
        """
        if (
            base_selected_trajectory is None
            or reference_anchor is None
            or mobility_tokens is None
            or field_ids is None
            or candidates.ndim != 4
        ):
            return None
        batch, num_candidates = candidates.shape[:2]
        if base_mobility_tokens is None:
            base_mobility_tokens = mobility_tokens
        if base_mobility_tokens.shape[0] != batch or mobility_tokens.shape[0] != batch:
            return None
        if reference_anchor.ndim == 4:
            reference_anchor = reference_anchor[:, 0]
        if reference_anchor.ndim != 3 or reference_anchor.shape[0] != batch or reference_anchor.shape[-1] != 2:
            return None

        dtype = candidates.dtype
        device = candidates.device
        base = self.prepare_base_selected_trajectory(
            base_selected_trajectory,
            batch=batch,
            target_len=candidates.shape[-2],
            device=device,
            dtype=dtype,
            base_selected_index=base_selected_index,
        )
        if base is None:
            return None
        cf = candidates.to(device=device, dtype=dtype)
        anchor = reference_anchor.to(device=device, dtype=dtype)
        if anchor.shape[-2] != cf.shape[-2]:
            anchor = self.resample_temporal(anchor, cf.shape[-2])

        # Reference distance must use point-to-OD-chord-segment distance, the
        # same quantity used by Table 3/4 geometry evaluation. Pointwise
        # distance to the same temporal anchor index over-penalizes valid
        # middle-route deviations on curved macro trajectories.
        base_ref = self._mean_reference_polyline_distance(base, anchor).view(batch, 1)
        cf_ref = self._mean_reference_polyline_distance(
            cf.reshape(batch * num_candidates, cf.shape[-2], 2),
            anchor[:, None].expand(-1, num_candidates, -1, -1).reshape(batch * num_candidates, anchor.shape[-2], 2),
        ).view(batch, num_candidates)

        def _scaled_turn_density(points: torch.Tensor, scales: Optional[torch.Tensor]) -> torch.Tensor:
            diffs = points[..., 1:, :] - points[..., :-1, :]
            if diffs.shape[-2] < 2:
                return points.new_zeros(points.shape[:-2])
            v1 = diffs[..., :-1, :]
            v2 = diffs[..., 1:, :]
            cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
            dot = (v1 * v2).sum(dim=-1)
            total_abs = safe_atan2(cross, dot).abs().sum(dim=-1)
            length_norm = self._path_length(points)
            if scales is None:
                return total_abs / length_norm.clamp_min(1e-6)
            scale = scales.to(device=device, dtype=dtype)
            while scale.ndim < length_norm.ndim:
                scale = scale.unsqueeze(1)
            length_m = length_norm * scale
            return total_abs / torch.clamp(length_m / 100.0, min=1.0)

        def _scaled_strong_turn_density(points: torch.Tensor, scales: Optional[torch.Tensor]) -> torch.Tensor:
            diffs = points[..., 1:, :] - points[..., :-1, :]
            if diffs.shape[-2] < 2:
                return points.new_zeros(points.shape[:-2])
            v1 = diffs[..., :-1, :]
            v2 = diffs[..., 1:, :]
            cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
            dot = (v1 * v2).sum(dim=-1)
            abs_angle = safe_atan2(cross, dot).abs()
            strong_total = (abs_angle * (abs_angle > math.radians(35.0)).to(dtype=dtype)).sum(dim=-1)
            length_norm = self._path_length(points)
            if scales is None:
                return strong_total / length_norm.clamp_min(1e-6)
            scale = scales.to(device=device, dtype=dtype)
            while scale.ndim < length_norm.ndim:
                scale = scale.unsqueeze(1)
            length_m = length_norm * scale
            return strong_total / torch.clamp(length_m / 100.0, min=1.0)

        scales_for_pair = None if route_scales is None else route_scales[:batch].to(device=device, dtype=dtype).view(batch)
        anchor_length_norm = self._path_length(anchor).view(batch, 1).clamp_min(1e-6)
        base_length_norm = self._path_length(base).view(batch, 1)
        cf_length_norm = self._path_length(cf)
        if scales_for_pair is None:
            base_distance = base_length_norm
            cf_distance = cf_length_norm
        else:
            base_distance = base_length_norm * scales_for_pair.view(batch, 1)
            cf_distance = cf_length_norm * scales_for_pair.view(batch, 1)
        base_curv = _scaled_turn_density(base, scales_for_pair).view(batch, 1)
        cf_curv = _scaled_turn_density(cf, scales_for_pair)
        base_contract_curv = _scaled_strong_turn_density(base, scales_for_pair).view(batch, 1)
        cf_contract_curv = _scaled_strong_turn_density(cf, scales_for_pair)
        base_curv_energy = self._curvature_energy(base).view(batch, 1)
        cf_curv_energy = self._curvature_energy(cf)
        if trip_duration_min is not None:
            duration = trip_duration_min.to(device=device, dtype=dtype).view(batch).clamp_min(1e-6)
            base_progress = ((base_distance / 1000.0) / (duration.view(batch, 1) / 60.0)).clamp(0.0, 120.0)
            cf_progress = ((cf_distance / 1000.0) / (duration.view(batch, 1) / 60.0)).clamp(0.0, 120.0)
        else:
            base_progress = (
                torch.linalg.norm(base[:, -1] - base[:, 0], dim=-1) / base_length_norm.view(batch).clamp_min(1e-6)
            ).view(batch, 1)
            cf_progress = torch.linalg.norm(cf[..., -1, :] - cf[..., 0, :], dim=-1) / cf_length_norm.clamp_min(1e-6)
        base_displacement_ratio = (
            torch.linalg.norm(base[:, -1] - base[:, 0], dim=-1) / base_length_norm.view(batch).clamp_min(1e-6)
        ).view(batch, 1)
        cf_displacement_ratio = (
            torch.linalg.norm(cf[..., -1, :] - cf[..., 0, :], dim=-1) / cf_length_norm.clamp_min(1e-6)
        )
        base_length_ratio = base_length_norm / anchor_length_norm
        cf_length_ratio = cf_length_norm / anchor_length_norm
        if scales_for_pair is None:
            base_ref_m = base_ref
            cf_ref_m = cf_ref
        else:
            base_ref_m = base_ref * scales_for_pair.view(batch, 1)
            cf_ref_m = cf_ref * scales_for_pair.view(batch, 1)
        base_heading = self._heading_features(base)
        base_heading.update(self._local_direction_features(base))
        base_heading.update(self._contract_direction_features(base))
        cf_heading = self._heading_features(cf)
        cf_heading.update(self._local_direction_features(cf))
        cf_heading.update(self._contract_direction_features(cf))

        field_index = {field: idx for idx, field in enumerate(DEFAULT_MOBILITY_VOCAB.fields)}
        cf_direction_id = mobility_tokens[:, field_index["direction"]].to(device=device, dtype=torch.long)
        cf_distance_id = mobility_tokens[:, field_index["distance"]].to(device=device, dtype=torch.long)
        cf_curv_id = mobility_tokens[:, field_index["curvature"]].to(device=device, dtype=torch.long)
        cf_progress_id = mobility_tokens[:, field_index["progress"]].to(device=device, dtype=torch.long)
        cf_ref_id = mobility_tokens[:, field_index["reference_adherence"]].to(device=device, dtype=torch.long)
        base_direction_id = base_mobility_tokens[:, field_index["direction"]].to(device=device, dtype=torch.long)
        base_distance_id = base_mobility_tokens[:, field_index["distance"]].to(device=device, dtype=torch.long)
        base_curv_id = base_mobility_tokens[:, field_index["curvature"]].to(device=device, dtype=torch.long)
        base_progress_id = base_mobility_tokens[:, field_index["progress"]].to(device=device, dtype=torch.long)
        base_ref_id = base_mobility_tokens[:, field_index["reference_adherence"]].to(device=device, dtype=torch.long)

        def _select_direction_value(features: dict[str, torch.Tensor], token_ids: torch.Tensor) -> torch.Tensor:
            metric_ids = self.calib_direction_metric.to(device=device)[token_ids]
            values = [
                features["signed_net_heading_rad"],
                features["abs_signed_heading_rad"],
                features["total_abs_heading_rad"],
                features["signed_total_turn_rad"],
                features["left_turn_ratio"],
                features["right_turn_ratio"],
                features["max_abs_turn_rad"],
                features["turn_count"],
                features["local_dx"],
                features["local_dy"],
                features["local_heading_diff_rad"],
                features["local_abs_heading_diff_rad"],
                features["local_signed_total_turn_rad"],
                features["local_total_abs_turn_rad"],
                features["local_left_turn_ratio"],
                features["local_right_turn_ratio"],
                features["local_max_abs_turn_rad"],
                features["local_turn_count"],
            ]
            selected = values[0]
            while metric_ids.ndim < selected.ndim:
                metric_ids = metric_ids.unsqueeze(-1)
            for metric_id, value in enumerate(values[1:], start=1):
                selected = torch.where(metric_ids == metric_id, value, selected)
            return selected

        def _select_scalar_value(values: list[torch.Tensor], metric_ids: torch.Tensor) -> torch.Tensor:
            selected = values[0]
            ids = metric_ids
            while ids.ndim < selected.ndim:
                ids = ids.unsqueeze(-1)
            for metric_id, value in enumerate(values[1:], start=1):
                selected = torch.where(ids == metric_id, value, selected)
            return selected

        base_scalar_values = [
            base_distance,
            base_ref,
            base_ref_m,
            base_curv,
            base_contract_curv,
            base_curv_energy,
            base_progress,
            base_displacement_ratio,
            base_length_ratio,
        ]
        cf_scalar_values = [
            cf_distance,
            cf_ref,
            cf_ref_m,
            cf_curv,
            cf_contract_curv,
            cf_curv_energy,
            cf_progress,
            cf_displacement_ratio,
            cf_length_ratio,
        ]

        def _base_token_score(
            field_center: torch.Tensor,
            field_scale: torch.Tensor,
            field_metric: torch.Tensor,
            token_ids: torch.Tensor,
        ) -> torch.Tensor:
            value = _select_scalar_value(base_scalar_values, field_metric.to(device=device)[token_ids])
            center = field_center.to(device=device, dtype=dtype)[token_ids].view(batch, 1)
            scale = field_scale.to(device=device, dtype=dtype)[token_ids].view(batch, 1).clamp_min(1e-6)
            return ((value - center) / scale).abs()

        def _cf_token_score(
            field_center: torch.Tensor,
            field_scale: torch.Tensor,
            field_metric: torch.Tensor,
            token_ids: torch.Tensor,
        ) -> torch.Tensor:
            value = _select_scalar_value(cf_scalar_values, field_metric.to(device=device)[token_ids])
            center = field_center.to(device=device, dtype=dtype)[token_ids].view(batch, 1)
            scale = field_scale.to(device=device, dtype=dtype)[token_ids].view(batch, 1).clamp_min(1e-6)
            return ((value - center) / scale).abs()

        def _scalar_metric_delta(
            field_scale: torch.Tensor,
            field_metric: torch.Tensor,
            token_ids: torch.Tensor,
        ) -> torch.Tensor:
            metric_ids = field_metric.to(device=device)[token_ids]
            base_value = _select_scalar_value(base_scalar_values, metric_ids)
            cf_value = _select_scalar_value(cf_scalar_values, metric_ids)
            scale = field_scale.to(device=device, dtype=dtype)[token_ids].view(batch, 1).clamp_min(1e-6)
            return (cf_value - base_value.expand_as(cf_value)).abs() / scale

        target_distance_score = _cf_token_score(
            self.calib_distance_center, self.calib_distance_scale, self.calib_distance_metric, cf_distance_id
        )
        target_ref_score = _cf_token_score(self.calib_ref_center, self.calib_ref_scale, self.calib_ref_metric, cf_ref_id)
        target_curv_score = _cf_token_score(
            self.calib_curv_center, self.calib_curv_scale, self.calib_curv_metric, cf_curv_id
        )
        target_progress_score = _cf_token_score(
            self.calib_progress_center, self.calib_progress_scale, self.calib_progress_metric, cf_progress_id
        )
        cf_direction_value = _select_direction_value(cf_heading, cf_direction_id)
        target_direction_score = (
            (cf_direction_value - self.calib_direction_center.to(device=device, dtype=dtype)[cf_direction_id].view(batch, 1))
            / self.calib_direction_scale.to(device=device, dtype=dtype)[cf_direction_id].view(batch, 1).clamp_min(1e-6)
        ).abs()

        base_target_distance_score = _base_token_score(
            self.calib_distance_center, self.calib_distance_scale, self.calib_distance_metric, cf_distance_id
        )
        base_target_ref_score = _base_token_score(self.calib_ref_center, self.calib_ref_scale, self.calib_ref_metric, cf_ref_id)
        base_target_curv_score = _base_token_score(
            self.calib_curv_center, self.calib_curv_scale, self.calib_curv_metric, cf_curv_id
        )
        base_target_progress_score = _base_token_score(
            self.calib_progress_center, self.calib_progress_scale, self.calib_progress_metric, cf_progress_id
        )
        base_direction_value = _select_direction_value(base_heading, cf_direction_id).view(batch, 1)
        base_target_direction_score = (
            (base_direction_value - self.calib_direction_center.to(device=device, dtype=dtype)[cf_direction_id].view(batch, 1))
            / self.calib_direction_scale.to(device=device, dtype=dtype)[cf_direction_id].view(batch, 1).clamp_min(1e-6)
        ).abs()

        target_scores = torch.stack(
            [target_distance_score, target_direction_score, target_curv_score, target_progress_score, target_ref_score],
            dim=-1,
        )
        base_target_scores = torch.stack(
            [
                base_target_distance_score.expand_as(target_distance_score),
                base_target_direction_score.expand_as(target_direction_score),
                base_target_curv_score.expand_as(target_curv_score),
                base_target_progress_score.expand_as(target_progress_score),
                base_target_ref_score.expand_as(target_ref_score),
            ],
            dim=-1,
        )
        target_feature_ids = field_ids.to(device=device, dtype=torch.long).clamp(0, 4).view(batch, 1, 1)
        target_score = target_scores.gather(2, target_feature_ids.expand(-1, num_candidates, 1)).squeeze(-1)
        base_target_score = base_target_scores.gather(2, target_feature_ids.expand(-1, num_candidates, 1)).squeeze(-1)
        target_effect = (base_target_score - target_score).clamp(min=-10.0, max=10.0)
        base_contract_direction_scores_for_target = self._contract_direction_all_token_scores(
            base_heading,
            dtype=dtype,
        )
        cf_contract_direction_scores_for_target = self._contract_direction_all_token_scores(
            cf_heading,
            dtype=dtype,
        )
        base_contract_target_direction_score = base_contract_direction_scores_for_target.gather(
            -1,
            cf_direction_id.view(batch, 1),
        ).squeeze(-1)
        cf_contract_target_direction_score = cf_contract_direction_scores_for_target.gather(
            -1,
            cf_direction_id.view(batch, 1, 1).expand(-1, num_candidates, 1),
        ).squeeze(-1)
        contract_direction_target_effect = (
            base_contract_target_direction_score.view(batch, 1).expand_as(cf_contract_target_direction_score)
            - cf_contract_target_direction_score
        ).clamp(min=-10.0, max=10.0)
        direction_field_idx = DEFAULT_MOBILITY_VOCAB.fields.index("direction")
        target_effect = torch.where(
            field_ids.to(device=device, dtype=torch.long).view(batch, 1) == int(direction_field_idx),
            contract_direction_target_effect,
            target_effect,
        )

        base_distance_score = _base_token_score(
            self.calib_distance_center, self.calib_distance_scale, self.calib_distance_metric, base_distance_id
        )
        distance_delta = _scalar_metric_delta(self.calib_distance_scale, self.calib_distance_metric, base_distance_id)
        base_ref_score = _base_token_score(self.calib_ref_center, self.calib_ref_scale, self.calib_ref_metric, base_ref_id)
        ref_delta = _scalar_metric_delta(self.calib_ref_scale, self.calib_ref_metric, base_ref_id)
        base_curv_score = _base_token_score(
            self.calib_curv_center, self.calib_curv_scale, self.calib_curv_metric, base_curv_id
        )
        curv_delta = _scalar_metric_delta(self.calib_curv_scale, self.calib_curv_metric, base_curv_id)
        base_progress_score = _base_token_score(
            self.calib_progress_center, self.calib_progress_scale, self.calib_progress_metric, base_progress_id
        )
        progress_delta = _scalar_metric_delta(self.calib_progress_scale, self.calib_progress_metric, base_progress_id)
        base_self_direction_value = _select_direction_value(base_heading, base_direction_id).view(batch, 1)
        base_direction_score = (
            (base_self_direction_value - self.calib_direction_center.to(device=device, dtype=dtype)[base_direction_id].view(batch, 1))
            / self.calib_direction_scale.to(device=device, dtype=dtype)[base_direction_id].view(batch, 1).clamp_min(1e-6)
        ).abs()
        cf_base_direction_value = _select_direction_value(cf_heading, base_direction_id)
        cf_direction_score = (
            (cf_base_direction_value - self.calib_direction_center.to(device=device, dtype=dtype)[base_direction_id].view(batch, 1))
            / self.calib_direction_scale.to(device=device, dtype=dtype)[base_direction_id].view(batch, 1).clamp_min(1e-6)
        ).abs()
        base_contract_direction_scores = self._contract_direction_all_token_scores(
            base_heading,
            dtype=dtype,
        )
        cf_contract_direction_scores = self._contract_direction_all_token_scores(
            cf_heading,
            dtype=dtype,
        )
        base_contract_direction_score = base_contract_direction_scores.gather(
            -1,
            base_direction_id.view(batch, 1),
        ).squeeze(-1)
        cf_contract_direction_score = cf_contract_direction_scores.gather(
            -1,
            base_direction_id.view(batch, 1, 1).expand(-1, num_candidates, 1),
        ).squeeze(-1)
        direction_delta = (
            cf_contract_direction_score
            - base_contract_direction_score.view(batch, 1).expand_as(cf_contract_direction_score)
        ).abs()
        collateral_terms = torch.stack(
            [
                distance_delta,
                direction_delta,
                curv_delta,
                progress_delta,
                ref_delta,
            ],
            dim=-1,
        )
        # Match Table-4 independent collateral: structurally coupled fields are
        # excluded instead of penalizing the intended geometry change.
        collateral_keep_by_field = candidates.new_tensor(
            [
                # distance edit: keep direction + curvature.
                [0.0, 1.0, 1.0, 0.0, 0.0],
                # direction edit: all other fields are structurally coupled.
                [0.0, 0.0, 0.0, 0.0, 0.0],
                # curvature edit: all other fields are structurally coupled.
                [0.0, 0.0, 0.0, 0.0, 0.0],
                # progress edit: keep direction + curvature + reference.
                [0.0, 1.0, 1.0, 0.0, 1.0],
                # reference edit: keep direction + curvature.
                [0.0, 1.0, 1.0, 0.0, 0.0],
            ],
            dtype=dtype,
        )
        if self.counterfactual_eval_selector_collateral_mode == "raw":
            collateral_keep_by_field = 1.0 - torch.eye(
                len(DEFAULT_MOBILITY_VOCAB.fields),
                device=device,
                dtype=dtype,
            )
        elif self.counterfactual_eval_selector_collateral_mode == "curvature_raw":
            raw_curvature_keep = candidates.new_tensor([1.0, 1.0, 0.0, 1.0, 1.0], dtype=dtype)
            collateral_keep_by_field = collateral_keep_by_field.clone()
            collateral_keep_by_field[2] = raw_curvature_keep
        elif self.counterfactual_eval_selector_collateral_mode != "independent":
            raise ValueError(
                "Unsupported counterfactual_eval_selector_collateral_mode="
                f"{self.counterfactual_eval_selector_collateral_mode}"
            )
        collateral_field_mask = collateral_keep_by_field[
            field_ids.to(device=device, dtype=torch.long).clamp(0, 4)
        ]
        collateral = (
            collateral_terms * collateral_field_mask.view(batch, 1, -1)
        ).sum(dim=-1) / collateral_field_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        collateral = collateral.clamp(max=10.0)
        raw_collateral_keep = (
            1.0
            - torch.eye(
                len(DEFAULT_MOBILITY_VOCAB.fields),
                device=device,
                dtype=dtype,
            )
        )
        raw_collateral_field_mask = raw_collateral_keep[
            field_ids.to(device=device, dtype=torch.long).clamp(0, 4)
        ]
        raw_collateral = (
            collateral_terms * raw_collateral_field_mask.view(batch, 1, -1)
        ).sum(dim=-1) / raw_collateral_field_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        raw_collateral = raw_collateral.clamp(max=10.0)
        if base_selected_index is None:
            base_selected_index = torch.zeros(batch, device=device, dtype=torch.long)
        mode_ids = torch.arange(num_candidates, device=device).view(1, -1)
        mode_switch = (mode_ids != base_selected_index.to(device=device, dtype=torch.long).view(-1, 1)).to(dtype=dtype)
        ref_scale = self.calib_ref_scale.to(device=device, dtype=dtype)[cf_ref_id].view(batch, 1).clamp_min(1e-6)
        ref_reduction = ((base_ref - cf_ref) / ref_scale).clamp(min=-10.0, max=10.0)
        ref_ratio = (cf_ref / base_ref.clamp_min(1e-6)).clamp(max=10.0)
        reference_field_idx = DEFAULT_MOBILITY_VOCAB.fields.index("reference_adherence")
        target_effect = torch.where(
            field_ids.to(device=device, dtype=torch.long).view(batch, 1) == int(reference_field_idx),
            ref_reduction,
            target_effect,
        )
        field_feature = field_ids.to(device=device, dtype=dtype).view(batch, 1).expand(-1, num_candidates)
        return torch.stack(
            [
                target_effect,
                target_score.clamp(max=10.0),
                collateral,
                mode_switch,
                field_feature,
                base_ref.expand_as(cf_ref).clamp(max=10.0),
                cf_ref.clamp(max=10.0),
                ref_reduction,
                ref_ratio,
                raw_collateral,
            ],
            dim=-1,
        )

    def counterfactual_eval_selector_logits(
        self,
        pair_features: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Table-4-style explicit selector built from inference-available pair features.

        pair_features[..., 0] is target effect; larger is better.
        pair_features[..., 1] is target-token score; lower is better.
        pair_features[..., 2] is collateral drift; lower is better.
        """
        if pair_features is None or pair_features.ndim != 3 or pair_features.shape[-1] < 3:
            return None, None, None
        target_effect = pair_features[..., 0]
        target_score = pair_features[..., 1]
        collateral = pair_features[..., 2]
        raw_collateral = pair_features[..., 9] if pair_features.shape[-1] > 9 else None
        mode_switch = pair_features[..., 3] if pair_features.shape[-1] > 3 else None
        field_id = pair_features[..., 4] if pair_features.shape[-1] > 4 else None
        ref_reduction = pair_features[..., 7] if pair_features.shape[-1] > 7 else None
        cost = (
            float(self.counterfactual_eval_selector_instruction_weight) * target_score
            - float(self.counterfactual_eval_selector_target_effect_weight) * target_effect
            + float(self.counterfactual_eval_selector_collateral_weight) * collateral
        )
        if raw_collateral is not None:
            cost = cost + cost.new_tensor(0.25) * raw_collateral
        if (
            field_id is not None
            and ref_reduction is not None
            and (
                float(self.counterfactual_eval_selector_reference_reduction_weight) != 0.0
                or float(self.counterfactual_eval_selector_reference_score_weight) != 0.0
            )
        ):
            reference_field_id = float(DEFAULT_MOBILITY_VOCAB.fields.index("reference_adherence"))
            ref_cost = (
                float(self.counterfactual_eval_selector_reference_score_weight) * target_score
                - float(self.counterfactual_eval_selector_target_effect_weight) * target_effect
                - float(self.counterfactual_eval_selector_reference_reduction_weight) * ref_reduction
                + float(self.counterfactual_eval_selector_collateral_weight) * collateral
            )
            cost = torch.where(field_id == cost.new_tensor(reference_field_id), ref_cost, cost)
        if field_id is not None:
            reference_field_id = float(DEFAULT_MOBILITY_VOCAB.fields.index("reference_adherence"))
            reference_mask = field_id == cost.new_tensor(reference_field_id)
            min_reference_target_effect = cost.new_tensor(0.02)
            near_noop_reference = reference_mask & (target_effect < min_reference_target_effect)
            cost = torch.where(
                near_noop_reference,
                cost + cost.new_tensor(0.35) * (min_reference_target_effect - target_effect),
                cost,
            )
        if mode_switch is not None and float(self.counterfactual_eval_selector_mode_switch_weight) != 0.0:
            cost = cost + float(self.counterfactual_eval_selector_mode_switch_weight) * mode_switch
        if (
            mode_switch is not None
            and field_id is not None
            and float(self.counterfactual_eval_selector_reference_mode_switch_weight) != 0.0
        ):
            reference_field_id = float(DEFAULT_MOBILITY_VOCAB.fields.index("reference_adherence"))
            reference_mask = field_id == cost.new_tensor(reference_field_id)
            cost = cost + torch.where(
                reference_mask,
                float(self.counterfactual_eval_selector_reference_mode_switch_weight) * mode_switch,
                cost.new_zeros(cost.shape),
            )
        feasible = (
            target_effect > float(self.counterfactual_eval_selector_success_margin)
        ) & (
            collateral <= float(self.counterfactual_eval_selector_collateral_margin)
        )
        if raw_collateral is not None:
            feasible = feasible & (raw_collateral <= cost.new_tensor(0.15))
        if bool(self.counterfactual_eval_selector_feasible_first):
            has_feasible = feasible.any(dim=1, keepdim=True)
            cost = torch.where(
                has_feasible & (~feasible),
                cost + cost.new_tensor(100.0),
                cost,
            )
        logits = -cost
        return logits, cost, feasible.to(dtype=pair_features.dtype)

    def build_candidate_family_metadata(
        self,
        batch: int,
        counterfactual_field_ids: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        control_family_prefixes: Optional[list[str]] = None,
        control_candidate_act_ids: Optional[torch.Tensor] = None,
    ) -> tuple[list[list[str]], torch.Tensor, torch.Tensor]:
        base_mask = torch.ones(batch, self.num_candidates, device=device, dtype=dtype)
        control_mask = torch.zeros(batch, self.num_candidates, device=device, dtype=dtype)
        family_ids = [["base:generated" for _ in range(self.num_candidates)] for _ in range(batch)]
        if (
            not bool(self.base_preserving_selector_enabled)
            or counterfactual_field_ids is None
            or self.num_candidates <= 1
        ):
            return family_ids, base_mask, control_mask

        split = max(1, self.num_candidates // 2)
        base_mask[:, split:] = 0.0
        control_mask[:, split:] = 1.0
        base_mask[:, :split] = 1.0
        control_mask[:, :split] = 0.0
        field_names = list(DEFAULT_MOBILITY_VOCAB.fields)
        for row in range(batch):
            field_idx = int(counterfactual_field_ids[row].item())
            field_name = field_names[field_idx] if 0 <= field_idx < len(field_names) else "unknown"
            row_ids = []
            for mode_idx in range(self.num_candidates):
                if mode_idx < split:
                    row_ids.append(f"base:{field_name}:preserve")
                else:
                    control_prefix = None
                    if control_family_prefixes is not None and row < len(control_family_prefixes):
                        control_prefix = control_family_prefixes[row]
                    if not control_prefix:
                        control_prefix = f"{field_name}:local_edit"
                    candidate_label = f"control:{control_prefix}"
                    if (
                        control_candidate_act_ids is not None
                        and control_candidate_act_ids.ndim == 2
                        and row < control_candidate_act_ids.shape[0]
                        and mode_idx < control_candidate_act_ids.shape[1]
                    ):
                        act_id = int(control_candidate_act_ids[row, mode_idx].item())
                        if act_id >= 0:
                            candidate_label = f"{candidate_label}:act{act_id}"
                    row_ids.append(candidate_label)
            family_ids[row] = row_ids
        return family_ids, base_mask, control_mask

    def build_field_control_routing(
        self,
        counterfactual_field_ids: Optional[torch.Tensor],
        mobility_tokens: Optional[torch.Tensor],
        base_mobility_tokens: Optional[torch.Tensor],
        field_control_logits: Optional[dict[str, torch.Tensor]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, object]:
        batch = 0 if counterfactual_field_ids is None else int(counterfactual_field_ids.shape[0])
        candidate_bonus = torch.zeros(batch, self.num_candidates, device=device, dtype=dtype)
        proposal_scale = torch.ones(batch, 1, 1, 1, device=device, dtype=dtype)
        proposal_shift = torch.zeros(batch, 1, 1, 1, device=device, dtype=dtype)
        control_family_prefixes = [None for _ in range(batch)]
        control_source_kinds = ["generated" for _ in range(batch)]
        control_candidate_act_ids = torch.full(
            (batch, self.num_candidates),
            -1,
            device=device,
            dtype=torch.long,
        )
        preferred_act_ids = torch.full((batch,), -1, device=device, dtype=torch.long)
        if counterfactual_field_ids is None or mobility_tokens is None:
            return {
                "candidate_logit_bonus": candidate_bonus,
                "proposal_scale": proposal_scale,
                "proposal_shift": proposal_shift,
                "control_family_prefixes": control_family_prefixes,
                "control_source_kinds": control_source_kinds,
                "control_candidate_act_ids": control_candidate_act_ids,
                "preferred_act_ids": preferred_act_ids,
            }

        direction_field_idx = DEFAULT_MOBILITY_VOCAB.fields.index("direction")
        curvature_field_idx = DEFAULT_MOBILITY_VOCAB.fields.index("curvature")
        direction_target_ids = mobility_tokens[:, direction_field_idx].to(device=device, dtype=torch.long)
        curvature_target_ids = mobility_tokens[:, curvature_field_idx].to(device=device, dtype=torch.long)
        if (
            base_mobility_tokens is not None
            and base_mobility_tokens.ndim == mobility_tokens.ndim
            and base_mobility_tokens.shape[0] == mobility_tokens.shape[0]
        ):
            direction_source_ids = base_mobility_tokens[:, direction_field_idx].to(device=device, dtype=torch.long)
            curvature_source_ids = base_mobility_tokens[:, curvature_field_idx].to(device=device, dtype=torch.long)
        else:
            direction_source_ids = direction_target_ids
            curvature_source_ids = curvature_target_ids

        multi_turn_id = int(DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]["multi_turn"])
        left_turn_id = int(DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]["left_turn"])
        right_turn_id = int(DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]["right_turn"])
        winding_id = int(DEFAULT_MOBILITY_VOCAB.token_to_id["curvature"]["winding"])
        smooth_id = int(DEFAULT_MOBILITY_VOCAB.token_to_id["curvature"]["smooth"])
        act_count_default = max(1, int(self.field_control_num_act_classes))

        for row in range(batch):
            field_idx = int(counterfactual_field_ids[row].item())
            field_name = (
                DEFAULT_MOBILITY_VOCAB.fields[field_idx]
                if 0 <= field_idx < len(DEFAULT_MOBILITY_VOCAB.fields)
                else "unknown"
            )
            if field_name == "direction":
                source_id = int(direction_source_ids[row].item())
                target_id = int(direction_target_ids[row].item())
                source_name = self.counterfactual_token_name("direction", source_id)
                target_name = self.counterfactual_token_name("direction", target_id)
                if target_id == multi_turn_id:
                    control_family_prefixes[row] = f"direction:{source_name}->{target_name}_local"
                    control_source_kinds[row] = "local_proposal"
                elif (
                    (source_id == left_turn_id and target_id == right_turn_id)
                    or (source_id == right_turn_id and target_id == left_turn_id)
                ):
                    control_family_prefixes[row] = f"direction:{source_name}->{target_name}_reversal"
                    control_source_kinds[row] = "shared_direction"
                else:
                    control_family_prefixes[row] = f"direction:{source_name}->{target_name}_compose"
                    control_source_kinds[row] = "shared_direction"
            elif field_name == "curvature":
                source_id = int(curvature_source_ids[row].item())
                target_id = int(curvature_target_ids[row].item())
                source_name = self.counterfactual_token_name("curvature", source_id)
                target_name = self.counterfactual_token_name("curvature", target_id)
                if target_id == winding_id:
                    control_family_prefixes[row] = f"curvature:{source_name}->{target_name}_local"
                    control_source_kinds[row] = "local_proposal"
                else:
                    control_family_prefixes[row] = f"curvature:{source_name}->{target_name}_compose"
                    control_source_kinds[row] = "shared_curvature"
            elif field_name == "reference_adherence":
                ref_field_idx = DEFAULT_MOBILITY_VOCAB.fields.index("reference_adherence")
                target_id = int(mobility_tokens[row, ref_field_idx].item())
                target_name = self.counterfactual_token_name("reference_adherence", target_id)
                control_family_prefixes[row] = f"reference_adherence:{target_name}_local"
                control_source_kinds[row] = "local_proposal"
            else:
                control_family_prefixes[row] = f"{field_name}:compose"
                control_source_kinds[row] = "generated"

            logits = None if field_control_logits is None else field_control_logits.get(field_name)
            if logits is None or logits.ndim != 2 or row >= logits.shape[0]:
                act_count = min(act_count_default, max(1, self.num_candidates))
                control_candidate_act_ids[row] = torch.arange(
                    self.num_candidates,
                    device=device,
                    dtype=torch.long,
                ) % act_count
                continue

            row_logits = logits[row].to(device=device, dtype=dtype).flatten()
            act_count = min(int(row_logits.numel()), act_count_default)
            if act_count <= 0:
                continue
            row_logits = row_logits[:act_count]
            act_ids = torch.arange(self.num_candidates, device=device, dtype=torch.long) % act_count
            control_candidate_act_ids[row] = act_ids
            preferred_act = int(torch.argmax(row_logits).item())
            preferred_act_ids[row] = preferred_act
            centered_logits = row_logits - row_logits.mean()
            scale = centered_logits.abs().amax().clamp_min(1e-6)
            candidate_bonus[row] = 0.25 * torch.tanh(centered_logits[act_ids] / scale)
            probs = torch.softmax(row_logits, dim=0)
            confidence = probs.max()
            proposal_scale[row, 0, 0, 0] = 0.95 + 0.35 * confidence
            if act_count > 1:
                proposal_shift[row, 0, 0, 0] = -0.35 + 0.70 * (
                    float(preferred_act) / float(act_count - 1)
                )

        return {
            "candidate_logit_bonus": candidate_bonus,
            "proposal_scale": proposal_scale,
            "proposal_shift": proposal_shift,
            "control_family_prefixes": control_family_prefixes,
            "control_source_kinds": control_source_kinds,
            "control_candidate_act_ids": control_candidate_act_ids,
            "preferred_act_ids": preferred_act_ids,
        }

    def build_base_preservation_scores(
        self,
        candidates: torch.Tensor,
        anchor: torch.Tensor,
        base_selected_trajectory: Optional[torch.Tensor],
    ) -> torch.Tensor:
        target = anchor if base_selected_trajectory is None else base_selected_trajectory.to(
            device=candidates.device,
            dtype=candidates.dtype,
        )
        diff = candidates - target.unsqueeze(1)
        return torch.linalg.norm(diff, dim=-1).mean(dim=-1)

    def build_pairwise_preference_labels(
        self,
        pair_features: Optional[torch.Tensor],
        feasible: Optional[torch.Tensor],
        base_mask: torch.Tensor,
        control_mask: torch.Tensor,
    ) -> Optional[dict[str, torch.Tensor]]:
        if (
            not bool(self.pairwise_feasible_preference_enabled)
            or pair_features is None
            or pair_features.ndim != 3
            or pair_features.shape[-1] < 3
        ):
            return None
        target_effect = pair_features[..., 0]
        collateral = pair_features[..., 2]
        utility = target_effect - collateral
        chosen_scores = utility.clone()
        if feasible is not None:
            chosen_scores = torch.where(
                feasible > 0,
                chosen_scores,
                chosen_scores - chosen_scores.new_tensor(100.0),
            )
        chosen_index = chosen_scores.argmax(dim=1)
        rejected_scores = target_effect + collateral
        rejected_index = rejected_scores.argmax(dim=1)
        weak_negative_scores = torch.where(
            base_mask > 0,
            -target_effect + 0.25 * collateral,
            rejected_scores.new_full(rejected_scores.shape, -1e6),
        )
        weak_negative_index = weak_negative_scores.argmax(dim=1)
        return {
            "chosen_index": chosen_index,
            "rejected_index": rejected_index,
            "weak_negative_index": weak_negative_index,
            "has_feasible": (
                feasible.any(dim=1)
                if feasible is not None
                else torch.zeros(pair_features.shape[0], device=pair_features.device, dtype=torch.bool)
            ),
            "control_candidate_count": (control_mask > 0).sum(dim=1).to(dtype=pair_features.dtype),
        }

    def build_selector_term_scores(
        self,
        pair_features: Optional[torch.Tensor],
        base_preservation_scores: torch.Tensor,
        feasible: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        target_term = base_preservation_scores.new_zeros(base_preservation_scores.shape)
        collateral_term = base_preservation_scores.new_zeros(base_preservation_scores.shape)
        feasible_term = base_preservation_scores.new_zeros(base_preservation_scores.shape)
        if pair_features is not None and pair_features.ndim == 3 and pair_features.shape[-1] >= 3:
            target_effect = pair_features[..., 0]
            target_score = pair_features[..., 1]
            collateral = pair_features[..., 2]
            target_term = (
                float(self.counterfactual_eval_selector_target_effect_weight) * target_effect
                - float(self.counterfactual_eval_selector_instruction_weight) * target_score
            )
            collateral_term = -float(self.counterfactual_eval_selector_collateral_weight) * collateral
        feasible_term = -base_preservation_scores
        if feasible is not None:
            feasible_term = feasible_term + 0.25 * feasible.to(dtype=base_preservation_scores.dtype)
        return {
            "target_effect": target_term,
            "collateral": collateral_term,
            "preserve_base": feasible_term,
        }

    @staticmethod
    def build_comparative_features(features: torch.Tensor, lower_is_better: bool = True) -> torch.Tensor:
        mean_delta = features - features.mean(dim=1, keepdim=True)
        min_delta = features - features.amin(dim=1, keepdim=True)
        order = torch.argsort(features, dim=1, descending=not bool(lower_is_better))
        rank = torch.empty_like(order, dtype=features.dtype)
        rank_values = torch.arange(features.shape[1], device=features.device, dtype=features.dtype).view(1, -1, 1)
        rank.scatter_(1, order, rank_values.expand(features.shape[0], -1, features.shape[-1]))
        rank = rank / float(max(features.shape[1] - 1, 1))
        return torch.cat([features, mean_delta, min_delta, rank], dim=-1)

    def forward(
        self,
        reference: torch.Tensor,
        fused_cond: torch.Tensor,
        cond_tokens: Optional[torch.Tensor] = None,
        route_tokens: Optional[torch.Tensor] = None,
        route_token_condition_weight: float = 1.0,
        route_anchor: Optional[torch.Tensor] = None,
        route_token_anchor_weight: float = 0.0,
        mode_route_anchor: Optional[torch.Tensor] = None,
        mode_route_anchor_weight: float = 0.0,
        mode_route_residual_scale: float = 1.0,
        mobility_tokens: Optional[torch.Tensor] = None,
        target_len: Optional[int] = None,
        counterfactual_mode: bool = False,
        counterfactual_adapter_scale: float = 0.0,
        counterfactual_field_ids: Optional[torch.Tensor] = None,
        base_selected_trajectory: Optional[torch.Tensor] = None,
        base_mobility_tokens: Optional[torch.Tensor] = None,
        route_scales: Optional[torch.Tensor] = None,
        base_selected_index: Optional[torch.Tensor] = None,
        trip_duration_min: Optional[torch.Tensor] = None,
        semantic_global_cond: Optional[torch.Tensor] = None,
        field_control_latents: Optional[dict[str, torch.Tensor]] = None,
        field_control_logits: Optional[dict[str, torch.Tensor]] = None,
        return_aux: bool = False,
    ):
        if fused_cond.ndim != 2 or fused_cond.shape[-1] != self.cond_dim:
            raise ValueError(f"fused_cond must be [B, {self.cond_dim}], got {tuple(fused_cond.shape)}")
        target_len = int(target_len or self.target_len)
        if target_len != self.target_len:
            raise ValueError(f"DiTAnchorDecoder currently expects target_len={self.target_len}, got {target_len}")
        mode_control_strength = float(max(0.0, min(1.0, self.mode_control_strength)))

        anchor = self.resample_reference(reference.float(), target_len)
        route_anchor_up = None
        if route_anchor is not None and float(route_token_anchor_weight) != 0.0:
            if route_anchor.ndim != 3 or route_anchor.shape[0] != reference.shape[0] or route_anchor.shape[-1] != 2:
                raise ValueError(f"route_anchor must be [B, P, 2], got {tuple(route_anchor.shape)}")
            route_anchor_up = self.resample_temporal(
                route_anchor.to(device=anchor.device, dtype=anchor.dtype),
                target_len,
            )
            route_anchor_delta = self.preserve_reference_endpoints(route_anchor_up - anchor)
            anchor = anchor + float(route_token_anchor_weight) * route_anchor_delta
        mode_route_anchor_up = None
        tangent, normal = self.tangent_normal(anchor)
        time = self.normalized_time(anchor.shape[0], target_len, anchor.device, anchor.dtype)
        point_features = torch.cat([anchor, tangent, normal, time, 1.0 - time], dim=-1)
        prepared_base_selected_trajectory = self.prepare_base_selected_trajectory(
            base_selected_trajectory,
            batch=anchor.shape[0],
            target_len=target_len,
            device=anchor.device,
            dtype=anchor.dtype,
            base_selected_index=base_selected_index,
        )

        cond = self.cond_proj(fused_cond)
        cf_adapter_scale = float(counterfactual_adapter_scale) if counterfactual_mode else 0.0
        cf_field_bias = None
        if cf_adapter_scale != 0.0:
            if counterfactual_field_ids is None:
                cf_field_bias = torch.zeros_like(cond)
            else:
                if counterfactual_field_ids.ndim != 1 or counterfactual_field_ids.shape[0] != cond.shape[0]:
                    raise ValueError(
                        "counterfactual_field_ids must be [B] when counterfactual_mode is enabled, "
                        f"got {tuple(counterfactual_field_ids.shape)}"
                    )
                cf_field_bias = self.counterfactual_field_embed(
                    counterfactual_field_ids.to(device=cond.device, dtype=torch.long)
                ).to(dtype=cond.dtype)
        if cf_adapter_scale != 0.0:
            cf_cond_input = cond + cf_field_bias
            cond = cond + cf_adapter_scale * (
                self.counterfactual_cond_adapter(cf_cond_input)
                + self.apply_counterfactual_field_adapter(
                    cf_cond_input,
                    counterfactual_field_ids,
                    self.counterfactual_field_cond_adapters,
                )
            )
        hidden_base = self.point_embed(point_features) + cond.unsqueeze(1) + self.pos_embed.to(anchor.dtype)
        route_skeleton_hidden_base = hidden_base
        if self.route_skeleton_condition_source == "od_only":
            route_skeleton_hidden_base = self.point_embed(point_features) + self.pos_embed.to(anchor.dtype)
        token_seq = None
        if cond_tokens is not None:
            if cond_tokens.ndim != 3 or cond_tokens.shape[0] != reference.shape[0]:
                raise ValueError(f"cond_tokens must be [B, F, D], got {tuple(cond_tokens.shape)}")
            if cond_tokens.shape[-1] != self.cond_dim:
                raise ValueError(f"cond_tokens hidden dim mismatch: expected {self.cond_dim}, got {cond_tokens.shape[-1]}")
            cond_tokens = self.token_cond_proj(cond_tokens)
            if cf_adapter_scale != 0.0:
                cf_token_input = cond_tokens + cf_field_bias.unsqueeze(1)
                cond_tokens = cond_tokens + cf_adapter_scale * (
                    self.counterfactual_token_adapter(cf_token_input)
                    + self.apply_counterfactual_field_adapter(
                        cf_token_input,
                        counterfactual_field_ids,
                        self.counterfactual_field_token_adapters,
                    )
                )
            token_seq = cond_tokens + self.cond_token_pos_embed(
                cond_tokens.shape[1],
                device=cond_tokens.device,
                dtype=cond_tokens.dtype,
            )
        route_seq = None
        if route_tokens is not None:
            if route_tokens.ndim != 3 or route_tokens.shape[0] != reference.shape[0]:
                raise ValueError(f"route_tokens must be [B, F, D], got {tuple(route_tokens.shape)}")
            if route_tokens.shape[-1] != self.cond_dim:
                raise ValueError(f"route_tokens hidden dim mismatch: expected {self.cond_dim}, got {route_tokens.shape[-1]}")
            route_seq = self.route_token_cond_proj(route_tokens)
            route_seq = route_seq + self.cond_token_pos_embed(
                route_seq.shape[1],
                device=route_seq.device,
                dtype=route_seq.dtype,
            )
        hidden = hidden_base.unsqueeze(1) + self.candidate_embed.view(1, self.num_candidates, 1, self.hidden_dim)
        route_skeleton_hidden = (
            route_skeleton_hidden_base.unsqueeze(1)
            + self.candidate_embed.view(1, self.num_candidates, 1, self.hidden_dim)
        )
        mode_role = None
        if self.mode_control_type in {"token_role_residual_gate", "mode_specific_heads", "role_gated_heads"}:
            if self.mode_role_proj is None:
                raise RuntimeError(f"{self.mode_control_type} requires mode_role_proj.")
            mode_role = self.mode_role_proj(
                self.mode_role_table.to(device=cond.device, dtype=cond.dtype)
            ).view(1, self.num_candidates, self.hidden_dim)
            hidden = hidden + (0.25 * mode_control_strength) * mode_role.unsqueeze(2)
        hidden = hidden.reshape(anchor.shape[0] * self.num_candidates, target_len, self.hidden_dim)
        route_skeleton_hidden = route_skeleton_hidden.reshape(
            anchor.shape[0] * self.num_candidates,
            target_len,
            self.hidden_dim,
        )
        cond_rep = cond.repeat_interleave(self.num_candidates, dim=0)
        token_seq_rep = token_seq.repeat_interleave(self.num_candidates, dim=0) if token_seq is not None else None
        route_seq_rep = route_seq.repeat_interleave(self.num_candidates, dim=0) if route_seq is not None else None
        anchor_rep = anchor.repeat_interleave(self.num_candidates, dim=0)
        tangent_rep = tangent.repeat_interleave(self.num_candidates, dim=0)
        normal_rep = normal.repeat_interleave(self.num_candidates, dim=0)

        for block in self.blocks:
            hidden = block(hidden, cond_rep, token_seq_rep)
            if route_seq_rep is not None:
                route_gate = torch.sigmoid(self.route_token_gate_logit) * float(route_token_condition_weight)
                hidden = block.route_fuse(hidden, route_seq_rep, route_gate)
            if self.route_skeleton_condition_source == "od_only":
                route_skeleton_hidden = block(route_skeleton_hidden, torch.zeros_like(cond_rep), None)

        shift, scale = self.final_modulation(cond_rep).chunk(2, dim=1)
        hidden = modulate(self.final_norm(hidden), shift, scale)
        if self.route_skeleton_condition_source == "od_only":
            route_shift, route_scale = self.final_modulation(torch.zeros_like(cond_rep)).chunk(2, dim=1)
            route_skeleton_hidden = modulate(
                self.final_norm(route_skeleton_hidden),
                route_shift,
                route_scale,
            )
        else:
            route_skeleton_hidden = hidden
        raw = self.residual_head(hidden)
        mode_specific_raw_delta = None
        if self.mode_specific_residual_heads is not None:
            hidden_view = hidden.view(anchor.shape[0], self.num_candidates, target_len, self.hidden_dim)
            mode_specific_raw_delta = torch.stack(
                [
                    head(hidden_view[:, mode_idx])
                    for mode_idx, head in enumerate(self.mode_specific_residual_heads)
                ],
                dim=1,
            )
            raw = raw + mode_control_strength * mode_specific_raw_delta.reshape(
                anchor.shape[0] * self.num_candidates,
                target_len,
                3,
            )
        raw_residual = torch.tanh(raw[..., :2])
        gate_logit = raw[..., 2:3]
        mode_tangent_scale = None
        mode_normal_scale = None
        mode_gate_bias = None
        mode_role_score_bias = None
        mode_skeleton_params = None
        mode_role_budget_delta = None
        mode_role_adapter_params = None
        mode_role_budget = self.mode_role_budget_table.to(device=cond.device, dtype=cond.dtype)
        mode_role_control_prior = self.mode_role_control_table.to(device=cond.device, dtype=cond.dtype)
        if self.mode_control is not None:
            if self.mode_control_type in {
                "token_residual_gate",
                "token_role_residual_gate",
                "mode_specific_heads",
                "role_gated_heads",
            }:
                token_context = cond
                if token_seq is not None:
                    token_context = token_context + token_seq.mean(dim=1)
                mode_embed = self.candidate_embed.view(1, self.num_candidates, self.hidden_dim).expand(
                    anchor.shape[0],
                    -1,
                    -1,
                )
                mode_context = token_context.unsqueeze(1).expand(-1, self.num_candidates, -1)
                if self.mode_control_type in {
                    "token_role_residual_gate",
                    "mode_specific_heads",
                    "role_gated_heads",
                }:
                    if mode_role is None:
                        raise RuntimeError(f"{self.mode_control_type} requires mode_role.")
                    mode_role = mode_role.expand(anchor.shape[0], -1, -1)
                    mode_control_input = torch.cat([mode_embed, mode_context, mode_role], dim=-1)
                    if self.mode_role_score is not None:
                        mode_role_score_bias = self.mode_role_score(mode_control_input).squeeze(-1)
                    if self.mode_skeleton is not None:
                        mode_skeleton_params = self.mode_skeleton(mode_control_input)
                    if self.mode_role_budget_head is not None:
                        mode_role_budget_delta = self.mode_role_budget_head(mode_control_input)
                    if self.mode_role_residual_adapter is not None:
                        mode_role_adapter_params = self.mode_role_residual_adapter(mode_control_input)
                else:
                    mode_control_input = torch.cat([mode_embed, mode_context], dim=-1)
                mode_control = self.mode_control(mode_control_input)
                if self.mode_control_type == "role_gated_heads":
                    mode_control = mode_control + mode_role_control_prior.view(1, self.num_candidates, 3)
            else:
                mode_control = self.mode_control(self.candidate_embed).view(1, self.num_candidates, 3).expand(
                    anchor.shape[0],
                    -1,
                    -1,
                )
                if self.mode_control_type == "role_gated_heads":
                    mode_control = mode_control + mode_role_control_prior.view(1, self.num_candidates, 3)
            mode_control = mode_control.view(anchor.shape[0], self.num_candidates, 1, 3).expand(-1, -1, target_len, -1)
            mode_control = mode_control.reshape(anchor.shape[0] * self.num_candidates, target_len, 3)
            mode_tangent_scale = 1.0 + (0.5 * mode_control_strength) * torch.tanh(mode_control[..., 0:1])
            mode_normal_scale = 1.0 + (0.5 * mode_control_strength) * torch.tanh(mode_control[..., 1:2])
            mode_gate_bias = (2.0 * mode_control_strength) * torch.tanh(mode_control[..., 2:3])
            gate_logit = gate_logit + mode_gate_bias
        gate = torch.sigmoid(gate_logit)
        delta_tangent = raw_residual[..., 0:1] * self.tangent_scale
        delta_normal = raw_residual[..., 1:2] * self.normal_scale
        if mode_tangent_scale is not None and mode_normal_scale is not None:
            delta_tangent = delta_tangent * mode_tangent_scale
            delta_normal = delta_normal * mode_normal_scale
        envelope = self.endpoint_envelope(target_len, anchor.device, anchor.dtype)
        residual = envelope * gate * (delta_tangent * tangent_rep + delta_normal * normal_rep)
        route_skeleton_delta = None
        route_skeleton = None
        route_skeleton_up = None
        route_skeleton_up_ungated = None
        route_skeleton_raw = None
        route_skeleton_gate = torch.sigmoid(self.route_skeleton_gate_logit).to(device=cond.device, dtype=cond.dtype)
        route_control_adapter_gate = None
        route_control_delta = None
        gated_route_control_delta = None
        route_control_budget = None
        route_control_budget_scale = None
        route_control_delta_norm_before_budget = None
        route_control_delta_norm_after_budget = None
        prototype_prior_delta = None
        gated_prototype_prior_delta = None
        route_proposal_logits = None
        route_control_selection_context = None
        route_control_selection_delta = None
        if self.route_skeleton_head is not None and self.route_control_mode_head is None:
            coarse_points = max(2, int(self.route_skeleton_points))
            coarse_hidden = self.resample_temporal(route_skeleton_hidden, coarse_points)
            route_skeleton_raw = torch.tanh(self.route_skeleton_head(coarse_hidden))
            coarse_anchor = self.resample_temporal(anchor, coarse_points)
            coarse_anchor_rep = coarse_anchor.repeat_interleave(self.num_candidates, dim=0)
            coarse_tangent, coarse_normal = self.tangent_normal(coarse_anchor_rep)
            coarse_envelope = self.endpoint_envelope(
                coarse_points,
                anchor.device,
                anchor.dtype,
            )
            coarse_gate = torch.sigmoid(route_skeleton_raw[..., 2:3])
            coarse_delta = coarse_envelope * coarse_gate * (
                route_skeleton_raw[..., 0:1] * self.route_skeleton_tangent_scale * coarse_tangent
                + route_skeleton_raw[..., 1:2] * self.route_skeleton_normal_scale * coarse_normal
            )
            coarse_delta = self.preserve_reference_endpoints(coarse_delta)
            route_skeleton_delta = float(self.route_skeleton_strength) * self.resample_temporal(
                coarse_delta,
                target_len,
            )
            route_skeleton_delta = self.preserve_reference_endpoints(route_skeleton_delta)
            route_skeleton_up_ungated = (anchor_rep + route_skeleton_delta).view(
                anchor.shape[0],
                self.num_candidates,
                target_len,
                2,
            )
            route_skeleton_delta = route_skeleton_gate * route_skeleton_delta
            residual = residual + route_skeleton_delta
            route_skeleton = (
                coarse_anchor_rep + route_skeleton_gate * float(self.route_skeleton_strength) * coarse_delta
            ).view(anchor.shape[0], self.num_candidates, coarse_points, 2)
            route_skeleton_up = (anchor_rep + route_skeleton_delta).view(
                anchor.shape[0],
                self.num_candidates,
                target_len,
                2,
            )
        elif self.route_wave_head is not None:
            pooled_hidden = route_skeleton_hidden.mean(dim=1)
            route_skeleton_raw = torch.tanh(self.route_wave_head(pooled_hidden))
            time_rep = time.repeat_interleave(self.num_candidates, dim=0)
            centered_time = 2.0 * time_rep - 1.0
            mode_role_table = self.mode_role_table.to(device=cond.device, dtype=cond.dtype)
            mode_role_rep = mode_role_table.view(1, self.num_candidates, self.mode_role_dim).expand(
                anchor.shape[0],
                -1,
                -1,
            )
            token_controls = self.token_wave_route_controls(
                mobility_tokens=mobility_tokens,
                batch=anchor.shape[0],
                device=cond.device,
                dtype=cond.dtype,
            )
            distance_scale = token_controls["distance"].expand(-1, self.num_candidates, -1, -1)
            curvature_scale = token_controls["curvature"].expand(-1, self.num_candidates, -1, -1)
            reference_scale = token_controls["reference"].expand(-1, self.num_candidates, -1, -1)
            progress_scale = token_controls["progress"].expand(-1, self.num_candidates, -1, -1)
            progress_skew = token_controls["progress_skew"].expand(-1, self.num_candidates, -1, -1)
            direction_sign = token_controls["direction_sign"].expand(-1, self.num_candidates, -1, -1)
            direction_cycles = token_controls["direction_cycles"].expand(-1, self.num_candidates, -1, -1)
            uturn_scale = token_controls["uturn"].expand(-1, self.num_candidates, -1, -1)

            close_ref = mode_role_rep[..., 0:1].unsqueeze(-2)
            smooth_role = mode_role_rep[..., 1:2].unsqueeze(-2)
            loose_ref = mode_role_rep[..., 2:3].unsqueeze(-2)
            winding_role = mode_role_rep[..., 3:4].unsqueeze(-2)
            fast_role = mode_role_rep[..., 4:5].unsqueeze(-2)
            slow_role = mode_role_rep[..., 5:6].unsqueeze(-2)

            mode_sign = torch.tensor(
                [1.0 if (idx % 2 == 0) else -1.0 for idx in range(self.num_candidates)],
                device=cond.device,
                dtype=cond.dtype,
            ).view(1, self.num_candidates, 1, 1).expand(anchor.shape[0], -1, -1, -1)
            signed_direction = torch.where(direction_sign.abs() > 1e-4, direction_sign, mode_sign)

            base_amp = (
                0.20
                + 0.32 * distance_scale
                + 0.22 * curvature_scale
                + 0.18 * reference_scale
                + 0.10 * progress_scale
                + 0.20 * loose_ref
                + 0.30 * winding_role
                + 0.08 * slow_role
                - 0.10 * close_ref
                - 0.08 * smooth_role
                + 0.10 * uturn_scale
            ).clamp(min=0.08)
            base_cycles = (
                0.55
                + 0.55 * direction_cycles
                + 0.28 * winding_role
                + 0.12 * loose_ref
                + 0.10 * uturn_scale
            ).clamp(min=0.45)

            route_skeleton_raw_view = route_skeleton_raw.view(anchor.shape[0], self.num_candidates, 1, 4)
            amp_gain = 1.0 + 0.60 * route_skeleton_raw_view[..., 0:1]
            cycle_gain = (1.0 + 0.35 * route_skeleton_raw_view[..., 1:2]).clamp(min=0.5)
            phase = math.pi * (
                0.20 * route_skeleton_raw_view[..., 2:3]
                + 0.15 * signed_direction
                + 0.10 * mode_sign
            )
            tangent_gain = route_skeleton_raw_view[..., 3:4]

            effective_amp = (base_amp * amp_gain).reshape(anchor.shape[0] * self.num_candidates, 1, 1)
            effective_cycles = (base_cycles * cycle_gain).reshape(anchor.shape[0] * self.num_candidates, 1, 1)
            signed_direction_flat = signed_direction.reshape(anchor.shape[0] * self.num_candidates, 1, 1)
            progress_skew_flat = progress_skew.reshape(anchor.shape[0] * self.num_candidates, 1, 1)
            uturn_flat = uturn_scale.reshape(anchor.shape[0] * self.num_candidates, 1, 1)
            phase_flat = phase.reshape(anchor.shape[0] * self.num_candidates, 1, 1)
            tangent_gain_flat = tangent_gain.reshape(anchor.shape[0] * self.num_candidates, 1, 1)

            bow_wave = torch.sin(math.pi * time_rep)
            main_wave = torch.sin(math.pi * effective_cycles * time_rep + phase_flat)
            multi_wave = torch.sin(2.0 * math.pi * effective_cycles * time_rep + 0.5 * phase_flat)
            uturn_wave = torch.sin(math.pi * time_rep).pow(2.0) * centered_time.sign()

            normal_curve = envelope * self.route_skeleton_normal_scale * effective_amp * (
                0.45 * signed_direction_flat * bow_wave
                + 0.55 * main_wave
                + 0.20 * winding_role.reshape(anchor.shape[0] * self.num_candidates, 1, 1) * multi_wave
                + 0.35 * uturn_flat * uturn_wave
            )
            tangent_curve = envelope * self.route_skeleton_tangent_scale * (
                0.22 * effective_amp * progress_skew_flat * centered_time
                + 0.18 * effective_amp * tangent_gain_flat * multi_wave
                + 0.16 * effective_amp * (distance_scale * progress_scale).reshape(anchor.shape[0] * self.num_candidates, 1, 1) * bow_wave.pow(2.0)
            )
            route_skeleton_delta = float(self.route_skeleton_strength) * (tangent_curve * tangent_rep + normal_curve * normal_rep)
            route_skeleton_delta = self.preserve_reference_endpoints(route_skeleton_delta)
            route_skeleton_up_ungated = (anchor_rep + route_skeleton_delta).view(
                anchor.shape[0],
                self.num_candidates,
                target_len,
                2,
            )
            route_skeleton_delta = route_skeleton_gate * route_skeleton_delta
            residual = residual + route_skeleton_delta
            route_skeleton = (anchor_rep + route_skeleton_delta).view(
                anchor.shape[0],
                self.num_candidates,
                target_len,
                2,
            )
            route_skeleton_up = route_skeleton
        elif self.route_transition_mode_head is not None:
            pooled_hidden = route_skeleton_hidden.mean(dim=1)
            route_skeleton_raw = torch.tanh(self.route_transition_mode_head(pooled_hidden))
            params = route_skeleton_raw.view(anchor.shape[0], self.num_candidates, 1, 8)
            time_rep = time.unsqueeze(1).expand(-1, self.num_candidates, -1, -1)
            centered_time = 2.0 * time_rep - 1.0
            envelope_view = envelope.view(1, 1, target_len, 1)
            bow_wave = torch.sin(math.pi * time_rep).clamp_min(0.0)

            transition_roles = self.route_transition_mode_table.to(device=cond.device, dtype=cond.dtype)
            transition_roles = transition_roles.view(1, self.num_candidates, 1, 6).expand(
                anchor.shape[0],
                -1,
                -1,
                -1,
            )
            direct_role = transition_roles[..., 0:1]
            left_role = transition_roles[..., 1:2]
            right_role = transition_roles[..., 2:3]
            uturn_role = transition_roles[..., 3:4]
            multi_role = transition_roles[..., 4:5]
            detour_role = transition_roles[..., 5:6]
            turn_role = (left_role + right_role).clamp(0.0, 1.0)
            role_signed = left_role - right_role

            token_controls = self.token_wave_route_controls(
                mobility_tokens=mobility_tokens,
                batch=anchor.shape[0],
                device=cond.device,
                dtype=cond.dtype,
            )
            distance_scale = token_controls["distance"].expand(-1, self.num_candidates, -1, -1)
            curvature_scale = token_controls["curvature"].expand(-1, self.num_candidates, -1, -1)
            reference_scale = token_controls["reference"].expand(-1, self.num_candidates, -1, -1)
            progress_scale = token_controls["progress"].expand(-1, self.num_candidates, -1, -1)
            progress_skew = token_controls["progress_skew"].expand(-1, self.num_candidates, -1, -1)
            direction_sign = token_controls["direction_sign"].expand(-1, self.num_candidates, -1, -1)
            direction_cycles = token_controls["direction_cycles"].expand(-1, self.num_candidates, -1, -1)
            uturn_scale = token_controls["uturn"].expand(-1, self.num_candidates, -1, -1)
            direction_known = token_controls["direction_known"].expand(-1, self.num_candidates, -1, -1)
            target_straight = token_controls["direction_straight"].expand(-1, self.num_candidates, -1, -1)
            target_left = token_controls["direction_left"].expand(-1, self.num_candidates, -1, -1)
            target_right = token_controls["direction_right"].expand(-1, self.num_candidates, -1, -1)
            target_uturn = token_controls["direction_uturn"].expand(-1, self.num_candidates, -1, -1)
            target_multi = token_controls["direction_multi"].expand(-1, self.num_candidates, -1, -1)

            keep_all_direction_roles = direction_known <= 0.5
            left_gate = (0.10 + 0.90 * (target_left + 0.25 * target_multi + 0.15 * target_uturn)).clamp(
                min=0.08,
                max=1.0,
            )
            right_gate = (0.10 + 0.90 * (target_right + 0.25 * target_multi + 0.15 * target_uturn)).clamp(
                min=0.08,
                max=1.0,
            )
            uturn_gate = (0.08 + 0.92 * (target_uturn + 0.20 * target_multi)).clamp(min=0.06, max=1.0)
            multi_gate = (0.12 + 0.88 * (target_multi + 0.20 * target_uturn)).clamp(min=0.08, max=1.0)
            detour_gate = (
                0.20
                + 0.80 * (target_multi + 0.35 * target_uturn + 0.25 * (1.0 - target_straight))
            ).clamp(min=0.12, max=1.0)
            left_role = torch.where(keep_all_direction_roles, left_role, left_role * left_gate)
            right_role = torch.where(keep_all_direction_roles, right_role, right_role * right_gate)
            uturn_role = torch.where(keep_all_direction_roles, uturn_role, uturn_role * uturn_gate)
            multi_role = torch.where(keep_all_direction_roles, multi_role, multi_role * multi_gate)
            detour_role = torch.where(keep_all_direction_roles, detour_role, detour_role * detour_gate)
            role_signed = left_role - right_role
            turn_role = (left_role + right_role).clamp(0.0, 1.0)

            mode_sign = torch.linspace(
                -1.0,
                1.0,
                steps=self.num_candidates,
                device=cond.device,
                dtype=cond.dtype,
            ).view(1, self.num_candidates, 1, 1).expand(anchor.shape[0], -1, -1, -1)
            role_direction = role_signed + 0.25 * mode_sign * (multi_role + detour_role)
            signed_direction = torch.where(direction_sign.abs() > 1e-4, direction_sign, role_direction)

            base_amp = (
                0.035
                + 0.060 * (1.0 - direct_role)
                + 0.170 * turn_role
                + 0.220 * uturn_role
                + 0.260 * multi_role
                + 0.320 * detour_role
            )
            token_amp = (
                0.70
                + 0.20 * distance_scale
                + 0.22 * curvature_scale
                + 0.16 * reference_scale
                + 0.10 * progress_scale
                + 0.16 * uturn_scale
            )
            amp_gain = 1.0 + 0.55 * params[..., 0:1]
            effective_amp = (base_amp * token_amp * amp_gain).clamp(min=0.015, max=1.25)

            base_cycles = (
                0.35
                + 0.35 * turn_role
                + 0.75 * uturn_role
                + 1.05 * multi_role
                + 0.80 * detour_role
                + 0.25 * direction_cycles
                + 0.16 * curvature_scale
            )
            cycle_gain = (1.0 + 0.25 * params[..., 1:2]).clamp(min=0.55, max=1.45)
            effective_cycles = (base_cycles * cycle_gain).clamp(min=0.35, max=3.50)
            phase = math.pi * (0.20 * params[..., 2:3] + 0.12 * mode_sign + 0.12 * role_signed)

            main_wave = torch.sin(math.pi * effective_cycles * time_rep + phase)
            multi_wave = torch.sin(2.0 * math.pi * effective_cycles * time_rep + 0.5 * phase)
            detour_wave = mode_sign * centered_time * bow_wave
            uturn_wave = torch.sign(centered_time + 1e-6) * bow_wave.pow(2.0)

            normal_pattern = (
                0.48 * signed_direction * bow_wave
                + 0.38 * (turn_role + 0.55 * detour_role) * main_wave
                + 0.28 * multi_role * multi_wave
                + 0.36 * (uturn_role + 0.35 * uturn_scale) * uturn_wave
                + 0.26 * detour_role * detour_wave
                + 0.12 * params[..., 3:4] * multi_wave
            )
            tangent_pattern = (
                0.18 * progress_skew * centered_time
                + 0.18 * detour_role * bow_wave.pow(2.0)
                - 0.20 * (uturn_role + 0.25 * uturn_scale) * bow_wave
                + 0.10 * params[..., 4:5] * centered_time * bow_wave
                + 0.08 * params[..., 5:6] * multi_wave
            )
            direct_damp = (1.0 - 0.55 * direct_role).clamp(min=0.35)
            normal_curve = (
                envelope_view
                * self.route_skeleton_normal_scale
                * effective_amp
                * direct_damp
                * normal_pattern
            )
            tangent_curve = (
                envelope_view
                * self.route_skeleton_tangent_scale
                * effective_amp
                * tangent_pattern
            )
            route_skeleton_delta = float(self.route_skeleton_strength) * (
                normal_curve.reshape(anchor.shape[0] * self.num_candidates, target_len, 1) * normal_rep
                + tangent_curve.reshape(anchor.shape[0] * self.num_candidates, target_len, 1) * tangent_rep
            )
            route_skeleton_delta = self.preserve_reference_endpoints(route_skeleton_delta)
            route_skeleton_up_ungated = (anchor_rep + route_skeleton_delta).view(
                anchor.shape[0],
                self.num_candidates,
                target_len,
                2,
            )
            route_skeleton_delta = route_skeleton_gate * route_skeleton_delta
            residual = residual + route_skeleton_delta
            route_skeleton = (anchor_rep + route_skeleton_delta).view(
                anchor.shape[0],
                self.num_candidates,
                target_len,
                2,
            )
            route_skeleton_up = route_skeleton
        elif self.route_control_mode_head is not None:
            coarse_prior_delta = None
            if self.route_skeleton_head is not None:
                coarse_points = max(2, int(self.route_skeleton_points))
                coarse_hidden = self.resample_temporal(route_skeleton_hidden, coarse_points)
                coarse_raw = torch.tanh(self.route_skeleton_head(coarse_hidden))
                coarse_anchor = self.resample_temporal(anchor, coarse_points)
                coarse_anchor_rep = coarse_anchor.repeat_interleave(self.num_candidates, dim=0)
                coarse_tangent, coarse_normal = self.tangent_normal(coarse_anchor_rep)
                coarse_envelope = self.endpoint_envelope(
                    coarse_points,
                    anchor.device,
                    anchor.dtype,
                )
                coarse_gate = torch.sigmoid(coarse_raw[..., 2:3])
                coarse_delta = coarse_envelope * coarse_gate * (
                    coarse_raw[..., 0:1] * self.route_skeleton_tangent_scale * coarse_tangent
                    + coarse_raw[..., 1:2] * self.route_skeleton_normal_scale * coarse_normal
                )
                coarse_delta = self.preserve_reference_endpoints(coarse_delta)
                coarse_prior_delta = float(self.route_skeleton_strength) * self.resample_temporal(
                    coarse_delta,
                    target_len,
                )
                coarse_prior_delta = self.preserve_reference_endpoints(coarse_prior_delta)
                residual = residual + route_skeleton_gate * coarse_prior_delta
            pooled_hidden = route_skeleton_hidden.mean(dim=1).view(
                anchor.shape[0],
                self.num_candidates,
                self.hidden_dim,
            )
            token_context = cond
            if token_seq is not None:
                token_context = token_context + token_seq.mean(dim=1)
            mode_role_for_route = self.route_control_role_proj(
                self.mode_role_table.to(device=cond.device, dtype=cond.dtype)
            ).view(1, self.num_candidates, self.hidden_dim).expand(
                anchor.shape[0],
                -1,
                -1,
            )
            route_context = torch.cat(
                [
                    pooled_hidden,
                    token_context.unsqueeze(1).expand(-1, self.num_candidates, -1),
                    mode_role_for_route,
                ],
                dim=-1,
            )
            route_control_selection_context = route_context
            route_skeleton_raw = torch.tanh(self.route_control_mode_head(route_context))
            params = route_skeleton_raw.view(anchor.shape[0], self.num_candidates, 1, 8)
            time_rep = time.unsqueeze(1).expand(-1, self.num_candidates, -1, -1)
            centered_time = 2.0 * time_rep - 1.0
            envelope_view = envelope.view(1, 1, target_len, 1)
            bow_wave = torch.sin(math.pi * time_rep).clamp_min(0.0)

            mode_roles = self.mode_role_table.to(device=cond.device, dtype=cond.dtype)
            mode_roles = mode_roles.view(1, self.num_candidates, 1, self.mode_role_dim).expand(
                anchor.shape[0],
                -1,
                -1,
                -1,
            )
            close_ref = mode_roles[..., 0:1]
            smooth_role = mode_roles[..., 1:2]
            loose_ref = mode_roles[..., 2:3]
            winding_role = mode_roles[..., 3:4]
            fast_role = mode_roles[..., 4:5]
            slow_role = mode_roles[..., 5:6]

            transition_roles = self.route_transition_mode_table.to(device=cond.device, dtype=cond.dtype)
            transition_roles = transition_roles.view(1, self.num_candidates, 1, 6).expand(
                anchor.shape[0],
                -1,
                -1,
                -1,
            )
            direct_role = transition_roles[..., 0:1]
            left_role = transition_roles[..., 1:2]
            right_role = transition_roles[..., 2:3]
            uturn_role = transition_roles[..., 3:4]
            multi_role = transition_roles[..., 4:5]
            detour_role = transition_roles[..., 5:6]
            role_signed = left_role - right_role

            token_controls = self.token_wave_route_controls(
                mobility_tokens=mobility_tokens,
                batch=anchor.shape[0],
                device=cond.device,
                dtype=cond.dtype,
            )
            distance_scale = token_controls["distance"].expand(-1, self.num_candidates, -1, -1)
            curvature_scale = token_controls["curvature"].expand(-1, self.num_candidates, -1, -1)
            reference_scale = token_controls["reference"].expand(-1, self.num_candidates, -1, -1)
            progress_scale = token_controls["progress"].expand(-1, self.num_candidates, -1, -1)
            progress_skew = token_controls["progress_skew"].expand(-1, self.num_candidates, -1, -1)
            direction_sign = token_controls["direction_sign"].expand(-1, self.num_candidates, -1, -1)
            direction_cycles = token_controls["direction_cycles"].expand(-1, self.num_candidates, -1, -1)
            uturn_scale = token_controls["uturn"].expand(-1, self.num_candidates, -1, -1)
            direction_known = token_controls["direction_known"].expand(-1, self.num_candidates, -1, -1)
            target_straight = token_controls["direction_straight"].expand(-1, self.num_candidates, -1, -1)
            target_left = token_controls["direction_left"].expand(-1, self.num_candidates, -1, -1)
            target_right = token_controls["direction_right"].expand(-1, self.num_candidates, -1, -1)
            target_uturn = token_controls["direction_uturn"].expand(-1, self.num_candidates, -1, -1)
            target_multi = token_controls["direction_multi"].expand(-1, self.num_candidates, -1, -1)

            keep_all_direction_roles = direction_known <= 0.5
            left_gate = (0.10 + 0.90 * (target_left + 0.25 * target_multi + 0.15 * target_uturn)).clamp(
                min=0.08,
                max=1.0,
            )
            right_gate = (0.10 + 0.90 * (target_right + 0.25 * target_multi + 0.15 * target_uturn)).clamp(
                min=0.08,
                max=1.0,
            )
            uturn_gate = (0.08 + 0.92 * (target_uturn + 0.20 * target_multi)).clamp(min=0.06, max=1.0)
            multi_gate = (0.12 + 0.88 * (target_multi + 0.20 * target_uturn)).clamp(min=0.08, max=1.0)
            detour_gate = (
                0.20
                + 0.80 * (target_multi + 0.35 * target_uturn + 0.25 * (1.0 - target_straight))
            ).clamp(min=0.12, max=1.0)
            left_role = torch.where(keep_all_direction_roles, left_role, left_role * left_gate)
            right_role = torch.where(keep_all_direction_roles, right_role, right_role * right_gate)
            uturn_role = torch.where(keep_all_direction_roles, uturn_role, uturn_role * uturn_gate)
            multi_role = torch.where(keep_all_direction_roles, multi_role, multi_role * multi_gate)
            detour_role = torch.where(keep_all_direction_roles, detour_role, detour_role * detour_gate)
            role_signed = left_role - right_role
            if float(self.route_proposal_logit_weight) != 0.0:
                direction_compat = (
                    target_straight * (direct_role + 0.25 * smooth_role + 0.15 * close_ref)
                    + target_left * left_role
                    + target_right * right_role
                    + target_uturn * (uturn_role + 0.20 * multi_role)
                    + target_multi * (multi_role + 0.25 * detour_role + 0.15 * winding_role)
                )
                route_role_compat = (
                    0.20 * direct_role
                    + 0.14 * (left_role + right_role)
                    + 0.12 * uturn_role
                    + 0.12 * multi_role
                    + 0.10 * detour_role
                    + 0.12 * close_ref
                    + 0.08 * smooth_role
                )
                route_proposal_logits = torch.where(
                    direction_known > 0.5,
                    direction_compat + route_role_compat,
                    route_role_compat,
                ).squeeze(-1).squeeze(-1)

            mode_sign = torch.linspace(
                -1.0,
                1.0,
                steps=self.num_candidates,
                device=cond.device,
                dtype=cond.dtype,
            ).view(1, self.num_candidates, 1, 1).expand(anchor.shape[0], -1, -1, -1)
            fallback_direction = role_signed + 0.20 * mode_sign * (multi_role + detour_role + winding_role)
            signed_direction = torch.where(direction_sign.abs() > 1e-4, direction_sign, fallback_direction)

            route_control_gain = (1.0 + 0.45 * params[..., 0:1]).clamp(min=0.35, max=1.70)
            route_control_cycles = (1.0 + 0.30 * params[..., 1:2]).clamp(min=0.55, max=1.55)
            route_control_phase = math.pi * (0.18 * params[..., 2:3] + 0.10 * mode_sign + 0.08 * role_signed)
            route_control_tangent = params[..., 3:4]
            route_control_detour = params[..., 4:5]
            route_control_ref = params[..., 5:6]
            route_control_turn = params[..., 6:7]
            route_control_progress = params[..., 7:8]

            base_amp = (
                0.045
                + 0.080 * loose_ref
                - 0.030 * close_ref
                - 0.020 * smooth_role
                + 0.150 * winding_role
                + 0.110 * (left_role + right_role)
                + 0.190 * uturn_role
                + 0.240 * multi_role
                + 0.260 * detour_role
            )
            token_amp = (
                0.62
                + 0.18 * distance_scale
                + 0.22 * curvature_scale
                + 0.14 * reference_scale
                + 0.08 * progress_scale
                + 0.12 * uturn_scale
            )
            effective_amp = (base_amp * token_amp * route_control_gain).clamp(min=0.012, max=1.10)
            base_cycles = (
                0.35
                + 0.25 * (left_role + right_role)
                + 0.80 * uturn_role
                + 0.95 * multi_role
                + 0.65 * detour_role
                + 0.45 * winding_role
                + 0.22 * direction_cycles
                + 0.18 * curvature_scale
            )
            effective_cycles = (base_cycles * route_control_cycles).clamp(min=0.30, max=3.75)

            main_wave = torch.sin(math.pi * effective_cycles * time_rep + route_control_phase)
            multi_wave = torch.sin(2.0 * math.pi * effective_cycles * time_rep + 0.5 * route_control_phase)
            detour_wave = mode_sign * centered_time * bow_wave
            uturn_wave = torch.sign(centered_time + 1e-6) * bow_wave.pow(2.0)
            ref_shrink = (1.0 - 0.28 * close_ref + 0.22 * loose_ref + 0.10 * route_control_ref).clamp(
                min=0.35,
                max=1.45,
            )

            normal_pattern = (
                0.50 * signed_direction * bow_wave
                + 0.36 * (left_role + right_role + 0.35 * route_control_turn) * main_wave
                + 0.30 * (multi_role + 0.45 * winding_role) * multi_wave
                + 0.34 * (uturn_role + 0.30 * uturn_scale) * uturn_wave
                + 0.24 * (detour_role + 0.30 * route_control_detour) * detour_wave
            )
            tangent_pattern = (
                0.16 * (fast_role - slow_role + progress_skew + 0.45 * route_control_progress) * centered_time
                + 0.14 * (detour_role + 0.35 * route_control_detour) * bow_wave.pow(2.0)
                - 0.16 * (uturn_role + 0.25 * uturn_scale) * bow_wave
                + 0.10 * route_control_tangent * multi_wave
            )
            route_control_delta = float(self.route_skeleton_strength) * (
                envelope_view
                * effective_amp
                * (
                    self.route_skeleton_normal_scale * ref_shrink * normal_pattern * normal_rep.view(
                        anchor.shape[0],
                        self.num_candidates,
                        target_len,
                        2,
                    )
                    + self.route_skeleton_tangent_scale * tangent_pattern * tangent_rep.view(
                        anchor.shape[0],
                        self.num_candidates,
                        target_len,
                        2,
                    )
                )
            ).reshape(anchor.shape[0] * self.num_candidates, target_len, 2)
            if float(self.route_control_budget_ratio) > 0.0:
                route_control_delta_view = route_control_delta.view(
                    anchor.shape[0],
                    self.num_candidates,
                    target_len,
                    2,
                )
                point_norm = torch.linalg.norm(route_control_delta_view, dim=-1)
                if self.route_control_budget_mode == "max":
                    route_control_delta_norm_before_budget = point_norm.amax(
                        dim=2,
                        keepdim=True,
                    ).unsqueeze(-1)
                else:
                    route_control_delta_norm_before_budget = torch.sqrt(
                        point_norm.pow(2.0).mean(dim=2, keepdim=True).clamp_min(0.0)
                    ).unsqueeze(-1)
                anchor_span_for_budget = torch.linalg.norm(anchor[:, -1, :] - anchor[:, 0, :], dim=-1)
                anchor_span_for_budget = anchor_span_for_budget.view(
                    anchor.shape[0],
                    1,
                    1,
                    1,
                ).clamp_min(1e-6)
                route_control_budget = float(self.route_control_budget_ratio) * anchor_span_for_budget
                if float(self.route_control_budget_min) > 0.0:
                    route_control_budget = torch.maximum(
                        route_control_budget,
                        torch.full_like(route_control_budget, float(self.route_control_budget_min)),
                    )
                if float(self.route_control_budget_max) > 0.0:
                    route_control_budget = torch.minimum(
                        route_control_budget,
                        torch.full_like(route_control_budget, float(self.route_control_budget_max)),
                    )
                route_control_budget_scale = (
                    route_control_budget / route_control_delta_norm_before_budget.clamp_min(1e-6)
                ).clamp(max=1.0)
                route_control_delta_view = route_control_delta_view * route_control_budget_scale
                route_control_delta_norm_after_budget = (
                    route_control_delta_norm_before_budget * route_control_budget_scale
                )
                route_control_delta = route_control_delta_view.reshape(
                    anchor.shape[0] * self.num_candidates,
                    target_len,
                    2,
                )
            route_control_delta = self.preserve_reference_endpoints(route_control_delta)
            if coarse_prior_delta is None:
                route_skeleton_delta = route_control_delta
            else:
                route_skeleton_delta = coarse_prior_delta + route_control_delta
            prototype_prior_delta = None
            prototype_prior_strength = max(float(self.route_prototype_prior_strength), 0.0)
            if prototype_prior_strength > 0.0 and self.route_prototype_bank is not None:
                prototype = self.route_prototype_bank.to(device=cond.device, dtype=cond.dtype)
                if prototype.shape[0] < self.num_candidates:
                    repeats = math.ceil(self.num_candidates / prototype.shape[0])
                    prototype = prototype.repeat(repeats, 1, 1)
                prototype = prototype[: self.num_candidates]
                prototype = self.resample_temporal(prototype, target_len)
                if self.route_prototype_prior_mix == "direction_role":
                    proto_direction_roles = self.direction_mode_role_table.to(device=cond.device, dtype=cond.dtype)
                    proto_transition_roles = self.route_transition_mode_table.to(device=cond.device, dtype=cond.dtype)
                    proto_mode_roles = self.mode_role_table.to(device=cond.device, dtype=cond.dtype)

                    direction_target = torch.cat(
                        [
                            target_straight,
                            target_left,
                            target_right,
                            target_uturn,
                            target_multi,
                        ],
                        dim=-1,
                    ).squeeze(2)
                    direction_score = torch.einsum("bkd,pd->bkp", direction_target, proto_direction_roles)
                    direction_score = torch.where(
                        direction_known.squeeze(2) > 0.5,
                        direction_score,
                        torch.zeros_like(direction_score),
                    )
                    transition_query = torch.cat(
                        [
                            0.35 * target_straight + 0.25 * close_ref + 0.15 * smooth_role,
                            left_role,
                            right_role,
                            uturn_role,
                            multi_role,
                            detour_role + 0.35 * loose_ref + 0.25 * winding_role,
                        ],
                        dim=-1,
                    ).squeeze(2)
                    transition_score = torch.einsum("bkr,pr->bkp", transition_query, proto_transition_roles)
                    role_query = torch.cat(
                        [close_ref, smooth_role, loose_ref, winding_role, fast_role, slow_role],
                        dim=-1,
                    ).squeeze(2)
                    role_score = torch.einsum("bkr,pr->bkp", role_query, proto_mode_roles)
                    mix_score = 2.2 * direction_score + 0.65 * transition_score + 0.35 * role_score
                    mix_score = mix_score - mix_score.amax(dim=-1, keepdim=True)
                    mix_weight = torch.softmax(mix_score, dim=-1)
                    prototype = torch.einsum("bkp,ptc->bktc", mix_weight, prototype)
                else:
                    prototype = prototype.view(1, self.num_candidates, target_len, 2).expand(
                        anchor.shape[0],
                        -1,
                        -1,
                        -1,
                    )
                proto_clip = max(float(self.route_prototype_prior_clip), 1e-6)
                proto_tangent = prototype[..., 0:1].clamp(min=-proto_clip, max=proto_clip)
                proto_normal = prototype[..., 1:2].clamp(min=-proto_clip, max=proto_clip)
                token_ref_shrink = (1.10 - 0.22 * close_ref + 0.18 * loose_ref).clamp(min=0.50, max=1.35)
                token_curve_gain = (0.65 + 0.20 * curvature_scale + 0.10 * distance_scale).clamp(
                    min=0.45,
                    max=1.30,
                )
                anchor_span = torch.linalg.norm(anchor[:, -1, :] - anchor[:, 0, :], dim=-1)
                anchor_span = anchor_span.view(anchor.shape[0], 1, 1, 1).clamp_min(1e-6)
                prototype_prior_delta = (
                    prototype_prior_strength
                    * anchor_span
                    * envelope_view
                    * (
                        self.route_skeleton_tangent_scale
                        * proto_tangent
                        * tangent_rep.view(anchor.shape[0], self.num_candidates, target_len, 2)
                        + self.route_skeleton_normal_scale
                        * token_ref_shrink
                        * token_curve_gain
                        * proto_normal
                        * normal_rep.view(anchor.shape[0], self.num_candidates, target_len, 2)
                    )
                ).reshape(anchor.shape[0] * self.num_candidates, target_len, 2)
                prototype_prior_delta = self.preserve_reference_endpoints(prototype_prior_delta)
                route_skeleton_delta = route_skeleton_delta + prototype_prior_delta
            route_skeleton_up_ungated = (anchor_rep + route_skeleton_delta).view(
                anchor.shape[0],
                self.num_candidates,
                target_len,
                2,
            )
            route_control_adapter_gate = torch.sigmoid(self.route_control_adapter_gate_logit).to(
                device=cond.device,
                dtype=cond.dtype,
            )
            gated_route_control_delta = route_control_adapter_gate * route_control_delta
            residual = residual + gated_route_control_delta
            route_skeleton_delta = (
                gated_route_control_delta
                if coarse_prior_delta is None
                else route_skeleton_gate * coarse_prior_delta + gated_route_control_delta
            )
            gated_prototype_prior_delta = None
            if prototype_prior_delta is not None:
                if self.route_prototype_prior_gate == "route_control_adapter":
                    prototype_prior_gate = route_control_adapter_gate
                else:
                    prototype_prior_gate = route_skeleton_gate
                gated_prototype_prior_delta = prototype_prior_gate * prototype_prior_delta
                residual = residual + gated_prototype_prior_delta
                route_skeleton_delta = route_skeleton_delta + gated_prototype_prior_delta
            route_skeleton = (anchor_rep + route_skeleton_delta).view(
                anchor.shape[0],
                self.num_candidates,
                target_len,
                2,
            )
            route_skeleton_up = route_skeleton
        elif self.route_proposal_head is not None:
            pooled_hidden = route_skeleton_hidden.mean(dim=1)
            route_skeleton_raw = torch.tanh(self.route_proposal_head(pooled_hidden))
            params = route_skeleton_raw.view(anchor.shape[0], self.num_candidates, 1, 6)
            time_rep = time.unsqueeze(1).expand(-1, self.num_candidates, -1, -1)
            centered_time = 2.0 * time_rep - 1.0
            envelope_view = envelope.view(1, 1, target_len, 1)
            bow_wave = torch.sin(math.pi * time_rep).clamp_min(0.0)
            s_wave = torch.sin(2.0 * math.pi * time_rep)
            detour_wave = centered_time * bow_wave

            base_roles = torch.tensor(
                [
                    [0.00, 0.00, 0.00, -0.10],
                    [1.00, 0.00, 0.25, 0.00],
                    [-1.00, 0.00, 0.25, 0.00],
                    [0.70, -1.00, 0.45, 0.18],
                    [-0.70, 1.00, 0.45, -0.18],
                    [0.35, 0.75, 0.80, 0.35],
                ],
                device=cond.device,
                dtype=cond.dtype,
            )
            if self.num_candidates > base_roles.shape[0]:
                repeats = math.ceil(self.num_candidates / base_roles.shape[0])
                base_roles = base_roles.repeat(repeats, 1)
            base_roles = base_roles[: self.num_candidates].view(1, self.num_candidates, 1, 4).expand(
                anchor.shape[0],
                -1,
                -1,
                -1,
            )
            lateral_role = base_roles[..., 0:1]
            s_role = base_roles[..., 1:2]
            detour_role = base_roles[..., 2:3]
            longitudinal_role = base_roles[..., 3:4]

            token_controls = self.token_wave_route_controls(
                mobility_tokens=mobility_tokens,
                batch=anchor.shape[0],
                device=cond.device,
                dtype=cond.dtype,
            )
            distance_scale = token_controls["distance"].expand(-1, self.num_candidates, -1, -1)
            curvature_scale = token_controls["curvature"].expand(-1, self.num_candidates, -1, -1)
            reference_scale = token_controls["reference"].expand(-1, self.num_candidates, -1, -1)
            progress_scale = token_controls["progress"].expand(-1, self.num_candidates, -1, -1)
            progress_skew = token_controls["progress_skew"].expand(-1, self.num_candidates, -1, -1)
            direction_sign = token_controls["direction_sign"].expand(-1, self.num_candidates, -1, -1)
            direction_cycles = token_controls["direction_cycles"].expand(-1, self.num_candidates, -1, -1)
            uturn_scale = token_controls["uturn"].expand(-1, self.num_candidates, -1, -1)

            instruction_turn = direction_sign.abs().clamp(0.0, 1.0)
            signed_lateral = torch.where(instruction_turn > 0.50, direction_sign, lateral_role)
            signed_lateral = signed_lateral + 0.20 * params[..., 1:2]
            proposal_amp = (
                0.055
                + 0.080 * lateral_role.abs()
                + 0.070 * s_role.abs()
                + 0.120 * detour_role
                + 0.050 * direction_cycles
                + 0.060 * uturn_scale
            )
            token_amp = (
                0.60
                + 0.18 * distance_scale
                + 0.22 * curvature_scale
                + 0.18 * reference_scale
                + 0.08 * progress_scale
            )
            amp_gain = (1.0 + 0.55 * params[..., 0:1]).clamp(min=0.35, max=1.75)
            effective_amp = (proposal_amp * token_amp * amp_gain).clamp(min=0.015, max=0.90)
            phase = math.pi * (0.15 * params[..., 2:3] + 0.08 * lateral_role)

            normal_pattern = (
                0.62 * signed_lateral * bow_wave
                + 0.38 * (s_role + 0.35 * params[..., 3:4]) * s_wave
                + 0.28 * detour_role * detour_wave
                + 0.30 * uturn_scale * torch.sign(centered_time + 1e-6) * bow_wave.pow(2.0)
                + 0.16 * torch.sin(math.pi * (1.0 + direction_cycles) * time_rep + phase)
            )
            tangent_pattern = (
                0.24 * longitudinal_role * bow_wave.pow(2.0)
                + 0.18 * progress_skew * centered_time * bow_wave
                + 0.12 * detour_role * bow_wave.pow(2.0)
                + 0.10 * params[..., 4:5] * s_wave
                + 0.10 * params[..., 5:6] * centered_time * bow_wave
            )
            route_skeleton_delta = float(self.route_skeleton_strength) * (
                envelope_view
                * effective_amp
                * (
                    self.route_skeleton_normal_scale * normal_pattern * normal_rep.view(
                        anchor.shape[0],
                        self.num_candidates,
                        target_len,
                        2,
                    )
                    + self.route_skeleton_tangent_scale * tangent_pattern * tangent_rep.view(
                        anchor.shape[0],
                        self.num_candidates,
                        target_len,
                        2,
                    )
                )
            ).reshape(anchor.shape[0] * self.num_candidates, target_len, 2)
            route_skeleton_delta = self.preserve_reference_endpoints(route_skeleton_delta)
            route_skeleton_up_ungated = (anchor_rep + route_skeleton_delta).view(
                anchor.shape[0],
                self.num_candidates,
                target_len,
                2,
            )
            route_skeleton_delta = route_skeleton_gate * route_skeleton_delta
            residual = residual + route_skeleton_delta
            route_skeleton = (anchor_rep + route_skeleton_delta).view(
                anchor.shape[0],
                self.num_candidates,
                target_len,
                2,
            )
            route_skeleton_up = route_skeleton
        elif self.route_prototype_head is not None:
            pooled_hidden = route_skeleton_hidden.mean(dim=1)
            route_skeleton_raw = torch.tanh(self.route_prototype_head(pooled_hidden))
            params = route_skeleton_raw.view(anchor.shape[0], self.num_candidates, 1, 4)
            prototype = self.route_prototype_bank.to(device=cond.device, dtype=cond.dtype)
            if prototype.shape[0] < self.num_candidates:
                repeats = math.ceil(self.num_candidates / prototype.shape[0])
                prototype = prototype.repeat(repeats, 1, 1)
            prototype = prototype[: self.num_candidates]
            prototype = self.resample_temporal(prototype, target_len)
            prototype = prototype.view(1, self.num_candidates, target_len, 2).expand(
                anchor.shape[0],
                -1,
                -1,
                -1,
            )

            time_view = time.view(anchor.shape[0], 1, target_len, 1)
            centered_time = 2.0 * time_view - 1.0
            bow_wave = torch.sin(math.pi * time_view).clamp_min(0.0)
            token_controls = self.token_wave_route_controls(
                mobility_tokens=mobility_tokens,
                batch=anchor.shape[0],
                device=cond.device,
                dtype=cond.dtype,
            )
            distance_scale = token_controls["distance"].expand(-1, self.num_candidates, -1, -1)
            curvature_scale = token_controls["curvature"].expand(-1, self.num_candidates, -1, -1)
            reference_scale = token_controls["reference"].expand(-1, self.num_candidates, -1, -1)
            progress_skew = token_controls["progress_skew"].expand(-1, self.num_candidates, -1, -1)
            direction_sign = token_controls["direction_sign"].expand(-1, self.num_candidates, -1, -1)

            anchor_span = torch.linalg.norm(anchor[:, -1, :] - anchor[:, 0, :], dim=-1)
            anchor_span = anchor_span.view(anchor.shape[0], 1, 1, 1).clamp_min(1e-6)
            proto_tangent = prototype[..., 0:1]
            proto_normal = prototype[..., 1:2]
            proto_sign = torch.sign(proto_normal.mean(dim=2, keepdim=True) + 1e-6)
            instruction_turn = direction_sign.abs().clamp(0.0, 1.0)
            normal_sign = torch.where(instruction_turn > 0.50, direction_sign, proto_sign)
            normal_abs = proto_normal.abs()
            token_gain = (
                0.72
                + 0.10 * distance_scale
                + 0.14 * curvature_scale
                + 0.10 * reference_scale
            ).clamp(min=0.45, max=1.35)
            learned_normal_gain = (1.0 + 0.45 * params[..., 0:1]).clamp(min=0.35, max=1.65)
            learned_tangent_gain = (1.0 + 0.30 * params[..., 1:2]).clamp(min=0.45, max=1.45)
            learned_phase = 0.04 * params[..., 2:3] * bow_wave
            learned_progress = 0.05 * params[..., 3:4] * centered_time * bow_wave

            tangent_pattern = proto_tangent * learned_tangent_gain + learned_progress + 0.04 * progress_skew * centered_time * bow_wave
            normal_pattern = normal_sign * normal_abs * token_gain * learned_normal_gain + learned_phase
            tangent_view = tangent_rep.view(anchor.shape[0], self.num_candidates, target_len, 2)
            normal_view = normal_rep.view(anchor.shape[0], self.num_candidates, target_len, 2)
            route_skeleton_delta = float(self.route_skeleton_strength) * anchor_span * (
                self.route_skeleton_tangent_scale * tangent_pattern * tangent_view
                + self.route_skeleton_normal_scale * normal_pattern * normal_view
            )
            route_skeleton_delta = route_skeleton_delta.reshape(anchor.shape[0] * self.num_candidates, target_len, 2)
            route_skeleton_delta = self.preserve_reference_endpoints(route_skeleton_delta)
            route_skeleton_up_ungated = (anchor_rep + route_skeleton_delta).view(
                anchor.shape[0],
                self.num_candidates,
                target_len,
                2,
            )
            route_skeleton_delta = route_skeleton_gate * route_skeleton_delta
            residual = residual + route_skeleton_delta
            route_skeleton = (anchor_rep + route_skeleton_delta).view(
                anchor.shape[0],
                self.num_candidates,
                target_len,
                2,
            )
            route_skeleton_up = route_skeleton
        mode_skeleton_delta = None
        dynamic_mode_role_budget = None
        if mode_skeleton_params is not None or mode_role_adapter_params is not None:
            budgets = mode_role_budget.view(1, self.num_candidates, 3).expand(anchor.shape[0], -1, -1)
            if mode_role_budget_delta is not None:
                budget_strength = max(float(self.mode_role_budget_strength), 0.0)
                budget_multiplier = 1.0 + budget_strength * mode_control_strength * torch.tanh(mode_role_budget_delta)
                budgets = budgets * budget_multiplier.clamp_min(0.05)
            dynamic_mode_role_budget = budgets
            endpoint_budget = budgets[..., 0:1]
            mid_budget = budgets[..., 1:2]
            skeleton_source = None
            if mode_skeleton_params is not None:
                skeleton_source = mode_skeleton_params
            if mode_role_adapter_params is not None:
                skeleton_source = (
                    mode_role_adapter_params
                    if skeleton_source is None
                    else skeleton_source + mode_role_adapter_params
                )
            skeleton = torch.tanh(skeleton_source).view(anchor.shape[0], self.num_candidates, 1, 4)
            skeleton = skeleton.expand(-1, -1, target_len, -1).reshape(anchor.shape[0] * self.num_candidates, target_len, 4)
            endpoint_ramp = time.repeat_interleave(self.num_candidates, dim=0).clamp(0.0, 1.0).pow(1.2)
            endpoint_delta = endpoint_ramp * (
                endpoint_budget.reshape(anchor.shape[0] * self.num_candidates, 1, 1)
                * (skeleton[..., 0:1] * self.tangent_scale * tangent_rep + skeleton[..., 1:2] * self.normal_scale * normal_rep)
            )
            mid_delta = 0.5 * envelope * (
                mid_budget.reshape(anchor.shape[0] * self.num_candidates, 1, 1)
                * (skeleton[..., 2:3] * self.tangent_scale * tangent_rep + skeleton[..., 3:4] * self.normal_scale * normal_rep)
            )
            mode_skeleton_delta = mode_control_strength * (endpoint_delta + mid_delta)
            residual = residual + mode_skeleton_delta
        base_residual_before_language = residual
        language_control_delta_sum = residual.new_zeros(residual.shape)
        language_control_budget_scale = residual.new_ones((anchor.shape[0] * self.num_candidates, 1, 1))
        language_control_delta_by_kind: dict[str, torch.Tensor] = {}
        language_control_scale_by_kind: dict[str, torch.Tensor] = {}

        def _apply_language_control_delta(kind: str, delta: torch.Tensor) -> None:
            nonlocal residual, language_control_delta_sum, language_control_delta_by_kind
            delta_flat = delta.reshape(anchor.shape[0] * self.num_candidates, target_len, 2)
            language_control_delta_sum = language_control_delta_sum + delta_flat
            language_control_delta_by_kind[kind] = (
                delta_flat
                if kind not in language_control_delta_by_kind
                else language_control_delta_by_kind[kind] + delta_flat
            )
            if self.language_control_residual_type == "direct":
                residual = residual + delta_flat

        curvature_field_idx_const = DEFAULT_MOBILITY_VOCAB.fields.index("curvature")
        base_preserving_curvature_counterfactual = False
        strict_geometry_preserving_curvature_counterfactual = False
        selector_side_base_preserving_curvature_counterfactual = False
        if (
            self.counterfactual_curvature_residual_type in {
                "base_preserving_local_turn_density",
                "base_selected_projected_normal_only",
                "base_selected_projected_normal_selector_compose",
                "direction_preserving_curvature_burst",
            }
            and bool(counterfactual_mode)
            and counterfactual_field_ids is not None
        ):
            cf_field_ids_for_scope = counterfactual_field_ids.to(device=cond.device, dtype=torch.long)
            base_preserving_curvature_counterfactual = bool(
                (cf_field_ids_for_scope == int(curvature_field_idx_const)).all().detach().item()
            )
            strict_geometry_preserving_curvature_counterfactual = bool(
                base_preserving_curvature_counterfactual
                and self.counterfactual_curvature_residual_type in {
                    "base_preserving_local_turn_density",
                    "base_selected_projected_normal_only",
                }
                and prepared_base_selected_trajectory is not None
            )
            selector_side_base_preserving_curvature_counterfactual = bool(
                base_preserving_curvature_counterfactual
                and self.counterfactual_curvature_residual_type
                == "base_selected_projected_normal_selector_compose"
                and prepared_base_selected_trajectory is not None
            )
        base_preserving_composed_residual = None
        base_preserving_composed_delta = None
        base_preserving_selected_residual = None
        base_selected_normal_scalar = None
        prepared_base_selected_residual = None
        selector_side_counterfactual_delta = None
        if prepared_base_selected_trajectory is not None:
            prepared_base_selected_residual = self.preserve_reference_endpoints(
                prepared_base_selected_trajectory - anchor
            )
            base_selected_residual_view = prepared_base_selected_residual.unsqueeze(1).expand(
                -1,
                self.num_candidates,
                -1,
                -1,
            )
            base_selected_normal_scalar = (
                base_selected_residual_view * normal.view(anchor.shape[0], 1, target_len, 2)
            ).sum(dim=-1, keepdim=True)
        field_control_routing = self.build_field_control_routing(
            counterfactual_field_ids,
            mobility_tokens,
            base_mobility_tokens,
            field_control_logits,
            device=cond.device,
            dtype=cond.dtype,
        )
        field_control_candidate_logit_bonus = field_control_routing["candidate_logit_bonus"]
        field_control_proposal_scale = field_control_routing["proposal_scale"]
        field_control_proposal_shift = field_control_routing["proposal_shift"]
        field_control_family_prefixes = field_control_routing["control_family_prefixes"]
        field_control_source_kinds = field_control_routing["control_source_kinds"]
        field_control_candidate_act_ids = field_control_routing["control_candidate_act_ids"]
        field_control_preferred_act_ids = field_control_routing["preferred_act_ids"]

        direction_control_delta = None
        direction_control_params = None
        direction_control_strength = float(max(0.0, self.direction_control_strength))
        direction_control_active = (
            direction_control_strength > 0.0
            and (self.direction_control_apply == "all" or counterfactual_mode)
        )
        if base_preserving_curvature_counterfactual:
            direction_control_active = False
        if self.direction_control_type == "direction_structural_residual":
            direction_control_active = direction_control_active and bool(counterfactual_mode)
        if (
            (
                self.direction_control_head is not None
                or self.direction_control_type in {"direction_role_basis", "direction_structural_residual"}
            )
            and mobility_tokens is not None
            and direction_control_active
        ):
            direction_ids = mobility_tokens[:, DEFAULT_MOBILITY_VOCAB.fields.index("direction")].long()
            direction_ids = direction_ids.to(device=cond.device)
            direction_one_hot = F.one_hot(direction_ids, num_classes=5).to(dtype=cond.dtype)
            direction_sign = direction_one_hot.new_tensor([0.0, 1.0, -1.0, 0.0, 0.0])
            straight_target = direction_one_hot[:, 0:1]
            turn_target = direction_one_hot[:, 1:2] + direction_one_hot[:, 2:3]
            uturn_target = direction_one_hot[:, 3:4]
            multi_target = direction_one_hot[:, 4:5]
            signed_target = (direction_one_hot * direction_sign.view(1, -1)).sum(dim=1, keepdim=True)
            mode_context = cond.unsqueeze(1).expand(-1, self.num_candidates, -1)
            mode_embed = self.candidate_embed.view(1, self.num_candidates, self.hidden_dim).expand(
                anchor.shape[0],
                -1,
                -1,
            )
            if self.direction_control_head is None:
                direction_control_params = cond.new_zeros(anchor.shape[0], self.num_candidates, 14)
            else:
                direction_control_params = self.direction_control_head(torch.cat([mode_embed, mode_context], dim=-1))
                direction_control_params = torch.tanh(direction_control_params)
            param_dim = int(direction_control_params.shape[-1])
            direction_control_params = direction_control_params.view(anchor.shape[0], self.num_candidates, 1, param_dim)
            time_basis = time.unsqueeze(1).expand(-1, self.num_candidates, -1, -1)
            signed_basis = (time_basis - 0.5) * 2.0
            mid_basis = torch.sin(math.pi * time_basis).clamp_min(0.0)
            s_curve_basis = torch.sin(2.0 * math.pi * time_basis)
            end_basis = time_basis.pow(1.2)
            early_uturn_window = (
                (time_basis / 0.10).clamp(0.0, 1.0)
                * (1.0 - time_basis / 0.28).clamp(0.0, 1.0).pow(2.0)
            )
            early_uturn_return = (
                ((time_basis - 0.18) / 0.26).clamp(0.0, 1.0)
                * (1.0 - time_basis / 0.72).clamp(0.0, 1.0).pow(1.3)
            )
            signed_target_view = signed_target.view(anchor.shape[0], 1, 1, 1)
            straight_target_view = straight_target.view(anchor.shape[0], 1, 1, 1)
            turn_target_view = turn_target.view(anchor.shape[0], 1, 1, 1)
            uturn_target_view = uturn_target.view(anchor.shape[0], 1, 1, 1)
            multi_target_view = multi_target.view(anchor.shape[0], 1, 1, 1)
            direction_role_table = self.direction_mode_role_table.to(device=cond.device, dtype=cond.dtype)
            direction_role_view = direction_role_table.view(1, self.num_candidates, 1, 5).expand(
                anchor.shape[0],
                -1,
                -1,
                -1,
            )
            target_role_strength = (direction_role_view * direction_one_hot.view(anchor.shape[0], 1, 1, 5)).sum(
                dim=-1,
                keepdim=True,
            )
            signed_role_target = (
                direction_role_view
                * direction_one_hot.view(anchor.shape[0], 1, 1, 5)
                * direction_sign.view(1, 1, 1, 5)
            ).sum(dim=-1, keepdim=True)
            role_turn_view = target_role_strength * turn_target_view
            role_uturn_view = target_role_strength * uturn_target_view
            role_multi_view = target_role_strength * multi_target_view
            role_straight_view = target_role_strength * straight_target_view
            # Deterministic role basis gives each candidate a stable MTR-like
            # directional hypothesis before scorer calibration.  Learned heads
            # can then refine it, but the candidate pool no longer starts with
            # all modes sharing the same straight/u-turn geometry.
            role_signed_turn_basis = signed_role_target * role_turn_view
            role_normal_curve = (
                3.20 * role_signed_turn_basis * mid_basis
                + 0.70 * role_signed_turn_basis * signed_basis * mid_basis
                + 0.65 * role_multi_view * s_curve_basis * mid_basis
                + 0.42 * role_uturn_view * torch.sign(signed_basis + 1e-6) * mid_basis
            )
            role_tangent_curve = (
                -0.75 * role_uturn_view * end_basis
                + 0.20 * role_turn_view * end_basis
                - 0.10 * role_straight_view * signed_basis.abs() * mid_basis
            )
            uturn_endpoint_safe_tangent_hook = role_uturn_view * (
                -1.80 * early_uturn_window
                + 0.34 * early_uturn_return
            )
            uturn_endpoint_safe_normal_hook = (
                0.10
                * role_uturn_view
                * torch.sign(
                    torch.linspace(
                        -1.0,
                        1.0,
                        steps=self.num_candidates,
                        device=cond.device,
                        dtype=cond.dtype,
                    ).view(1, self.num_candidates, 1, 1)
                    + 1e-6
                )
                * (early_uturn_window - early_uturn_return)
            )
            uturn_mode_sign = torch.sign(
                torch.linspace(
                    -1.0,
                    1.0,
                    steps=self.num_candidates,
                    device=cond.device,
                    dtype=cond.dtype,
                ).view(1, self.num_candidates, 1)
                + 1e-6
            ).expand(anchor.shape[0], -1, -1)
            uturn_template_delta = self.endpoint_uturn_template_delta(
                anchor.repeat_interleave(self.num_candidates, dim=0),
                side=uturn_mode_sign.reshape(anchor.shape[0] * self.num_candidates, 1, 1),
            ).view(anchor.shape[0], self.num_candidates, target_len, 2)
            uturn_template_delta = (
                role_uturn_view.clamp(0.0, 1.0)
                * uturn_target_view
                * 0.92
                * uturn_template_delta
            )
            multi_template_delta = self.endpoint_multiturn_template_delta(
                anchor.repeat_interleave(self.num_candidates, dim=0),
                side=uturn_mode_sign.reshape(anchor.shape[0] * self.num_candidates, 1, 1),
            ).view(anchor.shape[0], self.num_candidates, target_len, 2)
            multi_template_delta = (
                role_multi_view.clamp(0.0, 1.0)
                * multi_target_view
                * 0.78
                * multi_template_delta
            )
            if self.direction_control_type == "direction_structural_residual":
                mode_sign = torch.linspace(
                    -1.0,
                    1.0,
                    steps=self.num_candidates,
                    device=cond.device,
                    dtype=cond.dtype,
                ).view(1, self.num_candidates, 1, 1)
                raw_residual_view_for_control = raw_residual.view(
                    anchor.shape[0],
                    self.num_candidates,
                    target_len,
                    2,
                )
                existing_tangent = raw_residual_view_for_control[..., 0:1]
                existing_normal = raw_residual_view_for_control[..., 1:2]
                signed_turn_basis = signed_target_view * turn_target_view
                low_turn = (
                    signed_turn_basis
                    * (0.65 + 0.20 * target_role_strength)
                    * (0.75 + 0.25 * direction_control_params[..., 0:1].abs())
                    * mid_basis.pow(1.15)
                )
                multi_wave_basis = s_curve_basis + 0.35 * torch.sin(4.0 * math.pi * time_basis) * mid_basis
                low_multi = (
                    multi_target_view
                    * (0.72 + 0.35 * target_role_strength)
                    * (0.55 + 0.25 * direction_control_params[..., 1:2].abs())
                    * multi_wave_basis
                    * mid_basis.pow(1.20)
                )
                low_uturn = (
                    uturn_target_view
                    * torch.sign(mode_sign + 1e-6)
                    * (0.18 + 0.12 * mode_sign.abs())
                    * (0.60 + 0.25 * direction_control_params[..., 2:3].abs())
                    * torch.sign(signed_basis + 1e-6)
                    * mid_basis.pow(1.25)
                )
                straight_suppress = -straight_target_view * 0.25 * direction_control_params[..., 3:4].abs() * mid_basis
                # Straight targets need to reduce the generator's existing local
                # turn energy. Without this damping, multi_turn->straight cases
                # preserve the old residual and fail target-effect checks while
                # still looking structurally stable.
                straight_normal_damp = -straight_target_view * (0.42 + 0.18 * target_role_strength) * existing_normal * mid_basis
                straight_tangent_damp = -straight_target_view * 0.12 * existing_tangent * mid_basis
                normal_curve = low_turn + low_multi + low_uturn + straight_suppress + straight_normal_damp
                tangent_curve = (
                    0.05
                    * (
                        direction_control_params[..., 4:5] * turn_target_view
                        - direction_control_params[..., 5:6].abs() * uturn_target_view
                    )
                    * mid_basis
                    + straight_tangent_damp
                )
            elif self.direction_control_type in {"hybrid_rule_basis", "direction_role_basis"}:
                raw_residual_view_for_control = raw_residual.view(anchor.shape[0], self.num_candidates, target_len, 2)
                existing_tangent = raw_residual_view_for_control[..., 0:1]
                existing_normal = raw_residual_view_for_control[..., 1:2]
                straight_damp_strength = direction_control_params[..., 2:3].abs()
                straight_tangent_smooth = -straight_target_view * direction_control_params[..., 6:7].abs() * existing_tangent * mid_basis
                straight_normal_damp = -straight_target_view * straight_damp_strength * existing_normal * mid_basis
                # Table-3 direction is a low-frequency local-frame contract:
                # left/right should be a smooth single turn, not high-frequency
                # curvature that gets decoded as multi_turn.  The deterministic
                # role basis opens that candidate before the learned heads and
                # damps raw residual oscillations for the matching turn modes.
                signed_turn_basis = signed_target_view * turn_target_view
                deterministic_turn = signed_turn_basis * mid_basis
                turn_residual_normal_damp = -1.20 * role_turn_view * existing_normal * mid_basis
                turn_residual_tangent_damp = -0.35 * role_turn_view * existing_tangent * mid_basis
                turn_normal = (
                    signed_turn_basis
                    * (
                        direction_control_params[..., 0:1].abs() * mid_basis
                        + direction_control_params[..., 1:2] * signed_basis * mid_basis
                    )
                    * (turn_target_view + 0.25 * multi_target_view)
                )
                multi_curve = multi_target_view * direction_control_params[..., 3:4] * s_curve_basis * mid_basis
                # U-turn needs both backward longitudinal progress and enough
                # local heading change; a pure negative tangent often moves the
                # endpoint without satisfying the local-frame rule.
                uturn_tangent = -uturn_target_view * (
                    direction_control_params[..., 4:5].abs() * end_basis
                    + 0.5 * direction_control_params[..., 7:8].abs() * mid_basis
                )
                mode_sign = torch.linspace(
                    -1.0,
                    1.0,
                    steps=self.num_candidates,
                    device=cond.device,
                    dtype=cond.dtype,
                ).view(1, self.num_candidates, 1, 1)
                uturn_hook = (
                    uturn_target_view
                    * (0.35 + 0.65 * mode_sign.abs())
                    * torch.sign(mode_sign + 1e-6)
                    * direction_control_params[..., 8:9].abs()
                    * s_curve_basis
                    * mid_basis
                )
                uturn_end_hook = (
                    uturn_target_view
                    * torch.sign(mode_sign + 1e-6)
                    * direction_control_params[..., 9:10].abs()
                    * signed_basis
                    * end_basis
                )
                progress_tangent = direction_control_params[..., 5:6] * end_basis * (turn_target_view + straight_target_view)
                normal_curve = (
                    role_normal_curve
                    + straight_normal_damp
                    + deterministic_turn
                    + turn_residual_normal_damp
                    + turn_normal
                    + multi_curve
                    + uturn_hook
                    + uturn_end_hook
                )
                tangent_curve = (
                    role_tangent_curve
                    + straight_tangent_smooth
                    + turn_residual_tangent_damp
                    + uturn_tangent
                    + 0.25 * progress_tangent
                )
            else:
                signed_turn_basis = signed_target_view * turn_target_view
                normal_curve = (
                    signed_turn_basis
                    * (
                        direction_control_params[..., 0:1] * mid_basis
                        + direction_control_params[..., 1:2] * signed_basis * mid_basis
                    )
                    * (turn_target_view + 0.25 * multi_target_view)
                )
                straight_damp = -straight_target_view * direction_control_params[..., 2:3].abs() * mid_basis
                multi_curve = multi_target_view * direction_control_params[..., 3:4] * s_curve_basis * mid_basis
                uturn_tangent = -uturn_target_view * direction_control_params[..., 4:5].abs() * end_basis
                progress_tangent = direction_control_params[..., 5:6] * end_basis * (turn_target_view + straight_target_view)
                normal_curve = normal_curve + role_normal_curve + signed_turn_basis * mid_basis + straight_damp + multi_curve
                tangent_curve = role_tangent_curve + uturn_tangent + 0.25 * progress_tangent
            direction_control_delta = direction_control_strength * envelope.view(1, 1, target_len, 1) * (
                0.35
                * max(0.0, self.direction_control_tangent_weight)
                * self.tangent_scale
                * tangent_curve
                * tangent.view(anchor.shape[0], 1, target_len, 2)
                + 0.35 * self.normal_scale * normal_curve * normal.view(anchor.shape[0], 1, target_len, 2)
            )
            direction_control_delta = direction_control_delta + direction_control_strength * (
                max(0.0, self.direction_control_tangent_weight)
                * self.tangent_scale
                * uturn_endpoint_safe_tangent_hook
                * tangent.view(anchor.shape[0], 1, target_len, 2)
                + self.normal_scale
                * uturn_endpoint_safe_normal_hook
                * normal.view(anchor.shape[0], 1, target_len, 2)
            )
            direction_control_delta = direction_control_delta + direction_control_strength * uturn_template_delta
            direction_control_delta = direction_control_delta + direction_control_strength * multi_template_delta
            _apply_language_control_delta("direction", direction_control_delta)
        curvature_control_delta = None
        curvature_control_params = None
        curvature_control_strength = float(max(0.0, self.curvature_control_strength))
        curvature_control_active = (
            curvature_control_strength > 0.0
            and (self.curvature_control_apply == "all" or counterfactual_mode)
        )
        if strict_geometry_preserving_curvature_counterfactual:
            curvature_control_active = False
        if (
            self.curvature_control_head is not None
            and mobility_tokens is not None
            and curvature_control_active
        ):
            distance_ids = mobility_tokens[:, DEFAULT_MOBILITY_VOCAB.fields.index("distance")].long()
            distance_ids = distance_ids.to(device=cond.device).clamp(0, 3)
            curvature_ids = mobility_tokens[:, DEFAULT_MOBILITY_VOCAB.fields.index("curvature")].long()
            curvature_ids = curvature_ids.to(device=cond.device).clamp(0, 2)
            curvature_one_hot = F.one_hot(curvature_ids, num_classes=3).to(dtype=cond.dtype)
            direction_ids_for_curvature = mobility_tokens[:, DEFAULT_MOBILITY_VOCAB.fields.index("direction")].long()
            direction_ids_for_curvature = direction_ids_for_curvature.to(device=cond.device).clamp(0, 4)
            simple_turn_direction = (
                (direction_ids_for_curvature == DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]["left_turn"])
                | (direction_ids_for_curvature == DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]["right_turn"])
            ).to(dtype=cond.dtype)
            straight_direction = (
                direction_ids_for_curvature == DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]["straight"]
            ).to(dtype=cond.dtype)
            smooth_target = curvature_one_hot[:, 0:1].view(anchor.shape[0], 1, 1, 1)
            moderate_target = curvature_one_hot[:, 1:2].view(anchor.shape[0], 1, 1, 1)
            winding_target = curvature_one_hot[:, 2:3].view(anchor.shape[0], 1, 1, 1)
            simple_turn_direction_view = simple_turn_direction.view(anchor.shape[0], 1, 1, 1)
            straight_direction_view = straight_direction.view(anchor.shape[0], 1, 1, 1)
            curvature_cf_mask = cond.new_zeros(anchor.shape[0], 1, 1, 1)
            if counterfactual_mode and counterfactual_field_ids is not None:
                curvature_field_idx = DEFAULT_MOBILITY_VOCAB.fields.index("curvature")
                curvature_cf_mask = (
                    counterfactual_field_ids.to(device=cond.device, dtype=torch.long).view(anchor.shape[0], 1, 1, 1)
                    == int(curvature_field_idx)
                ).to(dtype=cond.dtype)
            preserve_direction_for_curvature = curvature_cf_mask * (straight_direction_view + simple_turn_direction_view).clamp(max=1.0)

            curvature_context = cond
            if token_seq is not None:
                curvature_context = curvature_context + token_seq.mean(dim=1)
            mode_context = curvature_context.unsqueeze(1).expand(-1, self.num_candidates, -1)
            mode_embed = self.candidate_embed.view(1, self.num_candidates, self.hidden_dim).expand(
                anchor.shape[0],
                -1,
                -1,
            )
            curvature_control_params = torch.tanh(
                self.curvature_control_head(torch.cat([mode_embed, mode_context], dim=-1))
            ).view(anchor.shape[0], self.num_candidates, 1, 8)

            time_basis = time.unsqueeze(1).expand(-1, self.num_candidates, -1, -1)
            mid_basis = torch.sin(math.pi * time_basis).clamp_min(0.0)
            signed_basis = (time_basis - 0.5) * 2.0
            curvature_local_window = (
                1.0 - ((time_basis - 0.5).abs() / 0.34)
            ).clamp(min=0.0, max=1.0)
            curvature_local_window = curvature_local_window * (0.80 + 0.20 * mid_basis)
            curvature_shape_window = (
                mid_basis * (1.0 - preserve_direction_for_curvature)
                + curvature_local_window * preserve_direction_for_curvature
            )
            # OD-chord normalization removes the exact meter length from the
            # decoder input. Use the distance token as a route-scale proxy so
            # long Porto routes get enough oscillation to be visible under the
            # RDP/strong-turn contract, mirroring the pseudo-target builder.
            moderate_cycles_lookup = cond.new_tensor([2.0, 5.0, 10.0, 16.0])
            winding_cycles_lookup = cond.new_tensor([4.0, 10.0, 20.0, 28.0])
            moderate_amp_lookup = cond.new_tensor([0.095, 0.065, 0.038, 0.022])
            winding_amp_lookup = cond.new_tensor([0.110, 0.075, 0.045, 0.024])
            moderate_cycles = moderate_cycles_lookup[distance_ids].view(anchor.shape[0], 1, 1, 1)
            winding_cycles = winding_cycles_lookup[distance_ids].view(anchor.shape[0], 1, 1, 1)
            moderate_amp = moderate_amp_lookup[distance_ids].view(anchor.shape[0], 1, 1, 1)
            winding_amp = winding_amp_lookup[distance_ids].view(anchor.shape[0], 1, 1, 1)
            # Curvature counterfactuals should add local shape without changing
            # the non-target direction token.  Under OD-chord conditioning a
            # high-frequency winding basis can otherwise be decoded as a global
            # direction change, which is the current raw-collateral failure.
            bounded_winding_cycles = winding_cycles * (1.0 - 0.35 * preserve_direction_for_curvature)
            bounded_winding_amp = winding_amp * (1.0 - 0.35 * preserve_direction_for_curvature)
            wave1 = torch.sin(math.pi * moderate_cycles * time_basis)
            wave2 = torch.sin(math.pi * bounded_winding_cycles * time_basis)
            wave3 = torch.sin(math.pi * (bounded_winding_cycles + 1.0) * time_basis)
            mode_phase = torch.linspace(
                -1.0,
                1.0,
                steps=self.num_candidates,
                device=cond.device,
                dtype=cond.dtype,
            ).view(1, self.num_candidates, 1, 1)
            mode_sign = torch.sign(mode_phase + 1e-6)

            raw_residual_view_for_curvature = raw_residual.view(anchor.shape[0], self.num_candidates, target_len, 2)
            existing_normal = raw_residual_view_for_curvature[..., 1:2]
            smooth_damp = -smooth_target * (0.40 + 0.35 * curvature_control_params[..., 0:1].abs()) * existing_normal * mid_basis
            moderate_curve = moderate_target * (
                moderate_amp
                * (
                    (1.00 + 0.20 * curvature_control_params[..., 1:2].abs()) * wave1
                    + 0.15 * curvature_control_params[..., 2:3] * wave2
                )
            ) * curvature_shape_window
            winding_curve = winding_target * (
                bounded_winding_amp
                * (
                    (0.85 + 0.20 * curvature_control_params[..., 3:4].abs()) * wave2
                    + mode_sign * (0.35 + 0.15 * curvature_control_params[..., 4:5].abs()) * wave3
                    + 0.10 * curvature_control_params[..., 5:6] * wave1
                )
            ) * curvature_shape_window * (1.0 - 0.65 * simple_turn_direction_view) * (1.0 - 0.15 * preserve_direction_for_curvature)
            normal_curve = smooth_damp + moderate_curve + winding_curve
            tangent_curve = (
                (moderate_target + winding_target)
                * (0.08 * curvature_control_params[..., 6:7] * signed_basis * curvature_shape_window)
                + winding_target * 0.06 * curvature_control_params[..., 7:8] * wave1
            ) * curvature_shape_window * (1.0 - 0.65 * preserve_direction_for_curvature)
            if (
                self.curvature_control_projection == "highfreq_preserve_direction"
                and float(self.curvature_control_projection_strength) > 0.0
            ):
                projection_mask = preserve_direction_for_curvature * curvature_shape_window
                projected_normal = self._project_curve_out_bases(
                    normal_curve,
                    [
                        curvature_shape_window,
                        signed_basis * curvature_shape_window,
                        mid_basis * curvature_shape_window,
                    ],
                    projection_mask,
                )
                projected_tangent = self._project_curve_out_bases(
                    tangent_curve,
                    [
                        signed_basis * curvature_shape_window,
                        wave1 * curvature_shape_window,
                    ],
                    projection_mask,
                )
                projection_strength = float(
                    max(0.0, min(1.0, self.curvature_control_projection_strength))
                )
                normal_curve = normal_curve + projection_strength * (projected_normal - normal_curve)
                tangent_curve = tangent_curve + projection_strength * (projected_tangent - tangent_curve)
            curvature_control_delta = curvature_control_strength * envelope.view(1, 1, target_len, 1) * (
                0.06 * self.tangent_scale * tangent_curve * tangent.view(anchor.shape[0], 1, target_len, 2)
                + normal_curve * normal.view(anchor.shape[0], 1, target_len, 2)
            )
            _apply_language_control_delta("curvature", curvature_control_delta)
        reference_control_delta = None
        reference_control_params = None
        reference_control_strength = float(max(0.0, self.reference_control_strength))
        reference_local_main_active = (
            str(getattr(self, "reference_distance_candidate_mode", "global_normal")) == "local_direction_preserving"
            and float(max(0.0, self.reference_distance_candidate_strength)) > 0.0
        )
        reference_control_active = (
            reference_control_strength > 0.0
            and (self.reference_control_apply == "all" or counterfactual_mode)
        )
        if reference_local_main_active and counterfactual_mode:
            reference_control_active = False
        if base_preserving_curvature_counterfactual:
            reference_control_active = False
        if (
            (self.reference_control_head is not None or self.reference_control_type == "mode_role_basis")
            and mobility_tokens is not None
            and reference_control_active
        ):
            reference_ids = mobility_tokens[:, DEFAULT_MOBILITY_VOCAB.fields.index("reference_adherence")].long()
            reference_ids = reference_ids.to(device=cond.device).clamp(0, 3)
            reference_one_hot = F.one_hot(reference_ids, num_classes=4).to(dtype=cond.dtype)
            ref_ids = DEFAULT_MOBILITY_VOCAB.token_to_id["reference_adherence"]
            unknown_target = reference_one_hot[:, ref_ids["unknown"] : ref_ids["unknown"] + 1].view(anchor.shape[0], 1, 1, 1)
            tight_target = reference_one_hot[:, ref_ids["tight"] : ref_ids["tight"] + 1].view(anchor.shape[0], 1, 1, 1)
            moderate_target = reference_one_hot[:, ref_ids["moderate"] : ref_ids["moderate"] + 1].view(anchor.shape[0], 1, 1, 1)
            loose_target = reference_one_hot[:, ref_ids["loose"] : ref_ids["loose"] + 1].view(anchor.shape[0], 1, 1, 1)

            reference_context = cond
            if token_seq is not None:
                reference_context = reference_context + token_seq.mean(dim=1)
            mode_context = reference_context.unsqueeze(1).expand(-1, self.num_candidates, -1)
            mode_embed = self.candidate_embed.view(1, self.num_candidates, self.hidden_dim).expand(
                anchor.shape[0],
                -1,
                -1,
            )
            if self.reference_control_head is None:
                reference_control_params = cond.new_zeros(anchor.shape[0], self.num_candidates, 1, 6)
            else:
                reference_control_params = torch.tanh(
                    self.reference_control_head(torch.cat([mode_embed, mode_context], dim=-1))
                ).view(anchor.shape[0], self.num_candidates, 1, 6)

            time_basis = time.unsqueeze(1).expand(-1, self.num_candidates, -1, -1)
            mid_basis = torch.sin(math.pi * time_basis).clamp_min(0.0)
            s_curve_basis = torch.sin(2.0 * math.pi * time_basis)
            mode_phase = torch.linspace(
                -1.0,
                1.0,
                steps=self.num_candidates,
                device=cond.device,
                dtype=cond.dtype,
            ).view(1, self.num_candidates, 1, 1)
            mode_sign = torch.sign(mode_phase + 1e-6)
            raw_residual_view_for_reference = raw_residual.view(anchor.shape[0], self.num_candidates, target_len, 2)
            existing_normal = raw_residual_view_for_reference[..., 1:2]

            if self.reference_control_type == "mode_role_basis":
                mode_roles = self.mode_role_table.to(device=cond.device, dtype=cond.dtype)
                mode_roles = mode_roles.view(1, self.num_candidates, 1, self.mode_role_dim)
                close_role = mode_roles[..., 0:1]
                loose_role = mode_roles[..., 2:3]
                winding_role = mode_roles[..., 3:4]
                moderate_role = (1.0 - torch.clamp(close_role + loose_role, 0.0, 1.0)).clamp(0.0, 1.0)
                reference_mode_sign = torch.where(
                    loose_role > close_role,
                    mode_sign,
                    0.35 * mode_sign,
                )
                tight_damp = -tight_target * (0.35 + 0.45 * close_role) * existing_normal
                moderate_damp = -moderate_target * (0.12 + 0.16 * moderate_role) * existing_normal
                loose_amp = loose_target * (0.025 + 0.085 * loose_role + 0.030 * winding_role)
                loose_wave = reference_mode_sign * mid_basis + 0.18 * winding_role * s_curve_basis * mid_basis
                moderate_wave = moderate_target * (0.010 + 0.020 * moderate_role) * reference_mode_sign * mid_basis
                unknown_damp = -unknown_target * (0.08 + 0.10 * close_role) * existing_normal
                normal_curve = (
                    (tight_damp + moderate_damp + unknown_damp) * mid_basis
                    + loose_amp * loose_wave
                    + moderate_wave
                )
                tangent_curve = loose_target * 0.012 * (1.0 + loose_role) * (time_basis - 0.5) * mid_basis
            else:
                tight_damp = -tight_target * (0.70 + 0.30 * reference_control_params[..., 0:1].abs()) * existing_normal
                moderate_damp = -moderate_target * (0.25 + 0.20 * reference_control_params[..., 1:2].abs()) * existing_normal
                loose_amp = loose_target * (
                    0.09
                    + 0.05 * reference_control_params[..., 2:3].abs()
                    + 0.03 * mode_sign.abs()
                )
                loose_wave = (
                    mode_sign * mid_basis
                    + 0.35 * reference_control_params[..., 3:4] * s_curve_basis * mid_basis
                )
                moderate_wave = moderate_target * 0.025 * reference_control_params[..., 4:5] * mode_sign * mid_basis
                unknown_damp = -unknown_target * 0.10 * existing_normal
                normal_curve = (tight_damp + moderate_damp + unknown_damp) * mid_basis + loose_amp * loose_wave + moderate_wave
                tangent_curve = loose_target * 0.03 * reference_control_params[..., 5:6] * (time_basis - 0.5) * mid_basis
            reference_control_delta = reference_control_strength * envelope.view(1, 1, target_len, 1) * (
                0.05 * self.tangent_scale * tangent_curve * tangent.view(anchor.shape[0], 1, target_len, 2)
                + normal_curve * normal.view(anchor.shape[0], 1, target_len, 2)
            )
            _apply_language_control_delta("reference_adherence", reference_control_delta)
        counterfactual_residual_delta = None
        counterfactual_residual_params = None
        counterfactual_edit_mask = None
        counterfactual_preserve_mask = None
        counterfactual_edit_slot_mask = None
        counterfactual_local_proposal_delta = None
        counterfactual_local_proposal_params = None
        counterfactual_local_proposal_family = None
        counterfactual_mask_family = None
        reference_local_direct_semantic_latent_used = None
        reference_local_direct_act_embedding_used = None
        reference_local_act_condition_ids = None
        reference_local_semantic_feature_norm = None
        reference_local_variant_kind = None
        reference_local_variant_uses_mode_sign = None
        reference_local_contraction_ladder = None
        if cf_adapter_scale != 0.0 and counterfactual_field_ids is not None:
            cf_field_ids = counterfactual_field_ids.to(device=cond.device, dtype=torch.long).clamp(
                0,
                len(DEFAULT_MOBILITY_VOCAB.fields) - 1,
            )
            mode_embed = self.candidate_embed.view(1, self.num_candidates, self.hidden_dim).expand(
                anchor.shape[0],
                -1,
                -1,
            )
            mode_context = cond.unsqueeze(1).expand(-1, self.num_candidates, -1)
            field_context = self.counterfactual_field_embed(cf_field_ids).to(dtype=cond.dtype)
            field_context = field_context.unsqueeze(1).expand(-1, self.num_candidates, -1)
            reference_field_idx_const = DEFAULT_MOBILITY_VOCAB.fields.index("reference_adherence")
            reference_field_rows = cf_field_ids == int(reference_field_idx_const)
            reference_local_direct_semantic_latent_used = torch.zeros(
                anchor.shape[0],
                device=cond.device,
                dtype=torch.bool,
            )
            reference_local_direct_act_embedding_used = torch.zeros(
                anchor.shape[0],
                device=cond.device,
                dtype=torch.bool,
            )
            reference_local_semantic_feature_norm = torch.zeros(
                anchor.shape[0],
                device=cond.device,
                dtype=cond.dtype,
            )
            reference_local_latent_tokens = mode_embed.new_zeros(
                anchor.shape[0],
                self.num_candidates,
                self.hidden_dim,
            )
            reference_local_act_tokens = mode_embed.new_zeros(
                anchor.shape[0],
                self.num_candidates,
                self.hidden_dim,
            )
            if (
                reference_field_rows.any()
                and field_control_latents is not None
                and self.reference_field_control_available
            ):
                raw_reference_latent = field_control_latents.get("reference_adherence")
                if (
                    raw_reference_latent is not None
                    and raw_reference_latent.ndim == 2
                    and raw_reference_latent.shape[0] == anchor.shape[0]
                ):
                    projected_reference_latent = self.reference_local_control_latent_proj(
                        raw_reference_latent.to(device=cond.device, dtype=cond.dtype)
                    )
                    reference_local_latent_tokens = projected_reference_latent.unsqueeze(1).expand(
                        -1,
                        self.num_candidates,
                        -1,
                    )
                    reference_local_direct_semantic_latent_used = reference_field_rows.clone()
                    reference_local_semantic_feature_norm = torch.linalg.vector_norm(
                        projected_reference_latent,
                        dim=-1,
                    )
            if field_control_candidate_act_ids is not None:
                reference_local_act_condition_ids = field_control_candidate_act_ids.to(
                    device=cond.device,
                    dtype=torch.long,
                )
            else:
                reference_local_act_condition_ids = torch.arange(
                    self.num_candidates,
                    device=cond.device,
                    dtype=torch.long,
                ).view(1, -1).expand(anchor.shape[0], -1)
            if reference_field_rows.any():
                safe_act_ids = reference_local_act_condition_ids.clamp(
                    0,
                    max(self.reference_local_control_act_embed.num_embeddings - 1, 0),
                )
                reference_local_act_tokens = self.reference_local_control_act_embed(safe_act_ids)
                reference_local_direct_act_embedding_used = reference_field_rows.clone()
            residual_input = torch.cat([mode_embed, mode_context, field_context], dim=-1)
            reference_residual_input = torch.cat(
                [
                    mode_embed,
                    mode_context,
                    field_context,
                    reference_local_latent_tokens,
                    reference_local_act_tokens,
                ],
                dim=-1,
            )
            counterfactual_residual_params = residual_input.new_zeros(anchor.shape[0], self.num_candidates, 6)
            for field_idx, head in enumerate(self.counterfactual_field_residual_heads):
                field_mask = cf_field_ids == int(field_idx)
                if bool(field_mask.any()):
                    field_params = torch.tanh(head(residual_input[field_mask]))
                    counterfactual_residual_params[field_mask] = field_params.to(
                        device=counterfactual_residual_params.device,
                        dtype=counterfactual_residual_params.dtype,
                    )

            time_basis = time.unsqueeze(1).expand(-1, self.num_candidates, -1, -1)
            mid_basis = torch.sin(math.pi * time_basis).clamp_min(0.0)
            signed_basis = 2.0 * time_basis - 1.0
            s_curve_basis = torch.sin(2.0 * math.pi * time_basis)
            high_curve_basis = torch.sin(4.0 * math.pi * time_basis)
            end_basis = time_basis.clamp_min(0.0).pow(1.2)
            curvature_local_window = (
                1.0 - ((time_basis - 0.5).abs() / 0.34)
            ).clamp(min=0.0, max=1.0)
            curvature_local_window = curvature_local_window * (0.80 + 0.20 * mid_basis)
            params = counterfactual_residual_params.view(anchor.shape[0], self.num_candidates, 1, 6)
            generic_tangent_curve = (
                params[..., 0:1] * mid_basis
                + params[..., 1:2] * end_basis * mid_basis
                + params[..., 4:5] * s_curve_basis * mid_basis
            )
            generic_normal_curve = (
                params[..., 2:3] * mid_basis
                + params[..., 3:4] * s_curve_basis * mid_basis
                + params[..., 5:6] * signed_basis * mid_basis
            )
            direction_tangent_curve = 0.20 * params[..., 1:2] * end_basis * mid_basis
            direction_normal_curve = (
                params[..., 2:3] * mid_basis
                + params[..., 3:4] * signed_basis * mid_basis
                + 0.40 * params[..., 5:6] * s_curve_basis * mid_basis
            )
            curvature_tangent_curve = 0.10 * params[..., 4:5] * s_curve_basis * curvature_local_window
            curvature_normal_curve = (
                params[..., 3:4] * s_curve_basis * curvature_local_window
                + params[..., 5:6] * high_curve_basis * curvature_local_window
                + 0.12 * params[..., 2:3] * signed_basis * curvature_local_window
            )
            reference_tangent_curve = 0.15 * params[..., 1:2] * signed_basis * mid_basis
            reference_normal_curve = (
                params[..., 2:3] * mid_basis
                + 0.50 * params[..., 3:4] * s_curve_basis * mid_basis
                + 0.20 * params[..., 5:6] * signed_basis * mid_basis
            )
            field_view = cf_field_ids.view(anchor.shape[0], 1, 1, 1)
            direction_field_idx = DEFAULT_MOBILITY_VOCAB.fields.index("direction")
            curvature_field_idx = DEFAULT_MOBILITY_VOCAB.fields.index("curvature")
            reference_field_idx = DEFAULT_MOBILITY_VOCAB.fields.index("reference_adherence")
            direction_mask = field_view == direction_field_idx
            curvature_mask = field_view == curvature_field_idx
            reference_mask = field_view == reference_field_idx
            if mobility_tokens is not None:
                ref_vocab = DEFAULT_MOBILITY_VOCAB.token_to_id["reference_adherence"]
                reference_target_ids = mobility_tokens[:, reference_field_idx].to(
                    device=cond.device,
                    dtype=torch.long,
                ).clamp(0, 3)
                reference_one_hot_cf = F.one_hot(reference_target_ids, num_classes=4).to(dtype=cond.dtype)
                unknown_ref_cf = reference_one_hot_cf[
                    :, ref_vocab["unknown"] : ref_vocab["unknown"] + 1
                ].view(anchor.shape[0], 1, 1, 1)
                tight_ref_cf = reference_one_hot_cf[
                    :, ref_vocab["tight"] : ref_vocab["tight"] + 1
                ].view(anchor.shape[0], 1, 1, 1)
                moderate_ref_cf = reference_one_hot_cf[
                    :, ref_vocab["moderate"] : ref_vocab["moderate"] + 1
                ].view(anchor.shape[0], 1, 1, 1)
                loose_ref_cf = reference_one_hot_cf[
                    :, ref_vocab["loose"] : ref_vocab["loose"] + 1
                ].view(anchor.shape[0], 1, 1, 1)
                residual_view_for_cf_reference = residual.view(
                    anchor.shape[0],
                    self.num_candidates,
                    target_len,
                    2,
                )
                tangent_view_for_cf_reference = tangent.view(anchor.shape[0], 1, target_len, 2)
                normal_view_for_cf_reference = normal.view(anchor.shape[0], 1, target_len, 2)
                existing_ref_tangent = (
                    residual_view_for_cf_reference * tangent_view_for_cf_reference
                ).sum(dim=-1, keepdim=True)
                existing_ref_normal = (
                    residual_view_for_cf_reference * normal_view_for_cf_reference
                ).sum(dim=-1, keepdim=True)
                reference_close_strength = (
                    0.95 * tight_ref_cf
                    + 0.55 * moderate_ref_cf
                    + 0.35 * unknown_ref_cf
                )
                reference_mode_phase = torch.linspace(
                    -1.0,
                    1.0,
                    steps=self.num_candidates,
                    device=cond.device,
                    dtype=cond.dtype,
                ).view(1, self.num_candidates, 1, 1)
                reference_mode_sign = torch.sign(reference_mode_phase + 1e-6)
                reference_tangent_curve = (
                    -0.35 * reference_close_strength * existing_ref_tangent * mid_basis
                    + loose_ref_cf
                    * 0.012
                    * (
                        0.5 * reference_mode_sign * s_curve_basis
                        + params[..., 1:2] * signed_basis
                    )
                    * mid_basis
                )
                reference_normal_curve = (
                    -reference_close_strength * existing_ref_normal * mid_basis
                    + loose_ref_cf
                    * (
                        0.055 * reference_mode_sign * mid_basis
                        + 0.020 * params[..., 3:4] * s_curve_basis * mid_basis
                    )
                )
                if self.counterfactual_reference_residual_type == "normal_mean_shift":
                    reference_shift_window = mid_basis.clamp_min(0.0)
                    mean_existing_ref_normal = (
                        existing_ref_normal * reference_shift_window
                    ).sum(dim=2, keepdim=True) / reference_shift_window.sum(dim=2, keepdim=True).clamp_min(1e-6)
                    reference_normal_curve = (
                        -(1.10 * tight_ref_cf + 0.62 * moderate_ref_cf + 0.42 * unknown_ref_cf)
                        * mean_existing_ref_normal
                        * reference_shift_window
                        + loose_ref_cf
                        * (
                            0.060 * reference_mode_sign * reference_shift_window
                            + 0.018 * params[..., 3:4] * s_curve_basis * reference_shift_window
                        )
                    )
                if self.counterfactual_reference_residual_type == "normal_hybrid_shift":
                    reference_shift_window = mid_basis.clamp_min(0.0)
                    mean_existing_ref_normal = (
                        existing_ref_normal * reference_shift_window
                    ).sum(dim=2, keepdim=True) / reference_shift_window.sum(dim=2, keepdim=True).clamp_min(1e-6)
                    smooth_existing_ref_normal = 0.55 * existing_ref_normal + 0.45 * mean_existing_ref_normal
                    reference_normal_curve = (
                        -(0.90 * tight_ref_cf + 0.52 * moderate_ref_cf + 0.35 * unknown_ref_cf)
                        * smooth_existing_ref_normal
                        * reference_shift_window
                        + loose_ref_cf
                        * (
                            0.070 * reference_mode_sign * reference_shift_window
                            + 0.018 * params[..., 3:4] * s_curve_basis * reference_shift_window
                        )
                    )
                if self.counterfactual_reference_residual_type in {
                    "normal_only",
                    "normal_mean_shift",
                    "normal_hybrid_shift",
                    "normal_only_projected",
                }:
                    reference_tangent_curve = torch.zeros_like(reference_tangent_curve)
                if self.counterfactual_reference_residual_type == "normal_only_projected":
                    reference_local_window = (
                        torch.exp(-0.5 * ((time_basis - 0.5).abs() / 0.28).pow(2)) * mid_basis
                    )
                    reference_normal_curve = (
                        -reference_close_strength * existing_ref_normal * reference_local_window
                        + loose_ref_cf
                        * (
                            0.050 * reference_mode_sign * torch.sin(2.0 * math.pi * time_basis)
                            + 0.014 * torch.sin(4.0 * math.pi * time_basis)
                        )
                        * reference_local_window
                    )
                    reference_projection_mask = reference_mask.to(dtype=cond.dtype) * reference_local_window
                    reference_normal_curve = self._project_curve_out_bases(
                        reference_normal_curve,
                        [
                            reference_local_window,
                            signed_basis * reference_local_window,
                            mid_basis * reference_local_window,
                        ],
                        reference_projection_mask,
                    )
            curvature_preserve_direction_mask = torch.zeros_like(curvature_mask)
            if mobility_tokens is not None:
                direction_target_ids = mobility_tokens[:, direction_field_idx].to(device=cond.device, dtype=torch.long)
                direction_preserve_ids = (
                    (direction_target_ids == DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]["straight"])
                    | (direction_target_ids == DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]["left_turn"])
                    | (direction_target_ids == DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]["right_turn"])
                ).view(anchor.shape[0], 1, 1, 1)
                curvature_preserve_direction_mask = curvature_mask & direction_preserve_ids
                if (
                    base_mobility_tokens is not None
                    and base_mobility_tokens.ndim == mobility_tokens.ndim
                    and base_mobility_tokens.shape[0] == mobility_tokens.shape[0]
                    and base_mobility_tokens.shape[1] > direction_field_idx
                ):
                    direction_source_ids = base_mobility_tokens[:, direction_field_idx].to(
                        device=cond.device,
                        dtype=torch.long,
                    )
                else:
                    direction_source_ids = direction_target_ids
                multi_turn_id = int(DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]["multi_turn"])
                straight_id = int(DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]["straight"])
                left_id = int(DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]["left_turn"])
                right_id = int(DEFAULT_MOBILITY_VOCAB.token_to_id["direction"]["right_turn"])
                direction_multi_mask = direction_mask & (
                    direction_target_ids.view(anchor.shape[0], 1, 1, 1) == multi_turn_id
                )
                source_to_multi_mask = direction_multi_mask & (
                    direction_source_ids.view(anchor.shape[0], 1, 1, 1) != multi_turn_id
                )
                source_straight_to_multi_mask = source_to_multi_mask & (
                    direction_source_ids.view(anchor.shape[0], 1, 1, 1) == straight_id
                )
                source_turn_to_multi_mask = source_to_multi_mask & (
                    (direction_source_ids.view(anchor.shape[0], 1, 1, 1) == left_id)
                    | (direction_source_ids.view(anchor.shape[0], 1, 1, 1) == right_id)
                )
            else:
                direction_multi_mask = torch.zeros_like(direction_mask)
                source_to_multi_mask = torch.zeros_like(direction_mask)
                source_straight_to_multi_mask = torch.zeros_like(direction_mask)
                source_turn_to_multi_mask = torch.zeros_like(direction_mask)
            if mobility_tokens is not None:
                curvature_target_ids = mobility_tokens[:, curvature_field_idx].to(
                    device=cond.device,
                    dtype=torch.long,
                ).clamp(0, 2)
                if (
                    base_mobility_tokens is not None
                    and base_mobility_tokens.ndim == mobility_tokens.ndim
                    and base_mobility_tokens.shape[0] == mobility_tokens.shape[0]
                    and base_mobility_tokens.shape[1] > curvature_field_idx
                ):
                    curvature_source_ids = base_mobility_tokens[:, curvature_field_idx].to(
                        device=cond.device,
                        dtype=torch.long,
                    ).clamp(0, 2)
                else:
                    curvature_source_ids = curvature_target_ids
                winding_id = int(DEFAULT_MOBILITY_VOCAB.token_to_id["curvature"]["winding"])
                smooth_id = int(DEFAULT_MOBILITY_VOCAB.token_to_id["curvature"]["smooth"])
                curvature_winding_mask = curvature_mask & (
                    curvature_target_ids.view(anchor.shape[0], 1, 1, 1) == winding_id
                )
                smooth_to_winding_mask = curvature_winding_mask & (
                    curvature_source_ids.view(anchor.shape[0], 1, 1, 1) == smooth_id
                )
            else:
                curvature_target_ids = None
                curvature_source_ids = None
                winding_id = int(DEFAULT_MOBILITY_VOCAB.token_to_id["curvature"]["winding"])
                smooth_id = int(DEFAULT_MOBILITY_VOCAB.token_to_id["curvature"]["smooth"])
                curvature_winding_mask = torch.zeros_like(curvature_mask)
                smooth_to_winding_mask = torch.zeros_like(curvature_mask)
            direction_multi_mask_f = direction_multi_mask.to(dtype=cond.dtype)
            source_to_multi_mask_f = source_to_multi_mask.to(dtype=cond.dtype)
            source_straight_to_multi_mask_f = source_straight_to_multi_mask.to(dtype=cond.dtype)
            source_turn_to_multi_mask_f = source_turn_to_multi_mask.to(dtype=cond.dtype)
            curvature_preserve_direction_f = curvature_preserve_direction_mask.to(dtype=cond.dtype)
            if (
                self.counterfactual_local_proposal_type == "mask_controlled_v1"
                and float(self.counterfactual_local_proposal_strength) > 0.0
                and mobility_tokens is not None
            ):
                reference_local_act_modulation = torch.tanh(
                    self.reference_local_act_to_proposal(reference_local_act_tokens)
                ).view(anchor.shape[0], self.num_candidates, 1, 4)
                reference_local_act_modulation = reference_local_act_modulation * (
                    reference_local_direct_act_embedding_used.to(dtype=cond.dtype).view(anchor.shape[0], 1, 1, 1)
                )
                proposal_params_raw = residual_input.new_zeros(anchor.shape[0], self.num_candidates, 4)
                direction_field_mask = direction_mask.view(anchor.shape[0])
                curvature_field_mask = curvature_mask.view(anchor.shape[0])
                multi_turn_creation_head_mask = direction_field_mask & source_to_multi_mask.view(anchor.shape[0])
                reversal_or_generic_direction_mask = direction_field_mask & (~multi_turn_creation_head_mask)
                winding_local_head_mask = curvature_field_mask & smooth_to_winding_mask.view(anchor.shape[0])
                generic_curvature_head_mask = curvature_field_mask & (~winding_local_head_mask)
                reference_field_mask = reference_mask.view(anchor.shape[0])
                field_head_specs = (
                    ("multi_turn_creation", multi_turn_creation_head_mask),
                    ("direction", reversal_or_generic_direction_mask),
                    ("winding_local", winding_local_head_mask),
                    ("curvature", generic_curvature_head_mask),
                    ("reference_adherence", reference_field_mask),
                )
                assigned_mask = torch.zeros(anchor.shape[0], device=cond.device, dtype=torch.bool)
                for head_name, field_mask in field_head_specs:
                    active = field_mask.to(device=cond.device, dtype=torch.bool)
                    if bool(active.any()):
                        if head_name == "reference_adherence":
                            active_params = torch.tanh(
                                self.reference_local_proposal_head(reference_residual_input[active])
                            )
                        else:
                            active_params = torch.tanh(
                                self.counterfactual_local_proposal_heads[head_name](residual_input[active])
                            )
                        proposal_params_raw[active] = active_params.to(
                            device=proposal_params_raw.device,
                            dtype=proposal_params_raw.dtype,
                        )
                        assigned_mask = assigned_mask | active
                fallback_mask = ~assigned_mask
                if bool(fallback_mask.any()):
                    fallback_params = torch.tanh(
                        self.counterfactual_local_proposal_heads["default"](residual_input[fallback_mask])
                    )
                    proposal_params_raw[fallback_mask] = fallback_params.to(
                        device=proposal_params_raw.device,
                        dtype=proposal_params_raw.dtype,
                    )
                counterfactual_local_proposal_params = proposal_params_raw.view(anchor.shape[0], self.num_candidates, 4)
                proposal_params = counterfactual_local_proposal_params.view(
                    anchor.shape[0],
                    self.num_candidates,
                    1,
                    4,
                )
                proposal_mode_phase = torch.linspace(
                    -1.0,
                    1.0,
                    steps=self.num_candidates,
                    device=cond.device,
                    dtype=cond.dtype,
                ).view(1, self.num_candidates, 1, 1)
                proposal_mode_phase = (
                    proposal_mode_phase
                    + 0.20 * field_control_proposal_shift.to(device=cond.device, dtype=cond.dtype)
                ).clamp(-1.0, 1.0)
                proposal_mode_sign = torch.sign(proposal_mode_phase + 1e-6)
                split_idx = max(1, self.num_candidates // 2)
                control_count = max(1, self.num_candidates - split_idx)
                candidate_pos = torch.arange(
                    self.num_candidates,
                    device=cond.device,
                    dtype=cond.dtype,
                ).view(1, self.num_candidates, 1, 1)
                control_rank = (
                    (candidate_pos - float(split_idx)) / float(max(control_count - 1, 1))
                ).clamp(0.0, 1.0)
                reference_local_contraction_ladder = torch.where(
                    candidate_pos >= float(split_idx),
                    0.35 + 0.65 * control_rank,
                    torch.zeros_like(control_rank),
                )
                narrow_window = torch.exp(-0.5 * ((time_basis - 0.5) / 0.18).pow(2))
                broad_window = torch.exp(-0.5 * ((time_basis - 0.5) / 0.28).pow(2))
                wide_window = torch.exp(-0.5 * ((time_basis - 0.5) / 0.36).pow(2))
                multiturn_edit_window = (
                    0.65 * broad_window + 0.35 * torch.sin(2.0 * math.pi * time_basis).pow(2)
                ) * mid_basis
                winding_edit_window = (
                    0.55 * broad_window + 0.45 * torch.sin(3.0 * math.pi * time_basis).pow(2)
                ) * mid_basis
                generic_edit_window = (0.70 * narrow_window + 0.30 * mid_basis).clamp(0.0, 1.0)
                active_direction_mask = source_to_multi_mask
                active_curvature_mask = curvature_winding_mask
                reference_edit_window = (
                    0.58 * broad_window
                    + 0.24 * torch.sin(2.0 * math.pi * time_basis).pow(2)
                    + 0.18 * torch.sin(4.0 * math.pi * time_basis).pow(2)
                ) * mid_basis
                active_reference_mask = reference_mask
                reference_target_ids_local = None
                reference_loose_mask = torch.zeros_like(reference_mask)
                reference_moderate_mask = torch.zeros_like(reference_mask)
                reference_tight_mask = torch.zeros_like(reference_mask)
                reference_unknown_mask = torch.zeros_like(reference_mask)
                reference_loose_to_tight_mask = torch.zeros_like(reference_mask)
                reference_loose_to_moderate_mask = torch.zeros_like(reference_mask)
                reference_moderate_to_tight_mask = torch.zeros_like(reference_mask)
                if mobility_tokens is not None:
                    ref_field_idx = DEFAULT_MOBILITY_VOCAB.fields.index("reference_adherence")
                    ref_vocab = DEFAULT_MOBILITY_VOCAB.token_to_id["reference_adherence"]
                    reference_target_ids_local = mobility_tokens[:, ref_field_idx].to(
                        device=cond.device,
                        dtype=torch.long,
                    )
                    reference_loose_mask = (
                        reference_target_ids_local == int(ref_vocab["loose"])
                    ).view(anchor.shape[0], 1, 1, 1)
                    reference_moderate_mask = (
                        reference_target_ids_local == int(ref_vocab["moderate"])
                    ).view(anchor.shape[0], 1, 1, 1)
                    reference_tight_mask = (
                        reference_target_ids_local == int(ref_vocab["tight"])
                    ).view(anchor.shape[0], 1, 1, 1)
                    reference_unknown_mask = (
                        reference_target_ids_local == int(ref_vocab["unknown"])
                    ).view(anchor.shape[0], 1, 1, 1)
                    if (
                        base_mobility_tokens is not None
                        and base_mobility_tokens.ndim == mobility_tokens.ndim
                        and base_mobility_tokens.shape[0] == mobility_tokens.shape[0]
                    ):
                        reference_source_ids_local = base_mobility_tokens[:, ref_field_idx].to(
                            device=cond.device,
                            dtype=torch.long,
                        )
                        reference_loose_to_tight_mask = (
                            (reference_source_ids_local == int(ref_vocab["loose"]))
                            & (reference_target_ids_local == int(ref_vocab["tight"]))
                        ).view(anchor.shape[0], 1, 1, 1)
                        reference_loose_to_moderate_mask = (
                            (reference_source_ids_local == int(ref_vocab["loose"]))
                            & (reference_target_ids_local == int(ref_vocab["moderate"]))
                        ).view(anchor.shape[0], 1, 1, 1)
                        reference_moderate_to_tight_mask = (
                            (reference_source_ids_local == int(ref_vocab["moderate"]))
                            & (reference_target_ids_local == int(ref_vocab["tight"]))
                        ).view(anchor.shape[0], 1, 1, 1)
                    reference_local_variant_kind = ["inactive" for _ in range(anchor.shape[0])]
                    reference_local_variant_uses_mode_sign = torch.zeros(
                        anchor.shape[0],
                        device=cond.device,
                        dtype=torch.bool,
                    )
                    for batch_idx in range(anchor.shape[0]):
                        if not bool(active_reference_mask[batch_idx, 0, 0, 0].item()):
                            continue
                        if bool(reference_loose_to_tight_mask[batch_idx, 0, 0, 0].item()):
                            reference_local_variant_kind[batch_idx] = "loose_to_tight_contraction"
                        elif bool(reference_loose_to_moderate_mask[batch_idx, 0, 0, 0].item()):
                            reference_local_variant_kind[batch_idx] = "loose_to_moderate_band"
                        elif bool(reference_moderate_to_tight_mask[batch_idx, 0, 0, 0].item()):
                            reference_local_variant_kind[batch_idx] = "moderate_to_tight_contraction"
                        elif bool(reference_tight_mask[batch_idx, 0, 0, 0].item()):
                            reference_local_variant_kind[batch_idx] = "tight_inward"
                        elif bool(reference_moderate_mask[batch_idx, 0, 0, 0].item()):
                            reference_local_variant_kind[batch_idx] = "moderate_inward"
                        elif bool(reference_loose_mask[batch_idx, 0, 0, 0].item()):
                            reference_local_variant_kind[batch_idx] = "loose_outward"
                            reference_local_variant_uses_mode_sign[batch_idx] = True
                        elif bool(reference_unknown_mask[batch_idx, 0, 0, 0].item()):
                            reference_local_variant_kind[batch_idx] = "unknown_signed"
                            reference_local_variant_uses_mode_sign[batch_idx] = True
                        else:
                            reference_local_variant_kind[batch_idx] = "reference_local_other"
                active_local_mask = active_direction_mask | active_curvature_mask | active_reference_mask
                active_direction_mask_f = active_direction_mask.to(dtype=cond.dtype)
                active_curvature_mask_f = active_curvature_mask.to(dtype=cond.dtype)
                active_reference_mask_f = active_reference_mask.to(dtype=cond.dtype)
                edit_window = torch.where(
                    active_direction_mask,
                    multiturn_edit_window,
                    torch.where(
                        active_curvature_mask,
                        winding_edit_window,
                        torch.where(active_reference_mask, reference_edit_window, generic_edit_window),
                    ),
                )
                counterfactual_edit_mask = active_local_mask.to(dtype=cond.dtype) * edit_window
                edge_guard = ((time_basis < 0.08) | (time_basis > 0.92)).to(dtype=cond.dtype)
                counterfactual_edit_mask = counterfactual_edit_mask * (1.0 - edge_guard)
                counterfactual_preserve_mask = torch.where(
                    edge_guard > 0.0,
                    torch.ones_like(counterfactual_edit_mask),
                    1.0 - counterfactual_edit_mask,
                ).clamp(0.0, 1.0)
                tangent_slot = torch.where(
                    active_direction_mask,
                    0.18 * counterfactual_edit_mask,
                    torch.where(
                        active_curvature_mask,
                        0.10 * counterfactual_edit_mask,
                        torch.where(active_reference_mask, 0.08 * counterfactual_edit_mask, 0.25 * counterfactual_edit_mask),
                    ),
                )
                normal_slot = torch.where(
                    active_direction_mask,
                    1.00 * counterfactual_edit_mask,
                    torch.where(
                        active_curvature_mask,
                        1.00 * counterfactual_edit_mask,
                        torch.where(active_reference_mask, 0.92 * counterfactual_edit_mask, 0.85 * counterfactual_edit_mask),
                    ),
                )
                counterfactual_edit_slot_mask = torch.cat([tangent_slot, normal_slot], dim=-1)
                multi_turn_base_curve = (
                    0.96 * torch.sin(4.0 * math.pi * time_basis)
                    - 0.62 * torch.sin(8.0 * math.pi * time_basis)
                    + 0.22 * proposal_mode_sign * torch.sin(10.0 * math.pi * time_basis)
                )
                multi_turn_tangent_curve = (
                    torch.sin(2.0 * math.pi * time_basis)
                    - 0.55 * torch.sin(6.0 * math.pi * time_basis)
                )
                proposal_winding_variant = (
                    0.35
                    + 0.45 * proposal_mode_phase.abs()
                    + 0.20 * proposal_params[..., 2:3].abs().clamp_max(1.0)
                    + 0.10 * field_control_proposal_shift.abs().to(device=cond.device, dtype=cond.dtype)
                )
                if self.counterfactual_curvature_residual_type == "midlocal_winding_pulse":
                    winding_base_curve, winding_active_window = self._build_midlocal_winding_pulse_curve(
                        time_basis,
                        winding_edit_window,
                        proposal_winding_variant,
                    )
                else:
                    winding_base_curve = self._build_signed_balanced_winding_curve(
                        time_basis,
                        winding_edit_window,
                        proposal_winding_variant,
                    )
                    winding_active_window = winding_edit_window
                winding_base_curve = self._stabilize_signed_balanced_winding_curve(
                    winding_base_curve,
                    time_basis,
                    winding_active_window,
                    active_curvature_mask_f * winding_active_window,
                    target_rms=0.92 if self.counterfactual_curvature_residual_type == "midlocal_winding_pulse" else 0.88,
                )
                raw_residual_view_for_local = raw_residual.view(
                    anchor.shape[0],
                    self.num_candidates,
                    target_len,
                    2,
                )
                local_base_normal_basis = base_selected_normal_scalar
                if local_base_normal_basis is None:
                    local_base_normal_basis = (
                        raw_residual_view_for_local * normal.view(anchor.shape[0], 1, target_len, 2)
                    ).sum(dim=-1, keepdim=True)
                winding_base_curve = self._project_heading_preserving_winding_curve(
                    winding_base_curve,
                    time_basis,
                    winding_active_window,
                    active_curvature_mask_f * winding_active_window,
                    base_normal_basis=local_base_normal_basis,
                    target_rms=0.74 if self.counterfactual_curvature_residual_type == "midlocal_winding_pulse" else 0.70,
                )
                winding_tangent_curve = torch.zeros_like(winding_base_curve)
                local_direction_normal = active_direction_mask_f * (
                    (0.65 + 0.20 * source_straight_to_multi_mask_f + 0.12 * source_turn_to_multi_mask_f)
                    + 0.35 * proposal_params[..., 0:1].abs()
                ) * multi_turn_base_curve * multiturn_edit_window
                local_direction_normal = (
                    local_direction_normal
                    * field_control_proposal_scale.to(device=cond.device, dtype=cond.dtype)
                )
                local_direction_tangent = active_direction_mask_f * (
                    0.10 * proposal_params[..., 1:2]
                ) * multi_turn_tangent_curve * multiturn_edit_window
                local_direction_tangent, local_direction_normal = self._stabilize_base_preserving_multiturn_curves(
                    local_direction_tangent,
                    local_direction_normal,
                    time_basis,
                    multiturn_edit_window,
                    active_direction_mask_f * multiturn_edit_window,
                    base_normal_basis=local_base_normal_basis,
                    tangent_target_rms=0.05,
                    normal_target_rms=0.46,
                )
                local_curvature_normal = active_curvature_mask_f * (
                    (0.60 + 0.18 * smooth_to_winding_mask.to(dtype=cond.dtype))
                    + 0.40 * proposal_params[..., 2:3].abs()
                ) * winding_base_curve
                local_curvature_normal = (
                    local_curvature_normal
                    * field_control_proposal_scale.to(device=cond.device, dtype=cond.dtype)
                )
                local_curvature_tangent = active_curvature_mask_f * (
                    0.08 * proposal_params[..., 3:4]
                ) * winding_tangent_curve * winding_edit_window
                reference_mode_sign = proposal_mode_sign
                inward_core = (
                    1.15 * mid_basis
                    + 0.35 * torch.sin(2.0 * math.pi * time_basis).pow(2)
                ) * reference_edit_window
                outward_core = (
                    0.75 * mid_basis
                    + 0.28 * reference_mode_sign * torch.sin(2.0 * math.pi * time_basis)
                    + 0.18 * torch.sin(4.0 * math.pi * time_basis)
                ) * reference_edit_window
                contraction_boost = (
                    1.35 * mid_basis
                    + 0.42 * torch.sin(2.0 * math.pi * time_basis).pow(2)
                    + 0.22 * torch.sin(4.0 * math.pi * time_basis).pow(2)
                ) * reference_edit_window
                reference_loose_to_tight_mask_f = reference_loose_to_tight_mask.to(dtype=cond.dtype)
                reference_loose_to_moderate_mask_f = reference_loose_to_moderate_mask.to(dtype=cond.dtype)
                reference_moderate_to_tight_mask_f = reference_moderate_to_tight_mask.to(dtype=cond.dtype)
                reference_metric_preserving_mask_f = (
                    reference_loose_to_tight_mask_f
                    + reference_loose_to_moderate_mask_f
                    + reference_moderate_to_tight_mask_f
                ).clamp(0.0, 1.0)
                loose_to_tight_amplitude = (
                    1.55
                    + 0.35 * proposal_params[..., 0:1].abs()
                    + 0.22 * reference_local_act_modulation[..., 0:1]
                ).clamp_min(0.35)
                loose_to_tight_shape = reference_loose_to_tight_mask_f * (
                    0.12 * reference_local_act_modulation[..., 1:2] * signed_basis
                    + 0.08 * reference_local_act_modulation[..., 2:3] * torch.sin(2.0 * math.pi * time_basis)
                ) * reference_edit_window
                reference_wave = (
                    -reference_tight_mask.to(dtype=cond.dtype) * inward_core * (1.10 + 0.30 * proposal_params[..., 0:1].abs())
                    -reference_moderate_mask.to(dtype=cond.dtype) * inward_core * (0.72 + 0.18 * proposal_params[..., 0:1].abs())
                    +reference_loose_mask.to(dtype=cond.dtype) * outward_core * (0.68 + 0.20 * proposal_params[..., 0:1].abs())
                    -reference_loose_to_tight_mask_f * contraction_boost * loose_to_tight_amplitude
                    +loose_to_tight_shape
                    +reference_unknown_mask.to(dtype=cond.dtype) * (
                        0.10 * reference_mode_sign * torch.sin(2.0 * math.pi * time_basis) * reference_edit_window
                    )
                )
                reference_tangent_wave = (
                    0.18 * torch.sin(2.0 * math.pi * time_basis)
                    - 0.12 * reference_mode_sign * signed_basis * mid_basis
                ) * reference_edit_window
                loose_to_tight_metric_delta = None
                reference_transition_metric_delta = None
                local_reference_normal = active_reference_mask_f * reference_wave
                local_reference_normal = self._project_curve_out_bases(
                    local_reference_normal,
                    [
                        reference_edit_window,
                        signed_basis * reference_edit_window,
                        end_basis * reference_edit_window,
                    ],
                    active_reference_mask_f * reference_edit_window,
                )
                local_reference_normal = self._normalize_masked_curve_rms(
                    local_reference_normal,
                    active_reference_mask_f * reference_edit_window,
                    target_rms=0.54,
                )
                reference_base_points = anchor.unsqueeze(1).expand(
                    -1,
                    self.num_candidates,
                    -1,
                    -1,
                )
                if prepared_base_selected_trajectory is not None:
                    reference_base_points = prepared_base_selected_trajectory.unsqueeze(1).expand(
                        -1,
                        self.num_candidates,
                        -1,
                        -1,
                    )
                if reference_base_points.shape[-2] != target_len:
                    reference_base_points = self.resample_temporal(reference_base_points, target_len)
                attraction_vector = self._nearest_reference_attraction_vector(reference_base_points, anchor)
                base_frame_tangent, base_frame_normal = self.tangent_normal(
                    reference_base_points.reshape(anchor.shape[0] * self.num_candidates, target_len, 2)
                )
                base_frame_tangent = base_frame_tangent.view(anchor.shape[0], self.num_candidates, target_len, 2)
                base_frame_normal = base_frame_normal.view(anchor.shape[0], self.num_candidates, target_len, 2)
                attraction_normal_scalar = (
                    attraction_vector * base_frame_normal
                ).sum(dim=-1, keepdim=True)
                attraction_tangent_scalar = (
                    attraction_vector * base_frame_tangent
                ).sum(dim=-1, keepdim=True)
                attraction_unit = attraction_vector / torch.linalg.vector_norm(
                    attraction_vector,
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-6)
                loose_to_tight_mid_window = reference_loose_to_tight_mask_f * (
                    reference_edit_window * mid_basis.pow(0.75)
                ).clamp(0.0, 1.0)
                attraction_act_strength = torch.where(
                    candidate_pos >= float(split_idx),
                    0.55 + 0.65 * control_rank,
                    torch.zeros_like(control_rank),
                )
                reference_direct_latent_gate = (
                    reference_local_direct_semantic_latent_used.to(
                        device=cond.device,
                        dtype=cond.dtype,
                    )
                    if reference_local_direct_semantic_latent_used is not None
                    else torch.ones(anchor.shape[0], device=cond.device, dtype=cond.dtype)
                ).view(anchor.shape[0], 1, 1, 1)
                reference_act_available_gate = (
                    (
                        field_control_candidate_act_ids.to(device=cond.device)
                        .view(anchor.shape[0], self.num_candidates, 1, 1)
                        >= 0
                    ).to(dtype=cond.dtype)
                    if field_control_candidate_act_ids is not None
                    else torch.ones(
                        anchor.shape[0],
                        self.num_candidates,
                        1,
                        1,
                        device=cond.device,
                        dtype=cond.dtype,
                    )
                )
                attraction_act_strength = attraction_act_strength * (
                    reference_direct_latent_gate
                    + (1.0 - reference_direct_latent_gate) * reference_act_available_gate
                )
                reference_field_gate = reference_mask.to(dtype=cond.dtype).view(anchor.shape[0], 1, 1, 1)
                reference_local_budget_gate = torch.tanh(
                    self.reference_local_budget_gate_head(reference_residual_input)
                ).view(anchor.shape[0], self.num_candidates, 1, 1)
                reference_local_budget_scale = 1.0 + 0.04 * reference_field_gate * reference_local_budget_gate
                # Keep the reference attraction as a local corridor edit. The
                # stronger pre-fix budget improved reference distance but
                # over-rotated loose->tight controls under the routed
                # base-preserving bank.
                attraction_step_limit = (
                    0.026
                    + 0.020 * attraction_act_strength
                    + 0.004 * reference_direct_latent_gate
                ) * reference_local_budget_scale
                attraction_step = loose_to_tight_mid_window * attraction_act_strength * attraction_step_limit
                attraction_normal_scalar = self.preserve_reference_endpoints(
                    attraction_step * attraction_normal_scalar.sign().clamp(-1.0, 1.0)
                )
                attraction_tangent_scalar = self.preserve_reference_endpoints(
                    0.12
                    * attraction_step
                    * attraction_tangent_scalar.sign().clamp(-1.0, 1.0)
                )
                strong_attraction_mask = (
                    reference_loose_to_tight_mask_f
                    * (candidate_pos >= float(split_idx)).to(dtype=cond.dtype)
                    * (control_rank >= 0.66).to(dtype=cond.dtype)
                )
                # The strongest loose->tight act should stay inside the local
                # corridor: otherwise it improves reference distance by leaving
                # the selector's raw/independent collateral-feasible region.
                strong_attraction_step_damping = 1.0 - 0.24 * strong_attraction_mask
                attraction_step = attraction_step * strong_attraction_step_damping
                attraction_normal_scalar = attraction_normal_scalar * strong_attraction_step_damping
                attraction_tangent_scalar = attraction_tangent_scalar * strong_attraction_step_damping
                loose_to_tight_metric_delta = (
                    attraction_step * attraction_unit
                    + (
                        attraction_tangent_scalar
                        + 0.04 * strong_attraction_mask * attraction_tangent_scalar
                    )
                    * base_frame_tangent
                )
                loose_to_tight_metric_delta = self.preserve_reference_endpoints(loose_to_tight_metric_delta)
                reference_transition_metric_delta = loose_to_tight_metric_delta
                reference_band_metric_mask_f = (
                    reference_loose_to_moderate_mask_f + reference_moderate_to_tight_mask_f
                ).clamp(0.0, 1.0)
                if bool((reference_band_metric_mask_f > 0.0).any().detach().item()):
                    expanded_reference = anchor.unsqueeze(1).expand(-1, self.num_candidates, -1, -1)
                    current_ref_distance = self._mean_reference_polyline_distance(
                        reference_base_points.reshape(
                            anchor.shape[0] * self.num_candidates,
                            target_len,
                            2,
                        ),
                        expanded_reference.reshape(
                            anchor.shape[0] * self.num_candidates,
                            target_len,
                            2,
                        ),
                    ).view(anchor.shape[0], self.num_candidates, 1, 1)
                    if reference_target_ids_local is not None:
                        target_ref_center = self.calib_ref_center.to(
                            device=cond.device,
                            dtype=cond.dtype,
                        )[
                            reference_target_ids_local.clamp(
                                0,
                                self.calib_ref_center.numel() - 1,
                            )
                        ].view(anchor.shape[0], 1, 1, 1)
                    else:
                        target_ref_center = current_ref_distance.new_zeros(anchor.shape[0], 1, 1, 1)
                    band_error = current_ref_distance - target_ref_center
                    band_direction = band_error.sign().clamp(-1.0, 1.0)
                    band_abs_error = band_error.abs()
                    band_act_strength = torch.where(
                        candidate_pos >= float(split_idx),
                        0.50 + 0.55 * control_rank,
                        torch.zeros_like(control_rank),
                    )
                    band_mid_window = reference_band_metric_mask_f * (
                        reference_edit_window * mid_basis.pow(0.80)
                    ).clamp(0.0, 1.0)
                    loose_to_moderate_step = (
                        reference_loose_to_moderate_mask_f
                        * band_act_strength
                        * torch.minimum(
                            0.72 * band_abs_error,
                            0.038 + 0.010 * band_act_strength,
                        )
                    )
                    moderate_to_tight_step = (
                        reference_moderate_to_tight_mask_f
                        * band_act_strength
                        * torch.minimum(
                            0.88 * band_abs_error,
                            0.022 + 0.006 * band_act_strength,
                        )
                    )
                    band_step = (
                        band_mid_window
                        * (loose_to_moderate_step + moderate_to_tight_step)
                        * reference_local_budget_scale
                    )
                    band_tangent_scalar = self.preserve_reference_endpoints(
                        0.14
                        * band_step
                        * band_direction
                        * attraction_tangent_scalar.sign().clamp(-1.0, 1.0)
                    )
                    reference_band_metric_delta = (
                        band_step * band_direction * attraction_unit
                        + band_tangent_scalar * base_frame_tangent
                    )
                    reference_band_metric_delta = self.preserve_reference_endpoints(
                        reference_band_metric_delta
                    )
                    reference_transition_metric_delta = torch.where(
                        reference_band_metric_mask_f > 0.0,
                        reference_band_metric_delta,
                        reference_transition_metric_delta,
                    )
                attraction_anchor_normal_scalar = (
                    attraction_normal_scalar * (base_frame_normal * normal.view(anchor.shape[0], 1, target_len, 2)).sum(
                        dim=-1,
                        keepdim=True,
                    )
                    + attraction_tangent_scalar * (base_frame_tangent * normal.view(anchor.shape[0], 1, target_len, 2)).sum(
                        dim=-1,
                        keepdim=True,
                    )
                )
                attraction_anchor_tangent_scalar = (
                    attraction_normal_scalar * (base_frame_normal * tangent.view(anchor.shape[0], 1, target_len, 2)).sum(
                        dim=-1,
                        keepdim=True,
                    )
                    + attraction_tangent_scalar * (base_frame_tangent * tangent.view(anchor.shape[0], 1, target_len, 2)).sum(
                        dim=-1,
                        keepdim=True,
                    )
                )
                attraction_anchor_normal_scalar = self._project_curve_out_bases(
                    attraction_anchor_normal_scalar,
                    [
                        reference_edit_window,
                        signed_basis * reference_edit_window,
                        end_basis * reference_edit_window,
                    ],
                    loose_to_tight_mid_window,
                )
                attraction_anchor_tangent_scalar = self._project_curve_out_bases(
                    attraction_anchor_tangent_scalar,
                    [
                        reference_edit_window,
                        signed_basis * reference_edit_window,
                        end_basis * reference_edit_window,
                    ],
                    loose_to_tight_mid_window,
                )
                local_reference_normal = torch.where(
                    (reference_loose_to_tight_mask_f * reference_direct_latent_gate) > 0.0,
                    attraction_anchor_normal_scalar,
                    local_reference_normal,
                )
                reference_contraction_gain = torch.where(
                    (reference_loose_to_tight_mask_f * reference_direct_latent_gate) > 0.0,
                    0.58
                    + 0.18 * torch.clamp(attraction_act_strength, 0.0, 1.0)
                    + 0.08 * reference_direct_latent_gate
                    + 0.10 * reference_local_contraction_ladder,
                    torch.ones_like(reference_local_contraction_ladder),
                )
                local_reference_normal = local_reference_normal * reference_contraction_gain
                local_reference_tangent = active_reference_mask_f * (
                    0.04 * proposal_params[..., 1:2]
                    + 0.10 * reference_loose_to_tight_mask_f * reference_local_act_modulation[..., 3:4]
                ) * reference_tangent_wave
                local_reference_tangent = self._project_curve_out_bases(
                    local_reference_tangent,
                    [
                        reference_edit_window,
                        signed_basis * reference_edit_window,
                        end_basis * reference_edit_window,
                    ],
                    active_reference_mask_f * reference_edit_window,
                )
                local_reference_tangent = self._normalize_masked_curve_rms(
                    local_reference_tangent,
                    active_reference_mask_f * reference_edit_window,
                    target_rms=0.04,
                )
                local_reference_tangent = (
                    local_reference_tangent
                    + 0.10 * strong_attraction_mask * attraction_anchor_tangent_scalar
                )
                counterfactual_local_proposal_delta = (
                    float(self.counterfactual_local_proposal_strength)
                    * envelope.view(1, 1, target_len, 1)
                    * (
                        0.018
                        * self.tangent_scale
                        * (local_direction_tangent + local_curvature_tangent + 0.55 * local_reference_tangent)
                        * tangent.view(anchor.shape[0], 1, target_len, 2)
                        + 0.16
                        * self.normal_scale
                        * (local_direction_normal + local_curvature_normal + 1.30 * local_reference_normal)
                        * normal.view(anchor.shape[0], 1, target_len, 2)
                    )
                )
                if reference_transition_metric_delta is not None:
                    reference_metric_override = (
                        float(self.counterfactual_local_proposal_strength)
                        * envelope.view(1, 1, target_len, 1)
                        * reference_transition_metric_delta
                    )
                    counterfactual_local_proposal_delta = torch.where(
                        reference_metric_preserving_mask_f > 0.0,
                        reference_metric_override,
                        counterfactual_local_proposal_delta,
                    )
                counterfactual_local_proposal_delta = self.preserve_reference_endpoints(
                    counterfactual_local_proposal_delta
                )
                counterfactual_local_proposal_family = []
                counterfactual_mask_family = []
                for batch_idx in range(anchor.shape[0]):
                    if bool(active_direction_mask[batch_idx, 0, 0, 0].item()):
                        routed_family = (
                            field_control_family_prefixes[batch_idx]
                            if batch_idx < len(field_control_family_prefixes)
                            else None
                        )
                        counterfactual_local_proposal_family.append(
                            routed_family or "direction:multi_turn_creation_local"
                        )
                        counterfactual_mask_family.append("multi_turn_creation_mask")
                    elif bool(active_curvature_mask[batch_idx, 0, 0, 0].item()):
                        routed_family = (
                            field_control_family_prefixes[batch_idx]
                            if batch_idx < len(field_control_family_prefixes)
                            else None
                        )
                        counterfactual_local_proposal_family.append(
                            routed_family or "curvature:smooth->winding_local"
                        )
                        counterfactual_mask_family.append("winding_burst_mask")
                    elif bool(active_reference_mask[batch_idx, 0, 0, 0].item()):
                        routed_family = (
                            field_control_family_prefixes[batch_idx]
                            if batch_idx < len(field_control_family_prefixes)
                            else None
                        )
                        counterfactual_local_proposal_family.append(
                            routed_family
                            or (
                                "reference_adherence:loose->tight_contraction_local"
                                if bool(reference_loose_to_tight_mask[batch_idx, 0, 0, 0].item())
                                else "reference_adherence:local_direction_preserving"
                            )
                        )
                        counterfactual_mask_family.append("reference_local_corridor_mask")
                    else:
                        field_name = DEFAULT_MOBILITY_VOCAB.fields[int(cf_field_ids[batch_idx].item())]
                        counterfactual_local_proposal_family.append(f"{field_name}:generic_local")
                        counterfactual_mask_family.append("generic_local_mask")
            # Counterfactual multi-turn requires a higher-frequency local
            # shape basis than left/right/straight. Keep it inside the
            # counterfactual direction path so ordinary Table-1/2/3 generation
            # is not distorted by a global oscillation prior.
            direction_multi_normal_boost = direction_multi_mask_f * (
                1.15 * s_curve_basis + 0.70 * high_curve_basis
            ) * mid_basis
            direction_multi_tangent_boost = 0.10 * direction_multi_mask_f * signed_basis * mid_basis
            if mobility_tokens is not None:
                source_sign_table = cond.new_tensor([0.0, 1.0, -1.0, 0.0, 0.0])
                source_sign = source_sign_table.gather(
                    0,
                    direction_source_ids.clamp(0, source_sign_table.numel() - 1),
                ).view(anchor.shape[0], 1, 1, 1)
            else:
                source_sign = cond.new_zeros(anchor.shape[0], 1, 1, 1)
            mode_sign_for_multi = torch.linspace(
                -1.0,
                1.0,
                steps=self.num_candidates,
                device=cond.device,
                dtype=cond.dtype,
            ).view(1, self.num_candidates, 1, 1)
            transition_sign = torch.where(source_sign.abs() > 1e-6, source_sign, torch.sign(mode_sign_for_multi + 1e-6))
            transition_wave = transition_sign * (1.25 * s_curve_basis - 0.90 * high_curve_basis) * mid_basis
            straight_wave = torch.sign(mode_sign_for_multi + 1e-6) * (
                0.80 * s_curve_basis + 0.95 * high_curve_basis
            ) * mid_basis
            direction_multi_transition_normal_boost = (
                source_turn_to_multi_mask_f * transition_wave
                + source_straight_to_multi_mask_f * straight_wave
                + 0.35 * source_to_multi_mask_f * torch.sign(signed_basis + 1e-6) * mid_basis
            )
            direction_multi_transition_tangent_boost = source_to_multi_mask_f * (
                -0.08 * end_basis + 0.08 * transition_sign * s_curve_basis * mid_basis
            )
            if self.counterfactual_direction_residual_type in {
                "transition_contract_pulse",
                "transition_arc_skeleton",
                "transition_arc_multiturn_burst",
                "transition_hybrid_skeleton_reflect",
                "transition_pivot_skeleton",
                "transition_reflect_skeleton",
                "transition_skeleton_reversal",
                "transition_skeleton_reversal_scurve",
                "transition_skeleton",
                "transition_template",
                "transition_template_balanced",
                "transition_template_contract",
                "transition_template_mean_projected",
                "transition_template_projected",
            }:
                direction_one_hot_cf = F.one_hot(
                    direction_target_ids.clamp(0, 4),
                    num_classes=5,
                ).to(dtype=cond.dtype)
                straight_cf = direction_one_hot_cf[:, 0:1].view(anchor.shape[0], 1, 1, 1)
                left_cf = direction_one_hot_cf[:, 1:2].view(anchor.shape[0], 1, 1, 1)
                right_cf = direction_one_hot_cf[:, 2:3].view(anchor.shape[0], 1, 1, 1)
                uturn_cf = direction_one_hot_cf[:, 3:4].view(anchor.shape[0], 1, 1, 1)
                multi_cf = direction_one_hot_cf[:, 4:5].view(anchor.shape[0], 1, 1, 1)
                signed_turn_cf = right_cf - left_cf
                raw_residual_view_for_cf_direction = raw_residual.view(
                    anchor.shape[0],
                    self.num_candidates,
                    target_len,
                    2,
                )
                existing_direction_tangent = raw_residual_view_for_cf_direction[..., 0:1]
                existing_direction_normal = raw_residual_view_for_cf_direction[..., 1:2]
                direction_mode_phase = torch.linspace(
                    -1.0,
                    1.0,
                    steps=self.num_candidates,
                    device=cond.device,
                    dtype=cond.dtype,
                ).view(1, self.num_candidates, 1, 1)
                direction_mode_sign = torch.sign(direction_mode_phase + 1e-6)
                if self.counterfactual_direction_residual_type == "transition_contract_pulse":
                    pulse_window = torch.exp(-0.5 * ((time_basis - 0.5) / 0.22).pow(2)) * mid_basis
                    signed_pulse = signed_turn_cf * (
                        0.95 * pulse_window
                        + 0.25 * direction_mode_sign * torch.sin(2.0 * math.pi * time_basis) * pulse_window
                    )
                    multi_pulse = multi_cf * (
                        0.70 * torch.sin(2.0 * math.pi * time_basis)
                        - 0.42 * torch.sin(4.0 * math.pi * time_basis)
                        + 0.24 * direction_mode_sign * torch.sin(6.0 * math.pi * time_basis)
                    ) * pulse_window
                    straight_pulse = -0.70 * straight_cf * existing_direction_normal * pulse_window
                    direction_normal_curve = signed_pulse + multi_pulse + straight_pulse
                    direction_tangent_curve = (
                        -0.08 * (left_cf + right_cf + multi_cf).clamp(0.0, 1.0) * existing_direction_tangent * pulse_window
                        -0.08 * uturn_cf * end_basis * pulse_window
                    )
                elif self.counterfactual_direction_residual_type in {
                    "transition_arc_skeleton",
                    "transition_arc_multiturn_burst",
                }:
                    arc_window = mid_basis.pow(1.10)
                    local_arc = (
                        1.10 * arc_window
                        + 0.26 * direction_mode_sign * s_curve_basis * mid_basis
                    )
                    reversal_source_left = (
                        direction_source_ids.view(anchor.shape[0], 1, 1, 1) == left_id
                    ).to(dtype=cond.dtype)
                    reversal_source_right = (
                        direction_source_ids.view(anchor.shape[0], 1, 1, 1) == right_id
                    ).to(dtype=cond.dtype)
                    reversal_mask = (left_cf * reversal_source_right + right_cf * reversal_source_left).clamp(0.0, 1.0)
                    reversal_gain = 1.0 + 0.45 * ((direction_mode_phase + 1.0) * 0.5)
                    reversal_arc = signed_turn_cf * reversal_gain * (
                        1.00 * arc_window
                        + 0.34 * torch.sin(2.0 * math.pi * time_basis) * mid_basis
                        + 0.16 * direction_mode_sign * high_curve_basis * mid_basis
                    )
                    turn_curve = signed_turn_cf * local_arc
                    uturn_curve = (
                        uturn_cf
                        * direction_mode_sign
                        * (
                            0.95 * torch.sin(2.0 * math.pi * time_basis)
                            - 0.54 * torch.sin(4.0 * math.pi * time_basis)
                        )
                        * arc_window
                    )
                    if self.counterfactual_direction_residual_type == "transition_arc_multiturn_burst":
                        multi_window = mid_basis.pow(1.35)
                        base_multi_curve = (
                            0.82 * torch.sin(4.0 * math.pi * time_basis)
                            - 0.58 * torch.sin(8.0 * math.pi * time_basis)
                            + 0.24 * direction_mode_sign * torch.sin(10.0 * math.pi * time_basis)
                        )
                        straight_to_multi_curve = (
                            1.10 * base_multi_curve
                            + 0.18 * torch.sin(2.0 * math.pi * time_basis)
                        )
                        turn_to_multi_curve = (
                            0.92 * base_multi_curve
                            + 0.16 * transition_sign * torch.sin(2.0 * math.pi * time_basis)
                        )
                        generic_multi_curve = 0.96 * base_multi_curve + 0.10 * torch.sin(6.0 * math.pi * time_basis)
                        multi_curve = (
                            multi_cf
                            * (
                                source_straight_to_multi_mask_f * straight_to_multi_curve
                                + source_turn_to_multi_mask_f * turn_to_multi_curve
                                + (1.0 - source_to_multi_mask_f) * generic_multi_curve
                            )
                            * multi_window
                        )
                    else:
                        multi_curve = (
                            multi_cf
                            * (
                                0.92 * s_curve_basis
                                + 0.78 * high_curve_basis
                                + 0.36 * direction_mode_sign * torch.sin(6.0 * math.pi * time_basis)
                            )
                            * arc_window
                        )
                    straight_curve = -0.88 * straight_cf * existing_direction_normal * arc_window
                    direction_normal_curve = (
                        torch.where(reversal_mask > 0.0, reversal_arc, turn_curve)
                        + uturn_curve
                        + multi_curve
                        + straight_curve
                        - 0.18 * (left_cf + right_cf + multi_cf).clamp(0.0, 1.0) * existing_direction_normal * arc_window
                    )
                    arc_abs = direction_normal_curve.abs().clamp(max=2.0)
                    tangent_compensation = -0.20 * arc_abs * arc_window
                    tangent_sign_bias = 0.035 * signed_turn_cf * signed_basis * mid_basis
                    direction_tangent_curve = (
                        tangent_compensation
                        + tangent_sign_bias
                        - 0.06 * straight_cf * existing_direction_tangent * arc_window
                        - 0.05 * uturn_cf * end_basis * arc_window
                    )
                    if self.counterfactual_direction_residual_type == "transition_arc_multiturn_burst":
                        direction_tangent_curve = direction_tangent_curve + (
                            multi_cf
                            * source_to_multi_mask_f
                            * 0.018
                            * (
                                torch.sin(4.0 * math.pi * time_basis)
                                - 0.75 * torch.sin(8.0 * math.pi * time_basis)
                            )
                            * mid_basis.pow(1.35)
                        )
                elif self.counterfactual_direction_residual_type in {
                    "transition_skeleton",
                    "transition_hybrid_skeleton_reflect",
                    "transition_pivot_skeleton",
                    "transition_reflect_skeleton",
                    "transition_skeleton_reversal",
                    "transition_skeleton_reversal_scurve",
                }:
                    interval_t = torch.linspace(
                        0.0,
                        1.0,
                        steps=max(target_len - 1, 1),
                        device=cond.device,
                        dtype=cond.dtype,
                    ).view(1, 1, max(target_len - 1, 1), 1)
                    smooth_interval = interval_t * interval_t * (3.0 - 2.0 * interval_t)
                    mode_strength = 0.82 + 0.36 * direction_mode_phase.abs()
                    turn_angle = math.radians(92.0) * mode_strength
                    turn_theta = signed_turn_cf * turn_angle * smooth_interval
                    if self.counterfactual_direction_residual_type in {
                        "transition_hybrid_skeleton_reflect",
                        "transition_pivot_skeleton",
                        "transition_reflect_skeleton",
                        "transition_skeleton_reversal",
                        "transition_skeleton_reversal_scurve",
                    }:
                        source_left_cf = (
                            direction_source_ids.view(anchor.shape[0], 1, 1, 1) == left_id
                        ).to(dtype=cond.dtype)
                        source_right_cf = (
                            direction_source_ids.view(anchor.shape[0], 1, 1, 1) == right_id
                        ).to(dtype=cond.dtype)
                        reversal_mask = (left_cf * source_right_cf + right_cf * source_left_cf).clamp(0.0, 1.0)
                        reversal_phase = (direction_mode_phase + 1.0) * 0.5
                        reversal_strength = 0.85 + 0.55 * reversal_phase
                        if self.counterfactual_direction_residual_type == "transition_pivot_skeleton":
                            pivot_phase = interval_t - 0.5
                            pivot_window = torch.sin(math.pi * interval_t).clamp_min(0.0)
                            pivot_strength = 0.95 + 0.70 * reversal_phase
                            reversal_theta = (
                                signed_turn_cf
                                * math.radians(120.0)
                                * pivot_strength
                                * (
                                    0.58 * torch.tanh(4.0 * pivot_phase)
                                    + 0.42 * torch.sin(2.0 * math.pi * interval_t) * pivot_window
                                )
                            )
                        elif self.counterfactual_direction_residual_type in {
                            "transition_hybrid_skeleton_reflect",
                            "transition_reflect_skeleton",
                        }:
                            reflect_window = torch.sin(math.pi * interval_t).clamp_min(0.0)
                            reflect_strength = 0.55 + 0.28 * reversal_phase
                            reversal_theta = (
                                signed_turn_cf
                                * math.radians(70.0)
                                * reflect_strength
                                * (
                                    0.68 * torch.sin(math.pi * interval_t).clamp_min(0.0)
                                    + 0.32 * torch.sin(2.0 * math.pi * interval_t) * reflect_window
                                )
                            )
                        else:
                            reversal_theta = (
                                signed_turn_cf
                                * math.radians(118.0)
                                * reversal_strength
                                * (
                                    0.72 * smooth_interval
                                    + 0.28 * torch.sin(math.pi * interval_t).clamp_min(0.0)
                                )
                            )
                        if self.counterfactual_direction_residual_type == "transition_skeleton_reversal_scurve":
                            reversal_wave = torch.sin(2.0 * math.pi * interval_t)
                            reversal_mid = torch.sin(math.pi * interval_t).clamp_min(0.0)
                            reversal_theta = (
                                signed_turn_cf
                                * math.radians(132.0)
                                * reversal_strength
                                * (
                                    0.54 * smooth_interval
                                    + 0.46 * reversal_wave * reversal_mid
                                )
                            )
                        turn_theta = torch.where(
                            reversal_mask > 0.0,
                            reversal_theta,
                            turn_theta,
                        )
                    uturn_theta = (
                        uturn_cf
                        * direction_mode_sign
                        * math.radians(160.0)
                        * torch.sin(math.pi * interval_t).clamp_min(0.0)
                    )
                    multi_theta = (
                        multi_cf
                        * mode_strength
                        * (
                            math.radians(82.0) * direction_mode_sign * torch.sin(2.0 * math.pi * interval_t)
                            - math.radians(58.0) * torch.sin(4.0 * math.pi * interval_t)
                            + math.radians(36.0) * direction_mode_sign * torch.sin(6.0 * math.pi * interval_t)
                        )
                    )
                    theta = turn_theta + uturn_theta + multi_theta
                    step_tangent = torch.cos(theta) - 1.0
                    step_normal = torch.sin(theta)
                    step_scale = theta.new_tensor(float(max(target_len - 1, 1)))
                    zero_step = theta.new_zeros(anchor.shape[0], self.num_candidates, 1, 1)
                    tangent_offset = torch.cat(
                        [zero_step, torch.cumsum(step_tangent / step_scale, dim=2)],
                        dim=2,
                    )
                    normal_offset = torch.cat(
                        [zero_step, torch.cumsum(step_normal / step_scale, dim=2)],
                        dim=2,
                    )
                    tangent_offset = tangent_offset - time_basis * tangent_offset[:, :, -1:, :]
                    normal_offset = normal_offset - time_basis * normal_offset[:, :, -1:, :]
                    skeleton_scale = 7.0
                    direction_tangent_curve = skeleton_scale * tangent_offset
                    direction_normal_curve = skeleton_scale * normal_offset
                    direction_normal_curve = (
                        direction_normal_curve
                        - 0.95 * straight_cf * existing_direction_normal * mid_basis
                    )
                    direction_tangent_curve = (
                        direction_tangent_curve
                        - 0.20 * straight_cf * existing_direction_tangent * mid_basis
                        - 0.05
                        * (left_cf + right_cf + multi_cf).clamp(0.0, 1.0)
                        * existing_direction_tangent
                        * mid_basis
                    )
                    if self.counterfactual_direction_residual_type == "transition_pivot_skeleton":
                        pivot_mask = reversal_mask.clamp(0.0, 1.0)
                        local_mid = mid_basis.pow(1.25)
                        pivot_normal_pulse = signed_turn_cf * (
                            0.85 * torch.sin(2.0 * math.pi * time_basis)
                            + 0.35 * direction_mode_sign * torch.sin(4.0 * math.pi * time_basis)
                        ) * local_mid
                        pivot_tangent_pulse = -0.08 * (
                            torch.cos(2.0 * math.pi * time_basis) - 1.0
                        ) * local_mid
                        direction_normal_curve = torch.where(
                            pivot_mask > 0.0,
                            direction_normal_curve + 1.75 * pivot_normal_pulse,
                            direction_normal_curve,
                        )
                        direction_tangent_curve = torch.where(
                            pivot_mask > 0.0,
                            direction_tangent_curve + pivot_tangent_pulse,
                            direction_tangent_curve,
                        )
                    elif self.counterfactual_direction_residual_type in {
                        "transition_hybrid_skeleton_reflect",
                        "transition_reflect_skeleton",
                    }:
                        reflect_mask = reversal_mask.clamp(0.0, 1.0)
                        if self.counterfactual_direction_residual_type == "transition_hybrid_skeleton_reflect":
                            skeleton_mode_mask = (
                                (direction_mode_phase > -0.30) & (direction_mode_phase < -0.05)
                            ).to(dtype=cond.dtype)
                            reflect_mode_mask = 1.0 - skeleton_mode_mask
                            reflect_mask = reflect_mask * reflect_mode_mask
                        target_side = signed_turn_cf
                        existing_away_from_target = torch.relu(-target_side * existing_direction_normal)
                        reflected_normal = 1.05 * target_side * existing_away_from_target * mid_basis
                        small_turn_basis = target_side * (
                            0.48 * mid_basis.pow(1.10)
                            + 0.22 * s_curve_basis * mid_basis
                        )
                        high_phase_boost = ((direction_mode_phase - 0.20) / 0.80).clamp(0.0, 1.0)
                        small_turn_basis = small_turn_basis + target_side * high_phase_boost * (
                            0.21 * mid_basis.pow(1.15)
                            + 0.07 * s_curve_basis * mid_basis
                        )
                        direction_normal_curve = torch.where(
                            reflect_mask > 0.0,
                            reflected_normal + small_turn_basis,
                            direction_normal_curve,
                        )
                        direction_tangent_curve = torch.where(
                            reflect_mask > 0.0,
                            -0.03 * existing_direction_tangent * mid_basis,
                            direction_tangent_curve,
                        )
                elif self.counterfactual_direction_residual_type == "transition_template_projected":
                    local_turn_basis = (
                        0.95 * s_curve_basis
                        + 0.55 * high_curve_basis
                        + 0.25 * direction_mode_sign * torch.sin(6.0 * math.pi * time_basis)
                    ) * mid_basis.pow(1.05)
                    turn_curve = signed_turn_cf * local_turn_basis
                    uturn_curve = (
                        uturn_cf
                        * direction_mode_sign
                        * (
                            0.70 * torch.sin(4.0 * math.pi * time_basis)
                            - 0.35 * torch.sin(8.0 * math.pi * time_basis)
                        )
                        * mid_basis.pow(1.10)
                    )
                    multi_curve = (
                        multi_cf
                        * (
                            1.00 * s_curve_basis
                            + 0.90 * torch.sin(4.0 * math.pi * time_basis)
                            + 0.45 * direction_mode_sign * torch.sin(8.0 * math.pi * time_basis)
                        )
                        * mid_basis.pow(1.10)
                    )
                    straight_curve = -0.65 * straight_cf * existing_direction_normal * mid_basis
                elif self.counterfactual_direction_residual_type == "transition_template_balanced":
                    balanced_turn_basis = (
                        1.20 * signed_basis * mid_basis
                        + 0.65 * s_curve_basis * mid_basis
                        + 0.35 * direction_mode_sign * high_curve_basis * mid_basis
                    )
                    turn_curve = signed_turn_cf * balanced_turn_basis
                    uturn_curve = (
                        uturn_cf
                        * direction_mode_sign
                        * (
                            0.80 * torch.sin(4.0 * math.pi * time_basis)
                            - 0.45 * torch.sin(8.0 * math.pi * time_basis)
                        )
                        * mid_basis.pow(1.10)
                    )
                    multi_curve = (
                        multi_cf
                        * (
                            0.95 * s_curve_basis
                            + 0.85 * torch.sin(4.0 * math.pi * time_basis)
                            + 0.55 * direction_mode_sign * torch.sin(6.0 * math.pi * time_basis)
                            - 0.25 * torch.sin(8.0 * math.pi * time_basis)
                        )
                        * mid_basis.pow(1.10)
                    )
                    straight_curve = -0.75 * straight_cf * existing_direction_normal * mid_basis
                elif self.counterfactual_direction_residual_type == "transition_template_contract":
                    local_heading_wave = (
                        1.10 * s_curve_basis
                        + 0.40 * high_curve_basis
                        + 0.25 * direction_mode_sign * torch.sin(6.0 * math.pi * time_basis)
                    ) * mid_basis.pow(0.80)
                    turn_curve = signed_turn_cf * local_heading_wave
                    uturn_curve = (
                        uturn_cf
                        * direction_mode_sign
                        * (
                            1.05 * torch.sin(2.0 * math.pi * time_basis)
                            - 0.60 * torch.sin(4.0 * math.pi * time_basis)
                            + 0.35 * torch.sin(6.0 * math.pi * time_basis)
                        )
                        * mid_basis.pow(0.90)
                    )
                    multi_curve = (
                        multi_cf
                        * (
                            0.85 * torch.sin(4.0 * math.pi * time_basis)
                            - 0.55 * torch.sin(6.0 * math.pi * time_basis)
                            + 0.45 * direction_mode_sign * torch.sin(8.0 * math.pi * time_basis)
                        )
                        * mid_basis.pow(0.95)
                    )
                    straight_curve = -1.10 * straight_cf * existing_direction_normal * mid_basis
                elif self.counterfactual_direction_residual_type == "transition_template_mean_projected":
                    turn_curve = signed_turn_cf * (
                        0.85 * mid_basis.pow(1.10)
                        + 0.60 * s_curve_basis * mid_basis
                        + 0.30 * direction_mode_sign * high_curve_basis * mid_basis
                    )
                    uturn_curve = (
                        uturn_cf
                        * direction_mode_sign
                        * (0.40 * torch.sign(signed_basis + 1e-6) + 0.35 * s_curve_basis)
                        * mid_basis.pow(1.15)
                    )
                    multi_curve = (
                        multi_cf
                        * (
                            1.05 * s_curve_basis
                            + 0.85 * torch.sin(4.0 * math.pi * time_basis)
                            + 0.40 * direction_mode_sign * torch.sin(6.0 * math.pi * time_basis)
                        )
                        * mid_basis.pow(1.10)
                    )
                    straight_curve = -0.80 * straight_cf * existing_direction_normal * mid_basis
                else:
                    turn_curve = signed_turn_cf * (
                        1.35 * mid_basis.pow(1.10)
                        + 0.28 * signed_basis * mid_basis
                    )
                    uturn_curve = (
                        uturn_cf
                        * direction_mode_sign
                        * (0.45 * torch.sign(signed_basis + 1e-6) + 0.25 * s_curve_basis)
                        * mid_basis.pow(1.15)
                    )
                    multi_curve = (
                        multi_cf
                        * (
                            1.10 * s_curve_basis
                            + 0.85 * torch.sin(4.0 * math.pi * time_basis)
                            + 0.30 * direction_mode_sign * torch.sin(6.0 * math.pi * time_basis)
                        )
                        * mid_basis.pow(1.10)
                    )
                    straight_curve = -straight_cf * existing_direction_normal * mid_basis
                if self.counterfactual_direction_residual_type not in {
                    "transition_contract_pulse",
                    "transition_arc_skeleton",
                    "transition_skeleton",
                    "transition_hybrid_skeleton_reflect",
                    "transition_pivot_skeleton",
                    "transition_reflect_skeleton",
                    "transition_skeleton_reversal",
                    "transition_skeleton_reversal_scurve",
                }:
                    direction_normal_curve = (
                        straight_curve
                        + turn_curve
                        + uturn_curve
                        + multi_curve
                        - 0.12 * (1.0 - straight_cf) * existing_direction_normal * mid_basis
                    )
                    direction_tangent_curve = (
                        -0.10 * straight_cf * existing_direction_tangent * mid_basis
                        - 0.10 * uturn_cf * end_basis
                        + 0.04 * signed_turn_cf * signed_basis * mid_basis
                        + 0.05 * multi_cf * s_curve_basis * mid_basis
                    )
                projection_mask = direction_mask.to(dtype=cond.dtype) * mid_basis
                if self.counterfactual_direction_residual_type == "transition_contract_pulse":
                    direction_normal_curve = self._project_curve_out_bases(
                        direction_normal_curve,
                        [
                            mid_basis,
                            end_basis,
                            signed_basis * mid_basis,
                        ],
                        projection_mask,
                    )
                    direction_tangent_curve = 0.20 * self._project_curve_out_bases(
                        direction_tangent_curve,
                        [
                            end_basis,
                            mid_basis,
                        ],
                        projection_mask,
                    )
                elif self.counterfactual_direction_residual_type in {
                    "transition_arc_skeleton",
                    "transition_arc_multiturn_burst",
                }:
                    direction_normal_curve = self._project_curve_out_bases(
                        direction_normal_curve,
                        (
                            [
                                mid_basis,
                                signed_basis * mid_basis,
                                s_curve_basis * mid_basis,
                            ]
                            if self.counterfactual_direction_residual_type == "transition_arc_multiturn_burst"
                            else [
                                mid_basis,
                                signed_basis * mid_basis,
                            ]
                        ),
                        projection_mask
                        * (
                            (straight_cf + uturn_cf + multi_cf).clamp(0.0, 1.0)
                            if self.counterfactual_direction_residual_type == "transition_arc_multiturn_burst"
                            else (straight_cf + uturn_cf).clamp(0.0, 1.0)
                        ),
                    )
                    direction_tangent_curve = 0.45 * self._project_curve_out_bases(
                        direction_tangent_curve,
                        (
                            [
                                end_basis,
                                mid_basis,
                                s_curve_basis * mid_basis,
                            ]
                            if self.counterfactual_direction_residual_type == "transition_arc_multiturn_burst"
                            else [
                                end_basis,
                                mid_basis,
                            ]
                        ),
                        projection_mask,
                    )
                elif self.counterfactual_direction_residual_type == "transition_template_projected":
                    direction_normal_curve = self._project_curve_out_bases(
                        direction_normal_curve,
                        [
                            mid_basis,
                            signed_basis * mid_basis,
                        ],
                        projection_mask,
                    )
                    direction_tangent_curve = 0.25 * self._project_curve_out_bases(
                        direction_tangent_curve,
                        [
                            end_basis,
                            signed_basis * mid_basis,
                        ],
                        projection_mask,
                    )
                elif self.counterfactual_direction_residual_type == "transition_template_mean_projected":
                    direction_normal_curve = self._project_curve_out_bases(
                        direction_normal_curve,
                        [mid_basis],
                        projection_mask,
                    )
                    direction_tangent_curve = 0.35 * self._project_curve_out_bases(
                        direction_tangent_curve,
                        [end_basis],
                        projection_mask,
                    )
                elif self.counterfactual_direction_residual_type == "transition_template_balanced":
                    direction_normal_curve = self._project_curve_out_bases(
                        direction_normal_curve,
                        [
                            mid_basis,
                            end_basis,
                        ],
                        projection_mask,
                    )
                elif self.counterfactual_direction_residual_type == "transition_template_contract":
                    direction_normal_curve = self._project_curve_out_bases(
                        direction_normal_curve,
                        [
                            mid_basis,
                            end_basis,
                            signed_basis * mid_basis,
                        ],
                        projection_mask,
                    )
                    direction_tangent_curve = 0.10 * self._project_curve_out_bases(
                        direction_tangent_curve,
                        [
                            end_basis,
                            mid_basis,
                        ],
                        projection_mask,
                    )
                elif self.counterfactual_direction_residual_type in {
                    "transition_skeleton",
                    "transition_hybrid_skeleton_reflect",
                    "transition_pivot_skeleton",
                    "transition_reflect_skeleton",
                    "transition_skeleton_reversal",
                    "transition_skeleton_reversal_scurve",
                }:
                    direction_tangent_curve = self._project_curve_out_bases(
                        direction_tangent_curve,
                        [end_basis],
                        projection_mask,
                    )
                    direction_tangent_curve = 0.20 * self._project_curve_out_bases(
                        direction_tangent_curve,
                        [
                            end_basis,
                            mid_basis,
                        ],
                        projection_mask,
                    )
                direction_normal_curve = self._project_curve_out_bases(
                    direction_normal_curve,
                    [mid_basis],
                    projection_mask * (uturn_cf + multi_cf).clamp(0.0, 1.0),
                )
            if self.counterfactual_curvature_residual_type in {
                "local_turn_density",
                "base_preserving_local_turn_density",
                "base_selected_projected_normal_only",
                "base_selected_projected_normal_selector_compose",
                "direction_neutral_turn_density",
                "direction_preserving_curvature_burst",
                "signed_balanced_winding_basis",
                "midlocal_winding_pulse",
            }:
                local_mode_phase = torch.linspace(
                    -1.0,
                    1.0,
                    steps=self.num_candidates,
                    device=cond.device,
                    dtype=cond.dtype,
                ).view(1, self.num_candidates, 1, 1)
                local_mode_sign = torch.sign(local_mode_phase + 1e-6)
                local_high_curve_basis = torch.sin(6.0 * math.pi * time_basis)
                local_window = curvature_local_window * mid_basis
                if self.counterfactual_curvature_residual_type in {
                    "base_preserving_local_turn_density",
                    "base_selected_projected_normal_only",
                    "base_selected_projected_normal_selector_compose",
                    "direction_neutral_turn_density",
                    "direction_preserving_curvature_burst",
                    "signed_balanced_winding_basis",
                    "midlocal_winding_pulse",
                }:
                    curvature_ids_for_cf = None
                    if mobility_tokens is not None:
                        curvature_ids_for_cf = mobility_tokens[:, curvature_field_idx].to(
                            device=cond.device,
                            dtype=torch.long,
                        ).clamp(0, 2)
                    if curvature_ids_for_cf is None:
                        curvature_one_hot_cf = cond.new_zeros(anchor.shape[0], 3)
                        curvature_one_hot_cf[:, 2] = 1.0
                    else:
                        curvature_one_hot_cf = F.one_hot(curvature_ids_for_cf, num_classes=3).to(dtype=cond.dtype)
                    smooth_cf = curvature_one_hot_cf[:, 0:1].view(anchor.shape[0], 1, 1, 1)
                    moderate_cf = curvature_one_hot_cf[:, 1:2].view(anchor.shape[0], 1, 1, 1)
                    winding_cf = curvature_one_hot_cf[:, 2:3].view(anchor.shape[0], 1, 1, 1)
                    raw_residual_view_for_cf_curvature = raw_residual.view(
                        anchor.shape[0],
                        self.num_candidates,
                        target_len,
                        2,
                    )
                    existing_normal_cf = raw_residual_view_for_cf_curvature[..., 1:2]
                    local_ultra_curve_basis = torch.sin(10.0 * math.pi * time_basis)
                    if self.counterfactual_curvature_residual_type == "direction_neutral_turn_density":
                        neutral_curve = (
                            0.55 * torch.sin(8.0 * math.pi * time_basis)
                            - 0.35 * torch.sin(12.0 * math.pi * time_basis)
                            + 0.20 * torch.sin(16.0 * math.pi * time_basis)
                        )
                        local_normal_curve = (
                            -0.45 * smooth_cf * existing_normal_cf * local_window
                            + moderate_cf * 0.85 * neutral_curve * local_window
                            + winding_cf * 1.55 * neutral_curve * local_window
                        )
                        local_tangent_curve = torch.zeros_like(local_normal_curve)
                    elif self.counterfactual_curvature_residual_type == "direction_preserving_curvature_burst":
                        burst_window = local_window * curvature_local_window
                        burst_curve = (
                            0.80 * torch.sin(8.0 * math.pi * time_basis)
                            - 0.62 * torch.sin(12.0 * math.pi * time_basis)
                            + 0.36 * torch.sin(16.0 * math.pi * time_basis)
                            - 0.16 * torch.sin(20.0 * math.pi * time_basis)
                        )
                        mode_burst_curve = (
                            0.22 * local_mode_sign * torch.sin(10.0 * math.pi * time_basis)
                            - 0.12 * local_mode_sign * torch.sin(18.0 * math.pi * time_basis)
                        )
                        local_normal_curve = (
                            -0.42 * smooth_cf * existing_normal_cf * burst_window
                            + moderate_cf * 0.82 * (burst_curve + 0.28 * mode_burst_curve) * burst_window
                            + winding_cf * 1.48 * (burst_curve + 0.78 * mode_burst_curve) * burst_window
                        )
                        local_tangent_curve = (
                            0.0030
                            * (moderate_cf + winding_cf)
                            * (
                                0.55 * torch.sin(8.0 * math.pi * time_basis)
                                - 0.35 * torch.sin(12.0 * math.pi * time_basis)
                            )
                            * burst_window
                        )
                    elif self.counterfactual_curvature_residual_type == "signed_balanced_winding_basis":
                        burst_window = local_window * curvature_local_window
                        winding_variant = 0.35 + 0.55 * local_mode_phase.abs()
                        balanced_curve = self._build_signed_balanced_winding_curve(
                            time_basis,
                            burst_window,
                            winding_variant,
                        )
                        balanced_curve = self._stabilize_signed_balanced_winding_curve(
                            balanced_curve,
                            time_basis,
                            burst_window,
                            curvature_mask.to(dtype=cond.dtype),
                            existing_normal_basis=existing_normal_cf,
                            target_rms=0.88,
                        )
                        local_normal_curve = (
                            -0.40 * smooth_cf * existing_normal_cf * burst_window
                            + moderate_cf * 0.78 * balanced_curve
                            + winding_cf * 1.36 * balanced_curve
                        )
                        local_tangent_curve = torch.zeros_like(local_normal_curve)
                    elif self.counterfactual_curvature_residual_type == "midlocal_winding_pulse":
                        winding_variant = 0.30 + 0.45 * local_mode_phase.abs()
                        pulse_curve, pulse_window = self._build_midlocal_winding_pulse_curve(
                            time_basis,
                            local_window * curvature_local_window,
                            winding_variant,
                        )
                        pulse_curve = self._stabilize_signed_balanced_winding_curve(
                            pulse_curve,
                            time_basis,
                            pulse_window,
                            curvature_mask.to(dtype=cond.dtype),
                            existing_normal_basis=existing_normal_cf,
                            target_rms=0.92,
                        )
                        burst_window = pulse_window
                        local_normal_curve = (
                            -0.34 * smooth_cf * existing_normal_cf * burst_window
                            + moderate_cf * 0.88 * pulse_curve
                            + winding_cf * 1.58 * pulse_curve
                        )
                        local_tangent_curve = torch.zeros_like(local_normal_curve)
                    elif self.counterfactual_curvature_residual_type in {
                        "base_selected_projected_normal_only",
                        "base_selected_projected_normal_selector_compose",
                    }:
                        base_normal_basis = existing_normal_cf
                        if base_selected_normal_scalar is not None:
                            base_normal_basis = base_selected_normal_scalar.to(
                                device=cond.device,
                                dtype=cond.dtype,
                            )
                        winding_variant = 0.25 + 0.50 * local_mode_phase.abs()
                        projected_curve, projected_window = self._build_base_selected_projected_normal_curve(
                            time_basis,
                            local_window * curvature_local_window,
                            base_normal_basis,
                            winding_variant,
                        )
                        projected_base_normal = self._project_curve_out_bases(
                            base_normal_basis * projected_window,
                            [
                                projected_window,
                                signed_basis * projected_window,
                                mid_basis * projected_window,
                                end_basis * projected_window,
                            ],
                            projected_window,
                        )
                        burst_window = projected_window
                        local_normal_curve = (
                            -0.20 * smooth_cf * projected_base_normal
                            + moderate_cf * 1.05 * projected_curve
                            + winding_cf * 1.65 * projected_curve
                        )
                        local_tangent_curve = torch.zeros_like(local_normal_curve)
                    else:
                        local_normal_curve = (
                            -0.60 * smooth_cf * existing_normal_cf * local_window
                            + moderate_cf
                            * (
                                0.55 * high_curve_basis
                                + 0.25 * local_mode_sign * local_high_curve_basis
                            )
                            * local_window
                            + winding_cf
                            * (
                                1.05 * high_curve_basis
                                + local_mode_sign * 0.75 * local_high_curve_basis
                                + 0.30 * local_ultra_curve_basis
                            )
                            * local_window
                        )
                        local_tangent_curve = (
                            0.006
                            * (moderate_cf + winding_cf)
                            * (high_curve_basis - local_mode_sign * local_high_curve_basis)
                            * local_window
                        )
                else:
                    local_normal_curve = (
                        (0.95 + 0.20 * params[..., 3:4].abs()) * high_curve_basis
                        + local_mode_sign * (0.70 + 0.20 * params[..., 5:6].abs()) * local_high_curve_basis
                        + 0.20 * params[..., 2:3] * s_curve_basis
                    ) * local_window
                    local_tangent_curve = (
                        0.015
                        * params[..., 4:5]
                        * (high_curve_basis - local_mode_sign * local_high_curve_basis)
                        * local_window
                    )
                projection_mask = curvature_mask.to(dtype=cond.dtype) * (
                    burst_window
                    if self.counterfactual_curvature_residual_type in {
                        "base_selected_projected_normal_only",
                        "base_selected_projected_normal_selector_compose",
                        "direction_preserving_curvature_burst",
                        "midlocal_winding_pulse",
                    }
                    else local_window
                )
                local_normal_curve = self._project_curve_out_bases(
                    local_normal_curve,
                    (
                        [
                            local_window,
                            signed_basis * local_window,
                            mid_basis * local_window,
                            (signed_basis.pow(2) - 0.35) * mid_basis * local_window,
                            end_basis * local_window,
                            (
                                existing_normal_cf
                                if base_selected_normal_scalar is None
                                else base_selected_normal_scalar.to(device=cond.device, dtype=cond.dtype)
                            ).abs() * burst_window,
                            signed_basis
                            * (
                                existing_normal_cf
                                if base_selected_normal_scalar is None
                                else base_selected_normal_scalar.to(device=cond.device, dtype=cond.dtype)
                            ).abs()
                            * burst_window,
                        ]
                        if self.counterfactual_curvature_residual_type in {
                            "base_selected_projected_normal_only",
                            "base_selected_projected_normal_selector_compose",
                        }
                        else
                        [
                            local_window,
                            signed_basis * local_window,
                            mid_basis * local_window,
                            s_curve_basis * local_window,
                            existing_normal_cf * burst_window,
                            end_basis * local_window,
                        ]
                        if self.counterfactual_curvature_residual_type == "direction_preserving_curvature_burst"
                        else [
                            local_window,
                            signed_basis * local_window,
                            mid_basis * local_window,
                            end_basis * local_window,
                            existing_normal_cf * local_window,
                        ]
                    ),
                    projection_mask,
                )
                local_tangent_curve = self._project_curve_out_bases(
                    local_tangent_curve,
                    (
                        [
                            signed_basis * burst_window,
                            mid_basis * burst_window,
                            end_basis * burst_window,
                        ]
                        if self.counterfactual_curvature_residual_type in {
                            "base_selected_projected_normal_only",
                            "base_selected_projected_normal_selector_compose",
                        }
                        else
                        [
                            signed_basis * local_window,
                            mid_basis * local_window,
                            s_curve_basis * local_window,
                            existing_normal_cf * burst_window,
                            end_basis * local_window,
                        ]
                        if self.counterfactual_curvature_residual_type == "direction_preserving_curvature_burst"
                        else [
                            signed_basis * local_window,
                            mid_basis * local_window,
                            end_basis * local_window,
                        ]
                    ),
                    projection_mask,
                )
                if self.counterfactual_curvature_residual_type == "signed_balanced_winding_basis":
                    local_normal_curve = self._normalize_masked_curve_rms(
                        local_normal_curve,
                        projection_mask,
                        target_rms=0.88,
                    )
                elif self.counterfactual_curvature_residual_type in {
                    "base_selected_projected_normal_only",
                    "base_selected_projected_normal_selector_compose",
                }:
                    local_normal_curve = self._normalize_masked_curve_rms(
                        local_normal_curve,
                        projection_mask,
                        target_rms=1.20,
                    )
                elif self.counterfactual_curvature_residual_type == "midlocal_winding_pulse":
                    local_normal_curve = self._normalize_masked_curve_rms(
                        local_normal_curve,
                        projection_mask,
                        target_rms=0.92,
                    )
                curvature_normal_curve = local_normal_curve
                curvature_tangent_curve = local_tangent_curve
            direction_extra_boost_scale = (
                cond.new_tensor(0.25)
                if self.counterfactual_direction_residual_type in {
                    "transition_contract_pulse",
                    "transition_hybrid_skeleton_reflect",
                    "transition_pivot_skeleton",
                    "transition_reflect_skeleton",
                    "transition_skeleton_reversal",
                    "transition_skeleton_reversal_scurve",
                    "transition_skeleton",
                    "transition_template_balanced",
                    "transition_template_contract",
                }
                else cond.new_tensor(1.0)
            )
            tangent_curve = torch.where(
                direction_mask,
                direction_tangent_curve
                + direction_extra_boost_scale * (direction_multi_tangent_boost + direction_multi_transition_tangent_boost),
                torch.where(curvature_mask, curvature_tangent_curve, torch.where(reference_mask, reference_tangent_curve, generic_tangent_curve)),
            )
            normal_curve = torch.where(
                direction_mask,
                direction_normal_curve
                + direction_extra_boost_scale * (direction_multi_normal_boost + direction_multi_transition_normal_boost),
                torch.where(curvature_mask, curvature_normal_curve, torch.where(reference_mask, reference_normal_curve, generic_normal_curve)),
            )
            tangent_scale = torch.where(
                direction_mask,
                tangent_curve.new_tensor(0.04),
                torch.where(
                    curvature_mask,
                    (
                        tangent_curve.new_tensor(0.0)
                        if self.counterfactual_curvature_residual_type in {
                            "base_selected_projected_normal_only",
                            "base_selected_projected_normal_selector_compose",
                        }
                        else tangent_curve.new_tensor(0.012)
                        if self.counterfactual_curvature_residual_type in {
                            "base_preserving_local_turn_density",
                            "base_selected_projected_normal_only",
                            "base_selected_projected_normal_selector_compose",
                            "direction_neutral_turn_density",
                            "direction_preserving_curvature_burst",
                            "signed_balanced_winding_basis",
                        }
                        else tangent_curve.new_tensor(0.02)
                    )
                    * (1.0 - 0.65 * curvature_preserve_direction_f),
                    torch.where(reference_mask, tangent_curve.new_tensor(1.0), tangent_curve.new_tensor(0.22)),
                ),
            )
            normal_scale = torch.where(
                direction_mask,
                normal_curve.new_tensor(0.16),
                torch.where(
                    curvature_mask,
                    (
                        normal_curve.new_tensor(0.22)
                        if self.counterfactual_curvature_residual_type in {
                            "base_selected_projected_normal_only",
                            "base_selected_projected_normal_selector_compose",
                        }
                        else normal_curve.new_tensor(0.16)
                        if self.counterfactual_curvature_residual_type in {
                            "base_preserving_local_turn_density",
                            "base_selected_projected_normal_only",
                            "base_selected_projected_normal_selector_compose",
                            "direction_neutral_turn_density",
                            "signed_balanced_winding_basis",
                        }
                        else (
                            normal_curve.new_tensor(0.16)
                            if self.counterfactual_curvature_residual_type == "direction_preserving_curvature_burst"
                            else normal_curve.new_tensor(0.13)
                        )
                    )
                    * (
                        normal_curve.new_tensor(1.0)
                        if self.counterfactual_curvature_residual_type in {
                            "base_selected_projected_normal_only",
                            "base_selected_projected_normal_selector_compose",
                        }
                        else (1.0 - 0.35 * curvature_preserve_direction_f)
                    ),
                    torch.where(
                        reference_mask,
                        (
                            normal_curve.new_tensor(0.85)
                            if self.counterfactual_reference_residual_type == "normal_only_projected"
                            else normal_curve.new_tensor(1.0)
                        ),
                        normal_curve.new_tensor(0.22),
                    ),
                ),
            )
            counterfactual_residual_delta = (
                cf_adapter_scale
                * self.counterfactual_residual_strength
                * envelope.view(1, 1, target_len, 1)
                * (
                tangent_scale * self.tangent_scale * tangent_curve * tangent.view(anchor.shape[0], 1, target_len, 2)
                + normal_scale * self.normal_scale * normal_curve * normal.view(anchor.shape[0], 1, target_len, 2)
            )
            )
            if counterfactual_local_proposal_delta is not None:
                counterfactual_residual_delta = counterfactual_residual_delta + counterfactual_local_proposal_delta
            if self.counterfactual_route_frame_budget_mode != "none":
                active_field_mask = torch.ones_like(field_view, dtype=torch.bool)
                if self.counterfactual_route_frame_fields is not None:
                    active_field_mask = torch.zeros_like(field_view, dtype=torch.bool)
                    for field_name in self.counterfactual_route_frame_fields:
                        field_idx = DEFAULT_MOBILITY_VOCAB.fields.index(field_name)
                        active_field_mask = active_field_mask | (field_view == int(field_idx))
                tangent_basis = tangent.view(anchor.shape[0], 1, target_len, 2)
                normal_basis = normal.view(anchor.shape[0], 1, target_len, 2)
                tangent_component = (counterfactual_residual_delta * tangent_basis).sum(dim=-1, keepdim=True)
                normal_component = (counterfactual_residual_delta * normal_basis).sum(dim=-1, keepdim=True)
                tangent_budget = counterfactual_residual_delta.new_tensor(
                    max(0.0, float(self.counterfactual_route_frame_tangent_budget))
                )
                normal_budget = counterfactual_residual_delta.new_tensor(
                    max(0.0, float(self.counterfactual_route_frame_normal_budget))
                )
                if self.counterfactual_route_frame_budget_mode == "soft_tanh":
                    tangent_bounded = tangent_budget * torch.tanh(
                        tangent_component / tangent_budget.clamp_min(1e-6)
                    )
                    normal_bounded = normal_budget * torch.tanh(
                        normal_component / normal_budget.clamp_min(1e-6)
                    )
                else:
                    tangent_bounded = tangent_component.clamp(
                        min=-float(self.counterfactual_route_frame_tangent_budget),
                        max=float(self.counterfactual_route_frame_tangent_budget),
                    )
                    normal_bounded = normal_component.clamp(
                        min=-float(self.counterfactual_route_frame_normal_budget),
                        max=float(self.counterfactual_route_frame_normal_budget),
                    )
                bounded_delta = tangent_bounded * tangent_basis + normal_bounded * normal_basis
                counterfactual_residual_delta = torch.where(
                    active_field_mask,
                    bounded_delta,
                    counterfactual_residual_delta,
                )
            if strict_geometry_preserving_curvature_counterfactual:
                base_preserving_selected_residual = prepared_base_selected_residual
                base_preserving_selected_residual = base_preserving_selected_residual.unsqueeze(1).expand(
                    -1,
                    self.num_candidates,
                    -1,
                    -1,
                ).reshape(anchor.shape[0] * self.num_candidates, target_len, 2)
                base_preserving_composed_residual = self.preserve_reference_endpoints(
                    base_preserving_selected_residual
                    + counterfactual_residual_delta.reshape(anchor.shape[0] * self.num_candidates, target_len, 2)
                )
                base_preserving_composed_delta = self.preserve_reference_endpoints(
                    base_preserving_composed_residual - base_residual_before_language
                )
            else:
                _apply_language_control_delta("counterfactual", counterfactual_residual_delta)
        language_control_delta_final = language_control_delta_sum
        if self.language_control_residual_type == "bounded_sum":
            base_norm = torch.linalg.vector_norm(
                base_residual_before_language.reshape(base_residual_before_language.shape[0], -1),
                dim=1,
                keepdim=True,
            ).view(-1, 1, 1)
            delta_norm = torch.linalg.vector_norm(
                language_control_delta_sum.reshape(language_control_delta_sum.shape[0], -1),
                dim=1,
                keepdim=True,
            ).view(-1, 1, 1)
            min_budget = residual.new_tensor(0.015)
            budget = (float(max(0.0, self.language_control_budget_ratio)) * base_norm).clamp_min(min_budget)
            language_control_budget_scale = (budget / delta_norm.clamp_min(1e-6)).clamp(max=1.0)
            language_control_delta_final = self.preserve_reference_endpoints(
                language_control_delta_sum * language_control_budget_scale
            )
            residual = residual + language_control_delta_final
        elif self.language_control_residual_type == "field_soft_gate":
            base_norm = torch.linalg.vector_norm(
                base_residual_before_language.reshape(base_residual_before_language.shape[0], -1),
                dim=1,
                keepdim=True,
            ).view(-1, 1, 1)
            min_budget = residual.new_tensor(0.015)
            soft_min = float(max(0.0, self.language_control_soft_gate_min))
            soft_max = float(max(soft_min, self.language_control_soft_gate_max))

            def _ratio_for_kind(kind: str) -> torch.Tensor:
                if kind == "counterfactual" and counterfactual_field_ids is not None:
                    cf_field_ids_local = counterfactual_field_ids.to(device=cond.device, dtype=torch.long).clamp(
                        0,
                        len(DEFAULT_MOBILITY_VOCAB.fields) - 1,
                    )
                    values = []
                    for field_idx in cf_field_ids_local.detach().cpu().tolist():
                        field_name = DEFAULT_MOBILITY_VOCAB.fields[int(field_idx)]
                        values.append(
                            float(
                                self.language_control_field_ratios.get(
                                    field_name,
                                    self.language_control_field_ratios.get("counterfactual", 0.70),
                                )
                            )
                        )
                    ratio = base_norm.new_tensor(values).view(anchor.shape[0], 1, 1)
                    return ratio.repeat_interleave(self.num_candidates, dim=0)
                ratio_value = float(
                    self.language_control_field_ratios.get(
                        kind,
                        self.language_control_field_ratios.get("default", 0.70),
                    )
                )
                return base_norm.new_full(base_norm.shape, ratio_value)

            language_control_delta_final = residual.new_zeros(residual.shape)
            field_scale_values = []
            gate_scope_active = self.language_control_gate_scope == "all" or (
                self.language_control_gate_scope == "counterfactual" and bool(counterfactual_mode)
            )
            for kind, kind_delta in language_control_delta_by_kind.items():
                should_gate = gate_scope_active and (
                    self.language_control_gate_fields is None or kind in self.language_control_gate_fields
                )
                if should_gate:
                    kind_norm = torch.linalg.vector_norm(
                        kind_delta.reshape(kind_delta.shape[0], -1),
                        dim=1,
                        keepdim=True,
                    ).view(-1, 1, 1)
                    field_ratio = _ratio_for_kind(kind).clamp_min(0.0)
                    budget = (
                        float(max(0.0, self.language_control_budget_ratio)) * field_ratio * base_norm
                    ).clamp_min(min_budget)
                    kind_scale = (budget / kind_norm.clamp_min(1e-6)).clamp(min=soft_min, max=soft_max)
                else:
                    kind_scale = base_norm.new_ones(base_norm.shape)
                language_control_scale_by_kind[kind] = kind_scale
                language_control_delta_final = language_control_delta_final + kind_delta * kind_scale
                field_scale_values.append(kind_scale)
            if field_scale_values:
                language_control_budget_scale = torch.stack(field_scale_values, dim=0).mean(dim=0)
            language_control_delta_final = self.preserve_reference_endpoints(language_control_delta_final)
            residual = residual + language_control_delta_final
        if (
            selector_side_base_preserving_curvature_counterfactual
            and "counterfactual" in language_control_delta_by_kind
        ):
            selector_side_counterfactual_delta = language_control_delta_by_kind["counterfactual"]
            if self.language_control_residual_type == "bounded_sum":
                selector_side_counterfactual_delta = (
                    selector_side_counterfactual_delta * language_control_budget_scale
                )
            elif "counterfactual" in language_control_scale_by_kind:
                selector_side_counterfactual_delta = (
                    selector_side_counterfactual_delta
                    * language_control_scale_by_kind["counterfactual"]
                )
            selector_side_counterfactual_delta = self.preserve_reference_endpoints(
                selector_side_counterfactual_delta
            )
        if base_preserving_composed_residual is not None and base_preserving_composed_delta is not None:
            residual = base_preserving_composed_residual
            language_control_delta_sum = base_preserving_composed_delta
            language_control_delta_final = base_preserving_composed_delta
            language_control_delta_by_kind = {"counterfactual": base_preserving_composed_delta}
            language_control_scale_by_kind = {
                "counterfactual": residual.new_ones((anchor.shape[0] * self.num_candidates, 1, 1))
            }
            language_control_budget_scale = residual.new_ones((anchor.shape[0] * self.num_candidates, 1, 1))
        residual = self.preserve_reference_endpoints(residual)
        language_control_base_norm = torch.linalg.vector_norm(
            base_residual_before_language.reshape(base_residual_before_language.shape[0], -1),
            dim=1,
        ).view(anchor.shape[0], self.num_candidates)
        language_control_raw_norm = torch.linalg.vector_norm(
            language_control_delta_sum.reshape(language_control_delta_sum.shape[0], -1),
            dim=1,
        ).view(anchor.shape[0], self.num_candidates)
        language_control_final_norm = torch.linalg.vector_norm(
            language_control_delta_final.reshape(language_control_delta_final.shape[0], -1),
            dim=1,
        ).view(anchor.shape[0], self.num_candidates)
        language_control_ratio = language_control_final_norm / language_control_base_norm.clamp_min(1e-6)
        language_control_budget_scale_view = language_control_budget_scale.view(
            anchor.shape[0],
            self.num_candidates,
        )
        language_control_field_scales = {
            kind: value.view(anchor.shape[0], self.num_candidates)
            for kind, value in language_control_scale_by_kind.items()
        }
        language_control_field_norms = {
            kind: torch.linalg.vector_norm(value.reshape(value.shape[0], -1), dim=1).view(
                anchor.shape[0],
                self.num_candidates,
            )
            for kind, value in language_control_delta_by_kind.items()
        }
        reference_positive_candidate_scale = residual.new_zeros(anchor.shape[0], self.num_candidates)
        reference_positive_strength = float(
            max(0.0, min(1.0, self.reference_positive_candidate_strength))
        )
        if reference_positive_strength > 0.0 and mobility_tokens is not None:
            field_index = DEFAULT_MOBILITY_VOCAB.fields.index("reference_adherence")
            tight_id = int(DEFAULT_MOBILITY_VOCAB.token_to_id["reference_adherence"]["tight"])
            scope = str(getattr(self, "reference_positive_candidate_scope", "all"))
            if scope == "all":
                scope_mask = residual.new_ones(anchor.shape[0])
            elif scope == "none":
                scope_mask = residual.new_zeros(anchor.shape[0])
            elif scope == "counterfactual_reference_adherence":
                if counterfactual_field_ids is None:
                    scope_mask = residual.new_zeros(anchor.shape[0])
                else:
                    scope_mask = (
                        counterfactual_field_ids.to(device=residual.device, dtype=torch.long)
                        == int(field_index)
                    ).to(dtype=residual.dtype)
            else:
                raise ValueError(
                    f"Unsupported reference_positive_candidate_scope={scope}."
                )
            tight_mask = (
                mobility_tokens[:, field_index].to(device=residual.device, dtype=torch.long) == tight_id
            ).to(dtype=residual.dtype) * scope_mask
            if bool((tight_mask > 0).any().detach().item()):
                mode_roles = self.mode_role_table.to(device=residual.device, dtype=residual.dtype)
                close_role = mode_roles[:, 0].clamp(min=0.0)
                close_role = close_role / close_role.max().clamp_min(1e-6)
                reference_positive_candidate_scale = (
                    reference_positive_strength * tight_mask.view(-1, 1) * close_role.view(1, -1)
                ).clamp(0.0, 1.0)
                residual_view_for_reference = residual.view(anchor.shape[0], self.num_candidates, target_len, 2)
                positive_scale = reference_positive_candidate_scale.view(anchor.shape[0], self.num_candidates, 1, 1)
                time_basis = time.view(anchor.shape[0], 1, target_len, 1).expand(
                    -1,
                    self.num_candidates,
                    -1,
                    -1,
                )
                mid_basis = torch.sin(math.pi * time_basis).clamp_min(0.0)
                signed_mid_basis = (time_basis - 0.5) * 2.0 * mid_basis
                active_mask = (positive_scale > 0.0).to(dtype=residual.dtype) * mid_basis
                # A positive reference-adherence candidate must actually cover
                # the close-reference label.  Low-frequency-only damping kept
                # too much local deviation and often still decoded as loose.
                highfreq_residual = residual_view_for_reference.new_zeros(residual_view_for_reference.shape)
                lowfreq_residual = residual_view_for_reference
                residual = (
                    highfreq_residual + lowfreq_residual * (1.0 - positive_scale)
                ).reshape(anchor.shape[0] * self.num_candidates, target_len, 2)
        reference_distance_candidate_scale = residual.new_zeros(anchor.shape[0], self.num_candidates)
        reference_distance_candidate_applied = residual.new_zeros(
            anchor.shape[0],
            device=residual.device,
            dtype=residual.dtype,
        )
        reference_distance_candidate_overridden_by_local_control = False
        reference_distance_strength = float(
            max(0.0, min(1.0, self.reference_distance_candidate_strength))
        )
        if reference_distance_strength > 0.0 and mobility_tokens is not None:
            field_index = DEFAULT_MOBILITY_VOCAB.fields.index("reference_adherence")
            ref_vocab = DEFAULT_MOBILITY_VOCAB.token_to_id["reference_adherence"]
            scope = str(getattr(self, "reference_distance_candidate_scope", "none"))
            if scope == "all":
                scope_mask = residual.new_ones(anchor.shape[0])
            elif scope == "none":
                scope_mask = residual.new_zeros(anchor.shape[0])
            elif scope == "instruction":
                scope_mask = residual.new_zeros(anchor.shape[0]) if counterfactual_mode else residual.new_ones(anchor.shape[0])
            elif scope == "counterfactual_reference_adherence":
                if counterfactual_field_ids is None:
                    scope_mask = residual.new_zeros(anchor.shape[0])
                else:
                    scope_mask = (
                        counterfactual_field_ids.to(device=residual.device, dtype=torch.long)
                        == int(field_index)
                    ).to(dtype=residual.dtype)
            else:
                raise ValueError(
                    f"Unsupported reference_distance_candidate_scope={scope}."
                )
            reference_ids = mobility_tokens[:, field_index].to(device=residual.device, dtype=torch.long)
            tight_mask = (reference_ids == int(ref_vocab["tight"])).to(dtype=residual.dtype) * scope_mask
            moderate_mask = (reference_ids == int(ref_vocab["moderate"])).to(dtype=residual.dtype) * scope_mask
            loose_mask = (reference_ids == int(ref_vocab["loose"])).to(dtype=residual.dtype) * scope_mask
            if bool(((tight_mask + moderate_mask + loose_mask) > 0).any().detach().item()):
                mode_roles = self.mode_role_table.to(device=residual.device, dtype=residual.dtype)
                close_role = mode_roles[:, 0].clamp(min=0.0)
                loose_role = mode_roles[:, 2].clamp(min=0.0)
                winding_role = mode_roles[:, 3].clamp(min=0.0)
                close_role = close_role / close_role.max().clamp_min(1e-6)
                loose_role = loose_role / loose_role.max().clamp_min(1e-6)
                moderate_role = (1.0 - torch.clamp(close_role + loose_role, 0.0, 1.0)).clamp(0.0, 1.0)
                if float(moderate_role.max().detach().item()) <= 1e-6:
                    moderate_role = 1.0 - close_role
                moderate_role = moderate_role / moderate_role.max().clamp_min(1e-6)
                mode_phase = torch.linspace(
                    -1.0,
                    1.0,
                    steps=self.num_candidates,
                    device=residual.device,
                    dtype=residual.dtype,
                )
                mode_sign = torch.sign(mode_phase + 1e-6).view(1, self.num_candidates, 1, 1)
                residual_view_for_reference = residual.view(anchor.shape[0], self.num_candidates, target_len, 2)
                time_basis = time.view(anchor.shape[0], 1, target_len, 1).expand(
                    -1,
                    self.num_candidates,
                    -1,
                    -1,
                )
                mid_basis = torch.sin(math.pi * time_basis).clamp_min(0.0)
                s_curve_basis = torch.sin(2.0 * math.pi * time_basis)
                tight_scale = (
                    reference_distance_strength
                    * tight_mask.view(-1, 1)
                    * close_role.view(1, -1)
                ).clamp(0.0, 1.0)
                moderate_scale = (
                    reference_distance_strength
                    * moderate_mask.view(-1, 1)
                    * moderate_role.view(1, -1)
                ).clamp(0.0, 1.0)
                loose_scale = (
                    reference_distance_strength
                    * loose_mask.view(-1, 1)
                    * torch.maximum(loose_role, winding_role).view(1, -1)
                ).clamp(0.0, 1.0)
                reference_distance_candidate_scale = torch.maximum(
                    tight_scale,
                    torch.maximum(moderate_scale, loose_scale),
                )
                tight_scale_v = tight_scale.view(anchor.shape[0], self.num_candidates, 1, 1)
                moderate_scale_v = moderate_scale.view(anchor.shape[0], self.num_candidates, 1, 1)
                loose_scale_v = loose_scale.view(anchor.shape[0], self.num_candidates, 1, 1)
                normal_basis = normal.view(anchor.shape[0], 1, target_len, 2)
                proposal_mode = str(
                    getattr(self, "reference_distance_candidate_mode", "global_normal")
                )
                if proposal_mode == "local_direction_preserving":
                    centered = (time_basis - 0.5).abs()
                    local_window = torch.exp(-0.5 * (centered / 0.22).pow(2)) * mid_basis
                    highfreq = torch.sin(2.0 * math.pi * time_basis) * local_window
                    counter_highfreq = torch.sin(4.0 * math.pi * time_basis) * local_window
                    tight_residual = residual_view_for_reference * (
                        1.0 - 0.35 * tight_scale_v * local_window
                    )
                    proposed = tight_residual
                    moderate_curve = 0.024 * mode_sign * highfreq
                    loose_curve = (
                        0.052 * mode_sign * highfreq
                        + 0.014 * mode_sign * counter_highfreq
                    )
                    proposed = proposed + moderate_scale_v * moderate_curve * normal_basis
                    proposed = proposed + loose_scale_v * loose_curve * normal_basis
                elif proposal_mode == "global_normal":
                    # Bounded endpoint-preserving proposals create reference-distance
                    # hypotheses without training a global normal-offset head.  The
                    # amplitudes are normalized OD-frame distances and stay smaller
                    # than the failed learned reference-control smoke.
                    proposed = residual_view_for_reference * (1.0 - 0.55 * tight_scale_v)
                    moderate_curve = 0.035 * mode_sign * mid_basis
                    loose_curve = (
                        0.090 * mode_sign * mid_basis
                        + 0.018 * mode_sign * s_curve_basis * mid_basis
                    )
                    proposed = proposed + moderate_scale_v * moderate_curve * normal_basis
                    proposed = proposed + loose_scale_v * loose_curve * normal_basis
                else:
                    raise ValueError(
                        "Unsupported reference_distance_candidate_mode="
                        f"{proposal_mode}"
                    )
                blend = torch.maximum(
                    tight_scale_v,
                    torch.maximum(moderate_scale_v, loose_scale_v),
                ).clamp(0.0, 1.0)
                # This proposal mutates the generated residual path only. Under
                # base-preserving reference control, the final control bank is
                # rebuilt later from the routed local-proposal branch, so this
                # path would otherwise create misleading diagnostics by seeming
                # "active" while not surviving into the final candidate pool.
                disable_generated_reference_distance_path = bool(
                    self.base_preserving_selector_enabled
                    and counterfactual_field_ids is not None
                    and prepared_base_selected_trajectory is not None
                )
                if disable_generated_reference_distance_path:
                    reference_distance_candidate_overridden_by_local_control = True
                else:
                    residual = (
                        residual_view_for_reference * (1.0 - blend) + proposed * blend
                    ).reshape(anchor.shape[0] * self.num_candidates, target_len, 2)
                    reference_distance_candidate_applied = (
                        blend.amax(dim=(1, 2, 3)) > 0
                    ).to(dtype=residual.dtype)
        if mode_route_anchor is not None and float(mode_route_anchor_weight) != 0.0:
            if (
                mode_route_anchor.ndim != 4
                or mode_route_anchor.shape[0] != anchor.shape[0]
                or mode_route_anchor.shape[-1] != 2
            ):
                raise ValueError(
                    "mode_route_anchor must be [B, K, P, 2], "
                    f"got {tuple(mode_route_anchor.shape)}"
                )
            mode_count = min(int(mode_route_anchor.shape[1]), int(self.num_candidates))
            if mode_count > 0:
                mode_route_anchor_up = self.resample_temporal(
                    mode_route_anchor[:, :mode_count].to(device=anchor.device, dtype=anchor.dtype),
                    target_len,
                )
                anchor_view = anchor.unsqueeze(1).expand(-1, mode_count, -1, -1)
                mode_route_anchor_delta = self.preserve_reference_endpoints(mode_route_anchor_up - anchor_view)
                residual_view_for_mode_anchor = residual.view(
                    anchor.shape[0],
                    self.num_candidates,
                    target_len,
                    2,
                )
                residual_view_for_mode_anchor = residual_view_for_mode_anchor.clone()
                residual_view_for_mode_anchor[:, :mode_count] = (
                    float(mode_route_residual_scale) * residual_view_for_mode_anchor[:, :mode_count]
                    + float(mode_route_anchor_weight) * mode_route_anchor_delta
                )
                residual = residual_view_for_mode_anchor.reshape(
                    anchor.shape[0] * self.num_candidates,
                    target_len,
                    2,
                )
        trajectory_flat = anchor_rep + residual
        trajectory_candidates = trajectory_flat.view(anchor.shape[0], self.num_candidates, target_len, 2)

        pooled = hidden.mean(dim=1).view(anchor.shape[0], self.num_candidates, self.hidden_dim)
        cond_pool = cond.unsqueeze(1).expand(-1, self.num_candidates, -1)
        residual_view = residual.view(anchor.shape[0], self.num_candidates, target_len, 2)
        raw_residual_view = raw_residual.view(anchor.shape[0], self.num_candidates, target_len, 2)
        gate_view = gate.view(anchor.shape[0], self.num_candidates, target_len, 1)
        scorer_trajectory_candidates = trajectory_candidates
        scorer_residual_view = residual_view
        scorer_raw_residual_view = raw_residual_view
        selector_candidate_mode = "generated"
        field_routed_control_candidates = None
        field_routed_control_active = False
        if (
            selector_side_base_preserving_curvature_counterfactual
            and prepared_base_selected_residual is not None
            and selector_side_counterfactual_delta is not None
        ):
            selector_base_residual = prepared_base_selected_residual.unsqueeze(1).expand(
                -1,
                self.num_candidates,
                -1,
                -1,
            )
            scorer_residual_view = self.preserve_reference_endpoints(
                selector_base_residual
                + selector_side_counterfactual_delta.view(
                    anchor.shape[0],
                    self.num_candidates,
                    target_len,
                    2,
                )
            )
            scorer_trajectory_candidates = anchor.unsqueeze(1) + scorer_residual_view
            scorer_raw_residual_view = torch.cat(
                [
                    (
                        scorer_residual_view * tangent.view(anchor.shape[0], 1, target_len, 2)
                    ).sum(dim=-1, keepdim=True),
                    (
                        scorer_residual_view * normal.view(anchor.shape[0], 1, target_len, 2)
                    ).sum(dim=-1, keepdim=True),
                ],
                dim=-1,
            )
            selector_candidate_mode = "base_selected_counterfactual_compose"
        if (
            bool(self.base_preserving_selector_enabled)
            and counterfactual_field_ids is not None
            and prepared_base_selected_trajectory is not None
            and self.num_candidates > 1
        ):
            base_control_source = prepared_base_selected_trajectory.unsqueeze(1).expand(
                -1,
                self.num_candidates,
                -1,
                -1,
            )
            routed_delta = torch.zeros_like(base_control_source)
            routed_any = False
            local_delta_view = (
                None
                if counterfactual_local_proposal_delta is None
                else counterfactual_local_proposal_delta.view(anchor.shape[0], self.num_candidates, target_len, 2)
            )
            shared_counterfactual_delta_view = (
                None
                if counterfactual_residual_delta is None
                else counterfactual_residual_delta.view(anchor.shape[0], self.num_candidates, target_len, 2)
            )
            edit_mask_view = (
                None
                if counterfactual_edit_mask is None
                else counterfactual_edit_mask.view(anchor.shape[0], self.num_candidates, target_len, 1)
            )
            preserve_mask_view = (
                None
                if counterfactual_preserve_mask is None
                else counterfactual_preserve_mask.view(anchor.shape[0], self.num_candidates, target_len, 1)
            )
            for batch_idx, source_kind in enumerate(field_control_source_kinds):
                if source_kind == "local_proposal" and local_delta_view is not None:
                    local_family = (
                        field_control_family_prefixes[batch_idx]
                        if batch_idx < len(field_control_family_prefixes)
                        else ""
                    ) or ""
                    base_batch = base_control_source[batch_idx : batch_idx + 1]
                    delta_batch = local_delta_view[batch_idx : batch_idx + 1]
                    edit_batch = (
                        edit_mask_view[batch_idx : batch_idx + 1]
                        if edit_mask_view is not None
                        else delta_batch.new_ones(delta_batch.shape[:-1] + (1,))
                    )
                    preserve_batch = (
                        preserve_mask_view[batch_idx : batch_idx + 1]
                        if preserve_mask_view is not None
                        else delta_batch.new_zeros(delta_batch.shape[:-1] + (1,))
                    )
                    if local_family.startswith("curvature:"):
                        composed_batch = self._compose_local_edit_on_base_path(
                            base_batch,
                            delta_batch,
                            edit_batch,
                            preserve_batch,
                            mode="curvature_direction_neutral",
                        )
                        routed_delta[batch_idx] = composed_batch[0] - base_batch[0]
                    elif "multi_turn" in local_family:
                        composed_batch = self._compose_local_edit_on_base_path(
                            base_batch,
                            delta_batch,
                            edit_batch,
                            preserve_batch,
                            mode="direction_multiturn_base_preserving",
                        )
                        routed_delta[batch_idx] = composed_batch[0] - base_batch[0]
                    elif local_family.startswith("reference_adherence:"):
                        reference_metric_compose = (
                            reference_loose_to_tight_mask is not None
                            and (
                                bool(reference_loose_to_tight_mask[batch_idx, 0, 0, 0].item())
                                or bool(reference_loose_to_moderate_mask[batch_idx, 0, 0, 0].item())
                                or bool(reference_moderate_to_tight_mask[batch_idx, 0, 0, 0].item())
                            )
                        )
                        reference_compose_mode = (
                            "reference_metric_attraction_preserving"
                            if reference_metric_compose
                            else "reference_local_direction_preserving"
                        )
                        composed_batch = self._compose_local_edit_on_base_path(
                            base_batch,
                            delta_batch,
                            edit_batch,
                            preserve_batch,
                            mode=reference_compose_mode,
                        )
                        routed_delta[batch_idx] = composed_batch[0] - base_batch[0]
                    else:
                        routed_delta[batch_idx] = local_delta_view[batch_idx]
                    routed_any = True
                elif source_kind in {"shared_direction", "shared_curvature"} and shared_counterfactual_delta_view is not None:
                    routed_delta[batch_idx] = shared_counterfactual_delta_view[batch_idx]
                    routed_any = True
            if routed_any:
                # `preserve_reference_endpoints` projects residuals/deltas. Applying it to
                # absolute coordinates removes the candidate's OD chord and detaches the
                # routed control bank from the base-selected trajectory.
                routed_delta = self.preserve_reference_endpoints(
                    routed_delta.reshape(anchor.shape[0] * self.num_candidates, target_len, 2)
                ).view(anchor.shape[0], self.num_candidates, target_len, 2)
                field_routed_control_candidates = base_control_source + routed_delta
                field_routed_control_active = True
        candidate_family_ids, candidate_base_family_mask, candidate_control_family_mask = (
            self.build_candidate_family_metadata(
                anchor.shape[0],
                counterfactual_field_ids,
                device=anchor.device,
                dtype=anchor.dtype,
                control_family_prefixes=field_control_family_prefixes,
                control_candidate_act_ids=field_control_candidate_act_ids,
            )
        )
        if (
            bool(self.base_preserving_selector_enabled)
            and counterfactual_field_ids is not None
            and self.num_candidates > 1
        ):
            if prepared_base_selected_trajectory is not None:
                base_family_candidates = prepared_base_selected_trajectory.unsqueeze(1).expand(
                    -1,
                    self.num_candidates,
                    -1,
                    -1,
                )
            else:
                base_family_candidates = trajectory_candidates
            control_family_candidates = (
                field_routed_control_candidates
                if field_routed_control_candidates is not None
                else scorer_trajectory_candidates
            )
            scorer_trajectory_candidates = (
                candidate_base_family_mask.view(anchor.shape[0], self.num_candidates, 1, 1)
                * base_family_candidates
                + candidate_control_family_mask.view(anchor.shape[0], self.num_candidates, 1, 1)
                * control_family_candidates
            )
            scorer_residual_view = scorer_trajectory_candidates - anchor.unsqueeze(1)
            scorer_raw_residual_view = torch.cat(
                [
                    (
                        scorer_residual_view * tangent.view(anchor.shape[0], 1, target_len, 2)
                    ).sum(dim=-1, keepdim=True),
                    (
                        scorer_residual_view * normal.view(anchor.shape[0], 1, target_len, 2)
                    ).sum(dim=-1, keepdim=True),
                ],
                dim=-1,
            )
            selector_candidate_mode = (
                "base_preserving_local_control_bank"
                if field_routed_control_active
                else "base_control_family_bank"
            )
        reference_control_effective_source = []
        reference_control_effective_source_detail = []
        for batch_idx in range(anchor.shape[0]):
            field_name = None
            if (
                counterfactual_field_ids is not None
                and batch_idx < counterfactual_field_ids.shape[0]
            ):
                field_idx = int(counterfactual_field_ids[batch_idx].item())
                if 0 <= field_idx < len(DEFAULT_MOBILITY_VOCAB.fields):
                    field_name = DEFAULT_MOBILITY_VOCAB.fields[field_idx]
            if field_name != "reference_adherence":
                reference_control_effective_source.append("inactive")
                reference_control_effective_source_detail.append("inactive")
                continue
            if field_routed_control_active:
                reference_control_effective_source.append("field_routed_local_proposal")
                source_kind = (
                    field_control_source_kinds[batch_idx]
                    if batch_idx < len(field_control_source_kinds)
                    else "unknown"
                )
                family = (
                    field_control_family_prefixes[batch_idx]
                    if batch_idx < len(field_control_family_prefixes)
                    else None
                )
                reference_control_effective_source_detail.append(
                    f"field_routed_local_proposal:{source_kind}:{family or 'unknown'}"
                )
            elif bool(reference_distance_candidate_applied[batch_idx].item()):
                reference_control_effective_source.append("generated_reference_distance_candidate")
                reference_control_effective_source_detail.append(
                    f"generated_reference_distance_candidate:{self.reference_distance_candidate_mode}"
                )
            elif reference_control_delta is not None:
                reference_control_effective_source.append("shared_reference_control_delta")
                reference_control_effective_source_detail.append(
                    f"shared_reference_control_delta:{self.reference_control_type}"
                )
            else:
                reference_control_effective_source.append("none")
                reference_control_effective_source_detail.append("none")
        mode_control_view = None
        if mode_tangent_scale is not None and mode_normal_scale is not None and mode_gate_bias is not None:
            mode_control_view = torch.cat([mode_tangent_scale, mode_normal_scale, mode_gate_bias], dim=-1).view(
                anchor.shape[0],
                self.num_candidates,
                target_len,
                3,
            )
        trajectory = (
            scorer_trajectory_candidates[:, 0]
            if self.num_candidates == 1
            else scorer_trajectory_candidates
        )
        residual_abs = scorer_residual_view.abs().mean(dim=(2, 3), keepdim=False).unsqueeze(-1)
        normal_abs = scorer_raw_residual_view[..., 1:2].abs().mean(dim=2)
        tangent_abs = scorer_raw_residual_view[..., 0:1].abs().mean(dim=2)
        gate_mean = gate_view.mean(dim=2)
        geometry_features = self.build_reranker_geometry_features(
            scorer_trajectory_candidates,
            anchor,
            scorer_residual_view,
            scorer_raw_residual_view,
            gate_view,
            mobility_tokens=mobility_tokens,
        )
        context_features = torch.cat([pooled, cond_pool, residual_abs, tangent_abs, normal_abs, gate_mean], dim=-1)
        pure_geometry_features = geometry_features[..., : self.reranker_pure_geometry_dim]
        token_consistency_features = geometry_features[..., self.reranker_pure_geometry_dim :]
        scorer_pure_features = pure_geometry_features
        scorer_token_features = token_consistency_features
        if self.reranker_type == "comparative_gated":
            scorer_pure_features = self.build_comparative_features(pure_geometry_features, lower_is_better=True)
            scorer_token_features = self.build_comparative_features(token_consistency_features, lower_is_better=True)
        token_gate = None
        residual_listwise_base_logits = None
        residual_listwise_delta_logits = None
        residual_listwise_alpha = None
        counterfactual_scorer_delta = None
        counterfactual_scorer_alpha = None
        counterfactual_policy_logits = None
        counterfactual_policy_gate = None
        counterfactual_metric_policy_logits = None
        counterfactual_metric_policy_feasible = None
        counterfactual_eval_selector_logits = None
        counterfactual_eval_selector_cost = None
        counterfactual_eval_selector_feasible = None
        selector_term_scores = None
        base_preservation_scores = self.build_base_preservation_scores(
            scorer_trajectory_candidates,
            anchor,
            prepared_base_selected_trajectory,
        )
        pairwise_preference_labels = None
        hybrid_policy_delta = None
        hybrid_policy_raw_delta = None
        hybrid_policy_alpha = None
        trajectory_reranker_embedding = None
        if self.reranker_type in {"split_gated", "comparative_gated"}:
            reranker_logits, token_gate = self.reranker(context_features, scorer_pure_features, scorer_token_features)
        elif self.reranker_type == "listwise_transformer":
            reranker_logits = self.reranker(context_features, scorer_pure_features, scorer_token_features)
        elif self.reranker_type == "residual_listwise":
            (
                reranker_logits,
                residual_listwise_base_logits,
                residual_listwise_delta_logits,
                residual_listwise_alpha,
            ) = self.reranker(context_features, scorer_pure_features, scorer_token_features)
        elif self.reranker_type == "trajectory_residual_listwise":
            (
                reranker_logits,
                residual_listwise_base_logits,
                residual_listwise_delta_logits,
                residual_listwise_alpha,
                trajectory_reranker_embedding,
            ) = self.reranker(
                context_features,
                scorer_pure_features,
                scorer_token_features,
                scorer_trajectory_candidates,
                anchor,
                tangent,
                normal,
            )
        else:
            score_features = torch.cat([context_features, geometry_features], dim=-1)
            reranker_logits = self.reranker(score_features).squeeze(-1)
        candidate_logits = reranker_logits + self.mode_logit_bias.view(1, -1)
        cf_score_features = torch.cat([context_features, geometry_features], dim=-1)
        counterfactual_pair_features = None
        eval_selector_enabled = float(self.counterfactual_eval_selector_weight) != 0.0
        if counterfactual_field_ids is not None and (counterfactual_mode or eval_selector_enabled):
            counterfactual_pair_features = self.build_counterfactual_pair_features(
                scorer_trajectory_candidates,
                reference_anchor=anchor,
                base_selected_trajectory=base_selected_trajectory,
                base_mobility_tokens=base_mobility_tokens,
                mobility_tokens=mobility_tokens,
                field_ids=counterfactual_field_ids,
                route_scales=route_scales,
                base_selected_index=base_selected_index,
                trip_duration_min=trip_duration_min,
            )
        if counterfactual_mode and counterfactual_field_ids is not None:
            hybrid_policy_delta, hybrid_policy_raw_delta, hybrid_policy_alpha = self.hybrid_policy_delta(
                cf_score_features,
                counterfactual_field_ids,
            )
            candidate_logits = candidate_logits + hybrid_policy_delta
        else:
            hybrid_policy_delta = candidate_logits.new_zeros(candidate_logits.shape)
            hybrid_policy_raw_delta = candidate_logits.new_zeros(candidate_logits.shape)
            hybrid_policy_alpha = self.hybrid_policy_delta.alpha.to(
                device=candidate_logits.device,
                dtype=candidate_logits.dtype,
            )
        if counterfactual_mode and counterfactual_field_ids is not None:
            counterfactual_scorer_delta, counterfactual_scorer_alpha = self.counterfactual_scorer_delta(
                cf_score_features,
                counterfactual_field_ids,
                pair_features=counterfactual_pair_features,
            )
            candidate_logits = candidate_logits + counterfactual_scorer_delta
            counterfactual_policy_logits, counterfactual_policy_gate = self.counterfactual_policy_head(
                cf_score_features,
                counterfactual_field_ids,
                pair_features=counterfactual_pair_features,
            )
            if self.counterfactual_policy_blend_mode == "override":
                candidate_logits = counterfactual_policy_logits
            else:
                candidate_logits = (
                    (1.0 - counterfactual_policy_gate) * candidate_logits
                    + counterfactual_policy_gate * counterfactual_policy_logits
                )
        if counterfactual_pair_features is not None and counterfactual_field_ids is not None:
            counterfactual_metric_policy_logits = self.counterfactual_metric_policy_head(
                counterfactual_pair_features
            )
            if counterfactual_metric_policy_logits is not None:
                target_effect_for_metric = counterfactual_pair_features[..., 0]
                collateral_for_metric = counterfactual_pair_features[..., 2]
                raw_collateral_for_metric = (
                    counterfactual_pair_features[..., 9]
                    if counterfactual_pair_features.shape[-1] > 9
                    else None
                )
                counterfactual_metric_policy_feasible = (
                    target_effect_for_metric > float(self.counterfactual_metric_policy_success_margin)
                ) & (
                    collateral_for_metric <= float(self.counterfactual_metric_policy_collateral_margin)
                )
                if raw_collateral_for_metric is not None:
                    counterfactual_metric_policy_feasible = counterfactual_metric_policy_feasible & (
                        raw_collateral_for_metric <= float(self.counterfactual_metric_policy_raw_collateral_margin)
                    )
                if bool(self.counterfactual_metric_policy_feasible_first):
                    has_feasible = counterfactual_metric_policy_feasible.any(dim=1, keepdim=True)
                    counterfactual_metric_policy_logits = torch.where(
                        has_feasible & (~counterfactual_metric_policy_feasible),
                        counterfactual_metric_policy_logits.new_tensor(-100.0),
                        counterfactual_metric_policy_logits,
                    )
            (
                counterfactual_eval_selector_logits,
                counterfactual_eval_selector_cost,
                counterfactual_eval_selector_feasible,
            ) = self.counterfactual_eval_selector_logits(counterfactual_pair_features)
            selector_term_scores = self.build_selector_term_scores(
                counterfactual_pair_features,
                base_preservation_scores,
                counterfactual_eval_selector_feasible,
            )
            pairwise_preference_labels = self.build_pairwise_preference_labels(
                counterfactual_pair_features,
                counterfactual_eval_selector_feasible,
                candidate_base_family_mask,
                candidate_control_family_mask,
            )
            if (
                counterfactual_eval_selector_logits is not None
                and eval_selector_enabled
            ):
                eval_weight = float(max(0.0, min(1.0, self.counterfactual_eval_selector_weight)))
                candidate_logits = (
                    (1.0 - eval_weight) * candidate_logits
                    + eval_weight * counterfactual_eval_selector_logits.to(
                        device=candidate_logits.device,
                        dtype=candidate_logits.dtype,
                    )
                )
            elif self.selector_noncollapse_mode == "additive":
                candidate_logits = (
                    candidate_logits
                    + selector_term_scores["target_effect"].to(
                        device=candidate_logits.device,
                        dtype=candidate_logits.dtype,
                    )
                    + selector_term_scores["collateral"].to(
                        device=candidate_logits.device,
                        dtype=candidate_logits.dtype,
                    )
                    + selector_term_scores["preserve_base"].to(
                        device=candidate_logits.device,
                        dtype=candidate_logits.dtype,
                    )
                )
            if (
                counterfactual_mode
                and counterfactual_metric_policy_logits is not None
                and float(self.counterfactual_metric_policy_weight) != 0.0
            ):
                metric_weight = float(max(0.0, min(1.0, self.counterfactual_metric_policy_weight)))
                candidate_logits = (
                    (1.0 - metric_weight) * candidate_logits
                    + metric_weight * counterfactual_metric_policy_logits.to(
                        device=candidate_logits.device,
                        dtype=candidate_logits.dtype,
                    )
                )
        candidate_edit_magnitude_vs_base = None
        candidate_reference_distance_delta = None
        candidate_reference_distance_improvement = None
        candidate_target_effect_feature = None
        candidate_collateral_feature = None
        candidate_ref_reduction_feature = None
        candidate_mode_switch_feature = None
        candidate_local_proposal_delta_magnitude = None
        control_candidate_pairwise_max_geom_diff_pre_compose = None
        control_candidate_pairwise_max_geom_diff_post_compose = None
        control_candidate_pairwise_max_geom_diff_selection = None
        reference_loose_to_tight_act_diagnostics = None
        reference_inward_act_diagnostics = None
        if prepared_base_selected_trajectory is not None:
            base_for_diag = prepared_base_selected_trajectory.unsqueeze(1).expand(
                -1,
                self.num_candidates,
                -1,
                -1,
            )
            candidate_edit_magnitude_vs_base = torch.linalg.vector_norm(
                scorer_trajectory_candidates - base_for_diag,
                dim=-1,
            ).mean(dim=-1)
        if counterfactual_pair_features is not None:
            candidate_target_effect_feature = counterfactual_pair_features[..., 0]
            candidate_collateral_feature = counterfactual_pair_features[..., 2]
            if counterfactual_pair_features.shape[-1] > 7:
                candidate_ref_reduction_feature = counterfactual_pair_features[..., 7]
            if counterfactual_pair_features.shape[-1] > 3:
                candidate_mode_switch_feature = counterfactual_pair_features[..., 3]
            if counterfactual_pair_features.shape[-1] > 6 and counterfactual_pair_features.shape[-1] > 5:
                candidate_reference_distance_delta = (
                    counterfactual_pair_features[..., 6] - counterfactual_pair_features[..., 5]
                )
                candidate_reference_distance_improvement = (
                    counterfactual_pair_features[..., 5] - counterfactual_pair_features[..., 6]
                )
        if counterfactual_local_proposal_delta is not None:
            local_delta_view_for_diag = counterfactual_local_proposal_delta.view(
                anchor.shape[0],
                self.num_candidates,
                target_len,
                2,
            )
            candidate_local_proposal_delta_magnitude = torch.linalg.vector_norm(
                local_delta_view_for_diag,
                dim=-1,
            ).mean(dim=-1)
            control_candidate_pairwise_max_geom_diff_pre_compose = torch.amax(
                torch.abs(
                    local_delta_view_for_diag.unsqueeze(2)
                    - local_delta_view_for_diag.unsqueeze(1)
                ),
                dim=(-1, -2),
            )
        if field_routed_control_candidates is not None:
            control_candidate_pairwise_max_geom_diff_post_compose = torch.amax(
                torch.abs(
                    field_routed_control_candidates.unsqueeze(2)
                    - field_routed_control_candidates.unsqueeze(1)
                ),
                dim=(-1, -2),
            )
        if scorer_trajectory_candidates is not None:
            control_candidate_pairwise_max_geom_diff_selection = torch.amax(
                torch.abs(
                    scorer_trajectory_candidates.unsqueeze(2)
                    - scorer_trajectory_candidates.unsqueeze(1)
                ),
                dim=(-1, -2),
            )
        if (
            counterfactual_local_proposal_delta is not None
            and prepared_base_selected_trajectory is not None
            and reference_loose_to_tight_mask is not None
        ):
            local_delta_view_for_diag = counterfactual_local_proposal_delta.view(
                anchor.shape[0],
                self.num_candidates,
                target_len,
                2,
            )
            base_diag = prepared_base_selected_trajectory.unsqueeze(1).expand(
                -1,
                self.num_candidates,
                -1,
                -1,
            )
            if base_diag.shape[-2] != target_len:
                base_diag = self.resample_temporal(base_diag, target_len)
            reference_diag = anchor.unsqueeze(1).expand(-1, self.num_candidates, -1, -1)
            base_ref_diag = self._mean_reference_polyline_distance(
                base_diag.reshape(anchor.shape[0] * self.num_candidates, target_len, 2),
                reference_diag.reshape(anchor.shape[0] * self.num_candidates, target_len, 2),
            ).view(anchor.shape[0], self.num_candidates)
            loose_to_tight_batch_mask = reference_loose_to_tight_mask.view(anchor.shape[0]).to(dtype=torch.bool)
            loose_to_moderate_batch_mask = reference_loose_to_moderate_mask.view(anchor.shape[0]).to(dtype=torch.bool)
            moderate_to_tight_batch_mask = reference_moderate_to_tight_mask.view(anchor.shape[0]).to(dtype=torch.bool)
            reference_inward_batch_mask = (
                loose_to_tight_batch_mask
                | loose_to_moderate_batch_mask
                | moderate_to_tight_batch_mask
            )
            if candidate_control_family_mask is not None:
                control_diag_mask = candidate_control_family_mask.to(device=anchor.device) > 0.5
            else:
                split_idx_diag = max(1, self.num_candidates // 2)
                control_diag_mask = torch.arange(
                    self.num_candidates,
                    device=anchor.device,
                ).view(1, self.num_candidates) >= split_idx_diag
                control_diag_mask = control_diag_mask.expand(anchor.shape[0], -1)
            def _candidate_reference_distance(points: torch.Tensor) -> torch.Tensor:
                if points.shape[-2] != target_len:
                    points = self.resample_temporal(points, target_len)
                return self._mean_reference_polyline_distance(
                    points.reshape(anchor.shape[0] * self.num_candidates, target_len, 2),
                    reference_diag.reshape(anchor.shape[0] * self.num_candidates, target_len, 2),
                ).view(anchor.shape[0], self.num_candidates)

            def _stage_reference_diag(
                stage_name: str,
                stage_candidates: Optional[torch.Tensor],
                valid_control_mask: torch.Tensor,
                stage_delta: Optional[torch.Tensor] = None,
            ) -> dict[str, torch.Tensor]:
                if stage_candidates is None:
                    nan_values = base_ref_diag.masked_fill(
                        torch.ones_like(valid_control_mask),
                        float("nan"),
                    )
                    return {
                        f"{stage_name}_candidate_reference_distance": nan_values,
                        f"{stage_name}_reference_distance_delta": nan_values,
                        f"{stage_name}_step_norm": nan_values,
                    }
                if stage_candidates.shape[-2] != target_len:
                    stage_candidates = self.resample_temporal(stage_candidates, target_len)
                candidate_ref = _candidate_reference_distance(stage_candidates)
                ref_delta = candidate_ref - base_ref_diag
                ref_improvement = base_ref_diag - candidate_ref
                if stage_delta is None:
                    delta_points = stage_candidates - base_diag
                else:
                    delta_points = stage_delta
                    if delta_points.shape[-2] != target_len:
                        delta_points = self.resample_temporal(delta_points, target_len)
                step_norm_values = torch.linalg.vector_norm(delta_points, dim=-1).mean(dim=-1)
                return {
                    f"{stage_name}_candidate_reference_distance": candidate_ref.masked_fill(
                        ~valid_control_mask,
                        float("nan"),
                    ),
                    f"{stage_name}_reference_distance_delta": ref_delta.masked_fill(
                        ~valid_control_mask,
                        float("nan"),
                    ),
                    f"{stage_name}_reference_distance_improvement": ref_improvement.masked_fill(
                        ~valid_control_mask,
                        float("nan"),
                    ),
                    f"{stage_name}_step_norm": step_norm_values.masked_fill(
                        ~valid_control_mask,
                        float("nan"),
                    ),
                }

            pre_compose_candidates = base_diag + local_delta_view_for_diag
            post_compose_delta = None
            if field_routed_control_candidates is not None:
                post_compose_delta = field_routed_control_candidates - base_diag
            selection_delta = None
            if scorer_trajectory_candidates is not None:
                selection_delta = scorer_trajectory_candidates - base_diag
            control_slot = (
                torch.arange(self.num_candidates, device=anchor.device).view(1, self.num_candidates)
                - max(1, self.num_candidates // 2)
            )
            reference_field_count = anchor.new_tensor(0.0)
            if counterfactual_field_ids is not None:
                reference_field_idx = DEFAULT_MOBILITY_VOCAB.fields.index("reference_adherence")
                reference_field_count = (
                    counterfactual_field_ids.to(device=anchor.device, dtype=torch.long)
                    == int(reference_field_idx)
                ).to(dtype=anchor.dtype).sum()

            def _build_reference_act_diagnostics(batch_mask: torch.Tensor) -> dict[str, torch.Tensor | None]:
                batch_diag_mask = batch_mask.view(anchor.shape[0], 1).expand_as(base_ref_diag)
                valid_control_mask = batch_diag_mask & control_diag_mask
                pre_stage_diag = _stage_reference_diag(
                    "pre_compose",
                    pre_compose_candidates,
                    valid_control_mask,
                    local_delta_view_for_diag,
                )
                post_stage_diag = _stage_reference_diag(
                    "post_compose",
                    field_routed_control_candidates,
                    valid_control_mask,
                    post_compose_delta,
                )
                selection_stage_diag = _stage_reference_diag(
                    "selection",
                    scorer_trajectory_candidates,
                    valid_control_mask,
                    selection_delta,
                )
                diagnostics = {
                    "reference_field_batch_count": reference_field_count,
                    "loose_to_tight_batch_count": loose_to_tight_batch_mask.to(dtype=anchor.dtype).sum(),
                    "loose_to_moderate_batch_count": loose_to_moderate_batch_mask.to(dtype=anchor.dtype).sum(),
                    "moderate_to_tight_batch_count": moderate_to_tight_batch_mask.to(dtype=anchor.dtype).sum(),
                    "inward_batch_count": reference_inward_batch_mask.to(dtype=anchor.dtype).sum(),
                    "control_valid_count": valid_control_mask.to(dtype=anchor.dtype).sum(),
                    "base_reference_distance": base_ref_diag.masked_fill(~valid_control_mask, float("nan")),
                    "candidate_reference_distance": pre_stage_diag[
                        "pre_compose_candidate_reference_distance"
                    ],
                    "reference_distance_delta": pre_stage_diag[
                        "pre_compose_reference_distance_delta"
                    ],
                    "reference_distance_improvement": pre_stage_diag[
                        "pre_compose_reference_distance_improvement"
                    ],
                    "step_norm": pre_stage_diag["pre_compose_step_norm"],
                    **pre_stage_diag,
                    **post_stage_diag,
                    **selection_stage_diag,
                    "target_effect": candidate_target_effect_feature.masked_fill(
                        ~valid_control_mask,
                        float("nan"),
                    )
                    if candidate_target_effect_feature is not None
                    else None,
                }
                for slot_idx in range(3):
                    slot_mask = valid_control_mask & (control_slot == slot_idx)
                    diagnostics[f"act{slot_idx}_valid_count"] = (
                        slot_mask.to(dtype=anchor.dtype).sum()
                    )
                    for stage_name in ("pre_compose", "post_compose", "selection"):
                        for value_name in (
                            "candidate_reference_distance",
                            "reference_distance_delta",
                            "reference_distance_improvement",
                            "step_norm",
                        ):
                            diag_key = f"{stage_name}_{value_name}"
                            values = diagnostics.get(diag_key)
                            if torch.is_tensor(values):
                                diagnostics[f"{stage_name}_act{slot_idx}_{value_name}"] = (
                                    values.masked_fill(~slot_mask, float("nan"))
                                )
                return diagnostics

            reference_loose_to_tight_act_diagnostics = _build_reference_act_diagnostics(
                loose_to_tight_batch_mask
            )
            reference_inward_act_diagnostics = _build_reference_act_diagnostics(
                reference_inward_batch_mask
            )
        candidate_control_feature_std = None
        if (
            candidate_control_family_mask is not None
            and candidate_target_effect_feature is not None
        ):
            control_mask_bool = candidate_control_family_mask > 0.5
            control_pair_mask = control_mask_bool.unsqueeze(2) & control_mask_bool.unsqueeze(1)

            def _masked_std(values: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                if values is None:
                    return None
                masked = values.masked_fill(~control_mask_bool, float("nan"))
                valid = torch.isfinite(masked)
                count = valid.sum(dim=1)
                safe = torch.where(valid, masked, masked.new_zeros(masked.shape))
                mean = safe.sum(dim=1) / count.clamp_min(1)
                centered = torch.where(valid, masked - mean.unsqueeze(1), masked.new_zeros(masked.shape))
                var = (centered.pow(2).sum(dim=1) / count.clamp_min(1)).to(dtype=masked.dtype)
                std = torch.sqrt(var)
                return torch.where(count > 0, std, std.new_full(std.shape, float("nan")))

            candidate_control_feature_std = {
                "target_effect": _masked_std(candidate_target_effect_feature),
                "collateral": _masked_std(candidate_collateral_feature),
                "ref_reduction": _masked_std(candidate_ref_reduction_feature),
                "edit_magnitude_vs_base": _masked_std(candidate_edit_magnitude_vs_base),
                "reference_distance_delta": _masked_std(candidate_reference_distance_delta),
                "reference_distance_improvement": _masked_std(candidate_reference_distance_improvement),
            }
            if control_candidate_pairwise_max_geom_diff_pre_compose is not None:
                control_candidate_pairwise_max_geom_diff_pre_compose = (
                    control_candidate_pairwise_max_geom_diff_pre_compose.masked_fill(
                        ~control_pair_mask,
                        float("nan"),
                    )
                )
            if control_candidate_pairwise_max_geom_diff_post_compose is not None:
                control_candidate_pairwise_max_geom_diff_post_compose = (
                    control_candidate_pairwise_max_geom_diff_post_compose.masked_fill(
                        ~control_pair_mask,
                        float("nan"),
                    )
                )
            if control_candidate_pairwise_max_geom_diff_selection is not None:
                control_candidate_pairwise_max_geom_diff_selection = (
                    control_candidate_pairwise_max_geom_diff_selection.masked_fill(
                        ~control_pair_mask,
                        float("nan"),
                    )
                )
        if mode_role_score_bias is not None:
            candidate_logits = candidate_logits + mode_role_score_bias * mode_role_budget[:, 2].to(
                device=candidate_logits.device, dtype=candidate_logits.dtype
            ).view(1, -1)
        if (
            field_control_candidate_logit_bonus is not None
            and field_control_candidate_logit_bonus.shape == candidate_logits.shape
        ):
            candidate_logits = candidate_logits + (
                candidate_control_family_mask.to(
                    device=candidate_logits.device,
                    dtype=candidate_logits.dtype,
                )
                * field_control_candidate_logit_bonus.to(
                    device=candidate_logits.device,
                    dtype=candidate_logits.dtype,
                )
            )
        if route_proposal_logits is not None and float(self.route_proposal_logit_weight) != 0.0:
            route_proposal_logits = route_proposal_logits.to(
                device=candidate_logits.device,
                dtype=candidate_logits.dtype,
            )
            route_proposal_logits = route_proposal_logits - route_proposal_logits.mean(dim=1, keepdim=True)
            candidate_logits = candidate_logits + float(self.route_proposal_logit_weight) * route_proposal_logits
        if (
            self.route_control_selection_adapter is not None
            and route_control_selection_context is not None
            and float(self.route_control_selection_adapter_weight) != 0.0
        ):
            route_control_selection_delta = self.route_control_selection_adapter(
                route_control_selection_context.to(device=candidate_logits.device, dtype=candidate_logits.dtype)
            ).squeeze(-1)
            route_control_selection_delta = (
                route_control_selection_delta - route_control_selection_delta.mean(dim=1, keepdim=True)
            )
            if float(self.route_control_selection_adapter_clip) > 0.0:
                clip = float(self.route_control_selection_adapter_clip)
                route_control_selection_delta = clip * torch.tanh(route_control_selection_delta / clip)
            route_control_selection_delta = (
                float(self.route_control_selection_adapter_weight) * route_control_selection_delta
            )
            candidate_logits = candidate_logits + route_control_selection_delta
        mode_role_prior_logits = None
        if (
            mobility_tokens is not None
            and self.num_candidates > 1
            and float(self.mode_role_prior_logit_weight) != 0.0
            and (
                self.mode_role_prior_scope == "all"
                or (self.mode_role_prior_scope == "counterfactual" and bool(counterfactual_mode))
                or (self.mode_role_prior_scope == "base" and not bool(counterfactual_mode))
            )
        ):
            mode_role_prior_logits = self.mode_role_prior_scores(
                mobility_tokens.to(device=candidate_logits.device),
                self.num_candidates,
                field_weights=self.mode_role_prior_field_weights,
            ).to(device=candidate_logits.device, dtype=candidate_logits.dtype)
            mode_role_prior_logits = mode_role_prior_logits - mode_role_prior_logits.mean(dim=1, keepdim=True)
            candidate_logits = candidate_logits + float(self.mode_role_prior_logit_weight) * mode_role_prior_logits

        if not return_aux:
            return trajectory
        return trajectory, {
            "anchor": anchor,
            "route_token_anchor": route_anchor_up,
            "mode_route_anchor": mode_route_anchor_up,
            "tangent": tangent,
            "normal": normal,
            "gate": gate_view,
            "mode_control": mode_control_view,
            "mode_role_score_bias": mode_role_score_bias,
            "mode_role_prior_logits": mode_role_prior_logits,
            "route_proposal_logits": route_proposal_logits,
            "route_control_selection_delta": route_control_selection_delta,
            "route_control_selection_adapter_weight": trajectory_candidates.new_tensor(
                float(self.route_control_selection_adapter_weight)
            ),
            "mode_skeleton_delta": (
                None
                if mode_skeleton_delta is None
                else mode_skeleton_delta.view(anchor.shape[0], self.num_candidates, target_len, 2)
            ),
            "route_skeleton": route_skeleton,
            "route_skeleton_up": route_skeleton_up,
            "route_skeleton_up_ungated": route_skeleton_up_ungated,
            "route_skeleton_gate": route_skeleton_gate.detach().view(1),
            "route_control_adapter_gate": (
                None
                if route_control_adapter_gate is None
                else route_control_adapter_gate.detach().view(1)
            ),
            "route_skeleton_delta": (
                None
                if route_skeleton_delta is None
                else route_skeleton_delta.view(anchor.shape[0], self.num_candidates, target_len, 2)
            ),
            "route_control_delta_ungated": (
                None
                if route_control_delta is None
                else route_control_delta.view(anchor.shape[0], self.num_candidates, target_len, 2)
            ),
            "route_control_delta_gated": (
                None
                if gated_route_control_delta is None
                else gated_route_control_delta.view(anchor.shape[0], self.num_candidates, target_len, 2)
            ),
            "route_control_budget": (
                None
                if route_control_budget is None
                else route_control_budget.detach().view(anchor.shape[0], 1, 1, 1)
            ),
            "route_control_budget_scale": (
                None
                if route_control_budget_scale is None
                else route_control_budget_scale.detach().view(anchor.shape[0], self.num_candidates, 1, 1)
            ),
            "route_control_delta_norm_before_budget": (
                None
                if route_control_delta_norm_before_budget is None
                else route_control_delta_norm_before_budget.detach().view(anchor.shape[0], self.num_candidates, 1, 1)
            ),
            "route_control_delta_norm_after_budget": (
                None
                if route_control_delta_norm_after_budget is None
                else route_control_delta_norm_after_budget.detach().view(anchor.shape[0], self.num_candidates, 1, 1)
            ),
            "prototype_prior_delta_ungated": (
                None
                if prototype_prior_delta is None
                else prototype_prior_delta.view(anchor.shape[0], self.num_candidates, target_len, 2)
            ),
            "prototype_prior_delta_gated": (
                None
                if gated_prototype_prior_delta is None
                else gated_prototype_prior_delta.view(anchor.shape[0], self.num_candidates, target_len, 2)
            ),
            "route_skeleton_raw": (
                None
                if route_skeleton_raw is None
                else route_skeleton_raw.view(anchor.shape[0], self.num_candidates, -1, route_skeleton_raw.shape[-1])
            ),
            "mode_role_budget": dynamic_mode_role_budget,
            "mode_role_budget_delta": mode_role_budget_delta,
            "mode_role_adapter_params": mode_role_adapter_params,
            "direction_control_delta": direction_control_delta,
            "direction_control_params": (
                None
                if direction_control_params is None
                else direction_control_params.squeeze(2)
            ),
            "curvature_control_delta": curvature_control_delta,
            "curvature_control_params": (
                None
                if curvature_control_params is None
                else curvature_control_params.squeeze(2)
            ),
            "reference_control_delta": reference_control_delta,
            "reference_control_params": (
                None
                if reference_control_params is None
                else reference_control_params.squeeze(2)
            ),
            "counterfactual_residual_delta": counterfactual_residual_delta,
            "counterfactual_residual_params": counterfactual_residual_params,
            "counterfactual_base_preserving_active": bool(base_preserving_composed_residual is not None),
            "counterfactual_selector_base_preserving_active": bool(
                selector_side_base_preserving_curvature_counterfactual
            ),
            "counterfactual_base_selected_residual": (
                None
                if base_preserving_selected_residual is None
                else base_preserving_selected_residual.view(anchor.shape[0], self.num_candidates, target_len, 2)
            ),
            "counterfactual_edit_mask": counterfactual_edit_mask,
            "counterfactual_preserve_mask": counterfactual_preserve_mask,
            "counterfactual_edit_slot_mask": counterfactual_edit_slot_mask,
            "counterfactual_local_proposal_delta": counterfactual_local_proposal_delta,
            "counterfactual_local_proposal_params": counterfactual_local_proposal_params,
            "counterfactual_local_proposal_family": counterfactual_local_proposal_family,
            "counterfactual_mask_family": counterfactual_mask_family,
            "reference_local_variant_kind": reference_local_variant_kind,
            "reference_local_variant_uses_mode_sign": reference_local_variant_uses_mode_sign,
            "reference_local_contraction_ladder": reference_local_contraction_ladder,
            "reference_local_direct_semantic_latent_used": (
                reference_local_direct_semantic_latent_used
            ),
            "reference_local_direct_act_embedding_used": (
                reference_local_direct_act_embedding_used
            ),
            "reference_local_act_condition_ids": reference_local_act_condition_ids,
            "reference_local_semantic_feature_norm": reference_local_semantic_feature_norm,
            "reference_distance_candidate_applied": reference_distance_candidate_applied,
            "reference_distance_candidate_overridden_by_local_control": (
                reference_distance_candidate_overridden_by_local_control
            ),
            "reference_control_effective_source": reference_control_effective_source,
            "reference_control_effective_source_detail": reference_control_effective_source_detail,
            "counterfactual_route_frame_budget_mode": self.counterfactual_route_frame_budget_mode,
            "counterfactual_route_frame_tangent_budget": self.counterfactual_route_frame_tangent_budget,
            "counterfactual_route_frame_normal_budget": self.counterfactual_route_frame_normal_budget,
            "counterfactual_route_frame_fields": (
                None
                if self.counterfactual_route_frame_fields is None
                else sorted(self.counterfactual_route_frame_fields)
            ),
            "reference_distance_candidate_mode": self.reference_distance_candidate_mode,
            "language_control_residual_type": self.language_control_residual_type,
            "language_control_soft_gate_min": self.language_control_soft_gate_min,
            "language_control_soft_gate_max": self.language_control_soft_gate_max,
            "language_control_field_ratios": self.language_control_field_ratios,
            "language_control_gate_fields": (
                None
                if self.language_control_gate_fields is None
                else sorted(self.language_control_gate_fields)
            ),
            "language_control_gate_scope": self.language_control_gate_scope,
            "language_control_delta_raw": language_control_delta_sum.view(
                anchor.shape[0],
                self.num_candidates,
                target_len,
                2,
            ),
            "language_control_delta": language_control_delta_final.view(
                anchor.shape[0],
                self.num_candidates,
                target_len,
                2,
            ),
            "language_control_budget_scale": language_control_budget_scale_view,
            "language_control_field_scales": language_control_field_scales,
            "language_control_field_norms": language_control_field_norms,
            "language_control_base_norm": language_control_base_norm,
            "language_control_raw_norm": language_control_raw_norm,
            "language_control_final_norm": language_control_final_norm,
            "language_control_ratio": language_control_ratio,
            "mode_specific_raw_delta": (
                None
                if mode_specific_raw_delta is None
                else mode_specific_raw_delta
            ),
            "raw_residual": raw_residual_view,
            "residual": residual_view,
            "reranker_geometry_features": geometry_features,
            "reranker_pure_geometry_features": pure_geometry_features,
            "reranker_token_features": token_consistency_features,
            "reranker_comparative_pure_features": scorer_pure_features,
            "reranker_comparative_token_features": scorer_token_features,
            "reranker_token_gate": token_gate,
            "reranker_base_logits": residual_listwise_base_logits,
            "reranker_delta_logits": residual_listwise_delta_logits,
            "reranker_residual_alpha": residual_listwise_alpha,
            "reranker_trajectory_embedding": trajectory_reranker_embedding,
            "counterfactual_scorer_delta": counterfactual_scorer_delta,
            "counterfactual_scorer_alpha": counterfactual_scorer_alpha,
            "counterfactual_policy_logits": counterfactual_policy_logits,
            "counterfactual_policy_gate": counterfactual_policy_gate,
            "counterfactual_metric_policy_logits": counterfactual_metric_policy_logits,
            "counterfactual_metric_policy_feasible": (
                None
                if counterfactual_metric_policy_feasible is None
                else counterfactual_metric_policy_feasible.to(dtype=candidate_logits.dtype)
            ),
            "counterfactual_eval_selector_logits": counterfactual_eval_selector_logits,
            "counterfactual_eval_selector_cost": counterfactual_eval_selector_cost,
            "counterfactual_eval_selector_feasible": counterfactual_eval_selector_feasible,
            "selector_term_scores": selector_term_scores,
            "base_preservation_scores": base_preservation_scores,
            "pairwise_preference_labels": pairwise_preference_labels,
            "counterfactual_pair_features": counterfactual_pair_features,
            "candidate_edit_magnitude_vs_base": candidate_edit_magnitude_vs_base,
            "candidate_reference_distance_delta": candidate_reference_distance_delta,
            "candidate_reference_distance_improvement": candidate_reference_distance_improvement,
            "candidate_target_effect_feature": candidate_target_effect_feature,
            "candidate_collateral_feature": candidate_collateral_feature,
            "candidate_ref_reduction_feature": candidate_ref_reduction_feature,
            "candidate_mode_switch_feature": candidate_mode_switch_feature,
            "candidate_local_proposal_delta_magnitude": candidate_local_proposal_delta_magnitude,
            "candidate_control_feature_std": candidate_control_feature_std,
            "control_candidate_pairwise_max_geom_diff_pre_compose": (
                control_candidate_pairwise_max_geom_diff_pre_compose
            ),
            "control_candidate_pairwise_max_geom_diff_post_compose": (
                control_candidate_pairwise_max_geom_diff_post_compose
            ),
            "control_candidate_pairwise_max_geom_diff_selection": (
                control_candidate_pairwise_max_geom_diff_selection
            ),
            "reference_loose_to_tight_act_diagnostics": reference_loose_to_tight_act_diagnostics,
            "reference_inward_act_diagnostics": reference_inward_act_diagnostics,
            "hybrid_policy_delta": hybrid_policy_delta,
            "hybrid_policy_raw_delta": hybrid_policy_raw_delta,
            "hybrid_policy_alpha": hybrid_policy_alpha,
            "selector_candidate_mode": selector_candidate_mode,
            "candidate_family_ids": candidate_family_ids,
            "candidate_base_family_mask": candidate_base_family_mask,
            "candidate_control_family_mask": candidate_control_family_mask,
            "field_control_candidate_logit_bonus": field_control_candidate_logit_bonus,
            "field_control_candidate_act_ids": field_control_candidate_act_ids,
            "field_control_proposal_scale": field_control_proposal_scale,
            "field_control_proposal_shift": field_control_proposal_shift,
            "field_control_preferred_act_ids": field_control_preferred_act_ids,
            "field_control_source_kinds": field_control_source_kinds,
            "field_routed_control_active": field_routed_control_active,
            "generated_candidates": trajectory_candidates,
            "selection_candidates": scorer_trajectory_candidates,
            "candidates": scorer_trajectory_candidates,
            "candidate_logits": candidate_logits,
            "counterfactual_adapter_scale": trajectory_candidates.new_tensor(cf_adapter_scale),
            "semantic_global_cond": semantic_global_cond,
            "local_proposal_direct_semantic_inputs": (
                "mode_embed+fused_cond+counterfactual_field_embed"
                if (
                    reference_local_direct_semantic_latent_used is None
                    or (
                        not bool(reference_local_direct_semantic_latent_used.any().item())
                        and (
                            reference_local_direct_act_embedding_used is None
                            or not bool(reference_local_direct_act_embedding_used.any().item())
                        )
                    )
                )
                else (
                    "mode_embed+fused_cond+counterfactual_field_embed"
                    "+reference_act_embedding"
                    if not bool(reference_local_direct_semantic_latent_used.any().item())
                    else (
                    "mode_embed+fused_cond+counterfactual_field_embed"
                    "+reference_field_control_latent+reference_act_embedding"
                    )
                )
            ),
            "local_proposal_direct_uses_field_control_latents": (
                False
                if reference_local_direct_semantic_latent_used is None
                else bool(reference_local_direct_semantic_latent_used.any().item())
            ),
            "local_proposal_direct_uses_field_control_logits": False,
            "field_control_latents": field_control_latents,
            "field_control_logits": field_control_logits,
        }
