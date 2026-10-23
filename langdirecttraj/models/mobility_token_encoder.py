from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn

from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB, MobilityTokenVocab


class MobilityTokenEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        output_dim: int = 128,
        vocab: MobilityTokenVocab = DEFAULT_MOBILITY_VOCAB,
        dropout: float = 0.0,
        fusion: str = "concat",
    ):
        super().__init__()
        self.vocab = vocab
        self.fields = list(vocab.fields)
        self.embed_dim = int(embed_dim)
        self.output_dim = int(output_dim)
        self.fusion = str(fusion)
        self.embeddings = nn.ModuleDict(
            {field: nn.Embedding(vocab.sizes[field], self.embed_dim) for field in self.fields}
        )
        self.field_embeddings = nn.Parameter(torch.zeros(len(self.fields), self.embed_dim))
        if self.fusion == "sum":
            in_dim = self.embed_dim
        elif self.fusion == "concat":
            in_dim = self.embed_dim * len(self.fields)
        else:
            raise ValueError(f"Unsupported fusion={fusion}")
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.output_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.output_dim, self.output_dim),
        )
        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.output_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.output_dim, self.output_dim),
        )
        nn.init.normal_(self.field_embeddings, std=0.02)

    def forward(self, mobility_tokens: torch.Tensor) -> torch.Tensor:
        if mobility_tokens.ndim != 2 or mobility_tokens.shape[1] != len(self.fields):
            raise ValueError(
                f"Expected tokens with shape [B, {len(self.fields)}], got {tuple(mobility_tokens.shape)}"
            )
        pieces = []
        for i, field in enumerate(self.fields):
            pieces.append(self.embeddings[field](mobility_tokens[:, i]))
        if self.fusion == "sum":
            fused = torch.stack(pieces, dim=0).sum(dim=0)
        else:
            fused = torch.cat(pieces, dim=-1)
        return self.proj(fused)

    def encode_token_sequence(self, mobility_tokens: torch.Tensor) -> torch.Tensor:
        if mobility_tokens.ndim != 2 or mobility_tokens.shape[1] != len(self.fields):
            raise ValueError(
                f"Expected tokens with shape [B, {len(self.fields)}], got {tuple(mobility_tokens.shape)}"
            )
        pieces = []
        for i, field in enumerate(self.fields):
            token = self.embeddings[field](mobility_tokens[:, i])
            token = token + self.field_embeddings[i].view(1, -1)
            pieces.append(token)
        return self.token_proj(torch.stack(pieces, dim=1))

    def encode_labels(self, labels: Mapping[str, str], device: Optional[torch.device] = None) -> torch.Tensor:
        tokens = torch.tensor([self.vocab.encode(labels)], dtype=torch.long, device=device)
        return self.forward(tokens)
