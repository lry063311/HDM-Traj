from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn

from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB, MobilityTokenVocab


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_'-]+")


def tokenize_instruction(text: str) -> List[str]:
    return [piece.lower() for piece in TOKEN_PATTERN.findall(str(text))]


def rule_token_overrides(
    text: str,
    vocab: MobilityTokenVocab = DEFAULT_MOBILITY_VOCAB,
) -> Dict[str, int]:
    lowered = f" {str(text).lower().replace('-', ' ')} "
    overrides: Dict[str, int] = {}

    def set_field(field: str, value: str) -> None:
        overrides[field] = vocab.token_to_id[field][value]

    if any(phrase in lowered for phrase in [" turn right", " turns right", " right turn", "right turning", " bend right", " bends right"]):
        set_field("direction", "right_turn")
    elif any(phrase in lowered for phrase in [" turn left", " turns left", " left turn", "left turning", " bend left", " bends left"]):
        set_field("direction", "left_turn")
    elif any(phrase in lowered for phrase in [" u turn", "u turn", "double back", "doubles back", "reverse direction", "reverses direction"]):
        set_field("direction", "u_turn")
    elif any(phrase in lowered for phrase in ["multi turn", "multiple turns", "several turns", "more than one major turn"]):
        set_field("direction", "multi_turn")
    elif any(phrase in lowered for phrase in [" mostly straight", " go straight", " continue straight", " no major turn", "almost straight"]):
        set_field("direction", "straight")

    if any(phrase in lowered for phrase in [" loose reference", "loosely follow", "looser route", "noticeable deviation", "deviate from the reference"]):
        set_field("reference_adherence", "loose")
    elif any(phrase in lowered for phrase in [" moderate reference", "roughly follow", "small deviations"]):
        set_field("reference_adherence", "moderate")
    elif any(phrase in lowered for phrase in [" close to the reference", "closely follow", "tight reference", "stay close", "stays close"]):
        set_field("reference_adherence", "tight")

    if any(phrase in lowered for phrase in [" winding", " curvy", " high curvature"]):
        set_field("curvature", "winding")
    elif any(phrase in lowered for phrase in [" smooth", "gently curved", "low curvature"]):
        set_field("curvature", "smooth")
    elif "moderate curvature" in lowered:
        set_field("curvature", "moderate")

    if any(phrase in lowered for phrase in [" fast progress", "aggressive progress", "more direct progress"]):
        set_field("progress", "fast")
    elif any(phrase in lowered for phrase in [" slow progress", "conservative progress", "less direct progress"]):
        set_field("progress", "slow")
    elif any(phrase in lowered for phrase in [" normal progress", "typical progress", "balanced progress"]):
        set_field("progress", "normal")

    if any(phrase in lowered for phrase in [" very short", " nearby", " brief"]):
        set_field("distance", "very_short")
    elif any(phrase in lowered for phrase in [" short route", " short trip", " short trajectory", " local route", " compact"]):
        set_field("distance", "short")
    elif any(phrase in lowered for phrase in [" long route", " long trip", " long trajectory", "long distance", " extended"]):
        set_field("distance", "long")
    elif any(phrase in lowered for phrase in [" medium", "moderate distance", "mid range"]):
        set_field("distance", "medium")

    return overrides


def rule_override_tokens(text: str, token_ids: List[int], vocab: MobilityTokenVocab = DEFAULT_MOBILITY_VOCAB) -> List[int]:
    ids = list(token_ids)
    for field, token_id in rule_token_overrides(text, vocab).items():
        ids[vocab.fields.index(field)] = token_id
    return ids


@dataclass
class TextVocab:
    token_to_id: Dict[str, int]
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        min_freq: int = 1,
        max_size: int = 20000,
    ) -> "TextVocab":
        counter: Counter = Counter()
        for text in texts:
            counter.update(tokenize_instruction(text))
        token_to_id = {"<pad>": 0, "<unk>": 1}
        for token, count in counter.most_common():
            if count < min_freq:
                continue
            if token in token_to_id:
                continue
            token_to_id[token] = len(token_to_id)
            if len(token_to_id) >= max_size:
                break
        return cls(token_to_id=token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    def encode(self, text: str, max_len: int) -> List[int]:
        tokens = tokenize_instruction(text)[:max_len]
        ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]
        if len(ids) < max_len:
            ids.extend([self.pad_id] * (max_len - len(ids)))
        return ids

    def to_dict(self) -> Dict:
        return {
            "token_to_id": self.token_to_id,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "TextVocab":
        return cls(
            token_to_id={str(k): int(v) for k, v in payload["token_to_id"].items()},
            pad_token=str(payload.get("pad_token", "<pad>")),
            unk_token=str(payload.get("unk_token", "<unk>")),
        )


class TextInstructionEncoder(nn.Module):
    """Lightweight local text encoder for text-to-mobility-token planning."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = int(pad_id)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self.pad_id)
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [B, L], got {tuple(input_ids.shape)}")
        mask = input_ids != self.pad_id
        embedded = self.embedding(input_ids)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(embedded.dtype)
        pooled = (embedded * mask.unsqueeze(-1).to(embedded.dtype)).sum(dim=1) / denom
        return self.proj(pooled)


class TextToMobilityTokenPlanner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        text_dim: int = 128,
        dropout: float = 0.1,
        pad_id: int = 0,
        mobility_vocab: MobilityTokenVocab = DEFAULT_MOBILITY_VOCAB,
    ):
        super().__init__()
        self.mobility_vocab = mobility_vocab
        self.fields = list(mobility_vocab.fields)
        self.encoder = TextInstructionEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=text_dim,
            dropout=dropout,
            pad_id=pad_id,
        )
        self.heads = nn.ModuleDict(
            {field: nn.Linear(text_dim, mobility_vocab.sizes[field]) for field in self.fields}
        )

    def forward(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        text_features = self.encoder(input_ids)
        return {field: self.heads[field](text_features) for field in self.fields}

    def predict_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_ids)
        return torch.stack([logits[field].argmax(dim=-1) for field in self.fields], dim=1)


def batch_encode_texts(texts: Sequence[str], vocab: TextVocab, max_len: int) -> torch.Tensor:
    return torch.tensor([vocab.encode(text, max_len=max_len) for text in texts], dtype=torch.long)
