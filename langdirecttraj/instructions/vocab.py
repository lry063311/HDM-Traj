from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping


LABEL_VOCAB: Dict[str, List[str]] = {
    "distance": ["very_short", "short", "medium", "long"],
    "direction": ["straight", "left_turn", "right_turn", "u_turn", "multi_turn"],
    "curvature": ["smooth", "moderate", "winding"],
    "progress": ["slow", "normal", "fast"],
    "reference_adherence": ["unknown", "tight", "moderate", "loose"],
}


@dataclass(frozen=True)
class MobilityTokenVocab:
    fields: List[str]
    token_to_id: Dict[str, Dict[str, int]]
    id_to_token: Dict[str, Dict[int, str]]

    @classmethod
    def default(cls) -> "MobilityTokenVocab":
        token_to_id = {field: {token: i for i, token in enumerate(tokens)} for field, tokens in LABEL_VOCAB.items()}
        id_to_token = {field: {i: token for token, i in mapping.items()} for field, mapping in token_to_id.items()}
        return cls(fields=list(LABEL_VOCAB.keys()), token_to_id=token_to_id, id_to_token=id_to_token)

    @property
    def sizes(self) -> Dict[str, int]:
        return {field: len(self.token_to_id[field]) for field in self.fields}

    def encode(self, labels: Mapping[str, str]) -> List[int]:
        ids = []
        for field in self.fields:
            value = str(labels.get(field, "unknown"))
            mapping = self.token_to_id[field]
            if value not in mapping:
                fallback = "unknown" if "unknown" in mapping else next(iter(mapping))
                value = fallback
            ids.append(mapping[value])
        return ids

    def decode(self, ids: Iterable[int]) -> Dict[str, str]:
        result = {}
        for field, token_id in zip(self.fields, ids):
            mapping = self.id_to_token[field]
            result[field] = mapping.get(int(token_id), "unknown")
        return result


DEFAULT_MOBILITY_VOCAB = MobilityTokenVocab.default()
