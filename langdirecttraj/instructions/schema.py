from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class InstructionRecord:
    sample_id: int
    dataset: str
    instruction: str
    labels: Dict[str, str]
    metrics: Dict[str, float]
    trajectory: List[List[float]]
    reference: List[List[float]]

    def to_dict(self) -> Dict:
        return asdict(self)
