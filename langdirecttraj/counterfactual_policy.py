from __future__ import annotations

from typing import Dict, Iterable, List, Set

from langdirecttraj.instructions.vocab import DEFAULT_MOBILITY_VOCAB


# Official Table-4 policy for the porto_od_chord_v1 protocol.
#
# These fields are structurally coupled under OD-chord conditioning, so they
# are still reported in raw collateral diagnostics but are excluded from the
# primary independent-collateral guardrail.
# In particular, direction/curvature edits change the path shape around the
# same OD chord, so length, pace, and chord-reference distance are not
# independent side effects in this protocol.
STRUCTURAL_COLLATERAL_EXCLUSIONS: Dict[str, Set[str]] = {
    "distance": {"progress", "reference_adherence"},
    "reference_adherence": {"distance", "progress"},
    "curvature": {"direction", "distance", "progress", "reference_adherence"},
    "direction": {"curvature", "distance", "progress", "reference_adherence"},
    "progress": {"distance"},
}


def independent_collateral_field_names(
    changed_field: str,
    fields: Iterable[str] | None = None,
) -> List[str]:
    """Return non-target fields used by Table-4 independent collateral."""

    field_list = list(fields) if fields is not None else list(DEFAULT_MOBILITY_VOCAB.fields)
    excluded = STRUCTURAL_COLLATERAL_EXCLUSIONS.get(str(changed_field), set())
    return [
        field
        for field in field_list
        if field != changed_field and field not in excluded
    ]


def serializable_collateral_exclusions() -> Dict[str, List[str]]:
    """Stable JSON representation used by diagnostics and eval outputs."""

    return {
        field: sorted(values)
        for field, values in sorted(STRUCTURAL_COLLATERAL_EXCLUSIONS.items())
    }
