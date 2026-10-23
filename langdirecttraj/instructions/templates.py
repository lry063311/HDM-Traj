from typing import Dict


DISTANCE_TEXT = {
    "very_short": "a very short urban trip",
    "short": "a short urban trip",
    "medium": "a medium-length urban trip",
    "long": "a long urban trip",
}

DIRECTION_TEXT = {
    "straight": "mostly continue straight",
    "left_turn": "make a left-turning movement",
    "right_turn": "make a right-turning movement",
    "u_turn": "make a U-turn-like movement",
    "multi_turn": "follow a multi-turn route",
}

CURVATURE_TEXT = {
    "smooth": "with a smooth path shape",
    "moderate": "with moderate curvature",
    "winding": "with a winding path shape",
}

REFERENCE_TEXT = {
    "tight": "closely follow the provided reference route",
    "moderate": "generally follow the provided reference route",
    "loose": "allow noticeable deviation from the provided reference route",
    "unknown": "use the provided reference route as weak context",
}

PROGRESS_TEXT = {
    "slow": "with conservative progress",
    "normal": "with normal progress",
    "fast": "with aggressive progress",
}


def build_instruction(labels: Dict[str, str]) -> str:
    distance = DISTANCE_TEXT.get(labels.get("distance"), "an urban trip")
    direction = DIRECTION_TEXT.get(labels.get("direction"), "follow the route")
    curvature = CURVATURE_TEXT.get(labels.get("curvature"), "with realistic geometry")
    reference = REFERENCE_TEXT.get(labels.get("reference_adherence"), "use the reference route")
    progress = PROGRESS_TEXT.get(labels.get("progress"), "with normal progress")

    return (
        f"Generate {distance} that should {direction}, {reference}, "
        f"and move {progress} {curvature}."
    )
