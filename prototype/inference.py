# inference.py

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List

from signals import get_name_signal, get_sport_signal, get_photo_signal, Signal
from weights import compute_weights, WeightedSignal


@dataclass
class InferenceConfig:
    """Config for inference behavior."""

    min_confidence: float = 0.6
    unknown_label: str = "Unknown"
    male_label: str = "Male"
    female_label: str = "Female"
    explicit_gender_field: str = "gender"          # input key for explicit gender
    explicit_gender_values_male: tuple = ("male", "m")
    explicit_gender_values_female: tuple = ("female", "f")


@dataclass
class InferenceResult:
    inferred_gender: str
    confidence: float
    p_male: float
    p_female: float
    attribution: List[Dict[str, Any]]
    skipped_due_to_explicit_gender: bool = False
    explicit_gender_value: Optional[str] = None


def has_explicit_gender(profile: Dict[str, Any], config: InferenceConfig) -> Optional[str]:
    """Return the normalized explicit gender if present, else None."""
    value = profile.get(config.explicit_gender_field)
    if value is None:
        return None

    v = str(value).strip().lower()
    if v in config.explicit_gender_values_male:
        return config.male_label
    if v in config.explicit_gender_values_female:
        return config.female_label

    # treat non binary or other as an explicit state where inference should not run
    return profile.get(config.explicit_gender_field)


def infer_gender(
    profile: Dict[str, Any],
    config: Optional[InferenceConfig] = None,
    photo_model: Any = None,
) -> InferenceResult:
    """
    Main inference pipeline.

    profile keys (for prototype):
      "first_name"
      "sport_gender"
      "photo_path"
      "gender" (explicit, optional)
    """

    if config is None:
        config = InferenceConfig()

    # 1. Respect explicit gender and non binary states
    explicit = has_explicit_gender(profile, config)
    if explicit is not None:
        return InferenceResult(
            inferred_gender=explicit,
            confidence=1.0,
            p_male=1.0 if explicit == config.male_label else 0.0,
            p_female=1.0 if explicit == config.female_label else 0.0,
            attribution=[],
            skipped_due_to_explicit_gender=True,
            explicit_gender_value=str(explicit),
        )

    # 2. Build raw signals
    first_name = profile.get("first_name")
    sport_gender = profile.get("sport_gender")
    photo_path = profile.get("photo_path")

    name_signal: Signal = get_name_signal(first_name)
    sport_signal: Signal = get_sport_signal(sport_gender)

    # context flags for weighting
    context: Dict[str, Any] = {}

    # simple heuristics for demo, you can pass real flags from upstream
    if sport_gender and str(sport_gender).lower() == "unknown":
        context["suspect_team_assignment"] = True

    # real system would pass in group photo detection etc
    context["group_photo"] = profile.get("group_photo", False)
    context["low_quality_photo"] = profile.get("low_quality_photo", False)

    photo_signal: Signal = get_photo_signal(photo_path, photo_model=photo_model)

    raw_signals = [name_signal, sport_signal, photo_signal]

    # 3. Compute weights
    weighted_signals: List[WeightedSignal] = compute_weights(raw_signals, context=context)

    # 4. Compute final probabilities
    p_male = sum(ws.weight * ws.p_male for ws in weighted_signals)
    p_female = sum(ws.weight * ws.p_female for ws in weighted_signals)

    # normalize final probabilities
    total = p_male + p_female
    if total <= 0:
        p_male = 0.5
        p_female = 0.5
    else:
        p_male /= total
        p_female /= total

    confidence = max(p_male, p_female)

    if confidence < config.min_confidence:
        inferred = config.unknown_label
    else:
        inferred = config.male_label if p_male > p_female else config.female_label

    # 5. Attribution
    attribution = []
    for ws in weighted_signals:
        contrib = ws.contribution()
        attribution.append(
            {
                "source": ws.source,
                "weight": ws.weight,
                "p_male": ws.p_male,
                "p_female": ws.p_female,
                "weighted_p_male": contrib["weighted_p_male"],
                "weighted_p_female": contrib["weighted_p_female"],
                "quality": ws.quality,
                "meta": ws.meta,
            }
        )

    return InferenceResult(
        inferred_gender=inferred,
        confidence=confidence,
        p_male=p_male,
        p_female=p_female,
        attribution=attribution,
        skipped_due_to_explicit_gender=False,
        explicit_gender_value=None,
    )


if __name__ == "__main__":
    # tiny manual test
    example_profile = {
        "first_name": "Alex",
        "sport_gender": "Male",
        "photo_path": "path/to/photo.jpg",
        "gender": None,
    }
    result = infer_gender(example_profile)
    print(asdict(result))
