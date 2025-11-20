from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from signals import Signal


@dataclass
class WeightedSignal(Signal):
    weight: float = 0.0

    def contribution(self) -> Dict[str, float]:
        return {
            "weighted_p_male": self.weight * self.p_male,
            "weighted_p_female": self.weight * self.p_female,
        }


def base_weight_for_source(source: str) -> float:
    if source == "sport":
        return 0.5
    if source == "name":
        return 0.3
    if source == "photo":
        return 0.2
    return 0.1


def quality_multiplier(quality: str) -> float:
    if quality == "high":
        return 1.0
    if quality == "medium":
        return 0.6
    return 0.2


def compute_weights(
    signals: List[Signal],
    context: Dict[str, Any] | None = None,
) -> List[WeightedSignal]:

    if context is None:
        context = {}

    weighted_signals: List[WeightedSignal] = []

    for signal in signals:
        if signal is None:
            continue

        # base + quality
        weight = base_weight_for_source(signal.source)
        weight *= quality_multiplier(signal.quality)

        # sport adjustments
        if signal.source == "sport" and context.get("suspect_team_assignment"):
            weight *= 0.3

        # NEW: group photo → almost zero weight
        if signal.source == "photo" and context.get("group_photo"):
            weight *= 0.05

        if signal.source == "photo" and context.get("low_quality_photo"):
            weight *= 0.5

        # ambiguous name reduced
        if signal.source == "name" and signal.meta and signal.meta.get("ambiguous"):
            weight *= 0.4

        weighted_signals.append(
            WeightedSignal(
                source=signal.source,
                p_male=signal.p_male,
                p_female=signal.p_female,
                quality=signal.quality,
                raw_value=signal.raw_value,
                meta=signal.meta or {},
                weight=weight,
            )
        )

    # Normalize
    total_weight = sum(ws.weight for ws in weighted_signals)
    if total_weight > 0:
        for ws in weighted_signals:
            ws.weight /= total_weight
    else:
        n = len(weighted_signals)
        if n > 0:
            for ws in weighted_signals:
                ws.weight = 1.0 / n

    return weighted_signals
