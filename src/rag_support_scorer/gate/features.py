from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class GateFeatures:
    answer_conditioned_probability: float
    top_bundle_overlap: float
    rank_correlation: float
    margin_disagreement: float
    contradiction_score: float
    draft_token_confidence: float | None = None
    draft_sample_agreement: float | None = None

    def names(self) -> tuple[str, ...]:
        return tuple(field.name for field in fields(self) if getattr(self, field.name) is not None)

    def as_mapping(self) -> dict[str, float]:
        return {
            field.name: float(value)
            for field in fields(self)
            if (value := getattr(self, field.name)) is not None
        }


def _ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = array[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def rank_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("score vectors must be aligned and contain at least two bundles")
    first_ranks = _ranks(first)
    second_ranks = _ranks(second)
    if np.std(first_ranks) == 0 or np.std(second_ranks) == 0:
        return 0.0
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def _top_margin(scores: Sequence[float]) -> float:
    if len(scores) < 2:
        return 0.0
    highest = sorted(scores, reverse=True)[:2]
    return highest[0] - highest[1]


def generate_gate_features(
    answer_conditioned_scores: Mapping[str, float],
    answer_free_scores: Mapping[str, float],
    *,
    calibrated_answer_conditioned_probability: float,
    contradiction_score: float,
    draft_token_confidence: float | None = None,
    draft_sample_agreement: float | None = None,
) -> GateFeatures:
    if answer_conditioned_scores.keys() != answer_free_scores.keys():
        raise ValueError("matched scorers must score identical bundles")
    bundle_ids = sorted(answer_conditioned_scores)
    conditioned = [answer_conditioned_scores[bundle_id] for bundle_id in bundle_ids]
    answer_free = [answer_free_scores[bundle_id] for bundle_id in bundle_ids]
    top_conditioned = max(
        bundle_ids,
        key=lambda bundle_id: (answer_conditioned_scores[bundle_id], bundle_id),
    )
    top_answer_free = max(
        bundle_ids,
        key=lambda bundle_id: (answer_free_scores[bundle_id], bundle_id),
    )
    return GateFeatures(
        answer_conditioned_probability=calibrated_answer_conditioned_probability,
        top_bundle_overlap=float(top_conditioned == top_answer_free),
        rank_correlation=rank_correlation(conditioned, answer_free),
        margin_disagreement=abs(_top_margin(conditioned) - _top_margin(answer_free)),
        contradiction_score=contradiction_score,
        draft_token_confidence=draft_token_confidence,
        draft_sample_agreement=draft_sample_agreement,
    )
