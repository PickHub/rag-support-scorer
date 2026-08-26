from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from rag_support_scorer.data.dedup import normalize_text

_ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*)*\b")


@dataclass(frozen=True)
class SurfaceFeatures:
    token_count: int
    character_count: int
    entity_count: int
    sentence_count: int
    question_overlap: float

    def as_vector(self) -> tuple[float, ...]:
        return (
            float(self.token_count),
            float(self.character_count),
            float(self.entity_count),
            float(self.sentence_count),
            self.question_overlap,
        )


def lexical_overlap(first: str, second: str) -> float:
    first_tokens = set(normalize_text(first).split())
    second_tokens = set(normalize_text(second).split())
    union = first_tokens | second_tokens
    return len(first_tokens & second_tokens) / len(union) if union else 0.0


def surface_features(text: str, *, question: str = "") -> SurfaceFeatures:
    return SurfaceFeatures(
        token_count=len(text.split()),
        character_count=len(text),
        entity_count=len(_ENTITY_PATTERN.findall(text)),
        sentence_count=sum(text.count(marker) for marker in ".!?"),
        question_overlap=lexical_overlap(text, question),
    )


def artifact_distance(first: SurfaceFeatures, second: SurfaceFeatures) -> float:
    scales = (
        max(first.token_count, second.token_count, 1),
        max(first.character_count, second.character_count, 1),
        max(first.entity_count, second.entity_count, 1),
        max(first.sentence_count, second.sentence_count, 1),
        1.0,
    )
    return sum(
        abs(left - right) / scale
        for left, right, scale in zip(first.as_vector(), second.as_vector(), scales, strict=True)
    ) / len(scales)


@dataclass(frozen=True)
class SurfaceProbeResult:
    balanced_accuracy: float
    feature_names: tuple[str, ...]


def fit_surface_probe(
    features: Sequence[SurfaceFeatures],
    labels: Sequence[int],
    *,
    regularization: float = 1.0,
) -> SurfaceProbeResult:
    if len(features) != len(labels) or len(set(labels)) < 2:
        raise ValueError("probe requires aligned features with both labels")
    matrix = np.asarray([feature.as_vector() for feature in features], dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    model = LogisticRegression(C=regularization, random_state=0, max_iter=1000)
    model.fit(matrix, target)
    predictions = model.predict(matrix)
    return SurfaceProbeResult(
        balanced_accuracy=float(balanced_accuracy_score(target, predictions)),
        feature_names=(
            "token_count",
            "character_count",
            "entity_count",
            "sentence_count",
            "question_overlap",
        ),
    )
