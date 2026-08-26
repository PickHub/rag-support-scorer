from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rag_support_scorer.eval.metrics import BinaryMetricSummary, binary_metric_summary
from rag_support_scorer.gate.features import GateFeatures


@dataclass(frozen=True)
class GateEvaluation:
    pooled: BinaryMetricSummary
    by_condition: dict[str, BinaryMetricSummary]


class LogisticHarmGate:
    def __init__(self, *, regularization: float = 1.0, seed: int = 0) -> None:
        self.regularization = regularization
        self.seed = seed
        self.feature_names: tuple[str, ...] = ()
        self._pipeline: Pipeline | None = None

    def fit(
        self,
        examples: Sequence[GateFeatures],
        labels: Sequence[int],
        *,
        include_features: Iterable[str] | None = None,
    ) -> LogisticHarmGate:
        if len(examples) != len(labels) or len(set(labels)) < 2:
            raise ValueError("gate training requires aligned examples with both labels")
        available = examples[0].names()
        if any(example.names() != available for example in examples):
            raise ValueError("missing-feature patterns cannot be used for gate training")
        requested = tuple(include_features) if include_features is not None else available
        if not requested or not set(requested) <= set(available):
            raise ValueError("requested gate features are unavailable")
        self.feature_names = requested
        matrix = np.asarray(
            [[example.as_mapping()[name] for name in requested] for example in examples],
            dtype=np.float64,
        )
        self._pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        C=self.regularization,
                        class_weight="balanced",
                        random_state=self.seed,
                        max_iter=1000,
                    ),
                ),
            ]
        )
        self._pipeline.fit(matrix, np.asarray(labels, dtype=np.int64))
        return self

    def predict_harm_probability(self, examples: Sequence[GateFeatures]) -> tuple[float, ...]:
        if self._pipeline is None:
            raise RuntimeError("gate must be fitted before prediction")
        matrix = np.asarray(
            [[example.as_mapping()[name] for name in self.feature_names] for example in examples],
            dtype=np.float64,
        )
        return tuple(float(value) for value in self._pipeline.predict_proba(matrix)[:, 1])

    def evaluate(
        self,
        examples: Sequence[GateFeatures],
        labels: Sequence[int],
        conditions: Sequence[str],
    ) -> GateEvaluation:
        if len(examples) != len(labels) or len(labels) != len(conditions):
            raise ValueError("evaluation inputs must be aligned")
        probabilities = self.predict_harm_probability(examples)
        grouped: dict[str, list[int]] = {}
        for index, condition in enumerate(conditions):
            grouped.setdefault(condition, []).append(index)
        by_condition = {}
        for condition, indices in grouped.items():
            condition_labels = [labels[index] for index in indices]
            if len(set(condition_labels)) < 2:
                continue
            by_condition[condition] = binary_metric_summary(
                condition_labels,
                [probabilities[index] for index in indices],
            )
        return GateEvaluation(
            pooled=binary_metric_summary(labels, probabilities),
            by_condition=by_condition,
        )


def feature_ablation_sets(features: GateFeatures) -> Mapping[str, tuple[str, ...]]:
    available = features.names()
    disagreement = ("top_bundle_overlap", "rank_correlation", "margin_disagreement")
    return {
        "confidence_only": ("answer_conditioned_probability",),
        "contradiction_only": ("contradiction_score",),
        "disagreement_only": tuple(name for name in disagreement if name in available),
        "full_without_contradiction": tuple(
            name for name in available if name != "contradiction_score"
        ),
        "full": available,
    }
