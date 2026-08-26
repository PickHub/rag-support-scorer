from __future__ import annotations

from dataclasses import dataclass

from rag_support_scorer.eval.calibration import (
    IsotonicCalibrator,
    PlattScaler,
    TemperatureScaler,
)
from rag_support_scorer.eval.rewardbench import RewardBenchItem, evaluate_rejection_gate


def test_scorer_calibrators_return_probabilities() -> None:
    scores = [-3.0, -1.0, 1.0, 3.0]
    labels = [0, 0, 1, 1]
    temperature = TemperatureScaler().fit(scores, labels)
    isotonic = IsotonicCalibrator().fit(scores, labels)
    platt = PlattScaler().fit(scores, labels)
    assert temperature.predict(scores) == tuple(sorted(temperature.predict(scores)))
    assert isotonic.predict(scores) == (0.0, 0.0, 1.0, 1.0)
    assert platt.predict(scores) == tuple(sorted(platt.predict(scores)))
    assert platt.bias != 0.0 or platt.scale != 1.0


@dataclass(frozen=True)
class _KnownScorer:
    def score(self, question: str, response: str) -> float:
        return float(response == "supported")


def test_optional_rewardbench_is_rejection_only() -> None:
    items = [
        RewardBenchItem(
            source_id="one",
            subset="conflict",
            question="Which evidence is reliable?",
            chosen="supported",
            rejected="a much longer unsupported response [1]",
        ),
        RewardBenchItem(
            source_id="two",
            subset="abstention",
            question="Can this be answered?",
            chosen="supported",
            rejected="unsupported [2]",
        ),
    ]
    result = evaluate_rejection_gate(items, _KnownScorer())
    assert result.scorer_accuracy == 1.0
    assert result.accepted
