from __future__ import annotations

from dataclasses import fields

import pytest

from rag_support_scorer.gate.features import (
    GateFeatures,
    generate_gate_features,
    rank_correlation,
)
from rag_support_scorer.gate.logistic import LogisticHarmGate, feature_ablation_sets


def test_gate_features_use_only_inference_visible_values() -> None:
    feature_names = {field.name for field in fields(GateFeatures)}
    assert "correct" not in " ".join(feature_names)
    features = generate_gate_features(
        {"a": 0.9, "b": 0.2, "c": 0.1},
        {"a": 0.1, "b": 0.8, "c": 0.2},
        calibrated_answer_conditioned_probability=0.7,
        contradiction_score=0.4,
    )
    assert features.top_bundle_overlap == 0.0
    assert features.margin_disagreement == pytest.approx(0.1)
    assert "draft_token_confidence" not in features.names()


def test_regularized_gate_and_feature_ablations() -> None:
    examples = [
        GateFeatures(0.9, 1.0, 0.9, 0.1, 0.1),
        GateFeatures(0.8, 1.0, 0.8, 0.2, 0.2),
        GateFeatures(0.7, 1.0, 0.7, 0.2, 0.2),
        GateFeatures(0.6, 1.0, 0.6, 0.3, 0.3),
        GateFeatures(0.4, 0.0, -0.2, 0.8, 0.8),
        GateFeatures(0.3, 0.0, -0.4, 0.9, 0.9),
        GateFeatures(0.2, 0.0, -0.6, 1.0, 0.9),
        GateFeatures(0.1, 0.0, -0.8, 1.1, 1.0),
    ]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    gate = LogisticHarmGate().fit(examples, labels)
    probabilities = gate.predict_harm_probability(examples)
    assert max(probabilities[:4]) < min(probabilities[4:])
    evaluation = gate.evaluate(
        examples,
        labels,
        ["correct", "correct", "wrong", "wrong", "correct", "correct", "wrong", "wrong"],
    )
    assert evaluation.pooled.pr_auc == 1.0
    assert set(feature_ablation_sets(examples[0])) == {
        "confidence_only",
        "contradiction_only",
        "disagreement_only",
        "full_without_contradiction",
        "full",
    }


def test_missing_feature_patterns_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing-feature"):
        LogisticHarmGate().fit(
            [
                GateFeatures(0.5, 0.0, 0.0, 0.0, 0.0),
                GateFeatures(0.5, 0.0, 0.0, 0.0, 0.0, draft_token_confidence=0.5),
            ],
            [0, 1],
        )


def test_rank_correlation_treats_constant_tied_scores_as_uninformative() -> None:
    assert rank_correlation([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]) == 0.0
