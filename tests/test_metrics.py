from __future__ import annotations

import pytest

from rag_support_scorer.eval.metrics import (
    area_under_risk_coverage,
    binary_metric_summary,
    exact_match,
    selective_regret,
    support_coverage_at_2,
    token_f1,
)


def test_deterministic_qa_metrics() -> None:
    assert exact_match("The Charles Babbage", ["Charles Babbage"]) == 1.0
    assert token_f1("Charles", ["Charles Babbage"]) == pytest.approx(2 / 3)
    assert support_coverage_at_2(["c0", "c1"], {"c0", "c1"}) == 1.0
    assert support_coverage_at_2(["c0", "c2"], {"c0", "c1"}) == 0.0
    assert selective_regret(0.8, 0.5, 0.5) == pytest.approx(0.3)


def test_binary_gate_metrics() -> None:
    summary = binary_metric_summary(
        [0, 0, 1, 1],
        [0.05, 0.2, 0.8, 0.95],
    )
    assert summary.pr_auc == 1.0
    assert summary.balanced_accuracy == 1.0
    assert summary.false_retain_rate == 0.0
    assert summary.prevalence == 0.5


def test_aurc_includes_the_first_coverage_interval() -> None:
    assert area_under_risk_coverage([1.0] * 4, [0.1, 0.2, 0.3, 0.4]) == 1.0
