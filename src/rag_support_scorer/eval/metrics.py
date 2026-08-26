from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score

from rag_support_scorer.data.dedup import normalize_text


def exact_match(prediction: str, gold_answers: Iterable[str]) -> float:
    normalized = normalize_text(prediction)
    return float(any(normalized == normalize_text(answer) for answer in gold_answers))


def _answer_f1(prediction: str, answer: str) -> float:
    prediction_tokens = normalize_text(prediction).split()
    answer_tokens = normalize_text(answer).split()
    common = Counter(prediction_tokens) & Counter(answer_tokens)
    overlap = sum(common.values())
    if not prediction_tokens or not answer_tokens:
        return float(prediction_tokens == answer_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, gold_answers: Iterable[str]) -> float:
    return max((_answer_f1(prediction, answer) for answer in gold_answers), default=0.0)


def support_coverage_at_2(
    selected_context_ids: Sequence[str],
    gold_support_ids: Iterable[str],
) -> float:
    return float(set(gold_support_ids) <= set(selected_context_ids[:2]))


def joint_success(coverage: float, answer_score: float) -> float:
    return float(coverage == 1.0 and answer_score == 1.0)


def selective_regret(
    answer_free_score: float,
    answer_conditioned_score: float,
    selected_score: float,
) -> float:
    return max(answer_free_score, answer_conditioned_score) - selected_score


def expected_calibration_error(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must be aligned and non-empty")
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    labels_array = np.asarray(labels)
    probabilities_array = np.asarray(probabilities)
    error = 0.0
    for index in range(bins):
        upper_inclusive = index == bins - 1
        mask = (probabilities_array >= boundaries[index]) & (
            probabilities_array <= boundaries[index + 1]
            if upper_inclusive
            else probabilities_array < boundaries[index + 1]
        )
        if mask.any():
            error += float(mask.mean()) * abs(
                float(labels_array[mask].mean()) - float(probabilities_array[mask].mean())
            )
    return error


def brier_score(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    labels_array = np.asarray(labels, dtype=np.float64)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    if labels_array.shape != probabilities_array.shape or not labels:
        raise ValueError("labels and probabilities must be aligned and non-empty")
    return float(np.mean((probabilities_array - labels_array) ** 2))


def area_under_risk_coverage(
    losses: Sequence[float],
    harm_probabilities: Sequence[float],
) -> float:
    if len(losses) != len(harm_probabilities) or not losses:
        raise ValueError("losses and probabilities must be aligned and non-empty")
    order = np.argsort(np.asarray(harm_probabilities))
    ordered_losses = np.asarray(losses, dtype=np.float64)[order]
    cumulative_risk = np.cumsum(ordered_losses) / np.arange(1, len(losses) + 1)
    return float(np.mean(cumulative_risk))


def false_retain_rate(
    labels: Sequence[int],
    harm_probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
) -> float:
    harmful = [
        probability < threshold
        for label, probability in zip(labels, harm_probabilities, strict=True)
        if label
    ]
    return sum(harmful) / len(harmful) if harmful else 0.0


@dataclass(frozen=True)
class BinaryMetricSummary:
    pr_auc: float
    balanced_accuracy: float
    brier: float
    ece: float
    aurc: float
    false_retain_rate: float
    prevalence: float


def binary_metric_summary(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
) -> BinaryMetricSummary:
    if len(set(labels)) < 2:
        raise ValueError("binary evaluation requires both classes")
    predictions = [int(probability >= threshold) for probability in probabilities]
    return BinaryMetricSummary(
        pr_auc=float(average_precision_score(labels, probabilities)),
        balanced_accuracy=float(balanced_accuracy_score(labels, predictions)),
        brier=brier_score(labels, probabilities),
        ece=expected_calibration_error(labels, probabilities),
        aurc=area_under_risk_coverage(labels, probabilities),
        false_retain_rate=false_retain_rate(labels, probabilities, threshold=threshold),
        prevalence=sum(labels) / len(labels),
    )
