from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator(Protocol):
    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> ProbabilityCalibrator: ...

    def predict(self, scores: Sequence[float]) -> tuple[float, ...]: ...


def _sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> TemperatureScaler:
        if len(scores) != len(labels) or len(set(labels)) < 2:
            raise ValueError("temperature scaling requires aligned scores with both labels")
        logits = np.asarray(scores, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.float64)
        candidates = np.geomspace(0.05, 20.0, 500)
        losses = []
        for candidate in candidates:
            probabilities = np.clip(_sigmoid(logits / candidate), 1e-12, 1 - 1e-12)
            loss = -np.mean(
                targets * np.log(probabilities) + (1 - targets) * np.log(1 - probabilities)
            )
            losses.append(float(loss))
        self.temperature = float(candidates[int(np.argmin(losses))])
        return self

    def predict(self, scores: Sequence[float]) -> tuple[float, ...]:
        probabilities = _sigmoid(np.asarray(scores, dtype=np.float64) / self.temperature)
        return tuple(float(probability) for probability in probabilities)


class IsotonicCalibrator:
    def __init__(self) -> None:
        self._model = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> IsotonicCalibrator:
        if len(scores) != len(labels) or len(set(labels)) < 2:
            raise ValueError("isotonic calibration requires aligned scores with both labels")
        self._model.fit(scores, labels)
        self._fitted = True
        return self

    def predict(self, scores: Sequence[float]) -> tuple[float, ...]:
        if not self._fitted:
            raise RuntimeError("calibrator must be fitted before prediction")
        return tuple(float(probability) for probability in self._model.predict(scores))


@dataclass
class PlattScaler:
    scale: float = 1.0
    bias: float = 0.0
    _model: LogisticRegression | None = None

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> PlattScaler:
        if len(scores) != len(labels) or len(set(labels)) < 2:
            raise ValueError("Platt scaling requires aligned scores with both labels")
        model = LogisticRegression(C=1_000_000, solver="lbfgs")
        model.fit(np.asarray(scores, dtype=np.float64).reshape(-1, 1), labels)
        self.scale = float(model.coef_[0, 0])
        self.bias = float(model.intercept_[0])
        self._model = model
        return self

    def predict(self, scores: Sequence[float]) -> tuple[float, ...]:
        logits = self.scale * np.asarray(scores, dtype=np.float64) + self.bias
        return tuple(float(probability) for probability in _sigmoid(logits))
