from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from rag_support_scorer.data.artifact_controls import lexical_overlap

ALLOWED_SUBSETS = frozenset({"citation", "conflict", "abstention"})
PINNED_RAG_REWARDBENCH_REVISION = "6dc0e802d41a0f4421e4477a37868ca8952c6691"
_CITATION_PATTERN = re.compile(r"\[(?:\d+|[A-Za-z][^\]]*)\]")


class RewardBenchItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    subset: str
    question: str
    chosen: str
    rejected: str


class PairwiseTextScorer(Protocol):
    def score(self, question: str, response: str) -> float: ...


@dataclass(frozen=True)
class RejectionGateResult:
    scorer_accuracy: float
    length_accuracy: float
    citation_count_accuracy: float
    lexical_overlap_accuracy: float
    accepted: bool


def _pairwise_accuracy(items: Sequence[RewardBenchItem], score: PairwiseTextScorer) -> float:
    correct = sum(
        score.score(item.question, item.chosen)
        > score.score(item.question, item.rejected)
        for item in items
    )
    return correct / len(items)


@dataclass(frozen=True)
class _HeuristicScorer:
    kind: str

    def score(self, question: str, response: str) -> float:
        if self.kind == "length":
            return float(len(response.split()))
        if self.kind == "citation_count":
            return float(len(_CITATION_PATTERN.findall(response)))
        if self.kind == "lexical_overlap":
            return lexical_overlap(question, response)
        raise ValueError(f"unknown heuristic {self.kind}")


def evaluate_rejection_gate(
    items: Sequence[RewardBenchItem],
    scorer: PairwiseTextScorer,
) -> RejectionGateResult:
    if not items:
        raise ValueError("rejection gate requires benchmark items")
    invalid = {item.subset for item in items} - ALLOWED_SUBSETS
    if invalid:
        raise ValueError(f"unsupported RAG-RewardBench subsets: {sorted(invalid)}")
    scorer_accuracy = _pairwise_accuracy(items, scorer)
    length_accuracy = _pairwise_accuracy(items, _HeuristicScorer("length"))
    citation_accuracy = _pairwise_accuracy(items, _HeuristicScorer("citation_count"))
    lexical_accuracy = _pairwise_accuracy(items, _HeuristicScorer("lexical_overlap"))
    return RejectionGateResult(
        scorer_accuracy=scorer_accuracy,
        length_accuracy=length_accuracy,
        citation_count_accuracy=citation_accuracy,
        lexical_overlap_accuracy=lexical_accuracy,
        accepted=scorer_accuracy > max(length_accuracy, citation_accuracy, lexical_accuracy),
    )
