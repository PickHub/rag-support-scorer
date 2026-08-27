from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from rag_support_scorer.data.dedup import normalize_text
from rag_support_scorer.schemas import ContextDocument, QuestionExample, TwoContextBundle


@dataclass(frozen=True)
class RankedBundle:
    bundle: TwoContextBundle
    score: float


class BundleRanker(Protocol):
    name: str

    def rank(
        self,
        example: QuestionExample,
        bundles: Sequence[TwoContextBundle],
        supplied_answer: str | None,
    ) -> tuple[RankedBundle, ...]: ...


class BundleScorerAdapter(Protocol):
    def score(
        self,
        question: str,
        contexts: tuple[ContextDocument, ContextDocument],
        supplied_answer: str | None,
    ) -> float: ...


class NLIAdapter(Protocol):
    def entailment_score(self, premise: str, hypothesis: str) -> float: ...


def _ordered(
    scored: Sequence[RankedBundle],
) -> tuple[RankedBundle, ...]:
    return tuple(
        sorted(
            scored,
            key=lambda item: (-item.score, item.bundle.context_ids),
        )
    )


def _contexts_by_id(example: QuestionExample) -> dict[str, ContextDocument]:
    return {context.source_id: context for context in example.contexts}


class DatasetOrderRanker:
    name = "dataset_order"

    def rank(
        self,
        example: QuestionExample,
        bundles: Sequence[TwoContextBundle],
        supplied_answer: str | None,
    ) -> tuple[RankedBundle, ...]:
        positions = {context.source_id: context.position for context in example.contexts}
        return _ordered(
            [
                RankedBundle(
                    bundle,
                    -sum(positions[context_id] for context_id in bundle.context_ids),
                )
                for bundle in bundles
            ]
        )


class RandomRanker:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def rank(
        self,
        example: QuestionExample,
        bundles: Sequence[TwoContextBundle],
        supplied_answer: str | None,
    ) -> tuple[RankedBundle, ...]:
        scored = []
        for bundle in bundles:
            key = "\0".join((str(self.seed), example.source_id, *bundle.context_ids))
            score = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
            scored.append(RankedBundle(bundle, score))
        return _ordered(scored)


def _bm25_like_score(query: str, documents: Sequence[str]) -> list[float]:
    query_terms = normalize_text(query).split()
    tokenized = [normalize_text(document).split() for document in documents]
    average_length = sum(map(len, tokenized)) / max(len(tokenized), 1)
    document_frequency = Counter(
        term for terms in tokenized for term in set(terms)
    )
    scores = []
    for terms in tokenized:
        frequencies = Counter(terms)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            inverse_frequency = math.log(
                1
                + (len(tokenized) - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            denominator = frequency + 1.2 * (
                0.25 + 0.75 * len(terms) / max(average_length, 1.0)
            )
            score += inverse_frequency * frequency * 2.2 / denominator
        scores.append(score)
    return scores


class LexicalRanker:
    name = "lexical_question_only"

    def rank(
        self,
        example: QuestionExample,
        bundles: Sequence[TwoContextBundle],
        supplied_answer: str | None,
    ) -> tuple[RankedBundle, ...]:
        contexts = _contexts_by_id(example)
        query = example.question
        texts = [
            " ".join(contexts[context_id].text for context_id in bundle.context_ids)
            for bundle in bundles
        ]
        return _ordered(
            [
                RankedBundle(bundle, score)
                for bundle, score in zip(bundles, _bm25_like_score(query, texts), strict=True)
            ]
        )


class AdapterRanker:
    def __init__(
        self,
        name: str,
        adapter: BundleScorerAdapter,
        *,
        answer_conditioned: bool,
    ) -> None:
        self.name = name
        self.adapter = adapter
        self.answer_conditioned = answer_conditioned

    def rank(
        self,
        example: QuestionExample,
        bundles: Sequence[TwoContextBundle],
        supplied_answer: str | None,
    ) -> tuple[RankedBundle, ...]:
        if self.answer_conditioned and supplied_answer is None:
            raise ValueError("answer-conditioned ranker requires a supplied answer")
        contexts = _contexts_by_id(example)
        context_bundles = tuple(
            (contexts[bundle.context_ids[0]], contexts[bundle.context_ids[1]])
            for bundle in bundles
        )
        score_many = getattr(self.adapter, "score_many", None)
        if callable(score_many):
            scores = score_many(
                example.question,
                context_bundles,
                supplied_answer if self.answer_conditioned else None,
            )
            return _ordered(
                [
                    RankedBundle(bundle, score)
                    for bundle, score in zip(bundles, scores, strict=True)
                ]
            )
        return _ordered(
            [
                RankedBundle(
                    bundle,
                    self.adapter.score(
                        example.question,
                        context_bundle,
                        supplied_answer if self.answer_conditioned else None,
                    ),
                )
                for bundle, context_bundle in zip(
                    bundles,
                    context_bundles,
                    strict=True,
                )
            ]
        )


class AnswerShuffledRanker(AdapterRanker):
    def __init__(self, adapter: BundleScorerAdapter, shuffled_answers: dict[str, str]) -> None:
        super().__init__("answer_shuffled", adapter, answer_conditioned=True)
        self.shuffled_answers = shuffled_answers

    def rank(
        self,
        example: QuestionExample,
        bundles: Sequence[TwoContextBundle],
        supplied_answer: str | None,
    ) -> tuple[RankedBundle, ...]:
        try:
            shuffled = self.shuffled_answers[example.source_id]
        except KeyError as error:
            raise ValueError(f"no shuffled answer for {example.source_id}") from error
        return super().rank(example, bundles, shuffled)


class OracleSupportRanker:
    name = "oracle_support"

    def rank(
        self,
        example: QuestionExample,
        bundles: Sequence[TwoContextBundle],
        supplied_answer: str | None,
    ) -> tuple[RankedBundle, ...]:
        return _ordered(
            [RankedBundle(bundle, float(bundle.contains_all_gold_support)) for bundle in bundles]
        )


class NLIRanker:
    name = "nli"

    def __init__(self, adapter: NLIAdapter) -> None:
        self.adapter = adapter

    def rank(
        self,
        example: QuestionExample,
        bundles: Sequence[TwoContextBundle],
        supplied_answer: str | None,
    ) -> tuple[RankedBundle, ...]:
        if supplied_answer is None:
            raise ValueError("NLI ranking requires a supplied answer")
        contexts = _contexts_by_id(example)
        return _ordered(
            [
                RankedBundle(
                    bundle,
                    self.adapter.entailment_score(
                        " ".join(contexts[context_id].text for context_id in bundle.context_ids),
                        supplied_answer,
                    ),
                )
                for bundle in bundles
            ]
        )
