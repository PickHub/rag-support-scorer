from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_support_scorer.schemas import (
    AnswerCondition,
    PairwiseRewardExample,
    ScorerKind,
    SuppliedAnswer,
    TwoContextBundle,
)


def test_two_context_bundle_rejects_duplicates() -> None:
    with pytest.raises(ValidationError):
        TwoContextBundle(
            question_id="q",
            context_ids=("c", "c"),
            contains_all_gold_support=False,
        )


def test_answer_presence_matches_condition() -> None:
    assert SuppliedAnswer(condition=AnswerCondition.ABSENT).text is None
    with pytest.raises(ValidationError):
        SuppliedAnswer(condition=AnswerCondition.CORRECT)


def test_reward_targets_are_separate() -> None:
    answer_free = PairwiseRewardExample(
        example_id="af",
        question_id="q",
        scorer_kind=ScorerKind.ANSWER_FREE,
        question="Question?",
        chosen_context_ids=("c0", "c1"),
        rejected_context_ids=("c0", "c2"),
    )
    assert answer_free.supplied_answer is None
    with pytest.raises(ValidationError):
        PairwiseRewardExample(
            example_id="bad",
            question_id="q",
            scorer_kind=ScorerKind.ANSWER_CONDITIONED,
            question="Question?",
            chosen_context_ids=("c0", "c1"),
            rejected_context_ids=("c0", "c2"),
        )
