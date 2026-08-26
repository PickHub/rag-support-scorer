from __future__ import annotations

import pytest

from rag_support_scorer.data.bundles import enumerate_two_context_bundles
from rag_support_scorer.data.preferences import build_pairwise_examples, bundle_key
from rag_support_scorer.schemas import QuestionExample, ScorerKind
from rag_support_scorer.train.reward import validate_training_targets


def test_answer_free_bundle_preferences_have_one_positive(sample_example: QuestionExample) -> None:
    bundles = enumerate_two_context_bundles(sample_example)
    examples = build_pairwise_examples(
        question_id=sample_example.source_id,
        question=sample_example.question,
        bundles=bundles,
        scorer_kind=ScorerKind.ANSWER_FREE,
    )
    assert len(examples) == 5
    assert {example.chosen_context_ids for example in examples} == {("q1:c0", "q1:c1")}
    validate_training_targets(examples)


def test_answer_conditioned_labels_are_explicit(sample_example: QuestionExample) -> None:
    bundles = enumerate_two_context_bundles(sample_example)
    labels = {bundle_key(bundle): "q1:c3" in bundle.context_ids for bundle in bundles}
    examples = build_pairwise_examples(
        question_id=sample_example.source_id,
        question=sample_example.question,
        bundles=bundles,
        scorer_kind=ScorerKind.ANSWER_CONDITIONED,
        supplied_answer="Alan Turing",
        answer_support_labels=labels,
    )
    assert examples
    assert all(example.supplied_answer == "Alan Turing" for example in examples)


def test_answer_conditioned_labels_must_cover_pool(sample_example: QuestionExample) -> None:
    with pytest.raises(ValueError, match="cover every bundle"):
        build_pairwise_examples(
            question_id=sample_example.source_id,
            question=sample_example.question,
            bundles=enumerate_two_context_bundles(sample_example),
            scorer_kind=ScorerKind.ANSWER_CONDITIONED,
            supplied_answer="Wrong",
            answer_support_labels={},
        )


def test_pairwise_preferences_require_both_classes(sample_example: QuestionExample) -> None:
    bundles = enumerate_two_context_bundles(sample_example)
    with pytest.raises(ValueError, match="at least one positive and one negative"):
        build_pairwise_examples(
            question_id=sample_example.source_id,
            question=sample_example.question,
            bundles=bundles,
            scorer_kind=ScorerKind.ANSWER_CONDITIONED,
            supplied_answer="Alan Turing",
            answer_support_labels={bundle_key(bundle): True for bundle in bundles},
        )
