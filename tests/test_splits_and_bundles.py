from __future__ import annotations

from rag_support_scorer.data.bundles import bundle_eligibility, enumerate_two_context_bundles
from rag_support_scorer.data.splits import assign_split
from rag_support_scorer.schemas import ContextDocument, QuestionExample


def test_hashed_split_is_stable_and_salt_sensitive() -> None:
    first = [assign_split(f"q{index}", salt="first") for index in range(100)]
    assert first == [assign_split(f"q{index}", salt="first") for index in range(100)]
    assert first != [assign_split(f"q{index}", salt="second") for index in range(100)]


def test_bundle_enumeration_and_gold_labels(sample_example: QuestionExample) -> None:
    bundles = enumerate_two_context_bundles(sample_example)
    assert len(bundles) == 6
    positives = [bundle for bundle in bundles if bundle.contains_all_gold_support]
    assert [bundle.context_ids for bundle in positives] == [("q1:c0", "q1:c1")]


def test_questions_requiring_three_contexts_are_excluded() -> None:
    contexts = tuple(
        ContextDocument(source_id=f"c{index}", title=f"T{index}", text="Text", position=index)
        for index in range(3)
    )
    example = QuestionExample(
        source_id="q",
        question="Question?",
        gold_answers=("Answer",),
        contexts=contexts,
        gold_support_ids=frozenset(context.source_id for context in contexts),
    )
    eligibility = bundle_eligibility(example)
    assert not eligibility.eligible
    assert eligibility.reason == "gold_support_requires_more_than_two_contexts"
    assert enumerate_two_context_bundles(example) == ()
