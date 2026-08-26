from __future__ import annotations

import pytest

from rag_support_scorer.schemas import ContextDocument, QuestionExample


@pytest.fixture
def sample_example() -> QuestionExample:
    contexts = (
        ContextDocument(
            source_id="q1:c0",
            title="Ada Lovelace",
            text="Ada Lovelace documented the Analytical Engine for the scientific community.",
            position=0,
            supporting_sentences=(
                "Ada Lovelace documented the Analytical Engine for the scientific community.",
            ),
        ),
        ContextDocument(
            source_id="q1:c1",
            title="Analytical Engine",
            text="The Analytical Engine was designed by Charles Babbage in London.",
            position=1,
            supporting_sentences=(
                "The Analytical Engine was designed by Charles Babbage in London.",
            ),
        ),
        ContextDocument(
            source_id="q1:c2",
            title="Difference Engine",
            text="The Difference Engine was designed as a mechanical calculator in London.",
            position=2,
        ),
        ContextDocument(
            source_id="q1:c3",
            title="Alan Turing",
            text="Alan Turing studied mathematical machines at a university in Britain.",
            position=3,
        ),
    )
    return QuestionExample(
        source_id="q1",
        question="Who designed the machine documented by Ada Lovelace?",
        gold_answers=("Charles Babbage",),
        contexts=contexts,
        gold_support_ids=frozenset({"q1:c0", "q1:c1"}),
        question_type="bridge",
    )
