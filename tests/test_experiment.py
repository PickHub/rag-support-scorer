from __future__ import annotations

from pathlib import Path

import pytest

from rag_support_scorer.experiment.adapters import DeterministicMockReader
from rag_support_scorer.experiment.hf_adapters import render_reader_prompt
from rag_support_scorer.experiment.runner import (
    ExperimentConfig,
    RankingPolicy,
    controlled_conditions,
    run_controlled_experiment,
)
from rag_support_scorer.rank.rankers import LexicalRanker, OracleSupportRanker
from rag_support_scorer.schemas import AnswerCondition, QuestionExample


def test_controlled_conditions_and_reader_isolation(sample_example: QuestionExample) -> None:
    calls: list[tuple[str, tuple[str, str], int]] = []
    reader = DeterministicMockReader(
        {sample_example.question: "Charles Babbage"},
        calls=calls,
    )
    conditions = controlled_conditions(
        sample_example,
        plausible_wrong_answer="Alan Turing",
    )
    results = run_controlled_experiment(
        sample_example,
        conditions,
        {
            "lexical": RankingPolicy(LexicalRanker()),
            "oracle": RankingPolicy(OracleSupportRanker()),
        },
        reader,
    )
    assert {condition.condition for condition in conditions} == {
        AnswerCondition.CORRECT,
        AnswerCondition.PLAUSIBLE_WRONG,
        AnswerCondition.ABSENT,
    }
    assert len(results) == 6
    assert len(calls) == 6
    assert all(call[0] == sample_example.question for call in calls)
    assert all(len(call[1]) == 2 for call in calls)
    assert all(result.final_answer == "Charles Babbage" for result in results)


def test_transformers_reader_prompt_contains_only_question_and_contexts(
    sample_example: QuestionExample,
) -> None:
    prompt = render_reader_prompt(
        sample_example.question,
        (sample_example.contexts[0], sample_example.contexts[1]),
    )
    assert sample_example.question in prompt
    assert sample_example.contexts[0].text in prompt
    assert "Supplied answer" not in prompt
    assert "Alan Turing" not in prompt


def test_gpu_experiment_requires_immutable_revisions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="immutable"):
        ExperimentConfig(
            questions_path=tmp_path / "questions.jsonl",
            wrong_answers_path=tmp_path / "wrong.json",
            reader_model="ibm-granite/granite-3.3-2b-instruct",
            reader_revision="main",
            output_path=tmp_path / "results.jsonl",
        )
