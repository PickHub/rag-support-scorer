from __future__ import annotations

from collections.abc import Sequence
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
from rag_support_scorer.rank.rankers import (
    LexicalRanker,
    OracleSupportRanker,
    RankedBundle,
)
from rag_support_scorer.schemas import (
    AnswerCondition,
    GateTrainingExample,
    QuestionExample,
    SuppliedAnswer,
    TwoContextBundle,
)


class _FixedRanker:
    name = "fixed"

    def __init__(self, prefer_support: bool) -> None:
        self.prefer_support = prefer_support

    def rank(
        self,
        example: QuestionExample,
        bundles: Sequence[TwoContextBundle],
        supplied_answer: str | None,
    ) -> tuple[RankedBundle, ...]:
        del example, supplied_answer
        return tuple(
            sorted(
                (
                    RankedBundle(
                        bundle,
                        float(
                            bundle.contains_all_gold_support
                            if self.prefer_support
                            else not bundle.contains_all_gold_support
                        ),
                    )
                    for bundle in bundles
                ),
                key=lambda item: (-item.score, item.bundle.context_ids),
            )
        )


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


def test_gate_output_requires_matched_ranking_methods(tmp_path: Path) -> None:
    immutable = "0" * 40
    with pytest.raises(ValueError, match="both matched scorer ranking methods"):
        ExperimentConfig(
            questions_path=tmp_path / "questions.jsonl",
            wrong_answers_path=tmp_path / "wrong.json",
            reader_model="ibm-granite/granite-3.3-2b-instruct",
            reader_revision=immutable,
            answer_free_checkpoint=tmp_path / "free",
            answer_conditioned_checkpoint=tmp_path / "conditioned",
            scorer_tokenizer_model="Qwen/Qwen3-0.6B",
            scorer_tokenizer_revision=immutable,
            gate_output_path=tmp_path / "gate.jsonl",
            contradiction_scores_path=tmp_path / "contradiction.json",
            answer_conditioned_calibration_scale=1.0,
            answer_conditioned_calibration_bias=0.0,
            output_path=tmp_path / "results.jsonl",
        )


def test_controlled_experiment_emits_gold_derived_gate_labels(
    sample_example: QuestionExample,
) -> None:
    gate_examples: list[GateTrainingExample] = []
    reader = DeterministicMockReader({sample_example.question: "Charles Babbage"})
    run_controlled_experiment(
        sample_example,
        (
            SuppliedAnswer(
                condition=AnswerCondition.PLAUSIBLE_WRONG,
                text="Alan Turing",
            ),
        ),
        {
            "matched_answer_free": RankingPolicy(_FixedRanker(True)),
            "matched_answer_conditioned": RankingPolicy(
                _FixedRanker(False),
                requires_answer=True,
            ),
        },
        reader,
        gate_examples=gate_examples,
        answer_conditioned_calibration_scale=1.0,
        answer_conditioned_calibration_bias=0.0,
        contradiction_scores={"q1:plausible_wrong": 0.8},
    )
    assert len(gate_examples) == 1
    assert gate_examples[0].harmful
    assert gate_examples[0].features["contradiction_score"] == 0.8
