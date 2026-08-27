from __future__ import annotations

from rag_support_scorer.schemas import QuestionExample
from scripts.build_azure_smoke_data import (
    azure_split,
    build_records,
    conflict_enriched_examples,
)


def test_build_records_emits_matched_private_smoke_inputs(
    sample_example: QuestionExample,
) -> None:
    other = sample_example.model_copy(
        update={
            "source_id": "q2",
            "gold_answers": ("Alan Turing",),
        }
    )
    records, _, wrong_answers, contradiction_scores, artifact_rows = build_records(
        (sample_example, other),
        limit=1,
        max_negatives=2,
        split_seed=17,
    )
    assert {record["target"] for record in records} == {
        "answer_free",
        "answer_conditioned",
    }
    assert sum(record["target"] == "answer_free" for record in records) == 2
    assert sum(record["target"] == "answer_conditioned" for record in records) == 4
    assert {row["label"] for row in artifact_rows} == {0, 1}
    if azure_split(sample_example.source_id, 17) == "test":
        assert wrong_answers[sample_example.source_id] == "Alan Turing"
        assert contradiction_scores[f"{sample_example.source_id}:correct"] == 0.5


def test_conflict_pool_is_fixed_across_correct_and_wrong_labels(
    sample_example: QuestionExample,
) -> None:
    result = conflict_enriched_examples(sample_example, "Alan Turing")
    assert result is not None
    enriched, wrong_labeled = result
    assert enriched.contexts == wrong_labeled.contexts
    assert enriched.gold_support_ids != wrong_labeled.gold_support_ids
    counterfactuals = [
        context for context in enriched.contexts if ":counterfactual-" in context.source_id
    ]
    assert len(counterfactuals) == 1
    assert "Alan Turing" in counterfactuals[0].text
    assert "q1:c1" in enriched.gold_support_ids
    assert counterfactuals[0].source_id in wrong_labeled.gold_support_ids
