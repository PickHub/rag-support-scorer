from __future__ import annotations

from rag_support_scorer.schemas import QuestionExample
from scripts.build_azure_smoke_data import azure_split, build_records


def test_build_records_emits_matched_private_smoke_inputs(
    sample_example: QuestionExample,
) -> None:
    other = sample_example.model_copy(
        update={
            "source_id": "q2",
            "gold_answers": ("Alan Turing",),
        }
    )
    records, _, wrong_answers, contradiction_scores = build_records(
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
    if azure_split(sample_example.source_id, 17) == "test":
        assert wrong_answers[sample_example.source_id] == "Alan Turing"
        assert contradiction_scores[f"{sample_example.source_id}:correct"] == 0.5
