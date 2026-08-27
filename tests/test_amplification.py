from __future__ import annotations

from collections.abc import Sequence

from rag_support_scorer.experiment.amplification import run_study
from rag_support_scorer.schemas import ContextDocument, QuestionExample
from scripts.build_azure_smoke_data import conflict_enriched_examples


class _FakeScorer:
    def __init__(self, *, conditioned: bool) -> None:
        self.conditioned = conditioned

    def score_many(
        self,
        question: str,
        bundles: Sequence[tuple[ContextDocument, ContextDocument]],
        supplied_answer: str | None,
        *,
        batch_size: int,
    ) -> tuple[float, ...]:
        del question, batch_size
        scores = []
        for contexts in bundles:
            ids = {context.source_id for context in contexts}
            has_gold = {"q1:c0", "q1:c1"} <= ids
            has_counterfactual = any(":counterfactual-" in source_id for source_id in ids)
            if self.conditioned and supplied_answer == "Alan Turing":
                scores.append(float(has_counterfactual and "q1:c0" in ids))
            else:
                scores.append(float(has_gold))
        return tuple(scores)


class _FakeNLI:
    def contradiction_score(
        self,
        question: str,
        contexts: tuple[ContextDocument, ContextDocument],
        supplied_answer: str,
    ) -> float:
        del question, contexts
        return 0.9 if supplied_answer == "Alan Turing" else 0.1


def test_study_measures_counterfactual_displacement(
    sample_example: QuestionExample,
) -> None:
    conflict = conflict_enriched_examples(sample_example, "Alan Turing")
    assert conflict is not None
    enriched, _ = conflict
    rows = run_study(
        (enriched,),
        {enriched.source_id: "Alan Turing"},
        _FakeScorer(conditioned=False),
        _FakeScorer(conditioned=True),
        _FakeNLI(),
        calibration_scale=1.0,
        calibration_bias=0.0,
        batch_size=8,
    )
    by_condition = {row["condition"]: row for row in rows}
    assert by_condition["correct"]["conditioned_coverage_at_2"] == 1.0
    assert by_condition["correct"]["conditioned_counterfactual_top1"] == 0.0
    assert by_condition["plausible_wrong"]["conditioned_coverage_at_2"] == 0.0
    assert by_condition["plausible_wrong"]["conditioned_counterfactual_top1"] == 1.0
    assert by_condition["plausible_wrong"]["harmful"]
