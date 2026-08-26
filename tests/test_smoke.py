from __future__ import annotations

from rag_support_scorer.experiment.adapters import DeterministicMockReader
from rag_support_scorer.experiment.runner import (
    RankingPolicy,
    controlled_conditions,
    run_controlled_experiment,
)
from rag_support_scorer.experiment.smoke import synthetic_example
from rag_support_scorer.rank.rankers import LexicalRanker, OracleSupportRanker


def test_synthetic_end_to_end_smoke() -> None:
    example = synthetic_example()
    results = run_controlled_experiment(
        example,
        controlled_conditions(example, plausible_wrong_answer="Alan Turing"),
        {
            "lexical": RankingPolicy(LexicalRanker()),
            "oracle": RankingPolicy(OracleSupportRanker()),
        },
        DeterministicMockReader({example.question: "Charles Babbage"}),
    )
    assert len(results) == 6
    oracle_results = [result for result in results if result.ranking_method == "oracle"]
    assert all(result.support_coverage_at_2 == 1.0 for result in oracle_results)
    assert all(result.joint_success == 1.0 for result in oracle_results)
