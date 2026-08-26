from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_support_scorer.experiment.adapters import DeterministicMockReader
from rag_support_scorer.experiment.runner import (
    RankingPolicy,
    controlled_conditions,
    run_controlled_experiment,
)
from rag_support_scorer.rank.rankers import LexicalRanker, OracleSupportRanker, RandomRanker
from rag_support_scorer.schemas import ContextDocument, QuestionExample


def synthetic_example() -> QuestionExample:
    contexts = (
        ContextDocument(
            source_id="synthetic:c0",
            title="Ada Lovelace",
            text="Ada Lovelace wrote notes about the Analytical Engine.",
            position=0,
            supporting_sentences=("Ada Lovelace wrote notes about the Analytical Engine.",),
        ),
        ContextDocument(
            source_id="synthetic:c1",
            title="Analytical Engine",
            text="The Analytical Engine was designed by Charles Babbage.",
            position=1,
            supporting_sentences=("The Analytical Engine was designed by Charles Babbage.",),
        ),
        ContextDocument(
            source_id="synthetic:c2",
            title="Difference Engine",
            text="The Difference Engine was a mechanical calculator.",
            position=2,
        ),
    )
    return QuestionExample(
        source_id="synthetic:q0",
        question="Who designed the machine discussed in Ada Lovelace's notes?",
        gold_answers=("Charles Babbage",),
        contexts=contexts,
        gold_support_ids=frozenset({"synthetic:c0", "synthetic:c1"}),
        question_type="bridge",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    example = synthetic_example()
    reader = DeterministicMockReader({example.question: "Charles Babbage"})
    results = run_controlled_experiment(
        example,
        controlled_conditions(example, plausible_wrong_answer="Alan Turing"),
        {
            "random": RankingPolicy(RandomRanker(seed=0)),
            "lexical_bm25": RankingPolicy(LexicalRanker()),
            "oracle_support": RankingPolicy(OracleSupportRanker()),
        },
        reader,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([result.model_dump(mode="json") for result in results], indent=2) + "\n"
    )
    print(json.dumps({"results": len(results), "output": str(args.output)}))
