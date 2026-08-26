from __future__ import annotations

import argparse
from pathlib import Path

from rag_support_scorer.experiment.runner import ExperimentConfig, run_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--wrong-answers", type=Path, required=True)
    parser.add_argument("--answer-free-checkpoint", type=Path, required=True)
    parser.add_argument("--answer-conditioned-checkpoint", type=Path, required=True)
    parser.add_argument("--scorer-tokenizer-model", required=True)
    parser.add_argument("--scorer-tokenizer-revision", required=True)
    parser.add_argument("--reader-model", required=True)
    parser.add_argument("--reader-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig(
        questions_path=args.questions,
        wrong_answers_path=args.wrong_answers,
        reader_model=args.reader_model,
        reader_revision=args.reader_revision,
        answer_free_checkpoint=args.answer_free_checkpoint,
        answer_conditioned_checkpoint=args.answer_conditioned_checkpoint,
        scorer_tokenizer_model=args.scorer_tokenizer_model,
        scorer_tokenizer_revision=args.scorer_tokenizer_revision,
        output_path=args.output_dir / "experiment_results.jsonl",
        seed=args.seed,
        ranking_methods=(
            "dataset_order",
            "random",
            "lexical_bm25",
            "matched_answer_free",
            "matched_answer_conditioned",
            "oracle_support",
        ),
    )
    run_from_config(config)


if __name__ == "__main__":
    main()
