from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def analyze(
    result_paths: dict[str, Path],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 17,
) -> dict[str, object]:
    by_seed: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for seed_name, path in result_paths.items():
        questions: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                questions[row["question_id"]][row["condition"]] = row
        complete = {
            question_id: conditions
            for question_id, conditions in questions.items()
            if set(conditions) == {"correct", "plausible_wrong"}
        }
        if not complete:
            raise ValueError(f"{seed_name} has no complete paired questions")
        by_seed[seed_name] = complete

    shared_questions = set.intersection(
        *(set(questions) for questions in by_seed.values())
    )
    if not shared_questions:
        raise ValueError("seed results have no shared questions")

    question_metrics: dict[str, dict[str, float]] = {}
    for question_id in sorted(shared_questions):
        values: dict[str, list[float]] = defaultdict(list)
        for questions in by_seed.values():
            correct = questions[question_id]["correct"]
            wrong = questions[question_id]["plausible_wrong"]
            correct_effect = (
                float(correct["conditioned_coverage_at_2"])
                - float(correct["free_coverage_at_2"])
            )
            wrong_effect = (
                float(wrong["conditioned_coverage_at_2"])
                - float(wrong["free_coverage_at_2"])
            )
            values["coverage_interaction"].append(wrong_effect - correct_effect)
            values["counterfactual_top1_increase"].append(
                float(wrong["conditioned_counterfactual_top1"])
                - float(correct["conditioned_counterfactual_top1"])
            )
            values["margin_delta"].append(
                float(wrong["counterfactual_gold_margin"])
                - float(correct["counterfactual_gold_margin"])
            )
            values["lexical_counterfactual_increase"].append(
                float(wrong["lexical_counterfactual_top1"])
                - float(correct["lexical_counterfactual_top1"])
            )
            values["correct_counterfactual_top1"].append(
                float(correct["conditioned_counterfactual_top1"])
            )
        question_metrics[question_id] = {
            name: _mean(metric_values) for name, metric_values in values.items()
        }

    rng = np.random.default_rng(seed)
    question_ids = sorted(question_metrics)
    primary = np.asarray(
        [question_metrics[question_id]["coverage_interaction"] for question_id in question_ids]
    )
    bootstrap = np.asarray(
        [
            rng.choice(primary, size=len(primary), replace=True).mean()
            for _ in range(bootstrap_samples)
        ]
    )
    seed_summaries: dict[str, dict[str, float]] = {}
    for seed_name, questions in by_seed.items():
        metrics: dict[str, list[float]] = defaultdict(list)
        for question_id in shared_questions:
            correct = questions[question_id]["correct"]
            wrong = questions[question_id]["plausible_wrong"]
            metrics["coverage_interaction"].append(
                (
                    float(wrong["conditioned_coverage_at_2"])
                    - float(wrong["free_coverage_at_2"])
                )
                - (
                    float(correct["conditioned_coverage_at_2"])
                    - float(correct["free_coverage_at_2"])
                )
            )
            metrics["counterfactual_top1_increase"].append(
                float(wrong["conditioned_counterfactual_top1"])
                - float(correct["conditioned_counterfactual_top1"])
            )
        seed_summaries[seed_name] = {
            name: _mean(values) for name, values in metrics.items()
        }

    aggregate = {
        name: _mean([metrics[name] for metrics in question_metrics.values()])
        for name in next(iter(question_metrics.values()))
    }
    return {
        "questions": len(shared_questions),
        "seeds": sorted(by_seed),
        "primary_metric": "coverage_interaction",
        "primary_direction": "negative indicates answer-error amplification",
        "aggregate": aggregate,
        "coverage_interaction_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "seed_summaries": seed_summaries,
        "failure_checks": {
            "correct_counterfactual_top1_exceeds_0_15": (
                aggregate["correct_counterfactual_top1"] > 0.15
            ),
            "coverage_effect_sign_changes_across_seeds": len(
                {
                    np.sign(summary["coverage_interaction"])
                    for summary in seed_summaries.values()
                }
            )
            > 1,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="Seed and JSONL path as seed=path",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    result_paths = {}
    for value in args.result:
        seed_name, path = value.split("=", 1)
        result_paths[seed_name] = Path(path)
    report = analyze(result_paths, bootstrap_samples=args.bootstrap_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
