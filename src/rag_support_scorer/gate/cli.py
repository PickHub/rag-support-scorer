from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_support_scorer.gate.features import GateFeatures
from rag_support_scorer.gate.logistic import LogisticHarmGate, feature_ablation_sets
from rag_support_scorer.schemas import GateTrainingExample


def load_gate_examples(path: Path) -> tuple[GateTrainingExample, ...]:
    examples = tuple(
        GateTrainingExample.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    )
    if not examples:
        raise ValueError(f"no gate examples found in {path}")
    return examples


def evaluate_gate_ablations(
    train_examples: tuple[GateTrainingExample, ...],
    test_examples: tuple[GateTrainingExample, ...],
    *,
    seed: int,
) -> dict[str, object]:
    train_keys = {(example.question_id, example.condition) for example in train_examples}
    test_keys = {(example.question_id, example.condition) for example in test_examples}
    overlap = train_keys & test_keys
    if overlap:
        raise ValueError(f"gate train/test overlap detected for {len(overlap)} examples")
    train_features = tuple(GateFeatures(**example.features) for example in train_examples)
    test_features = tuple(GateFeatures(**example.features) for example in test_examples)
    train_labels = tuple(int(example.harmful) for example in train_examples)
    test_labels = tuple(int(example.harmful) for example in test_examples)
    conditions = tuple(example.condition.value for example in test_examples)
    reports: dict[str, object] = {}
    for name, feature_names in feature_ablation_sets(train_features[0]).items():
        gate = LogisticHarmGate(seed=seed).fit(
            train_features,
            train_labels,
            include_features=feature_names,
        )
        reports[name] = gate.evaluate(
            test_features,
            test_labels,
            conditions,
        )
    return {
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "harm_prevalence": sum(test_labels) / len(test_labels),
        "ablations": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    report = evaluate_gate_ablations(
        load_gate_examples(args.train),
        load_gate_examples(args.test),
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, default=lambda value: value.__dict__, indent=2, sort_keys=True)
        + "\n"
    )
