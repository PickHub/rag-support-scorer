from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_amplification import analyze


def _write_seed(path: Path, interaction: float) -> None:
    rows = []
    for index in range(10):
        free = 1.0
        correct_conditioned = 1.0
        wrong_conditioned = free + interaction
        for condition, conditioned, counterfactual in (
            ("correct", correct_conditioned, 0.0),
            ("plausible_wrong", wrong_conditioned, 1.0),
        ):
            rows.append(
                {
                    "question_id": f"q{index}",
                    "condition": condition,
                    "free_coverage_at_2": free,
                    "conditioned_coverage_at_2": conditioned,
                    "conditioned_counterfactual_top1": counterfactual,
                    "lexical_counterfactual_top1": 0.0,
                    "counterfactual_gold_margin": counterfactual,
                }
            )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_analyzer_clusters_repeated_seeds_by_question(tmp_path: Path) -> None:
    paths = {}
    for seed_name in ("17", "23", "42"):
        path = tmp_path / f"{seed_name}.jsonl"
        _write_seed(path, -1.0)
        paths[seed_name] = path
    report = analyze(paths, bootstrap_samples=100)
    assert report["questions"] == 10
    assert report["aggregate"]["coverage_interaction"] == -1.0  # type: ignore[index]
    assert report["coverage_interaction_ci95"] == [-1.0, -1.0]
    p_value = report["coverage_interaction_one_sided_permutation_p"]
    assert isinstance(p_value, float)
    assert p_value < 0.02
