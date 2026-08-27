from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Protocol

from rag_support_scorer.data.artifact_controls import lexical_overlap
from rag_support_scorer.data.bundles import enumerate_two_context_bundles
from rag_support_scorer.experiment.hf_adapters import (
    TransformersNLIAdapter,
    TransformersScorerAdapter,
)
from rag_support_scorer.gate.features import generate_gate_features
from rag_support_scorer.schemas import ContextDocument, QuestionExample, ScorerKind


class BatchScorer(Protocol):
    def score_many(
        self,
        question: str,
        context_bundles: tuple[tuple[ContextDocument, ContextDocument], ...],
        supplied_answer: str | None,
        *,
        batch_size: int,
    ) -> tuple[float, ...]: ...


class ContradictionScorer(Protocol):
    def contradiction_score(
        self,
        question: str,
        contexts: tuple[ContextDocument, ContextDocument],
        supplied_answer: str,
    ) -> float: ...


def _bundle_text(
    context_ids: tuple[str, str],
    contexts: dict[str, ContextDocument],
) -> str:
    return "\n\n".join(contexts[context_id].text for context_id in context_ids)


def _top_index(scores: tuple[float, ...], bundle_ids: tuple[tuple[str, str], ...]) -> int:
    return min(
        range(len(scores)),
        key=lambda index: (-scores[index], bundle_ids[index]),
    )


def run_study(
    examples: tuple[QuestionExample, ...],
    wrong_answers: dict[str, str],
    answer_free: BatchScorer,
    answer_conditioned: BatchScorer,
    nli: ContradictionScorer,
    *,
    calibration_scale: float,
    calibration_bias: float,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        bundles = enumerate_two_context_bundles(example)
        if not bundles:
            continue
        contexts = {context.source_id: context for context in example.contexts}
        context_bundles = tuple(
            (contexts[bundle.context_ids[0]], contexts[bundle.context_ids[1]])
            for bundle in bundles
        )
        bundle_ids = tuple(bundle.context_ids for bundle in bundles)
        gold_index = next(
            index for index, bundle in enumerate(bundles) if bundle.contains_all_gold_support
        )
        counterfactual_indices = tuple(
            index
            for index, bundle in enumerate(bundles)
            if any(":counterfactual-" in context_id for context_id in bundle.context_ids)
        )
        if not counterfactual_indices:
            raise ValueError(f"{example.source_id} has no counterfactual context")
        free_scores = answer_free.score_many(
            example.question,
            context_bundles,
            None,
            batch_size=batch_size,
        )
        free_top = _top_index(free_scores, bundle_ids)
        free_contexts = context_bundles[free_top]
        free_coverage = float(bundles[free_top].contains_all_gold_support)
        free_counterfactual = float(free_top in counterfactual_indices)
        for condition, supplied_answer in (
            ("correct", example.gold_answers[0]),
            ("plausible_wrong", wrong_answers[example.source_id]),
        ):
            conditioned_scores = answer_conditioned.score_many(
                example.question,
                context_bundles,
                supplied_answer,
                batch_size=batch_size,
            )
            conditioned_top = _top_index(conditioned_scores, bundle_ids)
            conditioned_coverage = float(
                bundles[conditioned_top].contains_all_gold_support
            )
            best_counterfactual_score = max(
                conditioned_scores[index] for index in counterfactual_indices
            )
            lexical_scores = tuple(
                lexical_overlap(
                    supplied_answer,
                    _bundle_text(bundle.context_ids, contexts),
                )
                for bundle in bundles
            )
            lexical_top = _top_index(lexical_scores, bundle_ids)
            calibrated_logit = (
                calibration_scale * conditioned_scores[conditioned_top]
                + calibration_bias
            )
            probability = (
                1 / (1 + math.exp(-calibrated_logit))
                if calibrated_logit >= 0
                else math.exp(calibrated_logit) / (1 + math.exp(calibrated_logit))
            )
            contradiction = nli.contradiction_score(
                example.question,
                free_contexts,
                supplied_answer,
            )
            features = generate_gate_features(
                {
                    "|".join(bundle_ids[index]): score
                    for index, score in enumerate(conditioned_scores)
                },
                {
                    "|".join(bundle_ids[index]): score
                    for index, score in enumerate(free_scores)
                },
                calibrated_answer_conditioned_probability=probability,
                contradiction_score=contradiction,
            )
            rows.append(
                {
                    "question_id": example.source_id,
                    "condition": condition,
                    "free_coverage_at_2": free_coverage,
                    "conditioned_coverage_at_2": conditioned_coverage,
                    "free_counterfactual_top1": free_counterfactual,
                    "conditioned_counterfactual_top1": float(
                        conditioned_top in counterfactual_indices
                    ),
                    "lexical_counterfactual_top1": float(
                        lexical_top in counterfactual_indices
                    ),
                    "counterfactual_gold_margin": (
                        best_counterfactual_score - conditioned_scores[gold_index]
                    ),
                    "harmful": conditioned_coverage < free_coverage,
                    "features": features.as_mapping(),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--wrong-answers", type=Path, required=True)
    parser.add_argument("--answer-free-checkpoint", required=True)
    parser.add_argument("--answer-conditioned-checkpoint", required=True)
    parser.add_argument("--scorer-model", required=True)
    parser.add_argument("--scorer-revision", required=True)
    parser.add_argument("--nli-model", required=True)
    parser.add_argument("--nli-revision", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    examples = tuple(
        QuestionExample.model_validate_json(line)
        for line in args.questions.read_text().splitlines()
        if line.strip()
    )
    wrong_answers = json.loads(args.wrong_answers.read_text())
    calibration = json.loads(args.calibration.read_text())
    answer_free = TransformersScorerAdapter(
        args.answer_free_checkpoint,
        args.scorer_model,
        args.scorer_revision,
        ScorerKind.ANSWER_FREE,
    )
    answer_conditioned = TransformersScorerAdapter(
        args.answer_conditioned_checkpoint,
        args.scorer_model,
        args.scorer_revision,
        ScorerKind.ANSWER_CONDITIONED,
    )
    nli = TransformersNLIAdapter(args.nli_model, args.nli_revision)
    rows = run_study(
        examples,
        wrong_answers,
        answer_free,
        answer_conditioned,
        nli,
        calibration_scale=float(calibration["scale"]),
        calibration_bias=float(calibration["bias"]),
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    )


if __name__ == "__main__":
    main()
