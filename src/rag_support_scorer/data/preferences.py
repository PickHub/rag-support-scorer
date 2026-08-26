from __future__ import annotations

from collections.abc import Mapping, Sequence

from rag_support_scorer.schemas import PairwiseRewardExample, ScorerKind, TwoContextBundle


def bundle_key(bundle: TwoContextBundle) -> str:
    return "|".join(bundle.context_ids)


def build_pairwise_examples(
    *,
    question_id: str,
    question: str,
    bundles: Sequence[TwoContextBundle],
    scorer_kind: ScorerKind,
    supplied_answer: str | None = None,
    answer_support_labels: Mapping[str, bool] | None = None,
    intervention_family: str | None = None,
    template_id: str | None = None,
) -> tuple[PairwiseRewardExample, ...]:
    if scorer_kind == ScorerKind.ANSWER_FREE:
        if supplied_answer is not None or answer_support_labels is not None:
            raise ValueError("answer-free preferences use only gold-support bundle labels")
        labels = {bundle_key(bundle): bundle.contains_all_gold_support for bundle in bundles}
    else:
        if not supplied_answer or answer_support_labels is None:
            raise ValueError("answer-conditioned preferences require answer-specific labels")
        labels = dict(answer_support_labels)
        if set(labels) != {bundle_key(bundle) for bundle in bundles}:
            raise ValueError("answer-specific labels must cover every bundle exactly once")
    positives = [bundle for bundle in bundles if labels[bundle_key(bundle)]]
    negatives = [bundle for bundle in bundles if not labels[bundle_key(bundle)]]
    if not positives or not negatives:
        raise ValueError(
            "pairwise preferences require at least one positive and one negative bundle"
        )
    examples = []
    for positive in positives:
        for negative in negatives:
            identifier = (
                f"{question_id}:{scorer_kind.value}:"
                f"{bundle_key(positive)}>{bundle_key(negative)}"
            )
            examples.append(
                PairwiseRewardExample(
                    example_id=identifier,
                    question_id=question_id,
                    scorer_kind=scorer_kind,
                    question=question,
                    chosen_context_ids=positive.context_ids,
                    rejected_context_ids=negative.context_ids,
                    supplied_answer=supplied_answer,
                    intervention_family=intervention_family,
                    template_id=template_id,
                )
            )
    return tuple(examples)
