from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from rag_support_scorer.schemas import QuestionExample, TwoContextBundle


@dataclass(frozen=True)
class BundleEligibility:
    eligible: bool
    reason: str | None


def bundle_eligibility(example: QuestionExample, bundle_size: int = 2) -> BundleEligibility:
    if len(example.contexts) < bundle_size:
        return BundleEligibility(False, "fewer_than_two_contexts")
    if len(example.contexts) > 10:
        return BundleEligibility(False, "candidate_pool_exceeds_ten")
    if len(example.gold_support_ids) > bundle_size:
        return BundleEligibility(False, "gold_support_requires_more_than_two_contexts")
    return BundleEligibility(True, None)


def enumerate_two_context_bundles(example: QuestionExample) -> tuple[TwoContextBundle, ...]:
    eligibility = bundle_eligibility(example)
    if not eligibility.eligible:
        return ()
    bundles = []
    for first, second in combinations(example.contexts, 2):
        context_ids = (first.source_id, second.source_id)
        bundles.append(
            TwoContextBundle(
                question_id=example.source_id,
                context_ids=context_ids,
                contains_all_gold_support=example.gold_support_ids <= set(context_ids),
            )
        )
    return tuple(bundles)
