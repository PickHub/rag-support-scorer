from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from rag_support_scorer.data.artifact_controls import (
    artifact_distance,
    lexical_overlap,
    surface_features,
)
from rag_support_scorer.data.dedup import normalize_text
from rag_support_scorer.schemas import ContextDocument, QuestionExample


class InterventionFamily(StrEnum):
    NECESSARY_EVIDENCE_REMOVAL = "necessary_evidence_removal"
    LEXICAL_DISTRACTOR_REPLACEMENT = "lexical_distractor_replacement"
    SAME_TYPE_WRONG_ANSWER = "same_type_wrong_answer"
    CONTRADICTION_INJECTION = "contradiction_injection"


@dataclass(frozen=True)
class InterventionTemplate:
    template_id: str
    family: InterventionFamily
    held_out: bool = False


@dataclass(frozen=True)
class InterventionResult:
    family: InterventionFamily
    template_id: str
    contexts: tuple[ContextDocument, ...]
    supplied_answer: str | None
    replaced_context_id: str | None
    artifact_distance: float


def _replace_text(context: ContextDocument, text: str, suffix: str) -> ContextDocument:
    return context.model_copy(
        update={
            "source_id": f"{context.source_id}:{suffix}",
            "text": text,
            "supporting_sentences": (),
        }
    )


def _match_token_length(text: str, reference: str) -> str:
    target = len(reference.split())
    tokens = text.split()
    if not tokens:
        tokens = ["context"]
    if len(tokens) >= target:
        return " ".join(tokens[:target])
    return " ".join((tokens * ((target + len(tokens) - 1) // len(tokens)))[:target])


def remove_necessary_evidence(
    context: ContextDocument,
    *,
    template: InterventionTemplate,
    replacement_text: str | None = None,
) -> InterventionResult:
    if template.family != InterventionFamily.NECESSARY_EVIDENCE_REMOVAL:
        raise ValueError("template family mismatch")
    if not context.supporting_sentences:
        raise ValueError("context has no marked supporting sentence")
    text = context.text
    for sentence in context.supporting_sentences:
        text = text.replace(sentence, "")
    replacement_source = " ".join((text, replacement_text or "")).strip()
    if not replacement_source:
        raise ValueError("removal requires non-support replacement text")
    replacement = _match_token_length(replacement_source, context.text)
    modified = _replace_text(context, replacement, template.template_id)
    distance = artifact_distance(surface_features(context.text), surface_features(modified.text))
    return InterventionResult(
        template.family,
        template.template_id,
        (modified,),
        None,
        context.source_id,
        distance,
    )


def replace_with_lexical_distractor(
    target: ContextDocument,
    candidates: Iterable[ContextDocument],
    *,
    question: str,
    template: InterventionTemplate,
) -> InterventionResult:
    if template.family != InterventionFamily.LEXICAL_DISTRACTOR_REPLACEMENT:
        raise ValueError("template family mismatch")
    eligible = [
        candidate
        for candidate in candidates
        if candidate.source_id != target.source_id and not candidate.supporting_sentences
    ]
    if not eligible:
        raise ValueError("no non-support distractor candidate")
    distractor = max(
        eligible,
        key=lambda candidate: (
            lexical_overlap(candidate.text, target.text),
            lexical_overlap(candidate.text, question),
            -abs(len(candidate.text) - len(target.text)),
            candidate.source_id,
        ),
    )
    replacement = _match_token_length(distractor.text, target.text)
    modified = _replace_text(distractor, replacement, template.template_id)
    distance = artifact_distance(
        surface_features(target.text, question=question),
        surface_features(modified.text, question=question),
    )
    return InterventionResult(
        template.family,
        template.template_id,
        (modified,),
        None,
        target.source_id,
        distance,
    )


def replace_with_same_type_wrong_answer(
    example: QuestionExample,
    wrong_answers: Iterable[str],
    *,
    template: InterventionTemplate,
) -> InterventionResult:
    if template.family != InterventionFamily.SAME_TYPE_WRONG_ANSWER:
        raise ValueError("template family mismatch")
    gold = example.gold_answers[0]
    gold_type = infer_answer_type(gold)
    normalized_gold_answers = {normalize_text(answer) for answer in example.gold_answers}
    candidates = [
        answer
        for answer in wrong_answers
        if normalize_text(answer) not in normalized_gold_answers
        and infer_answer_type(answer) == gold_type
    ]
    if not candidates:
        raise ValueError(f"no wrong-answer candidate with type {gold_type}")
    wrong = min(candidates, key=lambda answer: (abs(len(answer) - len(gold)), answer))
    return InterventionResult(template.family, template.template_id, (), wrong, None, 0.0)


def inject_contradiction(
    context: ContextDocument,
    *,
    gold_answer: str,
    wrong_answer: str,
    template: InterventionTemplate,
) -> InterventionResult:
    if template.family != InterventionFamily.CONTRADICTION_INJECTION:
        raise ValueError("template family mismatch")
    pattern = re.compile(re.escape(gold_answer), re.IGNORECASE)
    if not pattern.search(context.text):
        raise ValueError("gold answer is not present in context")
    if len(wrong_answer.split()) != len(gold_answer.split()):
        raise ValueError("contradiction answer must match the gold answer token count")
    modified_text = pattern.sub(wrong_answer, context.text, count=1)
    modified = _replace_text(context, modified_text, template.template_id)
    distance = artifact_distance(surface_features(context.text), surface_features(modified.text))
    return InterventionResult(
        template.family,
        template.template_id,
        (modified,),
        wrong_answer,
        context.source_id,
        distance,
    )


def held_out_template_ids(templates: Iterable[InterventionTemplate]) -> frozenset[str]:
    return frozenset(template.template_id for template in templates if template.held_out)


def infer_answer_type(answer: str) -> str:
    normalized = answer.strip()
    if normalized.casefold() in {"yes", "no"}:
        return "boolean"
    if re.fullmatch(r"\d{4}(?:-\d{2}-\d{2})?", normalized):
        return "date"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:\s*%)?", normalized):
        return "number"
    if len(normalized.split()) in {2, 3} and all(
        token[:1].isupper() for token in normalized.split()
    ):
        return "person_or_named_entity"
    return "text"
