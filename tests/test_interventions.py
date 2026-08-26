from __future__ import annotations

import pytest

from rag_support_scorer.data.artifact_controls import surface_features
from rag_support_scorer.data.interventions import (
    InterventionFamily,
    InterventionTemplate,
    held_out_template_ids,
    inject_contradiction,
    remove_necessary_evidence,
    replace_with_lexical_distractor,
    replace_with_same_type_wrong_answer,
)
from rag_support_scorer.schemas import QuestionExample


def _template(family: InterventionFamily, *, held_out: bool = False) -> InterventionTemplate:
    return InterventionTemplate(f"{family.value}-v1", family, held_out)


def test_necessary_evidence_removal_preserves_surface_length(
    sample_example: QuestionExample,
) -> None:
    original = sample_example.contexts[0]
    result = remove_necessary_evidence(
        original,
        template=_template(InterventionFamily.NECESSARY_EVIDENCE_REMOVAL),
        replacement_text=sample_example.contexts[2].text,
    )
    modified = result.contexts[0]
    assert modified.supporting_sentences == ()
    assert (
        surface_features(modified.text).token_count
        == surface_features(original.text).token_count
    )


def test_lexical_distractor_is_non_support_and_length_matched(
    sample_example: QuestionExample,
) -> None:
    target = sample_example.contexts[1]
    result = replace_with_lexical_distractor(
        target,
        sample_example.contexts,
        question=sample_example.question,
        template=_template(InterventionFamily.LEXICAL_DISTRACTOR_REPLACEMENT),
    )
    modified = result.contexts[0]
    assert modified.supporting_sentences == ()
    assert surface_features(modified.text).token_count == surface_features(target.text).token_count


def test_same_type_wrong_answer_is_deterministic(sample_example: QuestionExample) -> None:
    result = replace_with_same_type_wrong_answer(
        sample_example,
        ["Alan Turing", "Grace Hopper"],
        template=_template(InterventionFamily.SAME_TYPE_WRONG_ANSWER),
    )
    assert result.supplied_answer == "Grace Hopper"


def test_same_type_wrong_answer_excludes_every_gold_alias(
    sample_example: QuestionExample,
) -> None:
    example = sample_example.model_copy(
        update={"gold_answers": ("Charles Babbage", "C. Babbage")}
    )
    result = replace_with_same_type_wrong_answer(
        example,
        ["C. Babbage", "Alan Turing"],
        template=_template(InterventionFamily.SAME_TYPE_WRONG_ANSWER),
    )
    assert result.supplied_answer == "Alan Turing"


def test_contradiction_injection_preserves_token_count(sample_example: QuestionExample) -> None:
    context = sample_example.contexts[1]
    result = inject_contradiction(
        context,
        gold_answer="Charles Babbage",
        wrong_answer="Alan Turing",
        template=_template(InterventionFamily.CONTRADICTION_INJECTION),
    )
    assert "Alan Turing" in result.contexts[0].text
    assert (
        surface_features(result.contexts[0].text).token_count
        == surface_features(context.text).token_count
    )


def test_contradiction_rejects_mismatched_answer_length(
    sample_example: QuestionExample,
) -> None:
    with pytest.raises(ValueError, match="match the gold answer token count"):
        inject_contradiction(
            sample_example.contexts[1],
            gold_answer="Charles Babbage",
            wrong_answer="Turing",
            template=_template(InterventionFamily.CONTRADICTION_INJECTION),
        )


def test_held_out_templates_are_complete_units() -> None:
    templates = [
        _template(InterventionFamily.CONTRADICTION_INJECTION, held_out=True),
        _template(InterventionFamily.SAME_TYPE_WRONG_ANSWER),
    ]
    assert held_out_template_ids(templates) == frozenset({"contradiction_injection-v1"})
