from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AnswerCondition(StrEnum):
    CORRECT = "correct"
    PLAUSIBLE_WRONG = "plausible_wrong"
    ABSENT = "absent"
    NATURAL_DRAFT = "natural_draft"
    SHUFFLED = "shuffled"


class ScorerKind(StrEnum):
    ANSWER_FREE = "answer_free"
    ANSWER_CONDITIONED = "answer_conditioned"


class ContextDocument(FrozenModel):
    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    position: int = Field(ge=0)
    entities: tuple[str, ...] = ()
    supporting_sentences: tuple[str, ...] = ()


class QuestionExample(FrozenModel):
    source_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    gold_answers: tuple[str, ...] = Field(min_length=1)
    contexts: tuple[ContextDocument, ...] = Field(min_length=1, max_length=10)
    gold_support_ids: frozenset[str] = Field(min_length=1)
    question_type: str | None = None

    @model_validator(mode="after")
    def support_must_exist(self) -> QuestionExample:
        context_ids = {context.source_id for context in self.contexts}
        if not self.gold_support_ids <= context_ids:
            raise ValueError("gold_support_ids must reference contexts in the candidate pool")
        return self


class TwoContextBundle(FrozenModel):
    question_id: str
    context_ids: tuple[str, str]
    contains_all_gold_support: bool

    @model_validator(mode="after")
    def contexts_must_be_distinct(self) -> TwoContextBundle:
        if len(set(self.context_ids)) != 2:
            raise ValueError("a bundle must contain two distinct contexts")
        return self


class SuppliedAnswer(FrozenModel):
    condition: AnswerCondition
    text: str | None = None
    source_question_id: str | None = None

    @model_validator(mode="after")
    def answer_presence_matches_condition(self) -> SuppliedAnswer:
        if self.condition == AnswerCondition.ABSENT and self.text is not None:
            raise ValueError("absent answer condition cannot contain text")
        if self.condition != AnswerCondition.ABSENT and not self.text:
            raise ValueError("present answer conditions require text")
        return self


class PairwiseRewardExample(FrozenModel):
    example_id: str
    question_id: str
    scorer_kind: ScorerKind
    question: str
    chosen_context_ids: tuple[str, str]
    rejected_context_ids: tuple[str, str]
    supplied_answer: str | None = None
    chosen_supported: bool = True
    rejected_supported: bool = False
    intervention_family: str | None = None
    template_id: str | None = None

    @model_validator(mode="after")
    def target_is_consistent(self) -> PairwiseRewardExample:
        if self.chosen_context_ids == self.rejected_context_ids:
            raise ValueError("chosen and rejected bundles must differ")
        if not self.chosen_supported or self.rejected_supported:
            raise ValueError("pairwise reward labels must prefer supported over unsupported")
        if self.scorer_kind == ScorerKind.ANSWER_CONDITIONED and not self.supplied_answer:
            raise ValueError("answer-conditioned examples require a supplied answer")
        if self.scorer_kind == ScorerKind.ANSWER_FREE and self.supplied_answer is not None:
            raise ValueError("answer-free examples cannot include a supplied answer")
        return self


class ScorerOutput(FrozenModel):
    question_id: str
    bundle: TwoContextBundle
    scorer_name: str
    score: float
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    supplied_answer: SuppliedAnswer | None = None


class ReaderOutput(FrozenModel):
    answer: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentResult(FrozenModel):
    question_id: str
    condition: AnswerCondition
    ranking_method: str
    selected_context_ids: tuple[str, str]
    supplied_answer: str | None
    final_answer: str
    support_coverage_at_2: float = Field(ge=0.0, le=1.0)
    exact_match: float = Field(ge=0.0, le=1.0)
    token_f1: float = Field(ge=0.0, le=1.0)
    joint_success: float = Field(ge=0.0, le=1.0)
    scorer_metadata: dict[str, float] = Field(default_factory=dict)
