from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_support_scorer.data.bundles import enumerate_two_context_bundles
from rag_support_scorer.eval.metrics import (
    exact_match,
    joint_success,
    support_coverage_at_2,
    token_f1,
)
from rag_support_scorer.experiment.adapters import (
    DeterministicMockReader,
    DraftAdapter,
    ReaderAdapter,
)
from rag_support_scorer.experiment.hf_adapters import (
    TransformersReaderAdapter,
    TransformersScorerAdapter,
)
from rag_support_scorer.rank.rankers import (
    AdapterRanker,
    BundleRanker,
    DatasetOrderRanker,
    LexicalRanker,
    OracleSupportRanker,
    RandomRanker,
)
from rag_support_scorer.schemas import (
    AnswerCondition,
    ExperimentResult,
    QuestionExample,
    ScorerKind,
    SuppliedAnswer,
)

_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class RankingPolicy:
    ranker: BundleRanker
    requires_answer: bool = False


def controlled_conditions(
    example: QuestionExample,
    *,
    plausible_wrong_answer: str,
    draft_adapter: DraftAdapter | None = None,
    seed: int = 0,
) -> tuple[SuppliedAnswer, ...]:
    conditions = [
        SuppliedAnswer(condition=AnswerCondition.CORRECT, text=example.gold_answers[0]),
        SuppliedAnswer(condition=AnswerCondition.PLAUSIBLE_WRONG, text=plausible_wrong_answer),
        SuppliedAnswer(condition=AnswerCondition.ABSENT),
    ]
    if draft_adapter is not None:
        draft = draft_adapter.generate_draft(example.question, example.contexts, seed=seed)
        conditions.append(
            SuppliedAnswer(condition=AnswerCondition.NATURAL_DRAFT, text=draft.answer)
        )
    return tuple(conditions)


def run_controlled_experiment(
    example: QuestionExample,
    conditions: Sequence[SuppliedAnswer],
    ranking_policies: Mapping[str, RankingPolicy],
    reader: ReaderAdapter,
    *,
    seed: int = 0,
) -> tuple[ExperimentResult, ...]:
    bundles = enumerate_two_context_bundles(example)
    if not bundles:
        raise ValueError("question is ineligible for the two-context experiment")
    contexts = {context.source_id: context for context in example.contexts}
    results: list[ExperimentResult] = []
    for condition in conditions:
        for method_name, policy in ranking_policies.items():
            if policy.requires_answer and condition.text is None:
                continue
            ranked = policy.ranker.rank(example, bundles, condition.text)
            selected = ranked[0].bundle
            reader_contexts = (
                contexts[selected.context_ids[0]],
                contexts[selected.context_ids[1]],
            )
            reader_output = reader.generate(example.question, reader_contexts, seed=seed)
            coverage = support_coverage_at_2(selected.context_ids, example.gold_support_ids)
            em = exact_match(reader_output.answer, example.gold_answers)
            f1 = token_f1(reader_output.answer, example.gold_answers)
            results.append(
                ExperimentResult(
                    question_id=example.source_id,
                    condition=condition.condition,
                    ranking_method=method_name,
                    selected_context_ids=selected.context_ids,
                    supplied_answer=condition.text,
                    final_answer=reader_output.answer,
                    support_coverage_at_2=coverage,
                    exact_match=em,
                    token_f1=f1,
                    joint_success=joint_success(coverage, em),
                    scorer_metadata={"top_score": ranked[0].score},
                )
            )
    return tuple(results)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions_path: Path
    wrong_answers_path: Path
    reader_answers_path: Path | None = None
    reader_model: str | None = None
    reader_revision: str | None = None
    reader_max_new_tokens: int = Field(default=64, ge=1)
    answer_free_checkpoint: Path | None = None
    answer_conditioned_checkpoint: Path | None = None
    scorer_tokenizer_model: str | None = None
    scorer_tokenizer_revision: str | None = None
    scorer_max_sequence_length: int = Field(default=2048, ge=128)
    output_path: Path
    seed: int = 0
    ranking_methods: tuple[str, ...] = ("dataset_order", "random", "lexical_bm25", "oracle_support")

    @model_validator(mode="after")
    def runtime_configuration_is_complete(self) -> ExperimentConfig:
        mock_reader = self.reader_answers_path is not None
        model_reader = self.reader_model is not None or self.reader_revision is not None
        if mock_reader == model_reader:
            raise ValueError("configure exactly one mock or Transformers reader")
        if model_reader and (
            self.reader_model is None
            or self.reader_revision is None
            or not _IMMUTABLE_REVISION.fullmatch(self.reader_revision)
        ):
            raise ValueError("Transformers reader requires an immutable revision")
        scorer_fields = (
            self.answer_free_checkpoint,
            self.answer_conditioned_checkpoint,
            self.scorer_tokenizer_model,
            self.scorer_tokenizer_revision,
        )
        if any(value is not None for value in scorer_fields) and not all(
            value is not None for value in scorer_fields
        ):
            raise ValueError("matched scorer runtime fields must be supplied together")
        if self.scorer_tokenizer_revision is not None and not _IMMUTABLE_REVISION.fullmatch(
            self.scorer_tokenizer_revision
        ):
            raise ValueError("matched scorers require an immutable tokenizer revision")
        return self


def _ranking_policies(config: ExperimentConfig) -> dict[str, RankingPolicy]:
    available: dict[str, BundleRanker] = {
        "dataset_order": DatasetOrderRanker(),
        "random": RandomRanker(config.seed),
        "lexical_bm25": LexicalRanker(),
        "oracle_support": OracleSupportRanker(),
    }
    if config.answer_free_checkpoint is not None:
        assert config.answer_conditioned_checkpoint is not None
        assert config.scorer_tokenizer_model is not None
        assert config.scorer_tokenizer_revision is not None
        available["matched_answer_free"] = AdapterRanker(
            "matched_answer_free",
            TransformersScorerAdapter(
                str(config.answer_free_checkpoint),
                config.scorer_tokenizer_model,
                config.scorer_tokenizer_revision,
                ScorerKind.ANSWER_FREE,
                config.scorer_max_sequence_length,
            ),
            answer_conditioned=False,
        )
        available["matched_answer_conditioned"] = AdapterRanker(
            "matched_answer_conditioned",
            TransformersScorerAdapter(
                str(config.answer_conditioned_checkpoint),
                config.scorer_tokenizer_model,
                config.scorer_tokenizer_revision,
                ScorerKind.ANSWER_CONDITIONED,
                config.scorer_max_sequence_length,
            ),
            answer_conditioned=True,
        )
    missing = set(config.ranking_methods) - set(available)
    if missing:
        raise ValueError(f"unconfigured ranking methods: {sorted(missing)}")
    return {
        name: RankingPolicy(
            available[name],
            requires_answer=name == "matched_answer_conditioned",
        )
        for name in config.ranking_methods
    }


def run_from_config(config: ExperimentConfig) -> tuple[ExperimentResult, ...]:
    examples = tuple(
        QuestionExample.model_validate_json(line)
        for line in config.questions_path.read_text().splitlines()
        if line.strip()
    )
    wrong_answers: dict[str, str] = json.loads(config.wrong_answers_path.read_text())
    if config.reader_answers_path is not None:
        reader_answers: dict[str, str] = json.loads(config.reader_answers_path.read_text())
        reader: ReaderAdapter = DeterministicMockReader(reader_answers)
    else:
        assert config.reader_model is not None
        assert config.reader_revision is not None
        reader = TransformersReaderAdapter(
            config.reader_model,
            config.reader_revision,
            config.reader_max_new_tokens,
        )
    policies = _ranking_policies(config)
    results: list[ExperimentResult] = []
    for example in examples:
        results.extend(
            run_controlled_experiment(
                example,
                controlled_conditions(
                    example,
                    plausible_wrong_answer=wrong_answers[example.source_id],
                ),
                policies,
                reader,
                seed=config.seed,
            )
        )
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        "\n".join(result.model_dump_json() for result in results) + "\n"
    )
    return tuple(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = ExperimentConfig.model_validate_json(os.path.expandvars(args.config.read_text()))
    results = run_from_config(config)
    print(json.dumps({"results": len(results), "output": str(config.output_path)}))
