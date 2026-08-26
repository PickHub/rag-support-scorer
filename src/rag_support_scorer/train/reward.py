from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_support_scorer.schemas import PairwiseRewardExample, ScorerKind

LOGGER = logging.getLogger(__name__)
_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")


class TokenizerAdapter(Protocol):
    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        max_length: int,
        add_special_tokens: bool,
    ) -> Mapping[str, Sequence[int]]: ...


@dataclass(frozen=True)
class PreparedRewardRows:
    rows: list[dict[str, list[int]]]
    truncated_pairs: int


class RewardTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model: str
    model_revision: str
    tokenizer_revision: str
    scorer_kind: ScorerKind
    dataset_path: Path
    output_dir: Path
    seed: int = 0
    max_sequence_length: int = Field(default=2048, ge=128)
    learning_rate: float = Field(default=2e-5, gt=0)
    epochs: float = Field(default=1.0, gt=0)
    per_device_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=16, ge=1)
    use_4bit: bool = True
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    cpu_validation: bool = False
    bf16: bool = False
    fp16: bool = False

    @model_validator(mode="after")
    def revisions_must_be_explicit(self) -> RewardTrainingConfig:
        revisions = {
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
        }
        invalid = [
            name
            for name, revision in revisions.items()
            if not _IMMUTABLE_REVISION.fullmatch(revision)
        ]
        if invalid:
            raise ValueError(f"immutable 40-character commit SHA required for {invalid}")
        if self.cpu_validation and (self.bf16 or self.fp16):
            raise ValueError("CPU validation requires bf16=false and fp16=false")
        return self


def load_config(path: Path) -> RewardTrainingConfig:
    expanded = os.path.expandvars(path.read_text())
    return RewardTrainingConfig.model_validate(yaml.safe_load(expanded))


def load_reward_examples(path: Path) -> tuple[PairwiseRewardExample, ...]:
    return tuple(
        PairwiseRewardExample.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    )


def validate_training_targets(examples: Iterable[PairwiseRewardExample]) -> None:
    labels: dict[tuple[str, ScorerKind, tuple[str, str], str | None], bool] = {}
    for example in examples:
        for bundle, label in (
            (example.chosen_context_ids, example.chosen_supported),
            (example.rejected_context_ids, example.rejected_supported),
        ):
            key = (
                example.question_id,
                example.scorer_kind,
                bundle,
                example.supplied_answer,
            )
            if key in labels and labels[key] != label:
                raise ValueError(f"conflicting labels for {key}")
            labels[key] = label


def render_input(
    question: str,
    context_text: str,
    *,
    scorer_kind: ScorerKind,
    supplied_answer: str | None,
) -> str:
    if scorer_kind == ScorerKind.ANSWER_CONDITIONED:
        if not supplied_answer:
            raise ValueError("answer-conditioned input requires an answer")
        return (
            f"Question:\n{question}\n\nSupplied answer:\n{supplied_answer}\n\n"
            f"Context bundle:\n{context_text}"
        )
    if supplied_answer is not None:
        raise ValueError("answer-free input cannot include a supplied answer")
    return f"Question:\n{question}\n\nContext bundle:\n{context_text}"


def _training_rows(
    examples: Iterable[PairwiseRewardExample],
    context_lookup: dict[str, str],
) -> list[dict[str, str]]:
    rows = []
    for example in examples:
        chosen_text = "\n\n".join(
            context_lookup[context_id] for context_id in example.chosen_context_ids
        )
        rejected_text = "\n\n".join(
            context_lookup[context_id] for context_id in example.rejected_context_ids
        )
        rows.append(
            {
                "chosen": render_input(
                    example.question,
                    chosen_text,
                    scorer_kind=example.scorer_kind,
                    supplied_answer=example.supplied_answer,
                ),
                "rejected": render_input(
                    example.question,
                    rejected_text,
                    scorer_kind=example.scorer_kind,
                    supplied_answer=example.supplied_answer,
                ),
            }
        )
    return rows


def prepare_tokenized_rows(
    rows: Sequence[Mapping[str, str]],
    tokenizer: TokenizerAdapter,
    *,
    max_length: int,
) -> PreparedRewardRows:
    prepared = []
    for row in rows:
        chosen_full = list(
            tokenizer(
                row["chosen"],
                truncation=False,
                max_length=max_length,
                add_special_tokens=True,
            )["input_ids"]
        )
        rejected_full = list(
            tokenizer(
                row["rejected"],
                truncation=False,
                max_length=max_length,
                add_special_tokens=True,
            )["input_ids"]
        )
        if len(chosen_full) > max_length or len(rejected_full) > max_length:
            raise ValueError(
                "reward pair exceeds max_sequence_length; window the context without "
                "removing labeled evidence or increase the sequence budget"
            )
        prepared.append({"chosen_ids": chosen_full, "rejected_ids": rejected_full})
    if len(prepared) != len(rows):
        raise RuntimeError("reward preparation unexpectedly filtered examples")
    LOGGER.info(
        "Prepared %d/%d reward pairs without filtering or truncation",
        len(prepared),
        len(rows),
    )
    return PreparedRewardRows(prepared, 0)


def train(config: RewardTrainingConfig, context_lookup: dict[str, str]) -> None:
    examples = load_reward_examples(config.dataset_path)
    validate_training_targets(examples)
    if any(example.scorer_kind != config.scorer_kind for example in examples):
        raise ValueError("training data scorer kind does not match config")
    rows = _training_rows(examples, context_lookup)
    if config.cpu_validation:
        return
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from trl.trainer.reward_config import RewardConfig
        from trl.trainer.reward_trainer import RewardTrainer
    except ImportError as error:
        raise RuntimeError("install the train extra with `uv sync --extra train`") from error
    bitsandbytes_config = cast(Callable[..., Any], BitsAndBytesConfig)
    quantization = (
        bitsandbytes_config(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        if config.use_4bit
        else None
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        revision=config.model_revision,
        num_labels=1,
        quantization_config=quantization,
        dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        revision=config.tokenizer_revision,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules="all-linear",
        modules_to_save=["score"],
        revision=config.model_revision,
    )
    tokenized_rows = prepare_tokenized_rows(
        rows,
        tokenizer,
        max_length=config.max_sequence_length,
    )
    arguments = RewardConfig(
        output_dir=str(config.output_dir),
        seed=config.seed,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=None,
        center_rewards_coefficient=0.01,
        bf16=config.bf16,
        fp16=config.fp16,
        report_to="none",
    )
    trainer = RewardTrainer(
        model=model,
        args=arguments,
        processing_class=tokenizer,
        train_dataset=Dataset.from_list(tokenized_rows.rows),
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--context-lookup", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    context_lookup: dict[str, str] = {}
    if args.context_lookup:
        context_lookup = json.loads(args.context_lookup.read_text())
    if args.dry_run:
        if config.dataset_path.exists():
            examples = load_reward_examples(config.dataset_path)
            validate_training_targets(examples)
        print(config.model_dump_json(indent=2))
        return
    train(config, context_lookup)
