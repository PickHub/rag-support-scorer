from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class TrainingConfig:
    target: str
    model_name: str
    model_revision: str
    seed: int
    epochs: int
    learning_rate: float
    max_length: int
    gradient_accumulation_steps: int


class PreferenceDataset(Dataset[tuple[str, str]]):
    def __init__(self, path: Path) -> None:
        self.examples: list[tuple[str, str]] = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                chosen = record.get("chosen")
                rejected = record.get("rejected")
                if not isinstance(chosen, str) or not isinstance(rejected, str):
                    raise ValueError(
                        f"{path}:{line_number}: chosen and rejected must be strings"
                    )
                self.examples.append((chosen, rejected))
        if not self.examples:
            raise ValueError(f"No preference examples found in {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.examples[index]


class PairCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(
        self, examples: list[tuple[str, str]]
    ) -> tuple[dict[str, torch.Tensor], int]:
        chosen, rejected = zip(*examples, strict=True)
        texts = [*chosen, *rejected]
        unpadded = self.tokenizer(
            texts,
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )
        if any(len(token_ids) > self.max_length for token_ids in unpadded["input_ids"]):
            raise ValueError(
                "preference pair exceeds max_length; window contexts without removing "
                "labeled evidence before training"
            )
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        return encoded, len(chosen)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 4-bit QLoRA scalar scorer.")
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument(
        "--target", choices=("answer_free", "answer_conditioned"), required=True
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mixed_precision_dtype() -> torch.dtype:
    major_version, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major_version >= 8 else torch.float16


def build_quantization_config(compute_dtype: torch.dtype) -> Any:
    try:
        importlib.import_module("bitsandbytes")
    except ImportError as error:
        raise RuntimeError(
            "4-bit scorer training requires the optional bitsandbytes dependency"
        ) from error
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )


def move_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def pairwise_loss(
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
    pair_count: int,
) -> torch.Tensor:
    rewards = model(**encoded).logits.float().reshape(-1)
    chosen_rewards = rewards[:pair_count]
    rejected_rewards = rewards[pair_count:]
    return -functional.logsigmoid(chosen_rewards - rejected_rewards).mean()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[tuple[dict[str, torch.Tensor], int]],
    device: torch.device,
) -> tuple[float, list[dict[str, float | int]]]:
    model.eval()
    losses: list[float] = []
    score_rows: list[dict[str, float | int]] = []
    for encoded, pair_count in loader:
        rewards = model(**move_to_device(encoded, device)).logits.float().reshape(-1)
        chosen_rewards = rewards[:pair_count]
        rejected_rewards = rewards[pair_count:]
        loss = -functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
        losses.append(loss.item())
        score_rows.extend(
            {"score": float(score), "label": label}
            for values, label in ((chosen_rewards, 1), (rejected_rewards, 0))
            for score in values.tolist()
        )
    model.train()
    return sum(losses) / len(losses), score_rows


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for 4-bit QLoRA training")
    if args.epochs < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("epochs and gradient_accumulation_steps must be positive")
    if not IMMUTABLE_REVISION.fullmatch(args.model_revision):
        raise ValueError("model revision must be an immutable 40-character commit SHA")

    config = TrainingConfig(
        target=args.target,
        model_name=args.model_name,
        model_revision=args.model_revision,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    set_determinism(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, revision=config.model_revision, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = mixed_precision_dtype()
    quantization = build_quantization_config(compute_dtype)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        num_labels=1,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=compute_dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
            target_modules="all-linear",
            modules_to_save=["score"],
            revision=config.model_revision,
        ),
    )

    train_dataset = PreferenceDataset(
        args.prepared_data / config.target / "train.jsonl"
    )
    validation_dataset = PreferenceDataset(
        args.prepared_data / config.target / "validation.jsonl"
    )
    collator = PairCollator(tokenizer, config.max_length)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=1, shuffle=False, collate_fn=collator
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    device = torch.device("cuda:0")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    update_count = 0
    train_losses: list[float] = []
    for _ in range(config.epochs):
        for step, (encoded, pair_count) in enumerate(train_loader, start=1):
            loss = pairwise_loss(model, move_to_device(encoded, device), pair_count)
            train_losses.append(loss.item())
            (loss / config.gradient_accumulation_steps).backward()
            is_update = (
                step % config.gradient_accumulation_steps == 0
                or step == len(train_loader)
            )
            if is_update:
                accumulated_steps = (
                    step % config.gradient_accumulation_steps
                    or config.gradient_accumulation_steps
                )
                if accumulated_steps < config.gradient_accumulation_steps:
                    scale = config.gradient_accumulation_steps / accumulated_steps
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(scale)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_count += 1

    validation_loss, validation_scores = evaluate(model, validation_loader, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    metrics: dict[str, Any] = {
        "config": asdict(config),
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "optimizer_updates": update_count,
        "train_loss": sum(train_losses) / len(train_losses),
        "validation_loss": validation_loss,
        "validation_pairwise_perplexity": math.exp(min(validation_loss, 20.0)),
    }
    (args.metrics_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.metrics_dir / "validation_scores.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in validation_scores) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
