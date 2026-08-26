from __future__ import annotations

import argparse
import importlib
import json
import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone 4-bit QLoRA DPO.")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss-type", nargs="+", default=["sigmoid"])
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(arguments)


def validate_model_revision(revision: str) -> None:
    if not IMMUTABLE_REVISION.fullmatch(revision):
        raise ValueError("model revision must be an immutable 40-character commit SHA")


def load_preference_records(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value: Any = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            record: dict[str, str] = {}
            for field in ("prompt", "chosen", "rejected"):
                field_value = value.get(field)
                if not isinstance(field_value, str) or not field_value.strip():
                    raise ValueError(
                        f"{path}:{line_number}: {field} must be a non-empty string"
                    )
                record[field] = field_value
            records.append(record)
    if not records:
        raise ValueError(f"No DPO preferences found in {path}")
    return records


def load_training_stack() -> dict[str, Any]:
    try:
        importlib.import_module("bitsandbytes")
    except ImportError as error:
        raise RuntimeError(
            "4-bit DPO requires the optional bitsandbytes dependency"
        ) from error

    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import DPOConfig, DPOTrainer

    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "DPOConfig": DPOConfig,
        "DPOTrainer": DPOTrainer,
    }


def train(
    args: argparse.Namespace,
    train_records: list[dict[str, str]],
    eval_records: list[dict[str, str]] | None,
) -> None:
    stack = load_training_stack()
    torch = stack["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for 4-bit QLoRA DPO")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    major_version, _ = torch.cuda.get_device_capability(0)
    compute_dtype = torch.bfloat16 if major_version >= 8 else torch.float16
    tokenizer = stack["AutoTokenizer"].from_pretrained(
        args.model_name, revision=args.model_revision, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = stack["AutoModelForCausalLM"].from_pretrained(
        args.model_name,
        revision=args.model_revision,
        quantization_config=stack["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
    )
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model = stack["prepare_model_for_kbit_training"](
        model, use_gradient_checkpointing=True
    )
    train_dataset = stack["Dataset"].from_list(train_records)
    eval_dataset = (
        stack["Dataset"].from_list(eval_records) if eval_records is not None else None
    )
    training_args = stack["DPOConfig"](
        output_dir=str(args.output_dir),
        seed=args.seed,
        data_seed=args.seed,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        beta=args.beta,
        loss_type=list(args.loss_type),
        max_length=args.max_length,
        truncation_mode="keep_start",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = stack["DPOTrainer"](
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=stack["LoraConfig"](
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
            revision=args.model_revision,
        ),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)


def main() -> None:
    args = parse_args()
    validate_model_revision(args.model_revision)
    train_records = load_preference_records(args.train_data)
    eval_records = (
        load_preference_records(args.eval_data) if args.eval_data is not None else None
    )
    if args.validate_only:
        evaluation_count = len(eval_records) if eval_records is not None else 0
        print(
            f"Validated {len(train_records)} training and "
            f"{evaluation_count} evaluation preferences"
        )
        return
    train(args, train_records, eval_records)


if __name__ == "__main__":
    main()
