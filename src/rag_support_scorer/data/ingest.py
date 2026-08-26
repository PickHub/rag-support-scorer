from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_support_scorer.data.bundles import bundle_eligibility
from rag_support_scorer.data.dedup import (
    ContaminationMatch,
    contamination_matches,
    normalize_text,
)
from rag_support_scorer.data.splits import assign_split
from rag_support_scorer.schemas import ContextDocument, QuestionExample

TWO_WIKI_REVISION = "612bc5039a457880d9e7d84c3b0a4cf154b70e4f"
TWO_WIKI_PARQUET_URLS = {
    "train": (
        "hf://datasets/xanhho/2WikiMultihopQA@"
        f"{TWO_WIKI_REVISION}/train.parquet"
    ),
    "validation": (
        "hf://datasets/xanhho/2WikiMultihopQA@"
        f"{TWO_WIKI_REVISION}/dev.parquet"
    ),
}
_ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9'-]*(?:\s+[A-Z][A-Za-z0-9'-]*)*\b")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_id(question_id: str, position: int, title: str) -> str:
    digest = hashlib.sha256(f"{question_id}\0{position}\0{title}".encode()).hexdigest()[:16]
    return f"2wiki:{question_id}:context:{position}:{digest}"


def _decode_structured(value: Any, field: str) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{field} contains invalid JSON") from error
    return value


def _support_map(raw: Mapping[str, Any]) -> dict[str, set[int]]:
    support: dict[str, set[int]] = {}
    facts = _decode_structured(raw.get("supporting_facts", []), "supporting_facts")
    for fact in facts:
        if isinstance(fact, Sequence) and len(fact) >= 2:
            support.setdefault(str(fact[0]), set()).add(int(fact[1]))
    return support


def parse_2wiki_record(raw: Mapping[str, Any]) -> QuestionExample:
    question_id = str(raw.get("_id") or raw.get("id") or "")
    if not question_id:
        raise ValueError("2Wiki record is missing an id")
    support = _support_map(raw)
    context_values = _decode_structured(raw.get("context", []), "context")
    contexts = []
    for position, context_value in enumerate(context_values):
        if not isinstance(context_value, Sequence) or len(context_value) != 2:
            raise ValueError(f"invalid context at position {position}")
        title = str(context_value[0])
        sentences = tuple(str(sentence) for sentence in context_value[1])
        support_sentences = tuple(
            sentences[index]
            for index in sorted(support.get(title, set()))
            if index < len(sentences)
        )
        contexts.append(
            ContextDocument(
                source_id=_source_id(question_id, position, title),
                title=title,
                text=" ".join(sentences),
                position=position,
                entities=tuple(
                    sorted(
                        {
                            title,
                            *_ENTITY_PATTERN.findall(" ".join(sentences)),
                        }
                    )
                ),
                supporting_sentences=support_sentences,
            )
        )
    gold_titles = set(support)
    gold_ids = frozenset(
        context.source_id for context in contexts if context.title in gold_titles
    )
    answers_value = raw.get("answer")
    answers = (
        (str(answers_value),)
        if not isinstance(answers_value, list)
        else tuple(map(str, answers_value))
    )
    return QuestionExample(
        source_id=f"2wiki:{question_id}",
        question=str(raw["question"]),
        gold_answers=answers,
        contexts=tuple(contexts[:10]),
        gold_support_ids=gold_ids,
        question_type=str(raw["type"]) if raw.get("type") is not None else None,
    )


def load_2wiki_examples(path: Path) -> tuple[QuestionExample, ...]:
    raw = json.loads(path.read_text())
    records: Iterable[Mapping[str, Any]]
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict) and isinstance(raw.get("data"), list):
        records = raw["data"]
    else:
        raise ValueError("expected a JSON list or an object with a data list")
    return tuple(parse_2wiki_record(record) for record in records)


def load_2wiki_parquet(
    data_files: Mapping[str, str] | None = None,
) -> dict[str, tuple[QuestionExample, ...]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("install the train extra to load Parquet datasets") from error
    files = dict(data_files or TWO_WIKI_PARQUET_URLS)
    if set(files) != {"train", "validation"}:
        raise ValueError("2Wiki Parquet mapping must contain train and validation only")
    dataset = load_dataset("parquet", data_files=files)
    return {
        split: tuple(parse_2wiki_record(record) for record in dataset[split])
        for split in ("train", "validation")
    }


def load_2wiki_parquet_file(path: Path) -> tuple[QuestionExample, ...]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("install the train extra to load Parquet datasets") from error
    dataset = load_dataset("parquet", data_files={"source": str(path)})
    return tuple(parse_2wiki_record(record) for record in dataset["source"])


@dataclass(frozen=True)
class PreparationSummary:
    source_sha256: str
    included: int
    excluded: int
    split_counts: dict[str, int]


def prepare_manifest(
    examples: Iterable[QuestionExample],
    *,
    source_path: Path,
    dataset_revision: str,
    split_salt: str,
    excluded_source_ids: frozenset[str] = frozenset(),
) -> tuple[dict[str, Any], PreparationSummary]:
    records = []
    excluded = []
    split_counts = {"train": 0, "dev": 0, "test": 0}
    for example in sorted(examples, key=lambda item: item.source_id):
        if example.source_id in excluded_source_ids:
            excluded.append({"source_id": example.source_id, "reason": "contamination_match"})
            continue
        eligibility = bundle_eligibility(example)
        if not eligibility.eligible:
            assert eligibility.reason is not None
            excluded.append({"source_id": example.source_id, "reason": eligibility.reason})
            continue
        split_group = normalize_text(example.question)
        split = assign_split(split_group, salt=split_salt)
        split_counts[split.value] += 1
        records.append(
            {
                "source_id": example.source_id,
                "question_sha256": hashlib.sha256(example.question.encode()).hexdigest(),
                "split_group_sha256": hashlib.sha256(split_group.encode()).hexdigest(),
                "context_ids": [context.source_id for context in example.contexts],
                "context_sha256": [
                    hashlib.sha256(context.text.encode()).hexdigest()
                    for context in example.contexts
                ],
                "title_sha256": [
                    hashlib.sha256(context.title.encode()).hexdigest()
                    for context in example.contexts
                ],
                "entity_sha256": sorted(
                    {
                        hashlib.sha256(entity.encode()).hexdigest()
                        for context in example.contexts
                        for entity in context.entities
                    }
                ),
                "gold_support_ids": sorted(example.gold_support_ids),
                "split": split.value,
            }
        )
    source_sha256 = sha256_file(source_path)
    manifest: dict[str, Any] = {
        "dataset": "2WikiMultiHopQA",
        "dataset_revision": dataset_revision,
        "source_filename": source_path.name,
        "source_sha256": source_sha256,
        "split_salt": split_salt,
        "records": records,
        "exclusions": excluded,
    }
    return manifest, PreparationSummary(
        source_sha256=source_sha256,
        included=len(records),
        excluded=len(excluded),
        split_counts=split_counts,
    )


def metadata_overlap_report(
    source_manifest: Mapping[str, Any],
    comparison_manifest: Mapping[str, Any],
) -> dict[str, int]:
    fields = ("context_sha256", "title_sha256", "entity_sha256")
    report = {}
    for field in fields:
        source_values = {
            value
            for record in source_manifest.get("records", [])
            for value in record.get(field, [])
        }
        comparison_values = {
            value
            for record in comparison_manifest.get("records", [])
            for value in record.get(field, [])
        }
        report[f"{field}_overlap"] = len(source_values & comparison_values)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--split-salt", default="rag-support-scorer-v1")
    parser.add_argument("--locked-questions", type=Path)
    parser.add_argument("--comparison-manifest", type=Path)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.85)
    args = parser.parse_args()
    examples = (
        load_2wiki_parquet_file(args.input)
        if args.input.suffix == ".parquet"
        else load_2wiki_examples(args.input)
    )
    matches: tuple[ContaminationMatch, ...] = ()
    if args.locked_questions:
        locked_questions: dict[str, str] = json.loads(args.locked_questions.read_text())
        matches = contamination_matches(
            {example.source_id: example.question for example in examples},
            locked_questions,
            near_duplicate_threshold=args.near_duplicate_threshold,
        )
    manifest, summary = prepare_manifest(
        examples,
        source_path=args.input,
        dataset_revision=args.dataset_revision,
        split_salt=args.split_salt,
        excluded_source_ids=frozenset(match.source_id for match in matches),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "preparation_summary.json").write_text(
        json.dumps(summary.__dict__, indent=2, sort_keys=True) + "\n"
    )
    overlap = {}
    if args.comparison_manifest:
        overlap = metadata_overlap_report(
            manifest,
            json.loads(args.comparison_manifest.read_text()),
        )
    contamination_report = {
        "near_duplicate_threshold": args.near_duplicate_threshold,
        "matches": [match.__dict__ for match in matches],
        "metadata_overlap": overlap,
        "base_model_pretraining_contamination": "unknowable",
    }
    (args.output_dir / "contamination_report.json").write_text(
        json.dumps(contamination_report, indent=2, sort_keys=True) + "\n"
    )
