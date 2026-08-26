from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

TARGETS = ("answer_free", "answer_conditioned")
SPLITS = ("train", "validation", "test")
SUPPLIED_ANSWER_MARKER = "supplied answer:"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate preference JSONL and create deterministic question-level splits."
    )
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=17)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def validate_record(
    record: dict[str, Any], path: Path, line_number: int
) -> tuple[str, str]:
    split_key = record.get("question_id", record.get("id"))
    required_strings = {
        "question": record.get("question"),
        "chosen": record.get("chosen"),
        "rejected": record.get("rejected"),
        "target": record.get("target"),
    }
    if not isinstance(split_key, str) or not split_key.strip():
        raise ValueError(f"{path}:{line_number}: id or question_id must be a string")
    for name, value in required_strings.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}:{line_number}: {name} must be a non-empty string")
    target = required_strings["target"]
    if target not in TARGETS:
        raise ValueError(
            f"{path}:{line_number}: target must be one of {', '.join(TARGETS)}"
        )
    if target == "answer_conditioned":
        supplied_answer = record.get("supplied_answer")
        if not isinstance(supplied_answer, str) or not supplied_answer.strip():
            raise ValueError(
                f"{path}:{line_number}: supplied_answer is required for answer_conditioned"
            )
    else:
        if record.get("supplied_answer") not in (None, ""):
            raise ValueError(
                f"{path}:{line_number}: answer_free rows cannot contain supplied_answer"
            )
        rendered = f"{required_strings['chosen']}\n{required_strings['rejected']}".casefold()
        if SUPPLIED_ANSWER_MARKER in rendered:
            raise ValueError(
                f"{path}:{line_number}: answer_free rows contain a supplied-answer marker"
            )
    return str(split_key), target


def assign_split(split_key: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{split_key}".encode()).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(source_data: Path, output_dir: Path, split_seed: int) -> None:
    source_files = sorted(source_data.rglob("*.jsonl"))
    if not source_files:
        raise ValueError(f"No JSONL files found below {source_data}")

    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        (target, split): (output_dir / target / f"{split}.jsonl")
        for target in TARGETS
        for split in SPLITS
    }
    for path in handles.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    streams = {key: path.open("w", encoding="utf-8") for key, path in handles.items()}
    try:
        for source_file in source_files:
            for line_number, record in iter_jsonl(source_file):
                split_key, target = validate_record(record, source_file, line_number)
                split = assign_split(split_key, split_seed)
                streams[(target, split)].write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
                counts[f"{target}/{split}"] += 1
    finally:
        for stream in streams.values():
            stream.close()

    manifest = {
        "schema_version": 1,
        "passage_bearing": True,
        "distribution_policy": "private_datastore_only",
        "split_seed": split_seed,
        "split_policy": {"train": 80, "validation": 10, "test": 10},
        "counts": dict(sorted(counts.items())),
        "sources": [
            {
                "path": str(path.relative_to(source_data)),
                "sha256": sha256(path),
            }
            for path in source_files
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    prepare(args.source_data, args.output_dir, args.split_seed)


if __name__ == "__main__":
    main()
