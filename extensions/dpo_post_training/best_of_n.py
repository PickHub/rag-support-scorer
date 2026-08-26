from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a deterministic best answer from fixed candidates."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    return parser.parse_args()


def read_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def select_candidate(
    record: dict[str, Any], n: int, location: str
) -> tuple[int, str, float]:
    candidates = record.get("candidates")
    scores = record.get("scores")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{location}: candidates must be a non-empty list")
    if not isinstance(scores, list) or len(scores) != len(candidates):
        raise ValueError(f"{location}: scores must match candidates")
    if not all(isinstance(candidate, str) for candidate in candidates):
        raise ValueError(f"{location}: every candidate must be a string")
    if not all(
        isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(float(score))
        for score in scores
    ):
        raise ValueError(f"{location}: every score must be finite and numeric")
    if n < 1 or n > len(candidates):
        raise ValueError(f"{location}: n must be between 1 and {len(candidates)}")

    ranked = (
        (index, candidates[index], float(scores[index])) for index in range(n)
    )
    return min(ranked, key=lambda item: (-item[2], item[1], item[0]))


def run(input_path: Path, output_path: Path, n: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for line_number, record in read_records(input_path):
            index, candidate, score = select_candidate(
                record, n, f"{input_path}:{line_number}"
            )
            result = {
                "id": record.get("id", line_number),
                "n": n,
                "selected": candidate,
                "selected_index": index,
                "selected_score": score,
            }
            output.write(
                json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
            )


def main() -> None:
    args = parse_args()
    run(args.input, args.output, args.n)


if __name__ == "__main__":
    main()
