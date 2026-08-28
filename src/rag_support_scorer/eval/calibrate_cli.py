from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_support_scorer.eval.calibration import PlattScaler


def fit_platt_file(input_path: Path, output_path: Path) -> None:
    rows = [
        json.loads(line)
        for line in input_path.read_text().splitlines()
        if line.strip()
    ]
    scores = [float(row["score"]) for row in rows]
    labels = [int(row["label"]) for row in rows]
    scaler = PlattScaler().fit(scores, labels)
    destination = (
        output_path / "calibration.json"
        if output_path.suffix.casefold() != ".json"
        else output_path
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"scale": scaler.scale, "bias": scaler.bias}, indent=2, sort_keys=True)
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit_platt_file(args.input, args.output)
