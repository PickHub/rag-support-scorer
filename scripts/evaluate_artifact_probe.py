from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def evaluate(input_path: Path) -> dict[str, float | int]:
    rows = [
        json.loads(line)
        for line in input_path.read_text().splitlines()
        if line.strip()
    ]
    train = [row for row in rows if row["split"] in {"train", "validation"}]
    test = [row for row in rows if row["split"] == "test"]
    if not train or not test:
        raise ValueError("artifact probe requires train/validation and test rows")
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", random_state=17)),
        ]
    )
    pipeline.fit(
        np.asarray([row["features"] for row in train], dtype=np.float64),
        np.asarray([row["label"] for row in train], dtype=np.int64),
    )
    test_features = np.asarray([row["features"] for row in test], dtype=np.float64)
    labels = np.asarray([row["label"] for row in test], dtype=np.int64)
    probabilities = pipeline.predict_proba(test_features)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
