from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from extensions.dpo_post_training.best_of_n import read_records, select_candidate
from extensions.dpo_post_training.train_dpo import (
    load_preference_records,
    validate_preference_lengths,
)

EXTENSION_ROOT = Path(__file__).resolve().parents[1]


class OfflineConfigurationTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    def test_tiny_preferences_validate_without_training_imports(self) -> None:
        preferences = EXTENSION_ROOT / "fixtures" / "tiny_preferences.jsonl"
        records = load_preference_records(preferences)
        self.assertEqual(len(records), 2)

        result = subprocess.run(
            [
                sys.executable,
                str(EXTENSION_ROOT / "train_dpo.py"),
                "--train-data",
                str(preferences),
                "--model-name",
                "offline/tiny-causal-lm",
                "--model-revision",
                "0000000000000000000000000000000000000000",
                "--output-dir",
                str(EXTENSION_ROOT / ".offline-test-output"),
                "--max-length",
                "128",
                "--learning-rate",
                "5e-6",
                "--validate-only",
            ],
            check=True,
            capture_output=True,
            env=os.environ,
            text=True,
        )
        self.assertIn("Validated 2 training", result.stdout)
        self.assertFalse((EXTENSION_ROOT / ".offline-test-output").exists())

    def test_tiny_best_of_n_is_deterministic(self) -> None:
        candidates = EXTENSION_ROOT / "fixtures" / "tiny_candidates.jsonl"
        for line_number, record in read_records(candidates):
            first = select_candidate(record, 3, f"{candidates}:{line_number}")
            second = select_candidate(record, 3, f"{candidates}:{line_number}")
            self.assertEqual(first, second)

    def test_over_length_preferences_fail_before_dpo_truncation(self) -> None:
        class Tokenizer:
            def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
                return {"input_ids": list(range(len(text.split())))}

        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            validate_preference_lengths(
                [{"prompt": "one two", "chosen": "three four", "rejected": "three"}],
                Tokenizer(),
                2,
            )


if __name__ == "__main__":
    unittest.main()
