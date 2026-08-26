from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_azure_preparation_rejects_answer_leakage(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "pairs.jsonl").write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question": "Question?",
                "chosen": "Supplied answer: leaked\nContext bundle: chosen",
                "rejected": "Context bundle: rejected",
                "target": "answer_free",
                "supplied_answer": "leaked",
            }
        )
        + "\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "azureml/scripts/prepare_data.py",
            "--source-data",
            str(source),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "answer_free rows cannot contain supplied_answer" in result.stderr
