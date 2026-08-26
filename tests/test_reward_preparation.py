from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from rag_support_scorer.schemas import ScorerKind
from rag_support_scorer.train.reward import RewardTrainingConfig, prepare_tokenized_rows


@dataclass(frozen=True)
class _WhitespaceTokenizer:
    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        max_length: int,
        add_special_tokens: bool,
    ) -> Mapping[str, Sequence[int]]:
        token_ids = list(range(len(text.split()) + int(add_special_tokens)))
        if truncation:
            token_ids = token_ids[:max_length]
        return {"input_ids": token_ids}


def test_reward_pairs_fail_before_labeled_evidence_is_truncated() -> None:
    rows = [
        {"chosen": " ".join(["chosen"] * 20), "rejected": "short rejected"},
        {"chosen": "short chosen", "rejected": " ".join(["rejected"] * 30)},
    ]
    with pytest.raises(ValueError, match="exceeds max_sequence_length"):
        prepare_tokenized_rows(rows, _WhitespaceTokenizer(), max_length=8)


def test_reward_config_requires_immutable_revisions() -> None:
    with pytest.raises(ValueError, match="immutable 40-character commit SHA"):
        RewardTrainingConfig(
            base_model="Qwen/Qwen3-0.6B",
            model_revision="${SCORER_MODEL_REVISION}",
            tokenizer_revision="main",
            scorer_kind=ScorerKind.ANSWER_FREE,
            dataset_path=Path("pairs.jsonl"),
            output_dir=Path("output"),
        )
