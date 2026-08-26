from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from rag_support_scorer.schemas import ContextDocument, ReaderOutput


class ReaderAdapter(Protocol):
    name: str

    def generate(
        self,
        question: str,
        contexts: tuple[ContextDocument, ContextDocument],
        *,
        seed: int,
    ) -> ReaderOutput: ...


class DraftAdapter(Protocol):
    name: str

    def generate_draft(
        self,
        question: str,
        candidate_contexts: Sequence[ContextDocument],
        *,
        seed: int,
    ) -> ReaderOutput: ...


@dataclass
class DeterministicMockReader:
    answers: Mapping[str, str]
    name: str = "deterministic_mock_reader"
    calls: list[tuple[str, tuple[str, str], int]] | None = None

    def generate(
        self,
        question: str,
        contexts: tuple[ContextDocument, ContextDocument],
        *,
        seed: int,
    ) -> ReaderOutput:
        if self.calls is not None:
            self.calls.append((question, (contexts[0].source_id, contexts[1].source_id), seed))
        return ReaderOutput(answer=self.answers[question], metadata={"adapter": self.name})


@dataclass(frozen=True)
class DeterministicMockDraft:
    drafts: Mapping[str, str]
    name: str = "deterministic_mock_draft"

    def generate_draft(
        self,
        question: str,
        candidate_contexts: Sequence[ContextDocument],
        *,
        seed: int,
    ) -> ReaderOutput:
        context_hash = hashlib.sha256(
            "\0".join(context.source_id for context in candidate_contexts).encode()
        ).hexdigest()
        return ReaderOutput(
            answer=self.drafts[question],
            metadata={"adapter": self.name, "candidate_pool_sha256": context_hash, "seed": seed},
        )
