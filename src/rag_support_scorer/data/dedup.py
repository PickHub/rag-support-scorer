from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = _NON_ALNUM.sub(" ", normalized)
    normalized = _ARTICLES.sub(" ", normalized)
    return _WHITESPACE.sub(" ", normalized).strip()


def character_shingles(text: str, width: int = 5) -> frozenset[str]:
    normalized = normalize_text(text).replace(" ", "_")
    if not normalized:
        return frozenset()
    if len(normalized) <= width:
        return frozenset({normalized})
    return frozenset(
        normalized[index : index + width]
        for index in range(len(normalized) - width + 1)
    )


def minhash_signature(text: str, *, permutations: int = 64) -> tuple[int, ...]:
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    shingles = character_shingles(text)
    if not shingles:
        return tuple(0 for _ in range(permutations))
    signature = []
    for seed in range(permutations):
        minimum = min(
            int.from_bytes(
                hashlib.blake2b(
                    shingle.encode(),
                    digest_size=8,
                    person=seed.to_bytes(8, "big"),
                ).digest(),
                "big",
            )
            for shingle in shingles
        )
        signature.append(minimum)
    return tuple(signature)


def estimated_jaccard(first: tuple[int, ...], second: tuple[int, ...]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("signatures must be non-empty and have equal length")
    return sum(left == right for left, right in zip(first, second, strict=True)) / len(first)


@dataclass(frozen=True)
class ContaminationMatch:
    source_id: str
    comparison_id: str
    kind: str
    score: float


def contamination_matches(
    source_questions: Mapping[str, str],
    comparison_questions: Mapping[str, str],
    *,
    near_duplicate_threshold: float = 0.85,
    permutations: int = 64,
) -> tuple[ContaminationMatch, ...]:
    if not 0 <= near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be in [0, 1]")
    normalized_comparison: dict[str, list[str]] = {}
    comparison_signatures = {}
    for comparison_id, question in comparison_questions.items():
        normalized_comparison.setdefault(normalize_text(question), []).append(comparison_id)
        comparison_signatures[comparison_id] = minhash_signature(
            question, permutations=permutations
        )
    matches = []
    for source_id, question in source_questions.items():
        normalized = normalize_text(question)
        exact_ids = normalized_comparison.get(normalized, [])
        for comparison_id in exact_ids:
            matches.append(ContaminationMatch(source_id, comparison_id, "exact", 1.0))
        if exact_ids:
            continue
        signature = minhash_signature(question, permutations=permutations)
        for comparison_id, comparison_signature in comparison_signatures.items():
            score = estimated_jaccard(signature, comparison_signature)
            if score >= near_duplicate_threshold:
                matches.append(
                    ContaminationMatch(source_id, comparison_id, "minhash_near_duplicate", score)
                )
    return tuple(sorted(matches, key=lambda match: (match.source_id, match.comparison_id)))


def exact_normalized_duplicates(values: Iterable[str]) -> frozenset[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        normalized = normalize_text(value)
        if normalized in seen:
            duplicates.add(normalized)
        seen.add(normalized)
    return frozenset(duplicates)
