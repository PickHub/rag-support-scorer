from __future__ import annotations

from rag_support_scorer.data.dedup import (
    contamination_matches,
    exact_normalized_duplicates,
    minhash_signature,
    normalize_text,
)


def test_contamination_normalization() -> None:
    assert normalize_text("The Café's ANSWER!") == "caf s answer"
    assert exact_normalized_duplicates(["Who is the author?", "who is author"]) == frozenset(
        {"who is author"}
    )


def test_exact_and_minhash_style_matches() -> None:
    matches = contamination_matches(
        {
            "source-exact": "Who designed the Analytical Engine?",
            "source-near": "Where was the Analytical Engine designed and constructed?",
        },
        {
            "locked-exact": "who designed analytical engine",
            "locked-near": "Where was the Analytical Engine designed and later constructed?",
        },
        near_duplicate_threshold=0.4,
    )
    kinds = {(match.source_id, match.kind) for match in matches}
    assert ("source-exact", "exact") in kinds
    assert ("source-near", "minhash_near_duplicate") in kinds
    assert minhash_signature("stable") == minhash_signature("stable")
