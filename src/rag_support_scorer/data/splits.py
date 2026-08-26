from __future__ import annotations

import hashlib
from enum import StrEnum


class Split(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


def stable_fraction(source_id: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}\0{source_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_split(
    source_id: str,
    *,
    salt: str,
    train_fraction: float = 0.8,
    dev_fraction: float = 0.1,
) -> Split:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    if not 0 <= dev_fraction < 1 or train_fraction + dev_fraction >= 1:
        raise ValueError("train and dev fractions must leave a non-empty test split")
    value = stable_fraction(source_id, salt)
    if value < train_fraction:
        return Split.TRAIN
    if value < train_fraction + dev_fraction:
        return Split.DEV
    return Split.TEST
