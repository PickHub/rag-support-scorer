from __future__ import annotations

import json
from pathlib import Path

from rag_support_scorer.data.ingest import (
    TWO_WIKI_PARQUET_URLS,
    TWO_WIKI_REVISION,
    load_2wiki_examples,
    metadata_overlap_report,
    parse_2wiki_record,
    prepare_manifest,
)


def test_2wiki_ingestion_and_passage_free_manifest(tmp_path: Path) -> None:
    source = tmp_path / "train.json"
    source.write_text(
        json.dumps(
            [
                {
                    "_id": "abc",
                    "question": "Who designed it?",
                    "answer": "Charles Babbage",
                    "type": "bridge",
                    "context": [
                        ["Engine", ["It was designed by Charles Babbage."]],
                        ["Person", ["Ada wrote notes."]],
                        ["Other", ["Unrelated passage."]],
                    ],
                    "supporting_facts": [["Engine", 0], ["Person", 0]],
                }
            ]
        )
    )
    examples = load_2wiki_examples(source)
    assert examples[0].gold_support_ids == frozenset(
        {examples[0].contexts[0].source_id, examples[0].contexts[1].source_id}
    )
    manifest, summary = prepare_manifest(
        examples,
        source_path=source,
        dataset_revision="immutable-revision",
        split_salt="test",
    )
    serialized = json.dumps(manifest)
    assert "It was designed" not in serialized
    assert summary.included == 1
    assert manifest["source_sha256"]
    assert metadata_overlap_report(manifest, manifest) == {
        "context_sha256_overlap": 3,
        "title_sha256_overlap": 3,
        "entity_sha256_overlap": 7,
    }


def test_pinned_parquet_mapping_avoids_loading_script() -> None:
    assert set(TWO_WIKI_PARQUET_URLS) == {"train", "validation"}
    assert all(TWO_WIKI_REVISION in url for url in TWO_WIKI_PARQUET_URLS.values())
    assert all(url.endswith(".parquet") for url in TWO_WIKI_PARQUET_URLS.values())


def test_parquet_json_string_fields_are_decoded() -> None:
    example = parse_2wiki_record(
        {
            "_id": "encoded",
            "question": "Who designed it?",
            "answer": "Charles Babbage",
            "context": json.dumps(
                [
                    ["Engine", ["It was designed by Charles Babbage."]],
                    ["Person", ["Ada Lovelace wrote notes."]],
                ]
            ),
            "supporting_facts": json.dumps([["Engine", 0], ["Person", 0]]),
        }
    )
    assert len(example.contexts) == 2
    assert "Charles Babbage" in example.contexts[0].entities
    assert len(example.gold_support_ids) == 2


def test_normalized_duplicate_questions_share_a_split(tmp_path: Path) -> None:
    records = [
        {
            "_id": identifier,
            "question": question,
            "answer": "Charles Babbage",
            "type": "bridge",
            "context": [
                ["Engine", ["It was designed by Charles Babbage."]],
                ["Person", ["Ada wrote notes."]],
                ["Other", ["Unrelated passage."]],
            ],
            "supporting_facts": [["Engine", 0], ["Person", 0]],
        }
        for identifier, question in (
            ("abc", "Who designed it?"),
            ("def", "WHO designed it??"),
        )
    ]
    raw = tmp_path / "records.json"
    raw.write_text(json.dumps(records))
    examples = load_2wiki_examples(raw)
    manifest, _ = prepare_manifest(
        examples,
        source_path=raw,
        dataset_revision="immutable-revision",
        split_salt="test",
    )
    assert len({record["split"] for record in manifest["records"]}) == 1
    assert len({record["split_group_sha256"] for record in manifest["records"]}) == 1


def test_identical_support_sets_cannot_cross_splits(tmp_path: Path) -> None:
    first_support = "The first shared passage names Charles Babbage."
    second_support = "The second shared passage describes the machine."
    records = [
        {
            "_id": identifier,
            "question": question,
            "answer": "Charles Babbage",
            "type": "bridge",
            "context": [
                ["First shared", [first_support]],
                ["Second shared", [second_support]],
                ["Other", ["Unrelated passage."]],
            ],
            "supporting_facts": [["First shared", 0], ["Second shared", 0]],
        }
        for identifier, question in (
            ("abc", "Who designed the first machine?"),
            ("def", "Who designed the second machine?"),
        )
    ]
    raw = tmp_path / "records.json"
    raw.write_text(json.dumps(records))
    manifest, _ = prepare_manifest(
        load_2wiki_examples(raw),
        source_path=raw,
        dataset_revision="immutable-revision",
        split_salt="test",
    )
    assert len({record["split"] for record in manifest["records"]}) == 1
