from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from rag_support_scorer.data.artifact_controls import lexical_overlap, surface_features
from rag_support_scorer.data.bundles import enumerate_two_context_bundles
from rag_support_scorer.data.ingest import parse_2wiki_record
from rag_support_scorer.data.interventions import (
    InterventionFamily,
    InterventionTemplate,
    infer_answer_type,
    inject_contradiction,
)
from rag_support_scorer.schemas import (
    ContextDocument,
    QuestionExample,
    ScorerKind,
    TwoContextBundle,
)
from rag_support_scorer.train.reward import render_input


def azure_split(question_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{question_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _bundle_text(
    bundle: TwoContextBundle,
    contexts: dict[str, ContextDocument],
) -> str:
    return "\n\n".join(
        f"Title: {contexts[context_id].title}\n{contexts[context_id].text}"
        for context_id in bundle.context_ids
    )


def _preference_records(
    example: QuestionExample,
    *,
    scorer_kind: ScorerKind,
    supplied_answer: str | None,
    max_negatives: int,
    split: str,
) -> list[dict[str, object]]:
    bundles = enumerate_two_context_bundles(example)
    positives = [bundle for bundle in bundles if bundle.contains_all_gold_support]
    if len(positives) != 1:
        raise ValueError("smoke data requires exactly one gold-support bundle")
    contexts = {context.source_id: context for context in example.contexts}
    positive = positives[0]
    counterfactual_ids = {
        context.source_id
        for context in example.contexts
        if ":counterfactual-" in context.source_id
    }
    negatives = sorted(
        (bundle for bundle in bundles if not bundle.contains_all_gold_support),
        key=lambda bundle: (
            not bool(counterfactual_ids & set(bundle.context_ids)),
            -lexical_overlap(example.question, _bundle_text(bundle, contexts)),
            bundle.context_ids,
        ),
    )[:max_negatives]
    target = scorer_kind.value
    records = []
    for index, negative in enumerate(negatives):
        record: dict[str, object] = {
            "id": f"{example.source_id}:{target}:{index}",
            "question_id": example.source_id,
            "question": example.question,
            "chosen": render_input(
                example.question,
                _bundle_text(positive, contexts),
                scorer_kind=scorer_kind,
                supplied_answer=supplied_answer,
            ),
            "rejected": render_input(
                example.question,
                _bundle_text(negative, contexts),
                scorer_kind=scorer_kind,
                supplied_answer=supplied_answer,
            ),
            "target": target,
            "split": split,
        }
        if supplied_answer is not None:
            record["supplied_answer"] = supplied_answer
        records.append(record)
    return records


def _wrong_answer(
    example: QuestionExample,
    candidate_answers: Iterable[str],
) -> str | None:
    gold = example.gold_answers[0]
    gold_answers = {answer.casefold() for answer in example.gold_answers}
    candidates = sorted(
        {
            answer
            for answer in candidate_answers
            if answer.casefold() not in gold_answers
            and infer_answer_type(answer) == infer_answer_type(gold)
            and len(answer.split()) == len(gold.split())
        },
        key=lambda answer: (abs(len(answer) - len(gold)), answer),
    )
    return candidates[0] if candidates else None


def conflict_enriched_examples(
    example: QuestionExample,
    wrong_answer: str,
) -> tuple[QuestionExample, QuestionExample] | None:
    gold = example.gold_answers[0]
    target = next(
        (
            context
            for context in example.contexts
            if context.source_id in example.gold_support_ids
            and gold.casefold() in context.text.casefold()
        ),
        None,
    )
    if target is None:
        return None
    result = inject_contradiction(
        target,
        gold_answer=gold,
        wrong_answer=wrong_answer,
        template=InterventionTemplate(
            "counterfactual-v1",
            InterventionFamily.CONTRADICTION_INJECTION,
        ),
    )
    distractors = sorted(
        (
            context
            for context in example.contexts
            if context.source_id not in example.gold_support_ids
        ),
        key=lambda context: (
            lexical_overlap(example.question, context.text),
            context.source_id,
        ),
    )
    if not distractors:
        return None
    removed = distractors[0]
    modified = result.contexts[0].model_copy(update={"position": removed.position})
    contexts = tuple(
        modified if context.source_id == removed.source_id else context
        for context in example.contexts
    )
    enriched = example.model_copy(update={"contexts": contexts})
    wrong_support_ids = frozenset(
        modified.source_id if source_id == target.source_id else source_id
        for source_id in example.gold_support_ids
    )
    wrong_labeled = enriched.model_copy(
        update={
            "gold_answers": (wrong_answer,),
            "gold_support_ids": wrong_support_ids,
        }
    )
    return enriched, wrong_labeled


def build_records(
    examples: Sequence[QuestionExample],
    *,
    limit: int,
    max_negatives: int,
    split_seed: int,
    explicit_split_counts: tuple[int, int, int] | None = None,
) -> tuple[
    list[dict[str, object]],
    list[QuestionExample],
    dict[str, str],
    dict[str, float],
    list[dict[str, object]],
]:
    answers = tuple(example.gold_answers[0] for example in examples)
    candidates = []
    for example in examples:
        if not enumerate_two_context_bundles(example):
            continue
        wrong = _wrong_answer(example, answers)
        if wrong is None:
            continue
        conflict_examples = conflict_enriched_examples(example, wrong)
        if conflict_examples is None:
            continue
        enriched, wrong_labeled = conflict_examples
        key = hashlib.sha256(f"{split_seed}:{example.source_id}".encode()).hexdigest()
        candidates.append((key, enriched, wrong, wrong_labeled))
    selected = sorted(candidates, key=lambda item: item[0])[:limit]
    records: list[dict[str, object]] = []
    test_examples = []
    wrong_answers = {}
    contradiction_scores = {}
    artifact_rows = []
    for index, (_, example, wrong, wrong_labeled) in enumerate(selected):
        if explicit_split_counts is None:
            split = azure_split(example.source_id, split_seed)
        else:
            train_count, validation_count, _ = explicit_split_counts
            split = (
                "train"
                if index < train_count
                else "validation"
                if index < train_count + validation_count
                else "test"
            )
        records.extend(
            _preference_records(
                example,
                scorer_kind=ScorerKind.ANSWER_FREE,
                supplied_answer=None,
                max_negatives=max_negatives,
                split=split,
            )
        )
        counterfactual = next(
            context
            for context in example.contexts
            if ":counterfactual-" in context.source_id
        )
        original = next(
            context
            for context in example.contexts
            if context.title == counterfactual.title
            and ":counterfactual-" not in context.source_id
        )
        for context, label in ((original, 0), (counterfactual, 1)):
            artifact_rows.append(
                {
                    "question_id": example.source_id,
                    "split": split,
                    "label": label,
                    "features": surface_features(
                        context.text,
                        question=example.question,
                    ).as_vector(),
                }
            )
        records.extend(
            _preference_records(
                example,
                scorer_kind=ScorerKind.ANSWER_CONDITIONED,
                supplied_answer=example.gold_answers[0],
                max_negatives=max_negatives,
                split=split,
            )
        )
        records.extend(
            _preference_records(
                wrong_labeled,
                scorer_kind=ScorerKind.ANSWER_CONDITIONED,
                supplied_answer=wrong,
                max_negatives=max_negatives,
                split=split,
            )
        )
        if split == "test":
            test_examples.append(example)
            wrong_answers[example.source_id] = wrong
            contradiction_scores[f"{example.source_id}:correct"] = 0.5
            contradiction_scores[f"{example.source_id}:plausible_wrong"] = 0.5
    return records, test_examples, wrong_answers, contradiction_scores, artifact_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scan-limit", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--max-negatives", type=int, default=4)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--train-questions", type=int)
    parser.add_argument("--validation-questions", type=int)
    parser.add_argument("--test-questions", type=int)
    args = parser.parse_args()
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("install the train extra to read Parquet data") from error
    dataset = load_dataset(
        "parquet",
        data_files={"source": str(args.input)},
        split=f"source[:{args.scan_limit}]",
    )
    examples = tuple(parse_2wiki_record(row) for row in dataset)
    requested_counts = (
        args.train_questions,
        args.validation_questions,
        args.test_questions,
    )
    explicit_split_counts = None
    if any(value is not None for value in requested_counts):
        if not all(value is not None and value > 0 for value in requested_counts):
            raise ValueError("all explicit split counts must be positive")
        explicit_split_counts = (
            int(requested_counts[0]),
            int(requested_counts[1]),
            int(requested_counts[2]),
        )
        args.limit = sum(explicit_split_counts)
    (
        records,
        test_examples,
        wrong_answers,
        contradiction_scores,
        artifact_rows,
    ) = build_records(
        examples,
        limit=args.limit,
        max_negatives=args.max_negatives,
        split_seed=args.split_seed,
        explicit_split_counts=explicit_split_counts,
    )
    preferences_dir = args.output_dir / "preferences"
    experiment_dir = args.output_dir / "experiment"
    preferences_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (preferences_dir / "preferences.jsonl").write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n"
    )
    (experiment_dir / "questions.jsonl").write_text(
        "\n".join(example.model_dump_json() for example in test_examples) + "\n"
    )
    (experiment_dir / "wrong_answers.json").write_text(
        json.dumps(wrong_answers, indent=2, sort_keys=True) + "\n"
    )
    (experiment_dir / "contradiction_scores.json").write_text(
        json.dumps(contradiction_scores, indent=2, sort_keys=True) + "\n"
    )
    (experiment_dir / "artifact_probe.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in artifact_rows) + "\n"
    )
    manifest = {
        "scan_limit": args.scan_limit,
        "question_limit": args.limit,
        "max_negatives": args.max_negatives,
        "split_seed": args.split_seed,
        "explicit_split_counts": explicit_split_counts,
        "preference_records": len(records),
        "test_questions": len(test_examples),
        "contradiction_scores": "constant smoke-only baseline; not a research result",
        "candidate_pool": "fixed original gold plus matched counterfactual support",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
