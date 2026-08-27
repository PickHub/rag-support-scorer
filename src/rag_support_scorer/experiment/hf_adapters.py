from __future__ import annotations

import importlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rag_support_scorer.schemas import ContextDocument, ReaderOutput, ScorerKind
from rag_support_scorer.train.reward import render_input

_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")


def _require_immutable_revision(revision: str) -> None:
    if not _IMMUTABLE_REVISION.fullmatch(revision):
        raise ValueError("model revisions must be immutable 40-character commit SHAs")


def render_reader_prompt(
    question: str,
    contexts: tuple[ContextDocument, ContextDocument],
) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Context 1 ({contexts[0].title}):\n{contexts[0].text}\n\n"
        f"Context 2 ({contexts[1].title}):\n{contexts[1].text}\n\n"
        "Answer using only the contexts. Return only the shortest answer span, "
        "without a label, sentence, explanation, or citation."
    )


def normalize_reader_answer(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    answer = lines[0]
    for prefix in ("answer:", "response:"):
        if answer.casefold().startswith(prefix):
            answer = answer[len(prefix) :].strip()
    return answer


@dataclass
class TransformersScorerAdapter:
    checkpoint_path: str
    tokenizer_model: str
    tokenizer_revision: str
    scorer_kind: ScorerKind
    max_sequence_length: int = 2048
    name: str = "transformers_scorer"
    _model: Any = field(init=False, repr=False)
    _tokenizer: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _require_immutable_revision(self.tokenizer_revision)
        try:
            peft = importlib.import_module("peft")
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise RuntimeError("install the train extra for scorer inference") from error
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.tokenizer_model,
            revision=self.tokenizer_revision,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        base_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.tokenizer_model,
            revision=self.tokenizer_revision,
            num_labels=1,
            dtype="auto",
            device_map="auto",
        )
        self._model = peft.PeftModel.from_pretrained(
            base_model,
            self.checkpoint_path,
        )
        self._model.config.pad_token_id = self._tokenizer.pad_token_id
        self._model.eval()

    def score(
        self,
        question: str,
        contexts: tuple[ContextDocument, ContextDocument],
        supplied_answer: str | None,
    ) -> float:
        return self.score_many(
            question,
            (contexts,),
            supplied_answer,
            batch_size=1,
        )[0]

    def score_many(
        self,
        question: str,
        context_bundles: Sequence[tuple[ContextDocument, ContextDocument]],
        supplied_answer: str | None,
        *,
        batch_size: int = 8,
    ) -> tuple[float, ...]:
        rendered = [
            render_input(
                question,
                "\n\n".join(context.text for context in contexts),
                scorer_kind=self.scorer_kind,
                supplied_answer=supplied_answer,
            )
            for contexts in context_bundles
        ]
        scores: list[float] = []
        torch = importlib.import_module("torch")
        device = next(self._model.parameters()).device
        for offset in range(0, len(rendered), batch_size):
            encoded = self._tokenizer(
                rendered[offset : offset + batch_size],
                return_tensors="pt",
                truncation=False,
                padding=True,
            )
            if encoded["input_ids"].shape[-1] > self.max_sequence_length:
                raise ValueError("scorer input exceeds the locked sequence budget")
            encoded = {name: value.to(device) for name, value in encoded.items()}
            with torch.no_grad():
                scores.extend(
                    float(value)
                    for value in self._model(**encoded).logits.reshape(-1).tolist()
                )
        return tuple(scores)


@dataclass
class TransformersNLIAdapter:
    model_name: str
    model_revision: str
    max_sequence_length: int = 512
    _model: Any = field(init=False, repr=False)
    _tokenizer: Any = field(init=False, repr=False)
    _contradiction_index: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _require_immutable_revision(self.model_revision)
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise RuntimeError("install the train extra for NLI inference") from error
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.model_revision,
        )
        self._model = transformers.AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            revision=self.model_revision,
            dtype="auto",
            device_map="auto",
        )
        labels = {
            int(index): str(label).casefold()
            for index, label in self._model.config.id2label.items()
        }
        try:
            self._contradiction_index = next(
                index for index, label in labels.items() if "contradiction" in label
            )
        except StopIteration as error:
            raise ValueError("NLI model has no contradiction label") from error
        self._model.eval()

    def contradiction_score(
        self,
        question: str,
        contexts: tuple[ContextDocument, ContextDocument],
        supplied_answer: str,
    ) -> float:
        premise = "\n\n".join(
            f"{context.title}: {context.text}" for context in contexts
        )
        hypothesis = f'The answer to "{question}" is "{supplied_answer}".'
        encoded = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation="only_first",
            max_length=self.max_sequence_length,
        )
        device = next(self._model.parameters()).device
        encoded = {name: value.to(device) for name, value in encoded.items()}
        torch = importlib.import_module("torch")
        with torch.no_grad():
            probabilities = torch.softmax(self._model(**encoded).logits[0], dim=-1)
        return float(probabilities[self._contradiction_index].item())


@dataclass
class TransformersReaderAdapter:
    model_name: str
    model_revision: str
    max_new_tokens: int = 64
    name: str = "transformers_reader"
    _model: Any = field(init=False, repr=False)
    _tokenizer: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _require_immutable_revision(self.model_revision)
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise RuntimeError("install the train extra for reader inference") from error
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.model_revision,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.model_revision,
            dtype="auto",
            device_map="auto",
        )
        self._model.config.pad_token_id = self._tokenizer.pad_token_id
        self._model.eval()

    def generate(
        self,
        question: str,
        contexts: tuple[ContextDocument, ContextDocument],
        *,
        seed: int,
    ) -> ReaderOutput:
        prompt = render_reader_prompt(question, contexts)
        rendered = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self._tokenizer(rendered, return_tensors="pt")
        device = next(self._model.parameters()).device
        encoded = {name: value.to(device) for name, value in encoded.items()}
        output = self._model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        prompt_length = encoded["input_ids"].shape[-1]
        answer = normalize_reader_answer(
            self._tokenizer.decode(
                output[0, prompt_length:],
                skip_special_tokens=True,
            )
        )
        return ReaderOutput(
            answer=answer,
            metadata={"adapter": self.name, "seed": seed},
        )
