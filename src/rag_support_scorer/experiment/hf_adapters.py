from __future__ import annotations

import importlib
import re
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
        "Answer using only the contexts. Return only the answer."
    )


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
        rendered = render_input(
            question,
            "\n\n".join(context.text for context in contexts),
            scorer_kind=self.scorer_kind,
            supplied_answer=supplied_answer,
        )
        encoded = self._tokenizer(rendered, return_tensors="pt", truncation=False)
        if encoded["input_ids"].shape[-1] > self.max_sequence_length:
            raise ValueError("scorer input exceeds the locked sequence budget")
        device = next(self._model.parameters()).device
        encoded = {name: value.to(device) for name, value in encoded.items()}
        torch = importlib.import_module("torch")
        with torch.no_grad():
            return float(self._model(**encoded).logits.reshape(-1)[0].item())


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
        encoded = self._tokenizer(prompt, return_tensors="pt")
        device = next(self._model.parameters()).device
        encoded = {name: value.to(device) for name, value in encoded.items()}
        output = self._model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        prompt_length = encoded["input_ids"].shape[-1]
        answer = self._tokenizer.decode(
            output[0, prompt_length:],
            skip_special_tokens=True,
        ).strip()
        return ReaderOutput(
            answer=answer,
            metadata={"adapter": self.name, "seed": seed},
        )
