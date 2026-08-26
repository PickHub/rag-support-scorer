#!/usr/bin/env bash
set -euo pipefail

: "${SCORER_MODEL_REVISION:?Set an immutable model commit revision}"
: "${SCORER_TOKENIZER_REVISION:?Set an immutable tokenizer commit revision}"
: "${GRANITE_MODEL_REVISION:?Set an immutable Granite commit revision}"
export ANSWER_FREE_CHECKPOINT="${ANSWER_FREE_CHECKPOINT:-outputs/24gb/answer_free}"
export ANSWER_CONDITIONED_CHECKPOINT="${ANSWER_CONDITIONED_CHECKPOINT:-outputs/24gb/answer_conditioned}"

uv sync --extra train
uv run rag-support-train configs/24gb/scorer_answer_free.yaml --context-lookup data/cache/context_lookup.json
uv run rag-support-train configs/24gb/scorer_answer_conditioned.yaml --context-lookup data/cache/context_lookup.json
uv run rag-support-experiment configs/experiment_gpu.json
