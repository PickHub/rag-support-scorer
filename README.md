# rag-support-scorer

Single-GPU research code for testing when answer-conditioned context selection reinforces a wrong supplied answer.

The primary task freezes up to ten 2WikiMultiHopQA contexts, enumerates every two-context bundle, compares matched answer-free and answer-conditioned scorers, and evaluates an inference-visible fallback gate. Raw passages and model weights are never committed.

## Quick start

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run mypy src
```

Run the synthetic CPU pipeline:

```bash
uv run rag-support-smoke --output results/smoke.json
```

Prepare locally downloaded 2WikiMultiHopQA JSON without copying passage text into repository artifacts:

```bash
uv sync --extra train
uv run python scripts/fetch_2wiki.py --output-dir data/raw/2wiki
uv run rag-support-prepare \
  --input data/raw/2wiki/train.parquet \
  --output-dir data/manifests \
  --dataset-revision 612bc5039a457880d9e7d84c3b0a4cf154b70e4f
```

Validate scorer examples on CPU without loading a model:

```bash
export SCORER_MODEL_REVISION="<immutable-40-character-commit>"
export SCORER_TOKENIZER_REVISION="<immutable-40-character-commit>"
uv run rag-support-train configs/16gb/scorer_answer_free.yaml --dry-run
uv run rag-support-train configs/16gb/scorer_answer_conditioned.yaml --dry-run
```

GPU reproduction commands are in `scripts/reproduce_16gb.sh` and `scripts/reproduce_24gb.sh`. Azure ML setup is documented in `azureml/README.md`.

The scorer defaults to the text-only `Qwen/Qwen3-0.6B`; immutable model and tokenizer revisions are supplied through environment-expanded config. The GPU experiment loads both trained PEFT checkpoints and the automated `ibm-granite/granite-3.3-2b-instruct` reader. The synthetic config remains mock-only for CPU validation. Gemma is quarantined from automated runs because access requires manual license acceptance.

## Scope

- Stage 1: deterministic ingestion, contamination checks, controlled interventions, artifact probes, and matched-scorer smoke validation.
- Stage 2: fixed-pool bundle ranking, controlled answer conditions, final-reader isolation, and a calibrated logistic harm gate.
- Optional DPO work is isolated under `extensions/dpo_post_training/` and is not imported by the main pipeline.

See `PREREGISTRATION.md`, `DATA_LICENSES.md`, `CONTAMINATION.md`, and `LIMITATIONS.md` before running or publishing experiments.
