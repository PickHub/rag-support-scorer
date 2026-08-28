# rag-support-scorer

RAG systems can use a draft answer to select supporting context. When that draft is wrong, the ranking step can reinforce the mistake instead of correcting it.

This repository provides a reproducible, single-GPU toolkit for measuring that failure mode and learning when to fall back to answer-free ranking.

## What you can do

- Build controlled preference data with correct and plausible-wrong supplied answers.
- Train matched answer-free and answer-conditioned support scorers.
- Compare both scorers over identical two-context candidate bundles.
- Produce gold-derived harm labels and fit a calibrated fallback gate.
- Reproduce the pipeline locally or on a scale-to-zero Azure ML GPU cluster.
- Explore QLoRA DPO separately without conflating it with the primary experiment.

The experiment harness, tests, and Azure ML assets are implemented. Trained checkpoints and empirical findings are not published yet, so the current value is reproducible research infrastructure rather than a claimed model improvement.

The first end-to-end Azure execution is documented in
[`docs/azure-smoke-run-2026-08-27.md`](docs/azure-smoke-run-2026-08-27.md).
It validates the pipeline and reports a negative smoke finding rather than
claiming research performance.

A subsequent three-seed controlled study demonstrates answer-error
amplification in the constructed counterfactual setting. Read the
[`answer-error amplification report`](docs/answer-error-amplification-2026-08-28.md)
for the result and its limitations.

## Who this is for

Researchers and RAG engineers evaluating whether generated answers should influence evidence selection, especially when working with small models and limited GPU capacity.

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

Build bounded private inputs for an Azure smoke run:

```bash
uv run --extra train python scripts/build_azure_smoke_data.py \
  --input data/raw/2wiki/train.parquet \
  --output-dir data/azure-smoke/source \
  --scan-limit 4096 \
  --limit 128 \
  --max-negatives 4
```

The builder keeps passage-bearing preferences separate from experiment inputs and marks its constant contradiction score as smoke-only. Replace that baseline with independently computed contradiction scores before treating gate metrics as research results.

For the amplification study, use explicit question-disjoint splits and run the
artifact detector before GPU training:

```bash
uv run --extra train python scripts/build_azure_smoke_data.py \
  --input data/raw/2wiki/train.parquet \
  --output-dir data/amplification/source \
  --scan-limit 20000 \
  --train-questions 300 \
  --validation-questions 100 \
  --test-questions 150

uv run python scripts/evaluate_artifact_probe.py \
  --input data/amplification/source/experiment/artifact_probe.jsonl \
  --output data/amplification/artifact_report.json
```

Stop if passage-only artifact detection exceeds the preregistered `0.60` ROC
AUC threshold. The Azure amplification pipeline fits Platt calibration from the
conditioned scorer's validation outputs and uses the independently pinned
DeBERTa NLI model.

Validate scorer examples on CPU without loading a model:

```bash
export SCORER_MODEL_REVISION="<immutable-40-character-commit>"
export SCORER_TOKENIZER_REVISION="<immutable-40-character-commit>"
uv run rag-support-train configs/16gb/scorer_answer_free.yaml --dry-run
uv run rag-support-train configs/16gb/scorer_answer_conditioned.yaml --dry-run
```

GPU reproduction commands are in `scripts/reproduce_16gb.sh` and `scripts/reproduce_24gb.sh`. Azure ML setup is documented in `azureml/README.md`.

The scorer defaults to the text-only `Qwen/Qwen3-0.6B`; immutable model and tokenizer revisions are supplied through environment-expanded config. The GPU experiment loads both trained PEFT checkpoints and the automated `ibm-granite/granite-3.3-2b-instruct` reader. The synthetic config remains mock-only for CPU validation. Gemma is quarantined from automated runs because access requires manual license acceptance.

Fit Platt calibration on development scorer outputs formatted as `{"score": ..., "label": ...}` JSONL:

```bash
uv run rag-support-calibrate \
  --input results/dev_scorer_labels.jsonl \
  --output results/scorer_calibration.json
```

Pass the resulting scale and bias into the controlled GPU experiment. Each run emits deterministic answer results plus gate examples containing matched-scorer disagreement features and gold-derived harm labels. Fit and evaluate the gate on separate development and locked test outputs:

```bash
uv run rag-support-gate \
  --train results/dev_gate_examples.jsonl \
  --test results/test_gate_examples.jsonl \
  --output results/gate_report.json
```

The answer-conditioned Platt scale and bias must be fitted on development scorer labels before generating locked test gate examples. Contradiction scores are supplied as a separate immutable JSON mapping keyed by `<question-id>:<condition>`.

## Scope

- Stage 1: deterministic ingestion, contamination checks, controlled interventions, artifact probes, and matched-scorer smoke validation.
- Stage 2: fixed-pool bundle ranking, controlled answer conditions, final-reader isolation, and a calibrated logistic harm gate.
- Optional DPO work is isolated under `extensions/dpo_post_training/` and is not imported by the main pipeline.

See `PREREGISTRATION.md`, `DATA_LICENSES.md`, `CONTAMINATION.md`, and `LIMITATIONS.md` before running or publishing experiments.
