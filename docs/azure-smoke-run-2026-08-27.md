# Azure smoke run: 2026-08-27

This run validates the complete Azure execution path. It is not evidence for the
preregistered hypotheses.

## Environment

- Azure Machine Learning, Spain Central
- `Standard_NC4as_T4_v3`, one dedicated Tesla T4 with 15 GB VRAM
- Scale-to-zero cluster, one-node maximum
- CUDA 12.8, driver 550.163.01
- Qwen3-0.6B scorer revision `c1899de289a04d12100db370d81485cdf75e47ca`
- Granite 3.3 2B reader revision `707f574c62054322f6b5b04b6d075f0a8f05e0f0`

The infrastructure smoke job verified CUDA, the private data mount, and MLflow.
The cluster returned to zero nodes after the run.

## Data

- 128 deterministically selected 2WikiMultiHopQA questions
- 1,536 pairwise preference records
- 404/48 answer-free train/validation pairs
- 808/96 answer-conditioned train/validation pairs
- 15 held-out questions
- Maximum scorer input length: 951 tokens against a 2,048-token budget

The contradiction feature was fixed at `0.5`, and scorer calibration used an
identity scale and zero bias. Both are smoke-only placeholders.

## Training

| Scorer | Train pairs | Validation pairs | Train loss | Validation loss |
|---|---:|---:|---:|---:|
| Answer-free | 404 | 48 | 0.1169 | 0.00107 |
| Answer-conditioned | 808 | 96 | 0.0581 | 0.00043 |

The very low validation losses most likely reflect easy negatives and the small
template family. They are not evidence of generalization.

## Held-out context selection

| Condition and method | Support coverage@2 | EM | Token F1 |
|---|---:|---:|---:|
| Correct, answer-conditioned | 0.467 | 0.200 | 0.404 |
| Correct, answer-free | 0.400 | 0.200 | 0.381 |
| Wrong, answer-conditioned | 0.467 | 0.200 | 0.390 |
| Wrong, answer-free | 0.400 | 0.200 | 0.381 |
| Answer absent, answer-free | 0.400 | 0.200 | 0.381 |
| Lexical question-only | 0.067 | 0.200 | 0.230 |
| Oracle support | 1.000 | 0.267 | 0.481 |

Native Granite chat formatting corrected an earlier reader artifact where
explanatory output made exact match zero.

The answer-conditioned scorer behaved almost identically for correct and wrong
supplied answers. This smoke run therefore does not demonstrate answer-error
amplification or a useful fallback gate.

## Gate smoke check

A question-disjoint 22/8 split contained only one harmful test example.
Disagreement-only features ranked that example first, but the sample is too
small to interpret. Constant contradiction scores and identity calibration
make the full-gate metrics non-scientific.

## Required next run

1. Replace constant contradiction scores with an independent NLI model.
2. Fit Platt scale and bias on development scorer labels.
3. Mine lexically matched and entity-matched hard negatives.
4. Hold out intervention templates.
5. Run at least 1,000 test questions and three scorer seeds.
6. Complete the intervention and final-answer human audits.

Azure run identifiers are retained in the workspace but raw passage-bearing
inputs and model outputs are not published by this repository.
