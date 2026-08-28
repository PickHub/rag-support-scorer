# Answer-error amplification: controlled stress test

## Headline

In a controlled counterfactual-evidence stress test, answer-conditioned scoring
amplified supplied wrong answers by replacing gold evidence with
wrong-answer-supporting evidence, with a **-0.636 coverage@2 interaction** across
150 questions and three seeds.

This demonstrates a vulnerability under a constructed setting. It does not
estimate natural-world prevalence.

## Design

- Dataset: pinned 2WikiMultiHopQA
- Questions: 300 train, 100 calibration, 150 held out
- Scorer: Qwen3-0.6B with QLoRA
- Seeds: 17, 23, 42
- Context pool: identical under correct and plausible-wrong supplied answers
- Pool contents: original gold support, ordinary 2Wiki distractors, and one
  matched counterfactual passage supporting the wrong answer
- Primary truth anchor: original dataset gold-support passages
- Independent secondary signal:
  `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
- Calibration: separate Platt scale and bias fitted per seed on validation pairs
- Primary unit: question; seed replicates are averaged within question before
  bootstrap and sign-flip permutation tests

## Primary result

| Metric | Correct answer | Plausible wrong answer | Difference |
|---|---:|---:|---:|
| Answer-free gold coverage@2 | 0.687 | 0.687 | 0.000 |
| Answer-conditioned gold coverage@2 | 0.882 | 0.247 | -0.636 |
| Answer-conditioned counterfactual top-1 | 0.053 | 0.709 | +0.656 |
| Lexical counterfactual top-1 | 0.727 | 0.787 | +0.060 |
| NLI contradiction against free evidence | 0.176 | 0.406 | +0.230 |
| Calibrated scorer confidence | 0.935 | 0.919 | -0.016 |
| Harm rate relative to answer-free ranking | 0.056 | 0.516 | +0.460 |

Coverage interaction:

- Estimate: **-0.6356**
- Question-clustered bootstrap 95% CI: **[-0.7001, -0.5667]**
- One-sided question-level sign-flip permutation p: **0.0001**

Per-seed interactions:

| Seed | Coverage interaction | Counterfactual top-1 increase |
|---:|---:|---:|
| 17 | -0.653 | +0.687 |
| 23 | -0.707 | +0.720 |
| 42 | -0.547 | +0.560 |

The effect replicated in sign across three training seeds.

## Controls

- Passage-only surface detector ROC AUC: **0.562**, below the preregistered
  0.60 stop threshold.
- Correct-answer counterfactual top-1: **0.053**, below the 0.15 failure
  threshold.
- Lexical answer-overlap selection changed only six points between conditions,
  versus a 65.6-point conditioned-scorer shift.
- Answer-free coverage remained 0.687 under both supplied-answer conditions.

These probes do not rule out semantic or generation-style artifacts. They show
only that measured surface features and the simple lexical baseline do not
explain the full effect.

## Exploratory gate

A question-disjoint split with 41 unseen questions pooled the three seed
replicates. This analysis is exploratory and not a validated mitigation.

| Features | PR-AUC | Balanced accuracy | False-retain rate |
|---|---:|---:|---:|
| Confidence only | 0.293 | 0.448 | 0.442 |
| Independent NLI only | 0.603 | 0.685 | 0.506 |
| Scorer disagreement only | 0.654 | 0.793 | 0.065 |
| Full | 0.665 | 0.793 | 0.065 |

Scorer confidence alone did not identify harm. Disagreement features were more
informative, but this evaluation pools repeated seeds and is too small for a
deployment claim.

## Limitations

- Counterfactual evidence is synthetic and uses one intervention template.
- No human audit has been completed.
- Surface and lexical probes cannot exclude semantic synthetic artifacts.
- The answer-conditioned scorer is explicitly trained to support supplied
  answers, so this is a causal stress test rather than an estimate of ordinary
  RAG behavior.
- Only one model family, size, task, and dataset are tested.
- Platt calibration is fitted on pairwise validation scores and is not evidence
  of calibration under the counterfactual distribution shift.
- NLI is corroborating evidence, not ground truth.
- Raw passages and model checkpoints are not published because upstream
  redistribution terms require further review.

## Azure runs

- Seed 17 study: `salmon_garage_lrvykmfp8c`
- Seed 23 study: `funny_yacht_fj6rb2wbwp`
- Seed 42 study: `happy_fly_x4ytbhd43w`

All jobs ran on one scale-to-zero `Standard_NC4as_T4_v3` node in Azure Machine
Learning. The cluster returned to zero nodes after completion.

## Reproduction artifacts

Passage-free result rows and generated summaries are committed under
[`docs/results/amplification/`](results/amplification/):

- One JSONL file per training seed.
- Question-clustered aggregate and permutation result.
- Surface-artifact probe output.
- Exploratory gate output.

The exploratory gate split assigns whole questions with a deterministic SHA-256
hash before pooling their three seed replicates.
