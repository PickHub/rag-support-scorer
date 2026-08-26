# DPO post-training extension

This directory is a standalone experiment. It is not imported by, or required
for, the main scorer pipeline.

## QLoRA DPO

Install the extension dependencies in an isolated environment:

```bash
python -m venv .venv-dpo
.venv-dpo/bin/python -m pip install -r extensions/dpo_post_training/requirements.txt
.venv-dpo/bin/python -m pip install \
  -r extensions/dpo_post_training/requirements-bitsandbytes.txt
```

Training and optional evaluation inputs are JSONL with string-valued `prompt`,
`chosen`, and `rejected` fields. Run on a CUDA GPU:

```bash
.venv-dpo/bin/python extensions/dpo_post_training/train_dpo.py \
  --train-data data/dpo/train.jsonl \
  --eval-data data/dpo/validation.jsonl \
  --model-name "<generator-model>" \
  --model-revision "<immutable-revision>" \
  --output-dir outputs/dpo
```

The entry point uses 4-bit QLoRA and TRL's DPO trainer. It is deliberately
separate from scorer training and does not imply that scorer effects transfer
to post-training. The model ID and immutable revision remain command-line
inputs. Bitsandbytes is an optional dependency loaded only when GPU training
starts.

## Offline validation

Tiny fixtures exercise input validation and deterministic selection without
loading or downloading a model:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m unittest discover \
  -s extensions/dpo_post_training/tests \
  -p 'test_*.py'
```

The validation-only entry point is also suitable for offline CI:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python extensions/dpo_post_training/train_dpo.py \
  --train-data extensions/dpo_post_training/fixtures/tiny_preferences.jsonl \
  --model-name offline/tiny-causal-lm \
  --model-revision 0000000000000000000000000000000000000000 \
  --output-dir extensions/dpo_post_training/.offline-ci-output \
  --max-length 128 \
  --learning-rate 5e-6 \
  --validate-only
```

## Deterministic best-of-N

Freeze candidate generation before comparison. Each baseline input line needs
an `id`, a non-empty `candidates` string list, and an equal-length numeric
`scores` list. Scores can come from any predeclared selector.

```bash
python extensions/dpo_post_training/best_of_n.py \
  --input data/dpo/candidates.jsonl \
  --output outputs/best_of_4.jsonl \
  --n 4
```

Selection takes the highest score from the first N candidates. Score ties use
lexicographic answer order and then source order, making repeated runs
byte-identical for the same input.
