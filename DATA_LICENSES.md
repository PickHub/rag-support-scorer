# Data and model licenses

## 2WikiMultiHopQA

The primary source is the 2WikiMultiHopQA author mirror at revision `612bc5039a457880d9e7d84c3b0a4cf154b70e4f`. The mirror asserts Apache-2.0, while the underlying Wikipedia passage provenance is CC-BY-SA. This project loads the pinned train and development Parquet objects directly, maps development to validation, and does not use the unlabeled test split. It does not redistribute source passages. Public preparation outputs contain only source identifiers, revision pins, hashes, split assignments, and aggregate reports. Passage-bearing preferences required for private training are ignored by Git and must remain in access-controlled local or Azure storage. Verify downstream attribution and share-alike obligations before publishing any passage-bearing artifact.

## RAG-RewardBench

Optional rejection-gate evaluation may use revision `6dc0e802d41a0f4421e4477a37868ca8952c6691` for citation, conflict, and abstention subsets only. It is never training data.

## Models

Every run config must record the exact model repository, immutable revision, tokenizer revision, and license. The default scorer family is the text-only `Qwen/Qwen3-0.6B`. The automated final-reader path uses Apache-licensed `ibm-granite/granite-3.3-2b-instruct`. Gemma is manually gated, non-OSI, and quarantined from automated paths. External rerankers, NLI models, and derived adapters retain their own obligations. No checkpoint or weight is included here.

Generated manifests are metadata, not a grant to redistribute upstream text.
