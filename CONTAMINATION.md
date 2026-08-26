# Contamination policy

Preparation normalizes questions, removes exact matches to locked benchmark manifests, and supports character five-gram MinHash-style near-duplicate screening. Reports also record title, passage-hash, and entity overlap when comparison metadata is available.

Evaluation revisions, split salt, intervention-template holdouts, source hashes, and comparison manifest hashes are written to immutable JSON manifests. Dataset-name separation is not treated as independence. Base-model pretraining overlap is unknowable and must be disclosed.

Split groups join exact normalized questions and exact two-document gold-support sets. They do not transitively join examples that share only one frequently reused Wikipedia paragraph. On the pinned 167,454-row training snapshot, this produced 131,727 groups with a largest group of 11 and a measured 133,673/16,960/16,821 train/dev/test allocation for the default salt.

The default thresholds are conservative development defaults, not validated universal cutoffs. Publish `eval_manifest.json` and `contamination_report.json` with every result release, but do not publish source passages through those files.
