# Assumptions and limitations

- Candidate-context selection uses a fixed 2WikiMultiHopQA pool and is not first-stage retrieval.
- The two-context restriction excludes questions requiring broader support.
- Programmatic interventions can retain semantic ambiguity; human audit remains required.
- Lexical overlap and lightweight entity typing are artifact controls, not semantic equivalence proofs.
- Mock adapters validate orchestration, not model quality.
- Dry-run training validates labels, formatting, and configuration without validating GPU kernels.
- MinHash-style signatures are deterministic screening tools and can miss semantic paraphrases.
- Calibration and gate estimates are invalid if fitted or tuned on the locked test split.
- Findings are limited to Wikipedia-style multi-hop QA and the tested model revisions.
- Base-model pretraining contamination cannot be ruled out.
- GPU memory and runtime estimates must be measured on the target hardware.
