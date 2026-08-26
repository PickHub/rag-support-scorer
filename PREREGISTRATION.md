# Preregistration

## Primary question

Can inference-visible uncertainty and scorer disagreement identify when answer-conditioned context selection performs worse than matched answer-free selection?

## Fixed design

- Unit: one question with a fixed pool of at most ten contexts.
- Bundle size: two unordered contexts.
- Included questions: all required gold support fits in at most two contexts.
- Supplied-answer conditions: correct, plausible same-type wrong, absent.
- Natural drafts: observational replication only.
- Reader input: question and selected contexts only.
- Primary outcomes: support coverage@2, answer EM/F1, joint success, selective regret.
- Gate target: answer-conditioned ranking has worse gold-derived outcome than answer-free ranking.
- Gate features: calibrated answer-conditioned probability, top-bundle overlap, rank correlation, score-margin disagreement, contradiction score.
- Forbidden inference feature: draft or supplied-answer correctness.
- Primary gate comparisons: confidence-only, contradiction-only, disagreement-only, full-minus-contradiction, always-answer-free, always-answer-conditioned, oracle.

Development uses one scorer seed. Locked Stage 2 uses three scorer seeds, paired bootstrap confidence intervals, and at least 1,000 paired test questions when exclusions permit. Report mean, standard deviation, worst seed, harm prevalence, PR-AUC, balanced accuracy, calibration, AURC, and false-retain rate for correct, wrong, pooled, and natural-draft conditions.

Complete intervention templates are held out from test. The Stage 1 stop condition is failure to beat surface probes on held-out templates or inadequate intervention-audit validity.
