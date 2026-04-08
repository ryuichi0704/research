# Literature Notes (2026-04-05)

Scope: quick web pass to position the new contrast-mode results against prior BatchEnsemble theory and practice.

## Primary sources checked

- BatchEnsemble original paper:
  - https://openreview.net/forum?id=Sklf1yrYDr
  - Main takeaway: BatchEnsemble shares a weight matrix and uses rank-one member-specific fast weights; the paper emphasizes efficiency and competitive uncertainty, but does not provide a feature-learning mean-field theory for symmetry breaking near collapse.

- Embedded Ensembles theory:
  - https://proceedings.mlr.press/v151/velikanov22a.html
  - Main takeaway: in the NTK regime, shared-parameter ensembles can be in collective or independent regimes depending on architecture and initialization.
  - Relevance: this is the closest theoretical precedent, but it is kernel/lazy rather than feature-learning and does not isolate boundary-state contrast modes around a collapsed mean-field trajectory.

- TabM official venue page:
  - https://openreview.net/forum?id=Sd4wYYOhmY
  - Main takeaway: parameter-efficient ensembling has a large practical effect in tabular MLPs, so a theory that tells practitioners how to induce useful diversity is directly relevant.

## What I did not find in this pass

- I did not find a direct BatchEnsemble theory paper that:
  - linearizes the feature-learning collapsed baseline,
  - decomposes perturbations into consensus vs. contrast modes,
  - proves that only the contrast covariance matters for leading disagreement, or
  - proves an antithetic/rank-one optimality result for small-budget diversity.

- I also did not find a primary-source paper explicitly arguing that increasing ensemble size `K` fails to improve the leading disagreement coefficient near the collapse diagonal. The closest existing theory I found is the NTK collective/independent picture from Embedded Ensembles.

## Positioning of the new sandbox note

The new note in `contrast_mode_theory.tex` appears to add a distinct message:

- Existing draft: exact collapse under symmetric initialization, plus the `K`-accelerated shared-law baseline.
- New note: local theory of *how asymmetry should be designed* once we move off that diagonal.
- Main design claim: with a small asymmetry budget, centered contrast perturbations are the only ones that generate disagreement without moving the ensemble mean at first order, and the optimal covariance is rank-one / antithetic rather than iid isotropic.

## Next empirical questions suggested by the theory

- Estimate the dominant eigendirection of the propagated Gram matrix `G_t(x)` at initialization or early training.
- Compare:
  - iid random fast-weight perturbations,
  - balanced zero-mean perturbations,
  - antithetic two-cluster perturbations aligned with a learned or estimated dominant contrast direction.
- Measure:
  - disagreement,
  - change in ensemble mean prediction,
  - validation NLL / calibration.
