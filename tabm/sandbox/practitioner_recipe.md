# Practitioner Recipe: Contrast-Aware BatchEnsemble Initialization

This note translates the new theory in `contrast_mode_theory.tex` into an implementable recipe.

## Core message

Near the symmetric-collapse regime:

- diversity comes from centered contrast modes, not from consensus motion;
- fixed perturbation budget should be spent on contrast covariance, not on nonzero empirical mean;
- iid isotropic fast-weight noise is generally not optimal;
- CLT-scale asymmetry of size `1/sqrt(M)` produces only `1/M` disagreement on fixed horizons;
- more heads do not automatically create a larger first-order disagreement coefficient.
- for finite `K`, there are two different goals:
  - maximize total disagreement: concentrate on one top mode;
  - maximize directional robustness / avoid all members drifting together: spread budget across several modes.

## Minimal actionable recipe

1. Start from a shared baseline state.
   - Initialize all BatchEnsemble members identically in the boundary variables.
   - This gives a clean approximation to the collapsed trajectory.

2. Estimate a contrast sensitivity matrix.
   - On a minibatch or validation subset, estimate
     - `G_0 ~= E_x[J_q(0,x)^T J_q(0,x)]`
   - At very early time, this is the simplest proxy for the propagated Gram matrix from the theory.
   - If possible, repeat later in training to estimate a time-dependent version and average it over data/time.
   - The theory says any weighted time/data objective reduces to an eigenproblem for an integrated Gram matrix.

3. Compute the dominant eigenvector `v_max`.
   - Use the top eigenvector of the estimated `G_0`.
   - This is the direction that maximizes first-order disagreement per unit contrast budget.

4. Use antithetic coding across heads.
   - If `K` is even:
     - assign half the heads `+delta * v_max`
     - assign half the heads `-delta * v_max`
   - If `K` is odd:
     - leave one head at `0`
     - split the others evenly between positive and negative codes.

5. Keep the empirical mean exactly zero.
   - This prevents wasting budget on consensus motion.
   - It also keeps the ensemble mean predictor unchanged at first order.

6. Tune `delta` as a diversity budget, not as arbitrary noise scale.
   - Small `delta`: the theory should be accurate.
   - Large `delta`: may escape the local regime, which can be good or bad and should be evaluated empirically.

## Which geometry to use

Use different code geometries depending on the objective:

- If the goal is maximum total disagreement:
  - use rank-one antithetic coding along the top eigendirection.
- If the goal is robustness with small `K`:
  - spread the code over the top `r <= K-1` eigendirections.
  - with `K = 4`, the canonical balanced design is tetrahedral over `r = 3` directions.

Interpretation:

- antithetic pairing is best for one dominant uncertainty axis;
- simplex/tetrahedral coding is better when you want the members not to drift together in one direction.

## What to compare empirically

Compare these three initialization families:

- iid random fast-weight perturbations;
- balanced zero-mean random perturbations;
- antithetic top-eigenvector perturbations.

Measure:

- disagreement or predictive spread;
- shift of the ensemble mean prediction relative to the collapsed baseline;
- validation NLL / calibration metrics;
- downstream accuracy.

## Hypothesis suggested by the theory

If the model is operating near the symmetric-collapse regime, then antithetic top-eigenvector initialization should:

- produce more disagreement than iid random initialization at the same perturbation budget;
- perturb the ensemble mean less than unbalanced random initialization;
- give a better diversity-to-accuracy tradeoff than simply increasing `K`.

## Stronger variant

If one can afford an online estimate of contrast sensitivity during training:

- periodically re-estimate the top eigendirection of the empirical contrast Gram matrix;
- reparameterize boundary fast weights in a low-dimensional contrast subspace;
- learn only the coefficients of that subspace with a zero-mean constraint across heads.

This is the direct algorithmic interpretation of the covariance-design view from the theory.
