# Practical Protocol for Finite-K BatchEnsemble Diversity

Goal:

- estimate uncertainty meaningfully;
- keep the ensemble mean stable;
- prevent individual members from drifting too far from that mean;
- make the design concrete enough to implement.

This protocol is derived from the local theory in:

- `contrast_mode_theory.tex`
- `finite_k_geometry_note.tex`

## Recommendation in one sentence

Do **not** initialize members with arbitrary independent random noise. Instead:

1. start from a shared baseline,
2. for long nonlinear training, first train the collapsed baseline close to convergence,
3. estimate a small contrast subspace from the resulting stationary law,
4. place the `K` members on a centered simplex inside that subspace,
5. keep a zero-mean constraint during training,
6. optionally project member deviations to stay within a per-member budget.

## Important warning

If training is long and nonlinear, an initialization that is locally optimal near time zero need not remain relevant at the end of training.

So there are really two versions of the protocol:

- Early-time protocol:
  - estimate the contrast Gramian near initialization or after a short warmup.
  - useful when the model remains near that regime.
- Stationary-law protocol:
  - first train the collapsed baseline close to convergence,
  - then estimate the Gramian at the converged collapsed law,
  - then inject and maintain centered diversity.

For long nonlinear training, the stationary-law protocol is the safer theoretically motivated choice.

## What to parameterize

Let `q_alpha` denote the member-specific fast parameters that you actually allow to differ across ensemble members.

There are two practical choices:

- Minimal version:
  - only use the explicit boundary fast weights.
  - this is the cleanest match to the current theory.
- Broader version:
  - collect all member-specific fast weights you want to control into one vector.
  - this is more practical for real BatchEnsemble/TabM systems, but the theory then becomes an approximation.

If positivity matters for multiplicative parameters, use a log-parameterization:

- write the multiplicative factor as `exp(theta_alpha)`;
- build the simplex code in `theta_alpha`;
- initialize around `theta = 0`, which corresponds to multiplicative factor `1`.

## Protocol A: Theory-aligned two-stage procedure

### Stage 0: Build a collapsed baseline

- Initialize all member-specific parameters identically.
- In multiplicative form, this usually means all factors are `1`.
- If training will be long and strongly nonlinear:
  - train this collapsed system close to convergence.
- If you only want an early-time approximation:
  - a short warmup may be enough.

Result:

- you get either:
  - an early collapsed state, or
  - an approximate stationary collapsed law.

### Stage 1: Estimate a contrast Gramian

Choose a calibration batch or a small held-out subset.

For each sample `x`, compute the Jacobian of the predictor with respect to the member-specific state:

- `J(x) = d f(x) / d q`

Then estimate

- `G_hat = mean_x J(x)^T J(x)`

or, if you want a slightly more faithful time-averaged version,

- average the same quantity over a few early-training checkpoints.

In practice:

- if exact Jacobians are expensive, use Jacobian-vector products or per-sample gradients;
- if output dimension is large, use the Jacobian of logits or pre-activation outputs.
- if you trained the collapsed baseline to convergence, interpret `G_hat` as an estimate of the stationary Gramian.

### Stage 2: Pick the protected uncertainty rank

Set

- `r = min(K - 1, r_max)`

where `r_max` is how many uncertainty directions you want to actively protect.

Default:

- `K = 2`: use `r = 1`
- `K = 3`: use `r = 2`
- `K = 4`: use `r = 3` if the top 3 eigenvalues are meaningful
- if the spectrum decays sharply, use smaller `r`

### Stage 3: Compute the top eigenvectors

Let

- `lambda_1 >= ... >= lambda_r`
- `e_1, ..., e_r`

be the top `r` eigenpairs of `G_hat`.

### Stage 4: Build a centered simplex code

Choose unit-norm simplex vertices `v_alpha in R^r` with

- `sum_alpha v_alpha = 0`
- `(1/K) sum_alpha v_alpha v_alpha^T = (1/r) I_r`

Canonical examples:

- `K = 2`, `r = 1`:
  - `v_1 = +1`, `v_2 = -1`
- `K = 3`, `r = 2`:
  - equilateral triangle
- `K = 4`, `r = 3`:
  - tetrahedron:
    - `(1,1,1)/sqrt(3)`
    - `(1,-1,-1)/sqrt(3)`
    - `(-1,1,-1)/sqrt(3)`
    - `(-1,-1,1)/sqrt(3)`

### Stage 5: Map the simplex into parameter space

Pick a per-member deviation budget `tau > 0`.

Initialize

- `c_alpha = sqrt(tau) * sum_{i=1}^r lambda_i^(-1/2) * (v_alpha)_i * e_i`

and set

- `q_alpha = q_base + c_alpha`

or, in log-parameterization,

- `theta_alpha = c_alpha`
- `multiplier_alpha = exp(theta_alpha)`

Interpretation:

- the `lambda_i^(-1/2)` factor equalizes the effect size across the protected output directions;
- the simplex geometry keeps the ensemble mean centered;
- `tau` directly controls how far each member can sit from the mean.

## Protocol B: Cheap default when you cannot estimate Jacobians

If estimating `G_hat` is too expensive:

1. choose a low-dimensional member-specific subspace manually;
2. use centered simplex codes in that subspace;
3. keep the code norm small;
4. keep the zero-mean constraint during training.

This loses the Gramian optimality guarantee, but keeps the two most important principles:

- zero mean across members;
- non-collinear geometry rather than iid noise.

For `K = 4`, the cheap default is:

- choose a 3-dimensional member-specific subspace;
- initialize members on a tetrahedron in that subspace.

## Training rule

The simplest practical training parameterization is

- `q_alpha = q_shared + U a_alpha`

where

- `U` contains the chosen basis directions,
- `a_alpha in R^r` are member coefficients.

Use the constraints

- `sum_alpha a_alpha = 0`
- `||a_alpha||^2 <= tau`

There are two ways to enforce this:

### Option 1: Projection after each optimizer step

After each optimizer step:

1. subtract the mean coefficient:
   - `a_alpha <- a_alpha - mean_beta a_beta`
2. if needed, rescale each `a_alpha` to satisfy the norm budget.

This is simple and robust.

### Option 2: Reparameterize by construction

Represent the member coefficients as

- `a = V b`

where the columns of `V` span the zero-sum subspace over members.

Then optimize `b` directly.

This automatically enforces centering.

## What is actually proven

The following statements are supported by the current local theory:

- zero-sum codes keep the ensemble mean unchanged at first order;
- disagreement at first order depends on the contrast covariance;
- under a worst-member budget, simplex designs maximize the smallest covered uncertainty eigenvalue;
- for `K = r + 1`, simplex codes exactly realize any target rank-`r` local covariance.

## What is motivated but not yet proven end-to-end

These steps are theoretically motivated but not fully proved in the nonlinear finite-width setting:

- estimating `G_hat` from a finite calibration set and then freezing it;
- continuing long nonlinear training after the local initialization step;
- using all member-specific fast weights instead of only the boundary state from the cleanest theory;
- expecting better calibration on every task.

So the protocol is:

- rigorously justified locally;
- algorithmically plausible globally;
- still needs empirical confirmation beyond the local regime.

## Recommended defaults

### If you want the safest theory-matched setup

- vary only boundary fast weights;
- train the collapsed baseline near convergence;
- estimate `G_hat`;
- use simplex initialization;
- keep a zero-mean constraint during training.

### If you want the most practical finite-K setup

- use the member-specific fast-weight vector you already have;
- choose `r = min(K - 1, 3)`;
- use simplex/tetrahedral initialization rather than iid random signs;
- re-center member parameters after each step;
- clip or project to a small norm budget.

### If `K = 4`

Use:

- `r = 3` if the top-3 spectrum is not degenerate;
- tetrahedral initialization;
- zero-mean projection every step;
- a small member budget `tau` chosen so that members remain close to the mean in early training.
