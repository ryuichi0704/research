# Verification Memo for the Proposed Diversity Extension (2026-04-05)

Reviewed object:

- the mathematical sketch provided in the user message
- local files currently available in sandbox:
  - `contrast_mode_theory.tex`
  - `paper_insert_contrast_modes.tex`
  - `neurips_2026_with_contrast_extension.tex`

Important note:

- the file name `batchensemble_diversity_extension.tex` was not present in the workspace, so this review checks the math from the supplied text directly.

Overall verdict:

- The core idea is strong and mostly correct.
- The best parts are the consensus/contrast splitting, the zero-sum code principle, and the Gramian-based optimal design result.
- The main places that need tightening are:
  - the precise regularity assumptions on derivatives with respect to the law `rho`,
  - the distinction between local linear stability and full nonlinear collapse,
  - one rank bound that is missing an `nd_y` ceiling for finite evaluation sets,
  - literature interpretations that should be marked as heuristic rather than theorem.

## 1. Consensus vs. contrast mode splitting

Status:

- Verified as correct under the same additive/local structure already used in the current draft.

Needed assumptions:

- `B`, `C`, and `H` must be `C^1` along the collapsed trajectory.
- Because `rho` is a measure-valued state, the derivative `D_rho` should be specified as a Fréchet derivative on the chosen state space, or an equivalent notion appropriate for the mean-field formulation.

Comments:

- The derivation
  - average over heads to get the consensus equation
  - subtract the average to get the centered contrast equation
  is sound.
- This is a genuinely useful theorem, because it shows first-order disagreement is driven only by the centered contrast sector.

## 2. Zero-sum codes preserve the ensemble mean at first order

Status:

- Verified in the linearized system.

Needed assumptions:

- zero initial consensus perturbation:
  - `eta_0 = 0`
  - `bar xi_0 = 0`
- uniqueness of the linearized ODE.

Comments:

- Then `eta_t = 0` and `bar xi_t = 0` for all time in the linearized dynamics.
- Therefore the ensemble mean predictor is unchanged at order `epsilon`, while disagreement is order `epsilon`.
- The follow-up risk statement
  - `R(bar f^epsilon) = R(f_coll) + O(epsilon^2)`
  is also correct, but only with an additional `C^2` solution-map / loss assumption and a justified differentiation under the data expectation.

## 3. First-order disagreement rank ceiling

Status:

- Substantively correct, but the stated bound should be sharpened.

Correct finite-point bound:

- If the covariance is formed over `n` evaluation points with output dimension `d_y`, then
  - `rank(Sigma_t^epsilon) <= min{K-1, p, n d_y}`.

What is correct in the original claim:

- `rank(Sigma_t^epsilon) <= rank(S_0) <= min{K-1, p}` is true as an internal-state bound.

What needs fixing:

- Since `Sigma_t^epsilon` is an `n d_y x n d_y` matrix, the displayed bound should also include `n d_y`.

Interpretation:

- The practitioner message remains valid:
  - once `K-1` exceeds the effective explicit member-state dimension, more heads do not create new first-order directions.
- But this conclusion is local and tied to the explicit member-specific state used in the reduced theory.

## 4. Gramian-based optimal design

Status:

- Verified as correct under the centered, unforced linearized dynamics.

Comments:

- The objective
  - `int_0^T D_t^mu dt = epsilon^2 tr(W_T S_0) + o(epsilon^2)`
  is correct.
- Under
  - `S_0 >= 0`
  - `tr(S_0) <= B`
  - `rank(S_0) <= K-1`
  the optimizer is rank-one on the top eigendirection of `W_T`.

This is the strongest part of the extension:

- it turns diversity design into an explicit spectral optimization problem;
- it gives a principled alternative to iid random-sign initialization.

## 5. Isotropic random signs vs. optimal design

Status:

- Valid as a conditional comparison, not as an unconditional identification with the original BatchEnsemble initialization.

What is safe to say:

- If the induced initial contrast covariance is approximately isotropic in the reduced `q`-space, then the isotropic-vs-optimal ratio is
  - `p lambda_max(W_T) / tr(W_T)`
  and lies in `[1, p]`.

What is too strong without extra work:

- saying that the original BatchEnsemble random-sign scheme *is* exactly the isotropic design in this reduced boundary-state theory.

Reason:

- in the current paper setup, `q = (u, v)` collects boundary variables only;
- BatchEnsemble's original random signs are attached to member-specific fast weights more broadly, including neuron-level factors in the finite model.

So:

- the isotropic interpretation is a useful heuristic,
- but it should be explicitly labeled as an approximation or analogy unless one proves the mapping.

## 6. Contraction test via `A_t`

Status:

- Verified as a local linear stability criterion.

What is correct:

- if the symmetric part of `A_t` is uniformly upper bounded by `-gamma < 0`, then each centered mode decays exponentially in the linearized dynamics.

What should be stated carefully:

- this proves local linear decay of small contrast perturbations around the collapsed trajectory;
- it does **not** by itself prove full nonlinear global collapse for finite perturbations.

So the strongest safe phrasing is:

- "`A_t` controls local survival/decay of small symmetry breaking near the collapse diagonal."

## 7. Member-specific forcing from batches / augmentation / noise

Status:

- Structurally correct as a reduced linearized model, but the modeling step needs to be made explicit.

What is safe:

- if the centered subsystem is forced as
  - `dot zeta_alpha = A_t zeta_alpha + (b_alpha - bar b)`,
  then only centered forcing contributes directly to first-order disagreement.

What needs caution:

- different mini-batches or augmentation pipelines may also perturb the shared-law dynamics, not only the member-local `q_alpha` equations;
- so the mapping from training asymmetry to an additive forcing term `b_alpha` is a modeling assumption, not a theorem from the current mean-field equations alone.

## 8. Literature checks

These source claims do check out:

- BatchEnsemble original paper states that random-sign initialization of fast weights is sufficient for desired diversity and also notes that training members on different sub-batches can encourage diversity:
  - [OpenReview PDF](https://openreview.net/pdf?id=Sklf1yrYDr), lines around 781-784.
- The same paper describes dividing a mini-batch into member-specific sub-batches in the vectorized implementation:
  - [OpenReview PDF](https://openreview.net/pdf?id=Sklf1yrYDr), lines around 242-255.
- TabM states that BatchEnsemble-style adapters are usually random `±1`, and then introduces the improved initialization where all multiplicative adapters except the first are initialized to `1`:
  - [OpenReview PDF](https://openreview.net/pdf?id=Sd4wYYOhmY), lines around 171-174 and 305-308.

One interpretive caution:

- the claim that TabM's better initialization works *because* it avoids spending budget in low-sensitivity directions is plausible and interesting, but it is still an inference from the new theory, not something TabM proves directly.

## Recommended final phrasing

The safest summary is:

- The proposed extension is mathematically sound as a local first-order theory around the collapsed BatchEnsemble baseline.
- Its strongest verified claims are:
  - consensus/contrast mode separation,
  - zero-sum codes preserving the ensemble mean at first order,
  - Gramian-based optimal design of disagreement,
  - local contraction diagnostics via the symmetric part of `A_t`.
- The claims that need explicit caveats are:
  - any exact identification of random-sign initialization with isotropic covariance in the reduced boundary-state space,
  - any upgrade from local linear contraction to global nonlinear collapse,
  - any direct attribution of TabM's empirical gains to the Gramian mechanism without further proof.

## Bottom line

My recommendation is:

- keep this extension and push it forward;
- fix the rank bound to include `n d_y` when using finitely many evaluation points;
- add a short assumptions paragraph for differentiability in `rho`;
- mark the TabM/random-sign interpretations as theory-driven hypotheses rather than proved consequences.

With those changes, the extension is not only interesting but publication-credible.
