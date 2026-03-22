# K=1 Width-Rate Investigation

## Executive Summary

This note isolates the current rate bottleneck in `paper/k1_raw_lmc.tex` and
records the most plausible routes toward an `O(1/N)`-type barrier law in the
single-component (`K=1`) Gaussian setting.

The current situation is:

- The fully closed finite-horizon SGD-to-LMC theorem stack gives a generic
  barrier rate of `O(N^{-1/4} + eps^{1/4})`.
- Under conditional head regularity, the same finite-horizon theorem stack
  improves to `O(N^{-1/2} + sqrt(eps))`.
- The deterministic width-sequence route is already aiming at
  `O(sqrt(d/N) + 1/N)` in the baseline case and `O(d/N + 1/N)` under head
  regularity, i.e. `O(1/N)` for fixed `d`.

The main conclusion of this investigation is:

1. The `N^{-1/4}` rate is not coming from the Gaussian semi-convexity term.
   It is coming from the combination
   `W_1 common path -> B_N = O(W_1) -> transfer = O(sqrt(B_N))`.
2. There is an immediate deterministic improvement available inside the current
   framework: add a `Delta_N`-from-`(M_{Delta V}^infty, B_N)` proposition,
   exactly analogous to `paper/main.tex`.
3. Reaching a theorem-level `O(1/N)` finite-horizon rate requires a stronger
   training/alignment input than the present `W_1` common-path theorem. In
   practice this means some second-moment control, e.g. `W_2` control of the
   hidden empirical laws or a particlewise mean-square coupling.

## 1. Where the Current `N^{-1/4}` Rate Comes From

The current finite-horizon theorem stack in `k1_raw_lmc.tex` is:

1. Shared mean-field path:

   `W_1(rho_A^N, rho_B^N) <= C_T (N^{-1/2} + sqrt(eps))`.

2. Alignment from a `W_1` coupling:

   `B_N <= C W_1`,
   `Delta_N <= C W_1^2`.

3. Deterministic barrier theorem:

   `B <= C sqrt(B_N) + C Delta_N`.

Substituting step 2 into step 3 gives

`B <= C W_1^{1/2} + C W_1^2`,

so the transfer term dominates and we get

`B = O(N^{-1/4} + eps^{1/4})`.

This means the generic `N^{-1/4}` rate is structurally forced by the present
choice of metric and alignment reduction. It is not a Gaussian-specific defect.

## 2. Why the Semi-Convexity Term Is Not the Real Bottleneck

In the deterministic barrier theorem,

`B <= C_G T_N + (rho_G / 8) Delta_N`.

For `K=1`, the Gaussian semi-convexity constant is already favorable: there is
no mixture responsibility covariance term, so the bad `K >= 2` mechanism is
absent. In the current rate chain, the semi-convexity term is of order
`Delta_N`, while the baseline transfer term is only controlled by `sqrt(B_N)`.

Once `B_N = O(W_1)` and `Delta_N = O(W_1^2)`, the semi-convexity part is
strictly smaller than the transfer part. So improving the barrier exponent
cannot come from sharpening `rho_G`; it must come from sharpening the transfer
side or the metric that feeds it.

## 3. Immediate Deterministic Upgrade Available Now

There is a missing proposition in `k1_raw_lmc.tex` that should be added before
any larger theorem redesign.

### Candidate Proposition

Let

- `v_i^X = (a_i^X, c_i^X) in R^2`,
- `M_{Delta V}^infty = max_j ((1/N) sum_i ((v_i^A - v_i^B)_j)^2)^{1/2}`,
- `B_N = (1/N) sum_i ||xi_i^A - xi_i^B||^2`,
- `Delta_N = E[ || z_{Theta_A}(X) - z_{Theta_B}(X) ||^2 ]`.

Then one should be able to prove

`Delta_N^{1/2} <= C_Delta ( M_{Delta V}^infty + sqrt(B_N) )`

with `C_Delta` depending only on `K_phi`, `K_nabla`, and the endpoint envelope
`M_V^infty`.

At the level of rates this gives

`M_{Delta V}^infty = O(sqrt(B_N))  =>  Delta_N = O(B_N)`.

### Proof Sketch

For either coordinate `u_i in {a_i, c_i}`,

`f_A(x) - f_B(x)`

splits into

1. a head-difference term

   `(1/N) sum_i (u_i^A - u_i^B) phi(xi_i^A, x)`,

2. a feature-difference term

   `(1/N) sum_i u_i^B ( phi(xi_i^A, x) - phi(xi_i^B, x) )`.

The first term is bounded by `K_phi M_{Delta V}^infty` by Cauchy-Schwarz. The
second term is bounded by `K_nabla M_V^infty sqrt(B_N)`. Taking the two output
coordinates together gives the displayed bound up to an absolute constant.

### Consequence

After adding this proposition, the deterministic fast-rate corollary no longer
needs to assume `Delta_N = O(1/N)` separately. Under conditional head
regularity,

`R_N = O(sqrt(B_N))  =>  M_{Delta V}^infty = O(sqrt(B_N))  =>  Delta_N = O(B_N)`.

Hence the deterministic fast-rate statement can be simplified to:

- if `B_N = O(d/N)` and `R_N = O(sqrt(B_N))`, then
  `B(Theta_A, Theta_B; varpi) = O(d/N)`,
- in fixed dimension, `B = O(1/N)`.

This is a genuine theorem-level improvement that stays fully inside the current
deterministic framework.

## 4. Why This Still Does Not Upgrade the Generic Finite-Horizon Theorem to `O(1/N)`

Even after the improvement above, the finite-horizon theorem still only sees

`W_1(rho_A^N, rho_B^N) = O(N^{-1/2} + sqrt(eps))`.

If the only available bridge from the shared path to alignment is

`B_N <= C W_1`,

then conditional head regularity yields at best

`transfer = O(B_N) = O(W_1) = O(N^{-1/2} + sqrt(eps))`.

So the best finite-horizon rate obtainable from the present `W_1` common-path
input is exactly the already stated fast rate:

`B = O(N^{-1/2} + sqrt(eps))`.

This is the key obstruction:

- improving the deterministic barrier theorem is not enough;
- the training/alignment input itself must become second-order.

## 5. What Stronger Input Would Actually Give `O(1/N)`

There are two natural routes.

### Route A: Endpoint `W_2` Hidden-Law Concentration

If one can show, along a width sequence, that the hidden empirical laws satisfy

`W_2(hat nu_A^N, mu_{xi,*}) = O(sqrt(d/N))`,
`W_2(hat nu_B^N, mu_{xi,*}) = O(sqrt(d/N))`,

then hidden-weight optimal transport gives

`B_N = W_2(hat nu_A^N, hat nu_B^N)^2 = O(d/N)`.

Combined with conditional head regularity and the new `Delta_N = O(B_N)` lemma,
this closes the deterministic width-sequence barrier at `O(d/N)`.

This route is the most realistic way to match an observed width-slope near `-1`
for trained endpoints.

What is still needed here is a quantitative theorem saying that the trained
hidden empirical law is close, in `W_2`, to a common mean-field hidden law.

### Route B: Finite-Horizon `W_2` or Mean-Square Common Path

To upgrade the actual SGD-to-LMC theorem to `O(1/N)`, one needs a stronger
training-side statement than the current `W_1` four-dynamics theorem. For
example, a theorem of the form

`sup_k W_2(hat nu_{A,k}^N, mu_{xi,k}) <= C_T sqrt(d/N + eps)`

would imply

`B_{N,k} = O(d/N + eps)`.

Together with head regularity and `Delta_N = O(B_N)`, this would yield

`B(Theta_{A,k}, Theta_{B,k}; varpi_{N,k}) = O(d/N + eps)`,

hence `O(1/N)` at fixed `d` and fixed finite horizon.

At present this is not a consequence of the imported bounded-support
four-dynamics theorem, which is only formulated in `W_1`.

## 6. Role of the Exact Conditional-Risk Modulus

The exact Gaussian modulus is extremely important numerically, but by itself it
does not fix the exponent problem.

Why:

- it removes worst-case Lipschitz slack;
- it keeps the path defects in exact pointwise form;
- but its leading terms still depend on the actual mean/log-variance path
  defects `delta_{m,t}`, `delta_{r,t}`.

If those defects are only controlled through first-order quantities, the exact
modulus improves constants, not asymptotic exponents.

Therefore the exact-modulus route should be viewed as:

- the right route for non-vacuous certificates;
- not the standalone route to a new width exponent.

To get a better exponent from the exact-modulus theorem, it must be combined
with a second-order control of the path defects, e.g. `delta_{m,t}`,
`delta_{r,t} = O(B_N)` in expectation or pointwise envelope form.

## 7. Most Plausible Research Program

The most grounded path forward is:

1. Add the missing `Delta_N`-from-`(M_{Delta V}^infty, B_N)` proposition to
   `k1_raw_lmc.tex`.
2. Simplify the deterministic fast-rate corollary so it depends only on
   `B_N = O(d/N)` plus head regularity.
3. Recast the theoretical target as an endpoint width-sequence theorem rather
   than an immediate full-SGD theorem. This better matches what the experiments
   are measuring.
4. Develop a separate theorem for hidden-marginal `W_2` concentration around
   the common mean-field law at the trained endpoint.
5. Only after that, revisit whether a full finite-horizon `O(1/N)` SGD-to-LMC
   theorem is worth the additional technical machinery.

## 8. Local Diagnostic From the Saved Workspace Runs

The saved synthetic width sweep in
`experiments/results/width_sweep_meanvar_synthetic`
does not yet appear to be in a clean `-1` regime on the currently stored runs.
Using the saved aggregates:

- dense barrier slope is about `-0.70`,
- `B_N` slope is about `-0.33`,
- `Delta_s_N` and `Delta_raw_N` slopes are about `-0.46`,
- the timewise exact-modulus bound slope is about `-0.57`.

The exact modulus is dominated by its mean-path term on those saved runs. This
is consistent with the theory diagnosis above: the dominant issue is the
transfer/common-path side, not the endpoint Jensen term.

If newer experiments outside the repository are closer to slope `-1`, that
would support investing effort in the endpoint `W_2` route rather than in
further sharpening the Gaussian semi-convexity constants.

## 9. External Theory Signals Worth Using

Two external references are directly relevant:

- Fournier and Guillin provide non-asymptotic Wasserstein convergence results
  for empirical measures, which are the natural starting point for endpoint
  `W_2` concentration.
- De Bortoli, Durmus, Fontaine, and Simsekli give quantitative propagation of
  chaos for SGD in wide neural networks, which is the natural starting point
  for strengthening the training-side approximation.

These do not by themselves close the desired `O(1/N)` finite-horizon barrier
theorem, but they point to the right objects: empirical `W_2` concentration and
quantitative chaos bounds stronger than the present `W_1` common-path theorem.
