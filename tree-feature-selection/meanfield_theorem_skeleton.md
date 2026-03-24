# Theorem Skeleton for Fixed-M Two-Layer BatchEnsemble Mean-Field Analysis

## Baseline setup

Take fixed ensemble size `M`.

For `theta = (w, (a_alpha, u_alpha, v_alpha)_{alpha=1}^M) in Omega`, define

```math
\psi_\alpha(x;\theta) = a_\alpha u_\alpha \sigma(v_\alpha w^\top x),
\qquad
f_\alpha(x;\rho) = \int \psi_\alpha(x;\theta)\,\rho(d\theta).
```

Define the per-head population risk

```math
\mathcal R(\rho)
= \mathbb E_{(x,y)}\left[
\frac1M \sum_{\alpha=1}^M \ell(f_\alpha(x;\rho), y)
\right].
```

Assumptions for the first rigorous pass:

- data distribution supported on a compact subset of `R^d x R`,
- `sigma in C_b^2(R)` or at least `C^2` with controlled growth,
- `ell in C_b^2(R x R)` or `C^2` with Lipschitz derivative,
- initial law `rho_0` has finite second moment,
- if needed, add confining regularization to guarantee global well-posedness.

The mean-field potential is

```math
V(\theta;\rho)
= \mathbb E_{(x,y)}\left[
\frac1M\sum_{\alpha=1}^M
\partial_1 \ell(f_\alpha(x;\rho), y)\,\psi_\alpha(x;\theta)
\right].
```

The formal PDE is

```math
\partial_t \rho_t
= \nabla_\theta \cdot \left(\rho_t \nabla_\theta V(\theta;\rho_t)\right).
```

## Theorem 1: Mean-field limit

### Statement

Let `(theta_j(0))_{j=1}^n` be i.i.d. with law `rho_0`, and let the particle system evolve by

```math
\dot \theta_j(t) = - \nabla_\theta V(\theta_j(t); \rho_t^n),
\qquad
\rho_t^n = \frac1n \sum_{j=1}^n \delta_{\theta_j(t)}.
```

Then for every finite time horizon `T > 0`:

- the PDE has a unique solution `rho_t` on `[0,T]`,
- `rho_t^n` converges to `rho_t` in Wasserstein distance, uniformly on `[0,T]`,
- for every compact set `K subset R^d`,

```math
\sup_{t \in [0,T]} \sup_{x \in K}
\max_{\alpha \le M}
\left|f_\alpha(x;\rho_t^n) - f_\alpha(x;\rho_t)\right|
\to 0.
```

### Proof strategy

1. Rewrite the particle system as an interacting diffusion/ODE in `R^D` with `D = d + 3M`.
2. Show the velocity field is locally Lipschitz in `(theta, rho)` under the smoothness assumptions.
3. Apply standard propagation-of-chaos / McKean-Vlasov theory for deterministic interacting particles.
4. Deduce convergence of the outputs by continuity of `psi_alpha`.

### References to adapt

- Mei, Montanari, Nguyen (2018): two-layer mean-field PDE.
- Mei, Misiakiewicz, Montanari (2019): dimension-free bounds and kernel limit.
- Chen, Cao, Gu, Zhang (2020): non-asymptotic rates and generalization bounds.

## Theorem 2: Exchangeability collapse

### Statement

Assume `rho_0` is invariant under every permutation of the member index `alpha`, i.e.

```math
Pi_# rho_0 = rho_0
```

for all `Pi in S_M`.

Then the PDE solution remains exchangeable:

```math
Pi_# rho_t = rho_t
\qquad \forall t >= 0.
```

Hence

```math
f_1(x;\rho_t) = \cdots = f_M(x;\rho_t),
```

and the kernel reduces to diagonal/off-diagonal values

```math
K_d(x,x';t), \qquad K_o(x,x';t).
```

The common predictor `f(x,t)` solves

```math
\partial_t f(x,t)
= - \mathbb E_{(x',y)}\left[
\frac{K_d(x,x';t) + (M-1)K_o(x,x';t)}{M}
\partial_1 \ell(f(x',t), y)
\right].
```

### Proof strategy

1. Check equivariance of `V` under member-index permutations.
2. Show that if `rho_t` solves the PDE, then `Pi_# rho_t` also solves it.
3. Use uniqueness to conclude `Pi_# rho_t = rho_t`.
4. Symmetry of `f_alpha` and `K_{alpha,beta}` follows immediately.

### Interpretation

At deterministic mean-field level, the ensemble collapses to one scalar predictor with an effective
kernel. Diversity does not survive unless symmetry is broken or fluctuations are retained.
In particular, under this collapse the per-head loss and the loss of the mean prediction coincide.

## Proposition 3: Mean-field analogue of the `gamma(M)` dichotomy

Under the reduced equation,

```math
K_eff = \frac{K_d + (M-1)K_o}{M}.
```

Two regimes emerge:

- if `K_o = 0`, then `K_eff = K_d / M`, so one needs time/lr rescaling by `M`,
- if `K_o = Theta(1)`, then `K_eff = Theta(1)`, so no extra scaling by `M` is natural.

This is the direct mean-field counterpart of the `gamma(M)=M` versus `gamma(M)=1` rule in
Embedded Ensembles.

## Proposition 4: Creation of off-diagonal coupling

### Simplified model

Take

```math
\psi_\alpha(x;\theta) = a_\alpha \varphi(w,x),
```

with trainable `a_alpha` and `w`, and assume exchangeable initial law with

```math
\mathbb E[a_\alpha a_\beta G(w)] = 0
\qquad (\alpha \neq \beta)
```

for a class of test functions `G`.

### Formal claim

If the common drift

```math
c(w,t)
= \mathbb E_{(x,y)}\left[
\frac1M \partial_1 \ell(f(x,t), y)\,\varphi(w,x)
\right]
```

is not identically zero at `t=0`, then for generic `G`

```math
\mathbb E[a_\alpha(t)a_\beta(t)G(w)] = t^2 A_G + O(t^3),
\qquad
A_G > 0
```

for `alpha != beta`.

### Interpretation

Even when cross-member coupling vanishes at initialization, training can create it immediately.
This makes the independent regime fragile in the feature-learning mean-field setting.

## Corollary 5: Full BatchEnsemble vs TabM-like heads

### Full BatchEnsemble proxy

If

```math
\psi_\alpha(x;\theta) = a\,u_\alpha \sigma(v_\alpha w^\top x)
```

with shared `a`, then off-diagonal coupling contains a direct shared-output term

```math
\int
u_\alpha \sigma(v_\alpha w^\top x)\,
u_\beta \sigma(v_\beta w^\top x')\,\rho(d\theta),
```

which can be nonzero already at `t=0`.

### TabM-like proxy

If

```math
\psi_\alpha(x;\theta) = a_\alpha u_\alpha \sigma(v_\alpha w^\top x),
```

then the shared-output term disappears and initial off-diagonal coupling is easier to kill by centered
independent head initialization. The coupling can still appear dynamically via the shared `w`.

## What still needs real work

1. Pick the exact architecture to analyze first:
   - safest: TabM-mini style two-layer model with `u_alpha = 1`,
   - more general: trainable `u_alpha`, `v_alpha`.
2. Decide the theorem regularity class:
   - easiest rigorous version: smooth bounded activation,
   - practical version: ReLU via approximation / subgradient / growth estimates.
3. Prove uniqueness and stability of the PDE under our specific nonlinearity.
4. Decide whether to stop at deterministic mean-field or push to fluctuations.

If we push to fluctuations, the most relevant templates appear to be:

- Sirignano and Spiliopoulos (2020): CLT around the two-layer mean-field limit.
- Bordelon and Pehlevan (2024): finite-width kernel / prediction fluctuations in feature-learning regimes.

## Best near-term paper framing

The strongest current framing is not:

- "BatchEnsemble has a mean-field limit."

It is:

- "The fixed-`M` deterministic mean-field limit of symmetric BatchEnsemble/TabM collapses headwise diversity,
  reduces to a single effective kernel dynamics, and predicts when collective coupling is created."

This is both more precise and more distinctive.
