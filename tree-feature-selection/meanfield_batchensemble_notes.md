# Mean-Field Start for Two-Layer BatchEnsemble / TabM-like Models

## Goal

We want a mean-field formulation for a two-layer network with:

- hidden features shared across ensemble members,
- member-specific multiplicative adapters,
- fixed ensemble size `M`,
- width `n -> infinity`.

This is the natural fixed-`M` starting point before considering deep models or `M -> infinity`.

## Minimal model

For each ensemble member `alpha in {1, ..., M}`, define

```math
f^n_\alpha(x)
= \frac{1}{n}\sum_{j=1}^n \psi_\alpha(x; \theta_j),
\qquad
\theta_j = \bigl(w_j, (a_{\alpha,j}, u_{\alpha,j}, v_{\alpha,j})_{\alpha=1}^M\bigr).
```

Use the feature map

```math
\psi_\alpha(x; \theta)
= a_\alpha u_\alpha \sigma(v_\alpha w^\top x).
```

Interpretation:

- `w_j` is the shared hidden feature for neuron `j`,
- `v_{alpha,j}` is a pre-activation modulation,
- `u_{alpha,j}` is a post-activation modulation,
- `a_{alpha,j}` is the member-specific output head coefficient.

This includes:

- a TabM-mini style two-layer model if `u_\alpha = 1`,
- a simpler BatchEnsemble-style hidden layer with independent output heads,
- a good fixed-`M` proxy for two-layer TabM.

If some coordinates are frozen rather than trained, remove them from the particle state or set their velocity to zero.

## Why fixed `M` is friendly

For fixed `M`, each neuron is still one particle in a finite-dimensional space:

```math
\Omega = \mathbb{R}^d \times (\mathbb{R}^3)^M.
```

The fact that parameters are "shared across members" does not break exchangeability across neurons:

- particles `theta_j` are i.i.d. at initialization,
- the dynamics are permutation-invariant in the neuron index `j`.

So standard two-layer mean-field / propagation-of-chaos machinery should be applicable under usual smoothness assumptions.

## Training objective

The BatchEnsemble / TabM-style training objective is naturally

```math
\mathcal{R}_{\mathrm{head}}(\rho)
= \mathbb{E}_{(x,y)}\left[\frac{1}{M}\sum_{\alpha=1}^M
\ell\bigl(f_\alpha(x;\rho), y\bigr)\right],
```

where

```math
f_\alpha(x;\rho)
= \int_\Omega \psi_\alpha(x;\theta)\,\rho(d\theta).
```

For comparison, the "mean prediction" objective would be

```math
\mathcal{R}_{\mathrm{mean}}(\rho)
= \mathbb{E}_{(x,y)}\left[
\ell\left(\frac{1}{M}\sum_{\alpha=1}^M f_\alpha(x;\rho), y\right)
\right].
```

These are not the same. The fixed-`M` theory should keep them separate.

## First variation for per-head loss

For

```math
\mathcal{R}_{\mathrm{head}}(\rho)
= \mathbb{E}\left[\frac{1}{M}\sum_{\alpha=1}^M
\ell(f_\alpha(x;\rho), y)\right],
```

the variational potential is formally

```math
V(\theta; \rho)
= \mathbb{E}_{(x,y)}\left[
\frac{1}{M}\sum_{\alpha=1}^M
\partial_1 \ell(f_\alpha(x;\rho), y)\,\psi_\alpha(x;\theta)
\right].
```

For squared loss `ell(z, y) = (z-y)^2 / 2`, this becomes

```math
V(\theta; \rho)
= \mathbb{E}_{(x,y)}\left[
\frac{1}{M}\sum_{\alpha=1}^M
\bigl(f_\alpha(x;\rho)-y\bigr)\psi_\alpha(x;\theta)
\right].
```

## Particle dynamics

The formal gradient-flow particle system is

```math
\dot{\theta}_j(t) = - \nabla_\theta V(\theta_j(t); \rho_t^n),
\qquad
\rho_t^n = \frac{1}{n}\sum_{j=1}^n \delta_{\theta_j(t)}.
```

For squared loss, the componentwise velocities are

```math
\dot{a}_\alpha
= - \mathbb{E}\left[\frac{1}{M}(f_\alpha(x)-y)\,u_\alpha \sigma(v_\alpha w^\top x)\right],
```

```math
\dot{u}_\alpha
= - \mathbb{E}\left[\frac{1}{M}(f_\alpha(x)-y)\,a_\alpha \sigma(v_\alpha w^\top x)\right],
```

```math
\dot{v}_\alpha
= - \mathbb{E}\left[\frac{1}{M}(f_\alpha(x)-y)\,
a_\alpha u_\alpha \sigma'(v_\alpha w^\top x)\,(w^\top x)\right],
```

```math
\dot{w}
= - \mathbb{E}\left[
\frac{1}{M}\sum_{\alpha=1}^M
(f_\alpha(x)-y)\,
a_\alpha u_\alpha \sigma'(v_\alpha w^\top x)\,v_\alpha x
\right].
```

Important structural point:

- member-specific coordinates are pushed by the member-specific residual,
- the shared coordinate `w` is pushed by the average over all members.

This is the mean-field analogue of shared-weight coupling.

## Mean-field PDE

Formally, `rho_t^n -> rho_t` as `n -> infinity`, where `rho_t` solves

```math
\partial_t \rho_t
= \nabla_\theta \cdot \left(\rho_t \nabla_\theta V(\theta; \rho_t)\right).
```

With entropy regularization / noisy SGD one would obtain an added diffusion term.

## Nonlinear kernel dynamics

Define the mean-field kernel

```math
K_{\alpha\beta}(x,x';\rho)
= \int_\Omega
\langle \nabla_\theta \psi_\alpha(x;\theta),
\nabla_\theta \psi_\beta(x';\theta)\rangle
\rho(d\theta).
```

Then the output dynamics becomes

```math
\partial_t f_\alpha(x;\rho_t)
= - \mathbb{E}_{(x',y)}\left[
\frac{1}{M}\sum_{\beta=1}^M
K_{\alpha\beta}(x,x';\rho_t)\,
\partial_1 \ell(f_\beta(x';\rho_t),y)
\right].
```

This is the clean bridge to the NTK picture:

- NTK: the kernel is frozen at initialization,
- mean-field: the kernel evolves with `rho_t`.

## Exchangeability and symmetry reduction

Let `Pi` be a permutation of the ensemble indices `{1, ..., M}` acting on particle coordinates by
permuting `(a_alpha, u_alpha, v_alpha)_alpha`.

If:

- the initial law `rho_0` is exchangeable under all such permutations,
- the loss is the same for each member,
- the training objective is the symmetric per-head objective `R_head`,

then the mean-field PDE preserves exchangeability.

Reason:

- the velocity field is equivariant with respect to ensemble-index permutations,
- by uniqueness of the PDE solution, `rho_t` must remain permutation-invariant.

Consequences:

```math
f_1(x;\rho_t) = \cdots = f_M(x;\rho_t) =: f(x,t),
```

and the kernels reduce to two values:

```math
K_{\alpha\alpha}(x,x';\rho_t) =: K_{\mathrm{d}}(x,x';t),
\qquad
K_{\alpha\beta}(x,x';\rho_t) =: K_{\mathrm{o}}(x,x';t)\ \ (\alpha \neq \beta).
```

Hence the whole fixed-`M` mean-field system collapses to the single scalar equation

```math
\partial_t f(x,t)
= - \mathbb{E}_{(x',y)}\left[
\frac{K_{\mathrm{d}}(x,x';t) + (M-1)K_{\mathrm{o}}(x,x';t)}{M}
\partial_1\ell(f(x',t), y)
\right].
```

This is a critical structural fact:

- the leading-order mean-field limit does not retain headwise diversity under symmetric initialization,
- instead, it retains an effective evolving kernel

```math
K_{\mathrm{eff}} = \frac{K_{\mathrm{d}} + (M-1)K_{\mathrm{o}}}{M}.
```

So mean-field is naturally suited to study shared feature learning, but not ensemble diversity itself unless one:

- breaks symmetry at initialization/objective/data assignment,
- studies finite-width fluctuations around mean-field,
- or takes a different large-system limit.

Concrete ways symmetry could be broken at leading order:

- member-dependent initialization laws,
- member-dependent data distributions or fixed bootstrap samples,
- member-dependent losses or regularizers,
- global member-specific latent variables that do not average out over neurons.

## Cross-member coupling

For the minimal model above, if `alpha != beta`, then the only shared trainable coordinate is `w`, so

```math
K_{\alpha\beta}(x,x';\rho)
= \int
\bigl[a_\alpha u_\alpha v_\alpha \sigma'(v_\alpha w^\top x)\bigr]
\bigl[a_\beta u_\beta v_\beta \sigma'(v_\beta w^\top x')\bigr]
\,(x^\top x')\,\rho(d\theta).
```

Thus:

- off-diagonal coupling is caused by shared feature learning,
- diagonal terms contain additional contributions from `a_\alpha`, `u_\alpha`, `v_\alpha`.

This is the mean-field counterpart of off-diagonal NTK blocks in Embedded Ensembles.

## Full BatchEnsemble vs. TabM-like independent heads

The output-layer design matters a lot.

### Case A: shared output coefficient (full BatchEnsemble-style proxy)

Consider

```math
\psi_\alpha(x;\theta)
= a\,u_\alpha \sigma(v_\alpha w^\top x),
```

where `a` is shared across members.

Then `K_{\alpha\beta}` for `alpha != beta` contains at least the shared-output contribution

```math
K_{\alpha\beta}^{(\mathrm{shared\ out})}(x,x';\rho)
= \int
u_\alpha \sigma(v_\alpha w^\top x)\,
u_\beta \sigma(v_\beta w^\top x')\,
\rho(d\theta),
```

coming from derivatives with respect to the shared output parameter `a`.

This term is generically nonzero unless the modulation is centered in a way that forces cancellation.
So full BatchEnsemble-style models can be collective already at initialization.

### Case B: independent output heads (TabM-like proxy)

Consider instead

```math
\psi_\alpha(x;\theta)
= a_\alpha u_\alpha \sigma(v_\alpha w^\top x).
```

Now the shared-output contribution disappears entirely, and for `alpha != beta` the leading off-diagonal term
comes only from shared hidden features:

```math
K_{\alpha\beta}(x,x';\rho)
= \int
\bigl[a_\alpha u_\alpha v_\alpha \sigma'(v_\alpha w^\top x)\bigr]
\bigl[a_\beta u_\beta v_\beta \sigma'(v_\beta w^\top x')\bigr]
(x^\top x')\,\rho(d\theta).
```

Therefore, if the independent head coefficients `a_\alpha` are initialized independently with zero mean,
it is much easier for the initial off-diagonal coupling to vanish.

### Consequence

This suggests a meaningful theoretical difference:

- full BatchEnsemble can exhibit `K_o(0) != 0` already because the output layer is shared,
- TabM-like models with independent output heads may start with `K_o(0) = 0` but create `K_o(t) > 0`
  dynamically through shared feature learning.

So, in mean-field terms, TabM may be "less collective at initialization" than full BatchEnsemble while still
becoming collective during training.

## Relation to the `gamma(M)` scaling intuition

Under the exchangeable reduction above,

```math
\partial_t f
\sim - K_{\mathrm{eff}} * \partial_1 \ell.
```

This immediately recovers the same scaling intuition as in Embedded Ensembles:

- if `K_o = 0` (independent-like regime), then

```math
K_{\mathrm{eff}} = K_{\mathrm{d}}/M,
```

so one should rescale time / learning rate by a factor of `M` to match single-model dynamics;

- if `K_o` stays of the same order as `K_d` (collective regime), then

```math
K_{\mathrm{eff}} = O(1),
```

so no extra factor `M` is natural.

Thus the `gamma(M)=M` versus `gamma(M)=1` dichotomy has a direct mean-field analogue.

## A critical caveat: generic symmetry collapse

The exchangeable fixed-`M` mean-field limit predicts identical outputs across members.
This means that the practical advantage of an ensemble is not visible at leading order unless symmetry is broken.

This is not necessarily a bug of the formulation. Rather, it is telling us that:

- feature learning may survive the limit,
- diversity is likely a sub-leading effect in width for symmetric ensembles trained on the same task.

This is very similar in spirit to what happens in strict infinite-width NTK analyses.

## Small-time creation of off-diagonal coupling

Even if the off-diagonal coupling is zero at `t=0`, it is not obvious that it remains zero under mean-field
training.

For a simplified model with trainable shared feature `w`, trainable per-head coefficient `a_alpha`, and

```math
\psi_\alpha(x;\theta) = a_\alpha \varphi(w,x),
```

one has

```math
\dot a_\alpha(t)
= - \mathbb{E}_{(x,y)}\left[
\frac{1}{M}\partial_1 \ell(f(x,t),y)\,\varphi(w,x)
\right]
```

under the exchangeable reduction `f_1 = ... = f_M = f`.

The right-hand side does not depend on `a_\alpha`, so all heads are shifted in the same direction.
Therefore, even if

```math
\mathbb{E}[a_\alpha a_\beta] = 0
\qquad (\alpha \neq \beta)
```

at time `t=0`, one generically expects

```math
\mathbb{E}[a_\alpha(t)a_\beta(t)] > 0
```

for small `t > 0`, unless the common drift vanishes identically.

Heuristically,

```math
a_\alpha(t) = a_\alpha(0) - t c(w) + O(t^2),
```

so for `alpha != beta`,

```math
\mathbb{E}[a_\alpha(t)a_\beta(t)G(w)]
= t^2 \mathbb{E}[c(w)^2 G(w)] + O(t^3)
```

for suitable test functions `G`, provided the initial cross-moments vanish.

This suggests:

- `K_o(0)=0` does not imply `K_o(t)=0`,
- genuinely independent behavior is likely fragile in the mean-field feature-learning regime.

This is one of the main places where the mean-field picture may differ sharply from the frozen-NTK picture.

## Comparison of losses

For `R_head`, the residual entering the velocity field is memberwise:

```math
\partial_1 \ell(f_\alpha(x), y).
```

For `R_mean`, the residual is shared:

```math
\partial_1 \ell\left(\frac{1}{M}\sum_\beta f_\beta(x), y\right).
```

So `R_mean` couples members directly through the loss, while `R_head` couples them only through shared particle coordinates and the evolving kernel.

This distinction should matter for TabM because its practical objective is of the `R_head` type.

However, under the exchangeability collapse `f_1 = \cdots = f_M = f`, one has

```math
\mathcal R_{\mathrm{head}}(\rho_t)
= \mathbb E[\ell(f(x,t), y)]
= \mathcal R_{\mathrm{mean}}(\rho_t).
```

So, at deterministic mean-field level with symmetric initialization, even the distinction between

- "mean of per-head losses", and
- "loss of the mean prediction"

disappears.

This is another sign that:

- deterministic mean-field is good for shared feature learning,
- but too coarse to capture ensemble-specific training benefits that rely on headwise diversity.

## What seems rigorous vs. what is still open

Likely standard / accessible:

- existence of a fixed-`M` mean-field limit for smooth activations and losses,
- PDE formulation,
- nonlinear kernel dynamics,
- symmetry preservation under permutation of ensemble members.

Likely nontrivial / new:

- characterization of "independent vs collective" regimes in the mean-field PDE,
- proof that exchangeable initialization collapses headwise outputs in the deterministic limit,
- a fluctuation theory capturing ensemble diversity beyond deterministic mean-field,
- precise conditions under which off-diagonal kernels stay zero or become nonzero,
- analysis of TabM-style non-centered initialization (many adapters initialized at `1`),
- finite-width corrections and comparison with NTK / lazy training.

## Suggested next theorem targets

1. Mean-field limit theorem for fixed `M` two-layer BatchEnsemble / TabM-mini.
2. Exchangeability theorem implying `f_1 = ... = f_M` in the deterministic limit.
3. Explicit formula for `K_{alpha,beta}` and a criterion for vanishing / creation of off-diagonal coupling.
4. Comparison theorem between `R_head` and `R_mean`.
5. Fluctuation or CLT correction as the first place where ensemble diversity can survive.
