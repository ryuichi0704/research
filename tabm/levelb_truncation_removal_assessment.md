# Level-B Truncation Removal Assessment

## Executive summary

The truncation in the current Level-B theorem appears removable for the **unregularized**
TabM-like two-layer Level-B system, but not with the current proof architecture unchanged.

The cleanest route is:

- keep the current bounded-input, bounded-`C^2`-activation, bounded/Lipschitz loss assumptions;
- strengthen the initial-data assumptions to
  - uniformly bounded output-head coordinates `a_alpha` at time `0`,
  - finite second moment in `(w,b)`,
  - finite initial head vector `r_0`;
- replace the current global `W_1 + ||·||` metric by a **moment-aware** metric, most naturally
  `W_2 + ||·||` on bounded-moment sets.

Under these assumptions, the main obstruction created by the untruncated system is not blow-up of
the neuron drift itself. The real difficulty is the head drift

`G_alpha(rho,r) = - E[ (1/M) l'(f_alpha) ∫ a_alpha sigma'(w·(r_alpha⊙x)+b) (w⊙x) rho(dvartheta) ]`,

because it contains the unbounded observable `a_alpha w`. This breaks the current bounded-Lipschitz
contraction argument in `W_1`. However, the system still seems controllable by:

- a pointwise uniform bound on `a_alpha(t)`,
- first/second moment propagation for `w,b`,
- a Gronwall system coupling `∫|w| rho_t(dw db da)` and `||r_t||`,
- local contraction on bounded moment balls,
- continuation in time.

## Where the current proof uses truncation

The current Level-B theorem uses truncation in exactly the places where global boundedness and
global Lipschitzness are needed:

- `tabm_meanfield_initial_step.tex`, Theorem `thm:levelb-wellposed`
- `tabm_meanfield_initial_step.tex`, proof lines asserting that `Psi_{alpha,R}` and its first derivatives
  are globally bounded and globally Lipschitz
- the stability proposition `prop:levelb-stability`
- the empirical-limit corollary `cor:levelb-empirical-limit`

In the untruncated model:

- `Psi_alpha(x;vartheta,r) = a_alpha sigma(w·(r_alpha⊙x)+b)` is unbounded because of `a_alpha`;
- `G_alpha` contains `a_alpha w`, so even with bounded `sigma'`, the head drift is not globally bounded;
- the map `(rho,r) -> G_alpha(rho,r)` is not globally Lipschitz in `W_1`.

## Untruncated drift structure

Write

- `A(vartheta) := sum_alpha |a_alpha|`,
- `z_alpha := w·(r_alpha⊙x)+b`.

Under bounded inputs `||x|| <= X_*`, bounded `sigma, sigma', sigma''`, and bounded `l'`, we have:

- `|dot a_alpha| <= C`
- `|dot b| <= C A(vartheta)`
- `|dot w| <= C A(vartheta) ||r||`
- `|dot r_alpha| <= C ∫ |a_alpha| |w| rho(dvartheta)`

So the neuron coordinates `(w,b)` do **not** generate superlinear self-interaction.
The only new coupling is through the finite-dimensional head variable `r`, and that coupling is driven
by the moment `∫ |a| |w| d rho`.

## The key a priori observation

If the initial law satisfies a uniform bound

`|a_alpha(0)| <= A_0` almost surely for every `alpha`,

then along any characteristic of the untruncated Level-B system,

`|a_alpha(t)| <= A_0 + C t`.

Hence on every finite interval `[0,T]`, the output coefficients remain uniformly bounded by a
deterministic constant `A_T`.

This is the main reason truncation removal looks feasible: once `a_alpha` is uniformly bounded on
`[0,T]`, the neuron drift `B` becomes globally Lipschitz in the particle variable `vartheta`, provided
`||r_t||` stays bounded.

## Finite-horizon moment closure

Assume now:

- `sup_alpha |a_alpha(0)| <= A_0` almost surely,
- `rho_0` has finite second moment in `(w,b)`,
- `r_0` is finite.

Then on `[0,T]`:

1. Uniform `a` bound:

   `A(vartheta_t) <= A_T := M(A_0 + C T)`.

2. First-moment bound for `w`:

   `d/dt ∫ |w| rho_t(dvartheta) <= C_T ||r_t||`.

3. Head bound:

   `d/dt ||r_t|| <= C_T ∫ |w| rho_t(dvartheta)`.

Therefore, if

`m_w(t) := ∫ |w| rho_t(dvartheta)`,

then

- `m_w'(t) <= C_T ||r_t||`,
- `||r_t||' <= C_T m_w(t)`.

Hence `m_w(t) + ||r_t||` obeys a closed Gronwall inequality and stays finite on every finite horizon.

Once `||r_t||` is bounded, we also get:

- `|dot b| <= C_T`,
- `|dot w| <= C_T`,

and therefore second moments of `(w,b)` propagate as well.

## Why `W_1 + ||·||` is no longer enough

The current proof uses the fact that truncated observables are bounded-Lipschitz.
Without truncation, the head drift contains

`h_{alpha,x,r}(vartheta) = a_alpha sigma'(w·(r_alpha⊙x)+b) (w⊙x)`.

Even after `a_alpha` is uniformly bounded, the derivative of `h_{alpha,x,r}` with respect to `w`
contains a term of size `O(1 + |w|)`, because of the product of `w` with `sigma'(z_alpha)`.

So:

- `h_{alpha,x,r}` is not globally Lipschitz with a uniform constant,
- a pure `W_1` estimate does not close,
- the natural stability space is a bounded-moment subset of `P_2(Theta)`.

This is the main technical change forced by truncation removal.

## Candidate replacement theorem

### Candidate theorem (finite-horizon untruncated Level-B well-posedness)

Assume:

- `||x|| <= X_*` on `X`,
- `sigma in C_b^2(R)` with bounded `sigma, sigma', sigma''`,
- `partial_1 l(·,y)` globally Lipschitz and uniformly bounded in `y`,
- `rho_0 in P_2(Theta)`,
- the `a_alpha` coordinates of `rho_0` are essentially bounded,
- `r_0 in R = (R^d)^M`.

Then for every `T > 0`, the untruncated Level-B system admits a unique solution

`(rho_t, r_t)_{t in [0,T]} in C([0,T], P_2(Theta) x R)`.

Moreover:

- the `a_alpha` coordinates remain uniformly bounded on `[0,T]`,
- the second moments of `(w,b)` remain finite on `[0,T]`,
- permutation symmetry is preserved exactly as in the truncated theorem.

### Candidate stability theorem

On every finite interval `[0,T]`, if two solutions start from initial data in the same bounded-moment
class, then

`W_2(rho_t, mu_t) + ||r_t - s_t||`

is controlled by the same quantity at time `0`, with a constant depending on `T` and on the a priori
moment bounds.

### Candidate finite-width limit

The empirical particle/head system should converge deterministically to the untruncated Level-B limit
on every compact time interval, again in the metric `W_2 + ||·||`, provided the empirical initial data
converge in that metric and inherit the same uniform `a` bound / moment assumptions.

## Proof architecture that seems to work

1. **A priori bounds**

   Prove:

   - uniform finite-horizon bound on all `a_alpha`,
   - Gronwall closure for `m_w(t) + ||r_t||`,
   - propagation of second moments of `(w,b)`.

2. **Local Lipschitz estimates on bounded balls**

   On sets where:

   - `|a_alpha| <= A_T`,
   - `||r|| <= R_T`,
   - `∫ (|w|^2 + |b|^2) d rho <= M_T`,

   show:

   - `B(vartheta;rho,r)` is globally Lipschitz in `vartheta`,
   - `B` is Lipschitz in `(rho,r)` with respect to `W_2 + ||·||`,
   - `G(rho,r)` is Lipschitz in `(rho,r)` with the same metric.

3. **Short-time fixed point**

   Given a path `(rho_hat, r_hat)` in a bounded ball, solve:

   - the characteristic ODE with drift `B(·; rho_hat_t, r_hat_t)`,
   - the finite-dimensional ODE for `r`.

   Show that for small time this map sends the ball to itself and is a contraction.

4. **Continuation**

   Use the a priori bounds to iterate the local construction and obtain existence/uniqueness on
   arbitrary finite horizons.

5. **Stability and empirical limit**

   Couple two characteristic flows and use the bounded-ball Lipschitz estimates plus Gronwall.

## What is genuinely new in Level-B compared with standard two-layer mean-field

Relative to the standard two-layer mean-field arguments in Mei--Misiakiewicz--Montanari (2019),
Level-B has one extra obstacle:

- the global head variable `r_alpha` is not a particle coordinate,
- its drift depends on the unbounded mixed observable `a_alpha w`,
- this forces a coupled moment bootstrap between the particle law and the finite-dimensional head ODE.

This obstacle looks real but manageable.
It does **not** currently look like a fatal obstruction.

## What I believe is not currently justified

I do **not** think the following stronger statement is currently justified without more work:

> The present Level-B theorem can be upgraded to the fully untruncated system while keeping exactly the
> same assumptions and the same `W_1 + ||·||` proof.

That is too optimistic, because the head drift is not bounded-Lipschitz in the relevant sense.

## Strongest safe conclusion

The truncation-removal problem appears solvable, but the proof should be rewritten around:

- uniform finite-horizon bounds on the `a` coordinates,
- propagation of first/second moments for `(w,b)`,
- a bounded-moment `P_2` framework,
- local contraction + continuation.

In short:

**Removing the Level-B truncation looks mathematically viable without changing the model, but it
requires a different theorem statement and a different stability metric.**

## Sources consulted

- Mei, Misiakiewicz, Montanari (COLT 2019), *Mean-field theory of two-layers neural networks:
  dimension-free bounds and kernel limit*.
- Chizat, Bach (NeurIPS 2018 / arXiv 1805.09545), *On the Global Convergence of Gradient Descent
  for Over-parameterized Models using Optimal Transport*.
- Erny (SPA 2022), *Well-posedness and propagation of chaos for McKean–Vlasov equations with jumps
  and locally Lipschitz coefficients*.

