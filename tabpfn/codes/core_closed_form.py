"""Closed-form oracle calculations for Beta-Bernoulli synthetic prior experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln, logsumexp


EPS = 1e-12

BetaPrior = tuple[float, float]
Predictor = Callable[[int, int], float]


@dataclass(frozen=True)
class NamedPredictor:
    """Convenience wrapper used by the experiment scripts."""

    name: str
    predictor: Predictor

    def __call__(self, s: int, n: int) -> float:
        return self.predictor(s, n)


def _as_float_array(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional numeric sequence.")
    return array


def _validate_beta_parameters(a: float, b: float) -> None:
    if a <= 0 or b <= 0:
        raise ValueError(f"Beta parameters must be positive, got {(a, b)}.")


def _clip_probability(p: np.ndarray | float) -> np.ndarray | float:
    return np.clip(p, EPS, 1.0 - EPS)


def single_beta_predictive(a: float, b: float, s: int, n: int) -> float:
    """Exact predictive mean for Beta(a, b) after observing s successes in n trials."""

    _validate_beta_parameters(a, b)
    if n < 0:
        raise ValueError("n must be non-negative.")
    if s < 0 or s > n:
        raise ValueError(f"s must satisfy 0 <= s <= n, got s={s}, n={n}.")
    return float((a + s) / (a + b + n))


def make_single_predictor(a: float, b: float, name: str | None = None) -> NamedPredictor:
    """Returns a named predictor callable for a single Beta prior."""

    predictor_name = name or f"Beta({a},{b})"
    return NamedPredictor(
        name=predictor_name,
        predictor=lambda s, n: single_beta_predictive(a=a, b=b, s=s, n=n),
    )


def mixture_posterior_weights(
    components: Sequence[BetaPrior],
    weights: Sequence[float],
    s: int,
    n: int,
) -> np.ndarray:
    """Posterior component weights for a finite Beta mixture after observing (s, n)."""

    if n < 0:
        raise ValueError("n must be non-negative.")
    if s < 0 or s > n:
        raise ValueError(f"s must satisfy 0 <= s <= n, got s={s}, n={n}.")

    if len(components) == 0:
        raise ValueError("At least one component is required.")

    weights_array = _as_float_array(weights)
    if len(components) != len(weights_array):
        raise ValueError("components and weights must have the same length.")
    if np.any(weights_array < 0):
        raise ValueError("Mixture weights must be non-negative.")
    if np.all(weights_array == 0):
        raise ValueError("At least one mixture weight must be positive.")

    normalized_weights = weights_array / np.sum(weights_array)
    components_array = np.asarray(components, dtype=float)
    if components_array.ndim != 2 or components_array.shape[1] != 2:
        raise ValueError("components must be a sequence of (a, b) pairs.")

    a = components_array[:, 0]
    b = components_array[:, 1]
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("All Beta parameters must be positive.")

    log_posterior = (
        np.log(normalized_weights)
        + betaln(a + s, b + n - s)
        - betaln(a, b)
    )
    log_posterior -= logsumexp(log_posterior)
    return np.exp(log_posterior)


def mixture_beta_predictive(
    components: Sequence[BetaPrior],
    weights: Sequence[float],
    s: int,
    n: int,
) -> float:
    """Exact predictive mean for a finite mixture of Beta priors."""

    posterior_weights = mixture_posterior_weights(components=components, weights=weights, s=s, n=n)
    components_array = np.asarray(components, dtype=float)
    predictive_means = (components_array[:, 0] + s) / (components_array[:, 0] + components_array[:, 1] + n)
    return float(np.sum(posterior_weights * predictive_means))


def make_mixture_predictor(
    components: Sequence[BetaPrior],
    weights: Sequence[float],
    name: str,
) -> NamedPredictor:
    """Returns a named predictor callable for a finite Beta mixture."""

    normalized_weights = _as_float_array(weights)
    normalized_weights = normalized_weights / np.sum(normalized_weights)
    component_tuple = tuple((float(a), float(b)) for a, b in components)
    weight_tuple = tuple(float(w) for w in normalized_weights.tolist())
    return NamedPredictor(
        name=name,
        predictor=lambda s, n: mixture_beta_predictive(
            components=component_tuple,
            weights=weight_tuple,
            s=s,
            n=n,
        ),
    )


def beta_binomial_pmf(s: int | np.ndarray, n: int, c: float, d: float) -> np.ndarray:
    """Beta-binomial pmf under deployment prior Beta(c, d)."""

    _validate_beta_parameters(c, d)
    if n < 0:
        raise ValueError("n must be non-negative.")

    s_array = np.asarray(s, dtype=int)
    if np.any(s_array < 0) or np.any(s_array > n):
        raise ValueError("All s values must satisfy 0 <= s <= n.")

    log_pmf = (
        gammaln(n + 1)
        - gammaln(s_array + 1)
        - gammaln(n - s_array + 1)
        + betaln(c + s_array, d + n - s_array)
        - betaln(c, d)
    )
    return np.exp(log_pmf)


def bernoulli_kl(p: float | np.ndarray, q: float | np.ndarray) -> np.ndarray:
    """KL(Ber(p) || Ber(q))."""

    p_array = _clip_probability(np.asarray(p, dtype=float))
    q_array = _clip_probability(np.asarray(q, dtype=float))
    return p_array * np.log(p_array / q_array) + (1.0 - p_array) * np.log((1.0 - p_array) / (1.0 - q_array))


def delta_n(deploy_prior: BetaPrior, predictor: Predictor, n: int) -> float:
    """One-step oracle gap against the deployment oracle at prefix length n."""

    c, d = deploy_prior
    s_values = np.arange(n + 1, dtype=int)
    deployment_pmf = beta_binomial_pmf(s=s_values, n=n, c=c, d=d)
    oracle_predictive = np.array(
        [single_beta_predictive(c, d, int(s), n) for s in s_values],
        dtype=float,
    )
    candidate_predictive = np.array([predictor(int(s), n) for s in s_values], dtype=float)
    gaps = bernoulli_kl(oracle_predictive, candidate_predictive)
    return float(np.sum(deployment_pmf * gaps))


def cumulative_gap(
    deploy_prior: BetaPrior,
    predictor: Predictor,
    n_grid: Iterable[int],
) -> pd.DataFrame:
    """Returns delta_n and cumulative sums on the provided n-grid."""

    rows: list[dict[str, float | int]] = []
    running_total = 0.0
    for n in n_grid:
        gap = delta_n(deploy_prior=deploy_prior, predictor=predictor, n=int(n))
        running_total += gap
        rows.append(
            {
                "n": int(n),
                "delta_n": float(gap),
                "cumulative_gap": float(running_total),
            }
        )
    return pd.DataFrame(rows)


def oracle_budget(
    deploy_prior: BetaPrior,
    predictor: Predictor,
    horizon: int,
) -> float:
    """Returns the exact oracle budget K_horizon = sum_{n=0}^{horizon-1} delta_n."""

    if horizon < 0:
        raise ValueError("horizon must be non-negative.")

    running_total = 0.0
    for n in range(horizon):
        running_total += delta_n(deploy_prior=deploy_prior, predictor=predictor, n=n)
    return float(running_total)


def small_count_profile(
    predictor: Predictor,
    n_grid: Iterable[int],
    s_values: Sequence[int] = (0, 1, 2),
) -> pd.DataFrame:
    """Returns q_hat and n*q_hat for the specified small-count events."""

    rows: list[dict[str, float | int]] = []
    for n in n_grid:
        n_int = int(n)
        for s in s_values:
            s_int = int(s)
            if s_int > n_int:
                continue
            predictive = float(predictor(s_int, n_int))
            rows.append(
                {
                    "n": n_int,
                    "s": s_int,
                    "predictive": predictive,
                    "scaled_predictive": n_int * predictive,
                }
            )
    return pd.DataFrame(rows)
