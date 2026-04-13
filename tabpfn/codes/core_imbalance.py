"""Oracle utilities for TabPFN-free imbalance experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import betaln, digamma, expit, logit

from core_closed_form import beta_binomial_pmf


EPS = 1e-12

BetaPrior = tuple[float, float]


def _clip_probability(p: np.ndarray | float) -> np.ndarray | float:
    return np.clip(p, EPS, 1.0 - EPS)


def binary_entropy(p: np.ndarray | float) -> np.ndarray | float:
    p_array = _clip_probability(np.asarray(p, dtype=float))
    return -p_array * np.log(p_array) - (1.0 - p_array) * np.log(1.0 - p_array)


def cross_entropy(p: np.ndarray | float, q: np.ndarray | float) -> np.ndarray | float:
    p_array = np.asarray(p, dtype=float)
    q_array = _clip_probability(np.asarray(q, dtype=float))
    return -p_array * np.log(q_array) - (1.0 - p_array) * np.log(1.0 - q_array)


def brier_risk(p: np.ndarray | float, q: np.ndarray | float) -> np.ndarray | float:
    p_array = np.asarray(p, dtype=float)
    q_array = np.asarray(q, dtype=float)
    return p_array * (1.0 - 2.0 * q_array) + q_array**2


def beta_mean(a: float, b: float) -> float:
    return float(a / (a + b))


def beta_variance(a: float, b: float) -> float:
    total = a + b
    return float((a * b) / (total**2 * (total + 1.0)))


def beta_expected_bernoulli_variance(a: float, b: float) -> float:
    total = a + b
    return float((a * b) / (total * (total + 1.0)))


def beta_expected_binary_entropy(a: float, b: float) -> float:
    total = a + b
    return float(
        digamma(total + 1.0)
        - (a * digamma(a + 1.0) + b * digamma(b + 1.0)) / total
    )


def natural_predictive_mean(prior: BetaPrior, s: int, n: int) -> float:
    c, d = prior
    return beta_mean(c + s, d + n - s)


def natural_expected_metrics(prior: BetaPrior, n: int) -> dict[str, float]:
    c, d = prior
    s_values = np.arange(n + 1, dtype=int)
    pmf = beta_binomial_pmf(s=s_values, n=n, c=c, d=d)
    predictive = (c + s_values) / (c + d + n)
    posterior_variance = ((c + s_values) * (d + n - s_values)) / (
        (c + d + n) ** 2 * (c + d + n + 1.0)
    )
    return {
        "log_loss": float(np.sum(pmf * binary_entropy(predictive))),
        "brier": float(np.sum(pmf * predictive * (1.0 - predictive))),
        "theta_rmse": float(np.sqrt(np.sum(pmf * posterior_variance))),
    }


def balanced_expected_metrics(prior: BetaPrior) -> dict[str, float]:
    c, d = prior
    mean = beta_mean(c, d)
    return {
        "log_loss": float(binary_entropy(mean)),
        "brier": float(mean * (1.0 - mean)),
        "theta_rmse": float(np.sqrt(beta_variance(c, d))),
    }


def true_prevalence_token_metrics(prior: BetaPrior) -> dict[str, float]:
    c, d = prior
    return {
        "log_loss": beta_expected_binary_entropy(c, d),
        "brier": beta_expected_bernoulli_variance(c, d),
        "theta_rmse": 0.0,
    }


@dataclass(frozen=True)
class NoisyLogitTokenModel:
    prior: BetaPrior
    sigma: float
    z_grid: np.ndarray
    posterior_mean_grid: np.ndarray

    @classmethod
    def build(
        cls,
        prior: BetaPrior,
        sigma: float,
        theta_grid_size: int = 4096,
        z_grid_size: int = 2401,
        eps: float = 1e-8,
    ) -> "NoisyLogitTokenModel":
        if sigma <= 0:
            raise ValueError("sigma must be positive.")

        c, d = prior
        theta_grid = expit(np.linspace(logit(eps), logit(1.0 - eps), theta_grid_size))
        theta_logit = logit(theta_grid)
        prior_density = np.exp(
            (c - 1.0) * np.log(theta_grid)
            + (d - 1.0) * np.log1p(-theta_grid)
            - betaln(c, d)
        )

        z_min = float(theta_logit[0] - 5.0 * sigma)
        z_max = float(theta_logit[-1] + 5.0 * sigma)
        z_grid = np.linspace(z_min, z_max, z_grid_size)
        diff = (z_grid[:, None] - theta_logit[None, :]) / sigma
        gaussian = np.exp(-0.5 * diff**2) / (sigma * np.sqrt(2.0 * np.pi))
        joint_density = gaussian * prior_density[None, :]
        denominator = np.trapz(joint_density, theta_grid, axis=1)
        numerator = np.trapz(joint_density * theta_grid[None, :], theta_grid, axis=1)
        posterior_mean = numerator / np.maximum(denominator, EPS)
        return cls(
            prior=prior,
            sigma=float(sigma),
            z_grid=z_grid,
            posterior_mean_grid=posterior_mean,
        )

    def sample_token(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        theta_array = _clip_probability(np.asarray(theta, dtype=float))
        return logit(theta_array) + rng.normal(loc=0.0, scale=self.sigma, size=theta_array.shape)

    def predict(self, z: np.ndarray | float) -> np.ndarray:
        z_array = np.asarray(z, dtype=float)
        return np.interp(
            z_array,
            self.z_grid,
            self.posterior_mean_grid,
            left=self.posterior_mean_grid[0],
            right=self.posterior_mean_grid[-1],
        )


def noisy_prevalence_token_metrics(
    prior: BetaPrior,
    token_model: NoisyLogitTokenModel,
    num_samples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    c, d = prior
    theta = rng.beta(c, d, size=num_samples)
    z = token_model.sample_token(theta, rng)
    prediction = token_model.predict(z)
    return {
        "log_loss": float(np.mean(cross_entropy(theta, prediction))),
        "brier": float(np.mean(brier_risk(theta, prediction))),
        "theta_rmse": float(np.sqrt(np.mean((prediction - theta) ** 2))),
    }


def conditional_small_count_table(
    prior: BetaPrior,
    n: int,
    s_values: Sequence[int],
    token_model: NoisyLogitTokenModel,
    noisy_num_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    c, d = prior
    rows: list[dict[str, float | int | str]] = []
    prior_mean = beta_mean(c, d)

    for s in s_values:
        if s > n:
            continue

        post_a = c + s
        post_b = d + n - s
        q_nat = beta_mean(post_a, post_b)

        exact_rows = [
            {
                "predictor": "Natural prompt",
                "positive_count": int(s),
                "metric": "log_loss",
                "value": float(binary_entropy(q_nat)),
            },
            {
                "predictor": "Natural prompt",
                "positive_count": int(s),
                "metric": "brier",
                "value": float(q_nat * (1.0 - q_nat)),
            },
            {
                "predictor": "Balanced 1:1 prompt",
                "positive_count": int(s),
                "metric": "log_loss",
                "value": float(cross_entropy(q_nat, prior_mean)),
            },
            {
                "predictor": "Balanced 1:1 prompt",
                "positive_count": int(s),
                "metric": "brier",
                "value": float(brier_risk(q_nat, prior_mean)),
            },
            {
                "predictor": "Balanced + true prevalence token",
                "positive_count": int(s),
                "metric": "log_loss",
                "value": beta_expected_binary_entropy(post_a, post_b),
            },
            {
                "predictor": "Balanced + true prevalence token",
                "positive_count": int(s),
                "metric": "brier",
                "value": beta_expected_bernoulli_variance(post_a, post_b),
            },
        ]
        rows.extend(exact_rows)

        theta = rng.beta(post_a, post_b, size=noisy_num_samples)
        z = token_model.sample_token(theta, rng)
        noisy_prediction = token_model.predict(z)
        rows.append(
            {
                "predictor": "Balanced + noisy prevalence token",
                "positive_count": int(s),
                "metric": "log_loss",
                "value": float(np.mean(cross_entropy(theta, noisy_prediction))),
            }
        )
        rows.append(
            {
                "predictor": "Balanced + noisy prevalence token",
                "positive_count": int(s),
                "metric": "brier",
                "value": float(np.mean(brier_risk(theta, noisy_prediction))),
            }
        )

    return pd.DataFrame(rows)


def sample_prevalence_predictions(
    prior: BetaPrior,
    n: int,
    token_model: NoisyLogitTokenModel,
    num_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    c, d = prior
    theta = rng.beta(c, d, size=num_samples)
    s = rng.binomial(n=n, p=theta)
    natural_prediction = (c + s) / (c + d + n)
    balanced_prediction = np.full(num_samples, beta_mean(c, d), dtype=float)
    true_token_prediction = theta.copy()
    noisy_prediction = token_model.predict(token_model.sample_token(theta, rng))

    frames = [
        pd.DataFrame(
            {
                "support_size": n,
                "predictor": "Natural prompt",
                "theta": theta,
                "prediction": natural_prediction,
            }
        ),
        pd.DataFrame(
            {
                "support_size": n,
                "predictor": "Balanced 1:1 prompt",
                "theta": theta,
                "prediction": balanced_prediction,
            }
        ),
        pd.DataFrame(
            {
                "support_size": n,
                "predictor": "Balanced + true prevalence token",
                "theta": theta,
                "prediction": true_token_prediction,
            }
        ),
        pd.DataFrame(
            {
                "support_size": n,
                "predictor": "Balanced + noisy prevalence token",
                "theta": theta,
                "prediction": noisy_prediction,
            }
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def prevalence_calibration_table(samples_df: pd.DataFrame, num_bins: int = 12) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for (support_size, predictor), group in samples_df.groupby(["support_size", "predictor"], sort=False):
        unique_values = group["prediction"].nunique()
        if unique_values == 1:
            grouped = [(0, group)]
        else:
            quantile_bins = min(num_bins, unique_values)
            bin_ids = pd.qcut(group["prediction"], q=quantile_bins, labels=False, duplicates="drop")
            grouped = list(group.assign(bin_id=bin_ids).groupby("bin_id", sort=True))

        for bin_id, bucket in grouped:
            rows.append(
                {
                    "support_size": int(support_size),
                    "predictor": predictor,
                    "bin_id": int(bin_id),
                    "prediction_mean": float(bucket["prediction"].mean()),
                    "theta_mean": float(bucket["theta"].mean()),
                    "count": int(len(bucket)),
                    "abs_gap": float(np.abs(bucket["prediction"].mean() - bucket["theta"].mean())),
                }
            )
    return pd.DataFrame(rows)


def ranking_counterexample_table() -> pd.DataFrame:
    family_scores = {
        ("A", "x1"): 0.05,
        ("A", "x2"): 0.001,
        ("B", "x1"): 1.0e-4,
        ("B", "x2"): 0.9,
    }
    rows: list[dict[str, float | str | int]] = []
    for family in ["A", "B"]:
        for query in ["x1", "x2"]:
            full_score = family_scores[(family, query)]
            balanced_score = 0.5 * family_scores[("A", query)] + 0.5 * family_scores[("B", query)]
            rows.append(
                {
                    "family": family,
                    "query": query,
                    "prompt": "Full prompt",
                    "score": full_score,
                }
            )
            rows.append(
                {
                    "family": family,
                    "query": query,
                    "prompt": "Balanced prompt",
                    "score": balanced_score,
                }
            )

    table = pd.DataFrame(rows)
    order_df = (
        table.pivot_table(index=["family", "prompt"], columns="query", values="score")
        .reset_index()
        .rename_axis(columns=None)
    )
    order_df["rank_order"] = np.where(order_df["x1"] > order_df["x2"], "x1 > x2", "x2 > x1")
    order_df["order_correct_for_family"] = np.where(
        ((order_df["family"] == "A") & (order_df["rank_order"] == "x1 > x2"))
        | ((order_df["family"] == "B") & (order_df["rank_order"] == "x2 > x1")),
        1,
        0,
    )
    return table.merge(order_df[["family", "prompt", "rank_order", "order_correct_for_family"]], on=["family", "prompt"])


def family_a_balanced_log_loss_gap() -> float:
    q_x1_full = 0.05
    q_x2_full = 0.001
    q_x1_bal = 0.02505
    q_x2_bal = 0.4505
    return float(
        0.5 * (cross_entropy(q_x1_full, q_x1_bal) - cross_entropy(q_x1_full, q_x1_full))
        + 0.5 * (cross_entropy(q_x2_full, q_x2_bal) - cross_entropy(q_x2_full, q_x2_full))
    )
