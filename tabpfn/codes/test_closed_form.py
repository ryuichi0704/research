#!/usr/bin/env python3
"""Lightweight regression checks for the closed-form experiment utilities."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core_closed_form import (
    cumulative_gap,
    delta_n,
    make_mixture_predictor,
    make_single_predictor,
    mixture_posterior_weights,
    oracle_budget,
    single_beta_predictive,
)
from run_closed_form import _build_phase_tail_summary


def assert_close(actual: float, expected: float, tol: float = 1e-10) -> None:
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"Expected {expected}, got {actual}.")


def test_single_beta_predictive() -> None:
    assert_close(single_beta_predictive(2.0, 3.0, s=4, n=10), (2.0 + 4.0) / (2.0 + 3.0 + 10.0))
    assert_close(single_beta_predictive(0.5, 4.0, s=0, n=16), 0.5 / 20.5)


def test_mixture_weights_sum_to_one() -> None:
    weights = mixture_posterior_weights(
        components=[(0.5, 3.0), (0.2, 3.0), (2.0, 2.0)],
        weights=[0.2, 0.3, 0.5],
        s=1,
        n=8,
    )
    assert_close(float(np.sum(weights)), 1.0)
    if np.any(weights <= 0):
        raise AssertionError("Posterior weights should be strictly positive.")


def test_delta_is_non_negative() -> None:
    deployment = (0.5, 4.0)
    predictors = [
        make_single_predictor(0.5, 3.0, name="matched-single"),
        make_mixture_predictor(
            components=[(0.5, 3.0), (0.2, 3.0)],
            weights=[0.5, 0.5],
            name="collapsed-broad",
        ),
        make_single_predictor(2.0, 2.0, name="interior-single"),
    ]
    for predictor in predictors:
        for n in [4, 8, 16, 32, 64]:
            gap = delta_n(deploy_prior=deployment, predictor=predictor, n=n)
            if gap < -1e-12:
                raise AssertionError(f"delta_n must be non-negative, got {gap} for {predictor.name} at n={n}.")


def test_oracle_budget_matches_dense_sum() -> None:
    deployment = (0.5, 4.0)
    predictor = make_mixture_predictor(
        components=[(0.5, 3.0), (0.2, 3.0)],
        weights=[0.5, 0.5],
        name="collapsed-broad",
    )
    horizon = 32
    expected = cumulative_gap(deploy_prior=deployment, predictor=predictor, n_grid=range(horizon))["delta_n"].sum()
    assert_close(oracle_budget(deploy_prior=deployment, predictor=predictor, horizon=horizon), float(expected))


def test_theory_aligned_regressions() -> None:
    exp1_budget_horizon = 128
    families = [(5.0, 5.0), (8.0, 2.0), (3.0, 6.0)]
    singles = [
        make_single_predictor(*families[0], name="Single Beta(5,5)"),
        make_single_predictor(*families[1], name="Single Beta(8,2)"),
        make_single_predictor(*families[2], name="Single Beta(3,6)"),
    ]
    broad = make_mixture_predictor(components=families, weights=[1.0, 1.0, 1.0], name="Broad mixture")

    predictor_scores = {}
    for predictor in [*singles, broad]:
        predictor_scores[predictor.name] = []
        for deployment in families:
            predictor_scores[predictor.name].append(
                oracle_budget(deploy_prior=deployment, predictor=predictor, horizon=exp1_budget_horizon)
            )

    broad_worst = max(predictor_scores["Broad mixture"])
    for single in singles:
        if broad_worst >= max(predictor_scores[single.name]):
            raise AssertionError("Broad mixture should improve the worst-case global regret in Experiment 1.")

    deployment = (0.5, 4.0)
    oracle = make_single_predictor(*deployment, name="oracle")
    collapsed = make_mixture_predictor(
        components=[(0.5, 3.0), (0.2, 3.0)],
        weights=[0.5, 0.5],
        name="collapsed",
    )
    bucketed = make_mixture_predictor(
        components=[(0.5, 3.0), (0.5, 6.0)],
        weights=[1.0, 1.0],
        name="bucketed",
    )

    exp2_n_grid = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    exp3_budget_horizon = 256

    oracle_exp2 = cumulative_gap(deploy_prior=deployment, predictor=oracle, n_grid=exp2_n_grid)["delta_n"].sum()
    collapsed_exp2 = cumulative_gap(deploy_prior=deployment, predictor=collapsed, n_grid=exp2_n_grid)["delta_n"].sum()
    if collapsed_exp2 <= oracle_exp2:
        raise AssertionError("Collapsed mixture should be worse than the deployment oracle in Experiment 2.")

    collapsed_exp3 = oracle_budget(deploy_prior=deployment, predictor=collapsed, horizon=exp3_budget_horizon)
    bucketed_exp3 = oracle_budget(deploy_prior=deployment, predictor=bucketed, horizon=exp3_budget_horizon)
    if bucketed_exp3 >= collapsed_exp3:
        raise AssertionError("Boundary-aware bucket should improve over the collapsed mixture in Experiment 3.")

    phase_train = make_single_predictor(1.5, 4.0, name="phase-train")
    phase_grid = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    phase_deployments = {
        "interior": (2.0, 4.0),
        "critical": (1.0, 4.0),
        "singular_05": (0.5, 4.0),
        "singular_02": (0.2, 4.0),
    }
    tail_slopes: dict[str, float] = {}

    for name, deployment in phase_deployments.items():
        curve = cumulative_gap(deploy_prior=deployment, predictor=phase_train, n_grid=phase_grid)
        tail_n = curve["n"].to_numpy()[-4:]
        tail_delta = curve["delta_n"].to_numpy()[-4:]
        slope = float(np.polyfit(np.log(tail_n), np.log(tail_delta), deg=1)[0])
        tail_slopes[name] = slope

        c = deployment[0]
        if c > 1.0:
            scaled = curve["delta_n"] * curve["n"] ** 2
        elif math.isclose(c, 1.0):
            scaled = curve["delta_n"] * curve["n"] ** 2 / np.log(curve["n"])
        else:
            scaled = curve["delta_n"] * curve["n"] ** (1.0 + c)

        tail_scaled = scaled.to_numpy()[-4:]
        ratio = float(np.max(tail_scaled) / np.min(tail_scaled))
        if ratio >= 1.25:
            raise AssertionError(f"The theorem-aligned rescaling should flatten the tail for {name}, got ratio={ratio}.")

    if tail_slopes["interior"] >= -1.85:
        raise AssertionError("Interior phase-transition slope should be close to -2.")
    if tail_slopes["critical"] >= -1.7:
        raise AssertionError("Critical phase-transition slope should still decay faster than n^-1.7.")
    if not (-1.7 < tail_slopes["singular_05"] < -1.3):
        raise AssertionError("The c=0.5 singular slope should be close to -(1+c) = -1.5.")
    if not (-1.35 < tail_slopes["singular_02"] < -1.05):
        raise AssertionError("The c=0.2 singular slope should be close to -(1+c) = -1.2.")


def test_phase_tail_summary_shape() -> None:
    deployment_priors = {
        "Interior Beta(2,4)": (2.0, 4.0),
        "Critical Beta(1,4)": (1.0, 4.0),
        "Singular Beta(0.5,4)": (0.5, 4.0),
        "More singular Beta(0.2,4)": (0.2, 4.0),
    }
    phase_train = make_single_predictor(1.5, 4.0, name="phase-train")
    phase_grid = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    rows: list[dict[str, float | int | str]] = []
    for deployment_name, deployment in deployment_priors.items():
        curve = cumulative_gap(deploy_prior=deployment, predictor=phase_train, n_grid=phase_grid)
        curve["deployment"] = deployment_name
        c = deployment[0]
        if c > 1.0:
            curve["scaled_gap"] = curve["delta_n"] * curve["n"] ** 2
        elif math.isclose(c, 1.0):
            curve["scaled_gap"] = curve["delta_n"] * curve["n"] ** 2 / np.log(curve["n"])
        else:
            curve["scaled_gap"] = curve["delta_n"] * curve["n"] ** (1.0 + c)
        rows.extend(curve.to_dict(orient="records"))

    summary_df = _build_phase_tail_summary(
        delta_df=pd.DataFrame(rows),
        deployment_priors=deployment_priors,
    )
    if len(summary_df) != len(deployment_priors):
        raise AssertionError("Phase tail summary should contain one row per deployment regime.")
    if set(summary_df["deployment"]) != set(deployment_priors):
        raise AssertionError("Phase tail summary deployments do not match the expected regimes.")


def main() -> None:
    test_single_beta_predictive()
    test_mixture_weights_sum_to_one()
    test_delta_is_non_negative()
    test_oracle_budget_matches_dense_sum()
    test_theory_aligned_regressions()
    test_phase_tail_summary_shape()
    print("All closed-form checks passed.")


if __name__ == "__main__":
    main()
