#!/usr/bin/env python3
"""Run the closed-form oracle experiments for the NeurIPS 2026 draft."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "closed_form"
DEFAULT_MPLCONFIGDIR = SCRIPT_DIR / "results" / ".mplconfig"
DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(DEFAULT_MPLCONFIGDIR)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
from scipy.special import betaln

from core_closed_form import (
    NamedPredictor,
    cumulative_gap,
    make_mixture_predictor,
    make_single_predictor,
    oracle_budget,
    small_count_profile,
)


PNG_DPI = 220


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures and CSV summaries will be written.",
    )
    return parser.parse_args()


def configure_plotting() -> None:
    plt.style.use("default")
    sns.set_theme(style="whitegrid", context="talk")
    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")
    plt.rcParams.update(
        {
            "font.size": 15,
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{newtxtext}\usepackage{newtxmath}",
            "axes.unicode_minus": False,
        }
    )
    try:
        del matplotlib.font_manager.weight_dict["roman"]
        rebuild = getattr(matplotlib.font_manager, "_rebuild", None)
        if rebuild is not None:
            rebuild()
    except Exception:
        pass


def _beta_tex(a: float, b: float) -> str:
    return rf"$\operatorname{{Beta}}({a:g},{b:g})$"


def _single_beta_tex(a: float, b: float) -> str:
    return rf"Single {_beta_tex(a, b)}"


def _predictor_tex(label: str) -> str:
    mapping = {
        "Single Beta(5,5)": _single_beta_tex(5.0, 5.0),
        "Single Beta(8,2)": _single_beta_tex(8.0, 2.0),
        "Single Beta(3,6)": _single_beta_tex(3.0, 6.0),
        "Broad mixture": r"Uniform mixture",
        "Deployment oracle": rf"Oracle {_beta_tex(0.5, 4.0)}",
        "Matched single": _single_beta_tex(0.5, 3.0),
        "Collapsed broad mixture": r"Collapsed mixture",
        "Global collapsed mixture": r"Global collapsed mixture",
        "Boundary-aware bucket": r"Boundary-aware bucket",
        "Fixed training prior Beta(1.5,4)": rf"{_beta_tex(1.5, 4.0)}: Train prior",
    }
    return mapping.get(label, label)


def _deployment_tex(label: str) -> str:
    mapping = {
        "Beta(5,5)": _beta_tex(5.0, 5.0),
        "Beta(8,2)": _beta_tex(8.0, 2.0),
        "Beta(3,6)": _beta_tex(3.0, 6.0),
        "Beta(0.5,4)": _beta_tex(0.5, 4.0),
        "Interior Beta(2,4)": _beta_tex(2.0, 4.0),
        "Critical Beta(1,4)": _beta_tex(1.0, 4.0),
        "Singular Beta(0.5,4)": _beta_tex(0.5, 4.0),
        "More singular Beta(0.2,4)": _beta_tex(0.2, 4.0),
    }
    return mapping.get(label, label)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_summary_rows(
    raw_df: pd.DataFrame,
    experiment: str,
    metric: str,
) -> pd.DataFrame:
    rows = raw_df.copy()
    rows["experiment"] = experiment
    rows["metric"] = metric
    rows["aggregation"] = "per_deployment"

    aggregates: list[pd.DataFrame] = []
    average_df = (
        raw_df.groupby("predictor", as_index=False)["value"]
        .mean()
        .assign(
            experiment=experiment,
            metric=metric,
            aggregation="average",
            deployment="AVERAGE",
        )
    )
    aggregates.append(average_df)
    worst_case_df = (
        raw_df.groupby("predictor", as_index=False)["value"]
        .max()
        .assign(
            experiment=experiment,
            metric=metric,
            aggregation="worst_case",
            deployment="WORST_CASE",
        )
    )
    aggregates.append(worst_case_df)
    return pd.concat([rows, *aggregates], ignore_index=True)


def _fit_tail_loglog_slope(n_values: pd.Series, delta_values: pd.Series, tail_points: int = 4) -> float:
    tail_n = np.asarray(n_values.tail(tail_points), dtype=float)
    tail_delta = np.asarray(delta_values.tail(tail_points), dtype=float)
    slope, _intercept = np.polyfit(np.log(tail_n), np.log(tail_delta), deg=1)
    return float(slope)


def _phase_regime(c: float) -> str:
    if c > 1.0:
        return "interior"
    if np.isclose(c, 1.0):
        return "critical"
    return "singular"


def _scaled_gap_for_regime(c: float, n_values: pd.Series, delta_values: pd.Series) -> pd.Series:
    n_float = n_values.astype(float)
    if c > 1.0:
        return delta_values * (n_float**2)
    if np.isclose(c, 1.0):
        return delta_values * (n_float**2) / np.log(n_float)
    return delta_values * (n_float ** (1.0 + c))


def _reference_curve(n_grid: list[int], exponent: float, anchor_n: float, anchor_y: float) -> np.ndarray:
    n_array = np.asarray(n_grid, dtype=float)
    return anchor_y * (n_array / anchor_n) ** exponent


def _critical_reference_curve(n_grid: list[int], anchor_n: float, anchor_y: float) -> np.ndarray:
    n_array = np.asarray(n_grid, dtype=float)
    return anchor_y * (np.log(n_array) / np.log(anchor_n)) * (n_array / anchor_n) ** (-2.0)


def _anchored_theorem_curve(c: float, n_grid: list[int], anchor_n: float, anchor_y: float) -> np.ndarray:
    if c > 1.0:
        return _reference_curve(n_grid=n_grid, exponent=-2.0, anchor_n=anchor_n, anchor_y=anchor_y)
    if np.isclose(c, 1.0):
        return _critical_reference_curve(n_grid=n_grid, anchor_n=anchor_n, anchor_y=anchor_y)
    return _reference_curve(n_grid=n_grid, exponent=-(1.0 + c), anchor_n=anchor_n, anchor_y=anchor_y)


def _beta_density(theta_grid: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.exp((a - 1.0) * np.log(theta_grid) + (b - 1.0) * np.log1p(-theta_grid) - betaln(a, b))


def _build_phase_tail_summary(
    delta_df: pd.DataFrame,
    deployment_priors: dict[str, tuple[float, float]],
    tail_points: int = 4,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for deployment_name, (c, _d) in deployment_priors.items():
        deployment_df = delta_df[delta_df["deployment"] == deployment_name].copy()
        tail_curve_df = deployment_df.tail(tail_points).copy()
        rows.append(
            {
                "deployment": deployment_name,
                "regime": _phase_regime(c),
                "tail_loglog_slope": _fit_tail_loglog_slope(deployment_df["n"], deployment_df["delta_n"]),
                "tail_scaled_mean": float(tail_curve_df["scaled_gap"].mean()),
                "tail_scaled_ratio": float(tail_curve_df["scaled_gap"].max() / tail_curve_df["scaled_gap"].min()),
            }
        )
    return pd.DataFrame(rows)


def run_experiment_1(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    budget_horizon = 128
    n_grid = list(range(budget_horizon))
    families = {
        "Beta(5,5)": (5.0, 5.0),
        "Beta(8,2)": (8.0, 2.0),
        "Beta(3,6)": (3.0, 6.0),
    }
    predictors = [
        make_single_predictor(*families["Beta(5,5)"], name="Single Beta(5,5)"),
        make_single_predictor(*families["Beta(8,2)"], name="Single Beta(8,2)"),
        make_single_predictor(*families["Beta(3,6)"], name="Single Beta(3,6)"),
        make_mixture_predictor(
            components=list(families.values()),
            weights=[1.0, 1.0, 1.0],
            name="Broad mixture",
        ),
    ]

    detailed_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | str]] = []
    for deployment_name, deployment_prior in families.items():
        for predictor in predictors:
            curve_df = cumulative_gap(deploy_prior=deployment_prior, predictor=predictor, n_grid=n_grid)
            curve_df["predictor"] = predictor.name
            curve_df["deployment"] = deployment_name
            detailed_rows.extend(curve_df.to_dict(orient="records"))
            summary_rows.append(
                {
                    "predictor": predictor.name,
                    "deployment": deployment_name,
                    "value": float(curve_df["delta_n"].sum()),
                }
            )

    detailed_df = pd.DataFrame(detailed_rows)
    summary_raw_df = pd.DataFrame(summary_rows)
    summary_df = add_summary_rows(summary_raw_df, experiment="exp1_global_hedge", metric="K_128")

    heatmap_predictor_order = [
        "Single Beta(5,5)",
        "Single Beta(8,2)",
        "Single Beta(3,6)",
        "Broad mixture",
    ]
    heatmap_deployment_order = ["Beta(5,5)", "Beta(8,2)", "Beta(3,6)"]
    heatmap_df = (
        summary_raw_df.pivot(index="predictor", columns="deployment", values="value")
        .loc[heatmap_predictor_order, heatmap_deployment_order]
        .copy()
    )
    heatmap_df.index = [_predictor_tex(label) for label in heatmap_df.index]
    heatmap_df.columns = [_deployment_tex(label) for label in heatmap_df.columns]

    summary_plot_df = summary_df[summary_df["aggregation"].isin(["average", "worst_case"])].copy()
    summary_plot_df["criterion"] = summary_plot_df["aggregation"].map(
        {
            "average": "Average",
            "worst_case": "Worst case",
        }
    )
    summary_order = heatmap_predictor_order
    summary_plot_df["plot_predictor"] = pd.Categorical(
        summary_plot_df["predictor"].map(_predictor_tex),
        categories=[_predictor_tex(label) for label in summary_order],
        ordered=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), gridspec_kw={"width_ratios": [1.2, 1.0]})

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap=sns.light_palette("#b22222", as_cmap=True),
        linewidths=0.6,
        linecolor="white",
        cbar_kws={},
        annot_kws={"fontsize": 10},
        ax=axes[0],
    )
    axes[0].set_title(r"Cumulative oracle gap $K_{128}$ for each train-test pair")
    axes[0].set_xlabel("Test prior")
    axes[0].set_ylabel("Train prior")

    sns.barplot(
        data=summary_plot_df,
        y="plot_predictor",
        x="value",
        hue="criterion",
        orient="h",
        palette={"Average": "#4C78A8", "Worst case": "#E45756"},
        ax=axes[1],
    )
    axes[1].set_title("")
    axes[1].set_xlabel(r"Average / worst-case cumulative gap $K_{128}$")
    axes[1].set_ylabel("Train prior")
    axes[1].legend(title="", fontsize=10)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.3f", fontsize=9, padding=2)

    save_figure(fig, output_dir, "figure1_global_hedge")

    detailed_df.to_csv(output_dir / "experiment1_global_hedge.csv", index=False)
    return detailed_df, summary_df


def _plot_named_lines(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    palette: dict[str, tuple[float, float, float]],
) -> None:
    for label, color in palette.items():
        label_df = data[data[label_col] == label].copy()
        ax.plot(
            label_df[x_col],
            label_df[y_col],
            color=color,
            linewidth=1.9,
            alpha=0.95,
            marker="o",
            markersize=5.0,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.3,
            label=_predictor_tex(label),
        )


def _plot_small_count_panel(
    ax: plt.Axes,
    profile_df: pd.DataFrame,
    predictor_order: list[str],
    palette: dict[str, tuple[float, float, float]],
) -> None:
    style_map = {
        0: "-",
        1: "--",
        2: ":",
    }
    for predictor in predictor_order:
        predictor_df = profile_df[profile_df["predictor"] == predictor].copy()
        for s_value, linestyle in style_map.items():
            count_df = predictor_df[predictor_df["s"] == s_value].copy()
            ax.plot(
                count_df["n"],
                count_df["scaled_predictive"],
                color=palette[predictor],
                linestyle=linestyle,
                linewidth=1.9,
                alpha=0.95,
                marker="o",
                markersize=4.6,
                markerfacecolor="white",
                markeredgecolor=palette[predictor],
                markeredgewidth=1.1,
            )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Context length n")
    ax.set_ylabel(r"Scaled predictive $n\,\hat q(1\mid S_n=s)$")

    predictor_handles = [
        Line2D(
            [0],
            [0],
            color=palette[predictor],
            linewidth=1.9,
            marker="o",
            markersize=4.8,
            markerfacecolor="white",
            markeredgecolor=palette[predictor],
            markeredgewidth=1.1,
            label=_predictor_tex(predictor),
        )
        for predictor in predictor_order
    ]
    count_handles = [
        Line2D([0], [0], color="black", linestyle=style_map[s_value], linewidth=1.9, label=rf"$S_n={s_value}$")
        for s_value in [0, 1, 2]
    ]
    predictor_legend = ax.legend(handles=predictor_handles, loc="upper left", fontsize=8.5, title=None)
    count_legend = ax.legend(handles=count_handles, loc="lower right", fontsize=8.5, title=None)
    ax.add_artist(predictor_legend)
    ax.add_artist(count_legend)


def _plot_small_count_error_panel(
    ax: plt.Axes,
    profile_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
    predictor_order: list[str],
    palette: dict[str, tuple[float, float, float]],
) -> None:
    style_map = {
        0: "-",
        1: "--",
        2: ":",
    }
    oracle_subset = oracle_df[["n", "s", "scaled_predictive"]].rename(
        columns={"scaled_predictive": "oracle_scaled_predictive"}
    )
    merged_df = profile_df.merge(oracle_subset, on=["n", "s"], how="inner")
    merged_df["scaled_error"] = merged_df["scaled_predictive"] - merged_df["oracle_scaled_predictive"]

    for predictor in predictor_order:
        predictor_df = merged_df[merged_df["predictor"] == predictor].copy()
        for s_value, linestyle in style_map.items():
            count_df = predictor_df[predictor_df["s"] == s_value].copy()
            ax.plot(
                count_df["n"],
                count_df["scaled_error"],
                color=palette[predictor],
                linestyle=linestyle,
                linewidth=1.9,
                alpha=0.95,
                marker="o",
                markersize=4.6,
                markerfacecolor="white",
                markeredgecolor=palette[predictor],
                markeredgewidth=1.1,
            )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Context length n")
    ax.set_ylabel(r"Oracle-centered scaled error $n(\hat q-q_{\mathrm{oracle}})$")

    predictor_handles = [
        Line2D(
            [0],
            [0],
            color=palette[predictor],
            linewidth=1.9,
            marker="o",
            markersize=4.8,
            markerfacecolor="white",
            markeredgecolor=palette[predictor],
            markeredgewidth=1.1,
            label=_predictor_tex(predictor),
        )
        for predictor in predictor_order
    ]
    count_handles = [
        Line2D([0], [0], color="black", linestyle=style_map[s_value], linewidth=1.9, label=rf"$S_n={s_value}$")
        for s_value in [0, 1, 2]
    ]
    predictor_legend = ax.legend(handles=predictor_handles, loc="upper left", fontsize=8.5, title=None)
    count_legend = ax.legend(handles=count_handles, loc="lower right", fontsize=8.5, title=None)
    ax.add_artist(predictor_legend)
    ax.add_artist(count_legend)


def save_boundary_failure_and_fix_figure(
    output_dir: Path,
    exp2_delta_df: pd.DataFrame,
    exp2_profile_df: pd.DataFrame,
    exp3_delta_df: pd.DataFrame,
    exp3_profile_df: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.8))

    common_n_max = 256
    delta_plot_df = pd.concat(
        [
            exp2_delta_df[exp2_delta_df["predictor"] == "Matched single"].copy(),
            exp3_delta_df[exp3_delta_df["predictor"] == "Global collapsed mixture"].copy(),
            exp3_delta_df[exp3_delta_df["predictor"] == "Boundary-aware bucket"].copy(),
        ],
        ignore_index=True,
    )
    delta_plot_df = delta_plot_df[delta_plot_df["n"] <= common_n_max].copy()
    delta_palette = {
        "Matched single": "#4C78A8",
        "Global collapsed mixture": "#E45756",
        "Boundary-aware bucket": "#54A24B",
    }
    _plot_named_lines(
        ax=axes[0],
        data=delta_plot_df,
        x_col="n",
        y_col="delta_n",
        label_col="predictor",
        palette=delta_palette,
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log", base=10)
    axes[0].set_title("Bucketing repairs the boundary failure")
    axes[0].set_xlabel("Context length n")
    axes[0].set_ylabel(r"One-step gap $\Delta_n$")
    axes[0].legend(fontsize=8.8, title=None, loc="upper right")

    oracle_profile_df = exp2_profile_df[
        (exp2_profile_df["predictor"] == "Deployment oracle") & (exp2_profile_df["n"] <= common_n_max)
    ].copy()
    error_profile_df = pd.concat(
        [
            exp2_profile_df[exp2_profile_df["predictor"] == "Matched single"].copy(),
            exp3_profile_df[exp3_profile_df["predictor"] == "Global collapsed mixture"].copy(),
            exp3_profile_df[exp3_profile_df["predictor"] == "Boundary-aware bucket"].copy(),
        ],
        ignore_index=True,
    )
    error_profile_df = error_profile_df[error_profile_df["n"] <= common_n_max].copy()
    _plot_small_count_error_panel(
        ax=axes[1],
        profile_df=error_profile_df,
        oracle_df=oracle_profile_df,
        predictor_order=["Matched single", "Global collapsed mixture", "Boundary-aware bucket"],
        palette=delta_palette,
    )
    axes[1].set_title("Bucketing removes the boundary-profile mismatch")

    save_figure(fig, output_dir, "figure2_boundary_failure_and_fix")


def run_experiment_2(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_grid = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    deployment_prior = (0.5, 4.0)
    predictors: list[NamedPredictor] = [
        make_single_predictor(*deployment_prior, name="Deployment oracle"),
        make_mixture_predictor(
            components=[(0.5, 3.0), (0.2, 3.0)],
            weights=[0.5, 0.5],
            name="Collapsed broad mixture",
        ),
        make_single_predictor(0.5, 3.0, name="Matched single"),
    ]

    delta_rows: list[dict[str, float | int | str]] = []
    profile_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | str]] = []

    for predictor in predictors:
        delta_df = cumulative_gap(deploy_prior=deployment_prior, predictor=predictor, n_grid=n_grid)
        delta_df["predictor"] = predictor.name
        delta_rows.extend(delta_df.to_dict(orient="records"))

        profile_df = small_count_profile(predictor=predictor, n_grid=n_grid)
        profile_df["predictor"] = predictor.name
        profile_rows.extend(profile_df.to_dict(orient="records"))

        summary_rows.append(
            {
                "predictor": predictor.name,
                "deployment": "Beta(0.5,4)",
                "value": float(delta_df["delta_n"].sum()),
            }
        )

    delta_df = pd.DataFrame(delta_rows)
    profile_df = pd.DataFrame(profile_rows)
    summary_df = pd.DataFrame(summary_rows)
    summary_df["experiment"] = "exp2_boundary_failure"
    summary_df["metric"] = "K_grid"
    summary_df["aggregation"] = "deployment_fixed"

    delta_df.to_csv(output_dir / "experiment2_boundary_failure_delta.csv", index=False)
    profile_df.to_csv(output_dir / "experiment2_boundary_failure_profile.csv", index=False)
    return delta_df, profile_df, summary_df


def run_experiment_3(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    budget_horizon = 256
    n_grid = [4, 8, 16, 32, 64, 128, 256]
    deployment_prior = (0.5, 4.0)
    predictors: list[NamedPredictor] = [
        make_mixture_predictor(
            components=[(0.5, 3.0), (0.5, 6.0), (0.2, 3.0), (0.2, 6.0)],
            weights=[1.0, 1.0, 1.0, 1.0],
            name="Global collapsed mixture",
        ),
        make_mixture_predictor(
            components=[(0.5, 3.0), (0.5, 6.0)],
            weights=[1.0, 1.0],
            name="Boundary-aware bucket",
        ),
        make_single_predictor(*deployment_prior, name="Deployment oracle"),
    ]

    delta_rows: list[dict[str, float | int | str]] = []
    profile_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | str]] = []

    for predictor in predictors:
        delta_df = cumulative_gap(deploy_prior=deployment_prior, predictor=predictor, n_grid=n_grid)
        delta_df["predictor"] = predictor.name
        delta_rows.extend(delta_df.to_dict(orient="records"))

        profile_df = small_count_profile(predictor=predictor, n_grid=n_grid)
        profile_df["predictor"] = predictor.name
        profile_rows.extend(profile_df.to_dict(orient="records"))

        summary_rows.append(
            {
                "predictor": predictor.name,
                "deployment": "Beta(0.5,4)",
                "value": oracle_budget(
                    deploy_prior=deployment_prior,
                    predictor=predictor,
                    horizon=budget_horizon,
                ),
            }
        )

    delta_df = pd.DataFrame(delta_rows)
    profile_df = pd.DataFrame(profile_rows)
    summary_df = pd.DataFrame(summary_rows)
    summary_df["experiment"] = "exp3_boundary_fix"
    summary_df["metric"] = "K_256"
    summary_df["aggregation"] = "deployment_fixed"

    delta_df.to_csv(output_dir / "experiment3_boundary_fix_delta.csv", index=False)
    profile_df.to_csv(output_dir / "experiment3_boundary_fix_profile.csv", index=False)
    return delta_df, profile_df, summary_df


def run_experiment_4(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_grid = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    training_prior = (1.5, 4.0)
    predictor = make_single_predictor(*training_prior, name="Fixed training prior Beta(1.5,4)")
    deployment_priors = {
        "Interior Beta(2,4)": (2.0, 4.0),
        "Critical Beta(1,4)": (1.0, 4.0),
        "Singular Beta(0.5,4)": (0.5, 4.0),
        "More singular Beta(0.2,4)": (0.2, 4.0),
    }

    delta_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | str]] = []
    slope_rows: list[dict[str, float | str]] = []
    for deployment_name, deployment_prior in deployment_priors.items():
        c, d = deployment_prior
        curve_df = cumulative_gap(deploy_prior=deployment_prior, predictor=predictor, n_grid=n_grid)
        curve_df["deployment"] = deployment_name
        curve_df["c"] = c
        curve_df["d"] = d
        curve_df["regime"] = _phase_regime(c)
        curve_df["scaled_gap"] = _scaled_gap_for_regime(c, curve_df["n"], curve_df["delta_n"])
        slope = _fit_tail_loglog_slope(curve_df["n"], curve_df["delta_n"])
        tail_scaled = curve_df["scaled_gap"].tail(4)
        tail_mean = float(tail_scaled.mean())
        tail_ratio = float(tail_scaled.max() / tail_scaled.min())

        delta_rows.extend(curve_df.to_dict(orient="records"))
        summary_rows.append(
            {
                "predictor": predictor.name,
                "deployment": deployment_name,
                "value": float(curve_df["delta_n"].sum()),
                "experiment": "exp4_phase_transition",
                "metric": "K_phase_grid",
                "aggregation": "per_deployment",
            }
        )
        slope_rows.append(
            {
                "predictor": predictor.name,
                "deployment": deployment_name,
                "value": slope,
                "experiment": "exp4_phase_transition",
                "metric": "tail_loglog_slope",
                "aggregation": "per_deployment",
            }
        )
        slope_rows.append(
            {
                "predictor": predictor.name,
                "deployment": deployment_name,
                "value": tail_mean,
                "experiment": "exp4_phase_transition",
                "metric": "tail_scaled_mean",
                "aggregation": "per_deployment",
            }
        )
        slope_rows.append(
            {
                "predictor": predictor.name,
                "deployment": deployment_name,
                "value": tail_ratio,
                "experiment": "exp4_phase_transition",
                "metric": "tail_scaled_ratio",
                "aggregation": "per_deployment",
            }
        )

    delta_df = pd.DataFrame(delta_rows)
    summary_df = pd.concat([pd.DataFrame(summary_rows), pd.DataFrame(slope_rows)], ignore_index=True)
    tail_summary_df = _build_phase_tail_summary(
        delta_df=delta_df,
        deployment_priors=deployment_priors,
    )

    palette = dict(zip(deployment_priors.keys(), sns.color_palette("deep", n_colors=len(deployment_priors))))
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

    for deployment_name in deployment_priors:
        color = palette[deployment_name]
        deployment_df = delta_df[delta_df["deployment"] == deployment_name].copy()
        axes[0].plot(
            deployment_df["n"],
            deployment_df["delta_n"],
            color=color,
            linewidth=1.6,
            alpha=0.95,
            marker="o",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.4,
            zorder=3,
        )

    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log", base=10)
    axes[0].set_title("")
    axes[0].set_xlabel("Context length n")
    axes[0].set_ylabel(r"One-step gap $\Delta_n$")

    asymptotic_start_n = 256
    for deployment_name, deployment_prior in deployment_priors.items():
        deployment_df = delta_df[delta_df["deployment"] == deployment_name].copy()
        tail_curve_df = deployment_df[deployment_df["n"] >= asymptotic_start_n].copy()
        tail_n = tail_curve_df["n"].astype(int).tolist()
        tail_anchor_n = float(tail_n[-1])
        tail_anchor_y = float(tail_curve_df["delta_n"].iloc[-1])
        theorem_curve = _anchored_theorem_curve(
            c=deployment_prior[0],
            n_grid=tail_n,
            anchor_n=tail_anchor_n,
            anchor_y=tail_anchor_y,
        )
        axes[0].plot(
            tail_n,
            theorem_curve,
            linestyle="--",
            linewidth=4.0,
            color=palette[deployment_name],
            alpha=0.45,
            dash_capstyle="round",
            zorder=2,
        )

    deployment_handles = [
        Line2D(
            [0],
            [0],
            color=palette[deployment_name],
            linewidth=1.6,
            marker="o",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor=palette[deployment_name],
            markeredgewidth=1.4,
            label=_deployment_tex(deployment_name),
        )
        for deployment_name in deployment_priors
    ]
    deployment_legend = axes[0].legend(handles=deployment_handles, fontsize=9, title=None, ncol=1)
    style_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=1.6,
            marker="o",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.4,
            label="Observed one-step gap",
        ),
        Line2D([0], [0], color="black", linestyle="--", linewidth=4.0, alpha=0.45, label="Anchored asymptotic rate"),
    ]
    style_legend = axes[0].legend(handles=style_handles, loc="lower left", fontsize=9, title=None)
    axes[0].add_artist(deployment_legend)
    axes[0].add_artist(style_legend)

    theta_grid = np.linspace(1e-4, 1.0 - 1e-4, 600)
    train_c, train_d = training_prior
    axes[1].plot(
        theta_grid,
        _beta_density(theta_grid, train_c, train_d),
        color="black",
        linestyle="--",
        linewidth=2.2,
        label=_predictor_tex("Fixed training prior Beta(1.5,4)"),
    )
    for deployment_name, (c, d) in deployment_priors.items():
        axes[1].plot(
            theta_grid,
            _beta_density(theta_grid, c, d),
            color=palette[deployment_name],
            linewidth=2.0,
            alpha=0.95,
            label=_deployment_tex(deployment_name),
        )
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_yscale("log", base=10)
    axes[1].set_ylim(5e-2, 1e3)
    axes[1].set_title("")
    axes[1].set_xlabel(r"Task prevalence $\theta$")
    axes[1].set_ylabel(r"Density $f_{\Theta}(\theta)$")
    axes[1].legend(fontsize=8.5, title=None, loc="lower right")

    save_figure(fig, output_dir, "figure3_phase_transition")

    delta_df.to_csv(output_dir / "experiment4_phase_transition_delta.csv", index=False)
    tail_summary_df.to_csv(output_dir / "experiment4_phase_transition_tail_summary.csv", index=False)
    return delta_df, summary_df


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    _, exp1_summary = run_experiment_1(output_dir=output_dir)
    exp2_delta, exp2_profile, exp2_summary = run_experiment_2(output_dir=output_dir)
    exp3_delta, exp3_profile, exp3_summary = run_experiment_3(output_dir=output_dir)
    save_boundary_failure_and_fix_figure(
        output_dir=output_dir,
        exp2_delta_df=exp2_delta,
        exp2_profile_df=exp2_profile,
        exp3_delta_df=exp3_delta,
        exp3_profile_df=exp3_profile,
    )
    _, exp4_summary = run_experiment_4(output_dir=output_dir)

    summary_df = pd.concat([exp1_summary, exp2_summary, exp3_summary, exp4_summary], ignore_index=True)
    summary_columns = ["experiment", "predictor", "deployment", "metric", "aggregation", "value"]
    summary_df = summary_df[summary_columns]
    summary_df.to_csv(output_dir / "summary_closed_form.csv", index=False)

    print(f"Closed-form experiments completed. Results written to {output_dir}")


if __name__ == "__main__":
    main()
