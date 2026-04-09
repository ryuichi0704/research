#!/usr/bin/env python3
"""Run the closed-form oracle experiments for the NeurIPS 2026 draft."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "results" / "closed_form"
DEFAULT_MPLCONFIGDIR = SCRIPT_DIR.parent / "results" / ".mplconfig"
DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(DEFAULT_MPLCONFIGDIR)

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from core_closed_form import (
    NamedPredictor,
    cumulative_gap,
    make_mixture_predictor,
    make_single_predictor,
    two_component_mixture_oracle_budget_grid,
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
    }
    return mapping.get(label, label)


def _deployment_tex(label: str) -> str:
    mapping = {
        "Beta(5,5)": _beta_tex(5.0, 5.0),
        "Beta(8,2)": _beta_tex(8.0, 2.0),
        "Beta(3,6)": _beta_tex(3.0, 6.0),
        "Beta(4,5)": _beta_tex(4.0, 5.0),
        "Beta(7.5,2.5)": _beta_tex(7.5, 2.5),
        "Beta(2.5,5.5)": _beta_tex(2.5, 5.5),
    }
    return mapping.get(label, label)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def run_experiment_1(output_dir: Path) -> None:
    budget_horizon = 128
    n_grid = list(range(budget_horizon))
    train_families = {
        "Beta(5,5)": (5.0, 5.0),
        "Beta(8,2)": (8.0, 2.0),
        "Beta(3,6)": (3.0, 6.0),
    }
    deployment_families = {
        "Beta(4,5)": (4.0, 5.0),
        "Beta(7.5,2.5)": (7.5, 2.5),
        "Beta(2.5,5.5)": (2.5, 5.5),
    }
    predictors = [
        make_single_predictor(*train_families["Beta(5,5)"], name="Single Beta(5,5)"),
        make_single_predictor(*train_families["Beta(8,2)"], name="Single Beta(8,2)"),
        make_single_predictor(*train_families["Beta(3,6)"], name="Single Beta(3,6)"),
        make_mixture_predictor(
            components=list(train_families.values()),
            weights=[1.0, 1.0, 1.0],
            name="Broad mixture",
        ),
    ]

    summary_rows: list[dict[str, float | str]] = []
    for deployment_name, deployment_prior in deployment_families.items():
        for predictor in predictors:
            curve_df = cumulative_gap(deploy_prior=deployment_prior, predictor=predictor, n_grid=n_grid)
            summary_rows.append(
                {
                    "predictor": predictor.name,
                    "deployment": deployment_name,
                    "value": float(curve_df["delta_n"].sum()),
                }
            )

    summary_raw_df = pd.DataFrame(summary_rows)

    heatmap_predictor_order = [
        "Single Beta(5,5)",
        "Single Beta(8,2)",
        "Single Beta(3,6)",
        "Broad mixture",
    ]
    heatmap_deployment_order = ["Beta(4,5)", "Beta(7.5,2.5)", "Beta(2.5,5.5)"]
    heatmap_df = (
        summary_raw_df.pivot(index="predictor", columns="deployment", values="value")
        .loc[heatmap_predictor_order, heatmap_deployment_order]
        .copy()
    )
    heatmap_df.index = [_predictor_tex(label) for label in heatmap_df.index]
    heatmap_df.columns = [_deployment_tex(label) for label in heatmap_df.columns]

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.8))

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap=sns.light_palette("#b22222", as_cmap=True),
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Cumulative oracle budget"},
        annot_kws={"fontsize": 10},
        ax=ax,
    )
    ax.set_xlabel("Test prior")
    ax.set_ylabel("Train prior")

    save_figure(fig, output_dir, "figure1_global_hedge")


def run_experiment_2(output_dir: Path) -> None:
    specialist_components = [(8.0, 2.0), (3.0, 6.0)]
    alpha_grid = np.linspace(0.0, 1.0, 21)
    budget_horizon = 128

    curve_deployments = {
        "Beta(3,4)": (3.0, 4.0),
        "Beta(6,5)": (6.0, 5.0),
        "Beta(9,5)": (9.0, 5.0),
    }

    curve_rows: list[dict[str, float | str]] = []

    deployment_curves: dict[str, np.ndarray] = {}
    for deployment_name, deployment_prior in curve_deployments.items():
        deployment_curves[deployment_name] = two_component_mixture_oracle_budget_grid(
            deploy_prior=deployment_prior,
            components=specialist_components,
            alpha_grid=alpha_grid,
            horizon=budget_horizon,
        )

    for deployment_name in curve_deployments:
        values = deployment_curves[deployment_name]
        for alpha, value in zip(alpha_grid, values):
            curve_rows.append(
                {
                    "deployment": deployment_name,
                    "alpha": float(alpha),
                    "value": float(value),
                }
            )

    curve_df = pd.DataFrame(curve_rows)

    fig, ax = plt.subplots(1, 1, figsize=(8.6, 5.4))

    left_palette = {
        "Beta(3,4)": "#4C78A8",
        "Beta(6,5)": "#72B7B2",
        "Beta(9,5)": "#E45756",
    }

    for deployment_name, color in left_palette.items():
        deployment_curve = curve_df[curve_df["deployment"] == deployment_name].copy()
        ax.plot(
            deployment_curve["alpha"],
            deployment_curve["value"],
            color=color,
            linewidth=2.1,
            alpha=0.97,
            marker="o",
            markersize=4.8,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.1,
            label=deployment_name,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(r"Mixture weight $\alpha$ in $\alpha\,\operatorname{Beta}(8,2)+(1-\alpha)\,\operatorname{Beta}(3,6)$")
    ax.set_ylabel("Cumulative oracle budget")
    ax.legend(fontsize=9.0, title=None, loc="upper right")

    save_figure(fig, output_dir, "figure2_lambda_sweep")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    run_experiment_1(output_dir=output_dir)
    run_experiment_2(output_dir=output_dir)

    print(f"Closed-form experiments completed. Results written to {output_dir}")


if __name__ == "__main__":
    main()
