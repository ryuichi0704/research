#!/usr/bin/env python3
"""Run TabPFN-free oracle imbalance experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "closed_form"
DEFAULT_MPLCONFIGDIR = SCRIPT_DIR / "results" / ".mplconfig"
DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(DEFAULT_MPLCONFIGDIR)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from core_imbalance import (
    NoisyLogitTokenModel,
    balanced_expected_metrics,
    beta_mean,
    conditional_small_count_table,
    family_a_balanced_log_loss_gap,
    natural_expected_metrics,
    noisy_prevalence_token_metrics,
    prevalence_calibration_table,
    ranking_counterexample_table,
    sample_prevalence_predictions,
    true_prevalence_token_metrics,
)


PNG_DPI = 220
PALETTE = {
    "Natural prompt": "#4C78A8",
    "Balanced 1:1 prompt": "#E45756",
    "Balanced + true prevalence token": "#54A24B",
    "Balanced + noisy prevalence token": "#B279A2",
}
PROMPT_STYLE = {
    "Full prompt": ("#4C78A8", "o"),
    "Balanced prompt": ("#E45756", "s"),
}


def parse_support_sizes(value: str) -> list[int]:
    support_sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not support_sizes:
        raise argparse.ArgumentTypeError("Expected at least one support size.")
    if any(size <= 0 for size in support_sizes):
        raise argparse.ArgumentTypeError("Support sizes must be positive integers.")
    return support_sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where figures and CSV summaries will be written.",
    )
    parser.add_argument(
        "--support-sizes",
        type=parse_support_sizes,
        default=parse_support_sizes("8,16,32,64,128,256,512"),
        help="Comma-separated natural prompt sizes used in the prevalence experiment.",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=0.5,
        help="Beta prior shape c for the task prevalence prior.",
    )
    parser.add_argument(
        "--d",
        type=float,
        default=49.5,
        help="Beta prior shape d for the task prevalence prior.",
    )
    parser.add_argument(
        "--noisy-token-sigma",
        type=float,
        default=1.0,
        help="Standard deviation of the noisy prevalence token on the logit scale.",
    )
    parser.add_argument(
        "--num-token-samples",
        type=int,
        default=50000,
        help="Monte Carlo sample count for the noisy-token metrics.",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=40000,
        help="Monte Carlo sample count used for the prevalence calibration tables.",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=12,
        help="Number of quantile bins used in the prevalence calibration table.",
    )
    parser.add_argument(
        "--small-count-n",
        type=int,
        default=256,
        help="Support size used for the small-count conditional analysis.",
    )
    parser.add_argument(
        "--max-small-count",
        type=int,
        default=6,
        help="Largest positive-count stratum included in the small-count plot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for Monte Carlo components.",
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


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_metric_panel(ax: plt.Axes, df: pd.DataFrame, metric: str, ylabel: str) -> None:
    metric_df = df[df["metric"] == metric].copy()
    for predictor, color in PALETTE.items():
        predictor_df = metric_df[metric_df["predictor"] == predictor].copy()
        ax.plot(
            predictor_df["support_size"],
            predictor_df["value"],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=5.2,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.3,
            label=predictor,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Support size n")
    ax.set_ylabel(ylabel)


def plot_prevalence_curves(output_dir: Path, metrics_df: pd.DataFrame, prior: tuple[float, float]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.8))
    metric_specs = [
        ("log_loss", "Expected log loss"),
        ("brier", "Expected Brier score"),
        ("theta_rmse", r"Prevalence RMSE $\sqrt{\mathbb{E}[(\hat q-\theta)^2]}$"),
    ]
    for ax, (metric, ylabel) in zip(axes, metric_specs):
        _plot_metric_panel(ax=ax, df=metrics_df, metric=metric, ylabel=ylabel)
    axes[0].legend(fontsize=9.5, title=None, loc="upper right")
    axes[0].set_title("")
    axes[1].set_title("")
    axes[2].set_title("")
    c, d = prior
    fig.suptitle(
        rf"Beta--Bernoulli prevalence experiment with $\theta \sim \operatorname{{Beta}}({c:g},{d:g})$",
        y=1.03,
    )
    save_figure(fig, output_dir, "figure4_balancing_prevalence_curves")


def build_small_count_excess_table(small_count_df: pd.DataFrame) -> pd.DataFrame:
    natural_baseline = (
        small_count_df[small_count_df["predictor"] == "Natural prompt"][["positive_count", "metric", "value"]]
        .rename(columns={"value": "natural_value"})
        .copy()
    )
    excess_df = small_count_df.merge(natural_baseline, on=["positive_count", "metric"], how="left")
    excess_df["excess_over_natural"] = excess_df["value"] - excess_df["natural_value"]
    return excess_df


def plot_small_count(output_dir: Path, small_count_df: pd.DataFrame, prior: tuple[float, float], n: int) -> None:
    excess_df = build_small_count_excess_table(small_count_df)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    for ax, metric, ylabel in [
        (axes[0], "log_loss", "Excess log loss over natural prompt"),
        (axes[1], "brier", "Excess Brier score over natural prompt"),
    ]:
        metric_df = excess_df[excess_df["metric"] == metric].copy()
        for predictor, color in PALETTE.items():
            if predictor == "Natural prompt":
                continue
            predictor_df = metric_df[metric_df["predictor"] == predictor].copy()
            ax.plot(
                predictor_df["positive_count"],
                predictor_df["excess_over_natural"],
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=5.2,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.3,
                label=predictor,
            )
        ax.axhline(0.0, color="#333333", linewidth=1.1, linestyle="--", alpha=0.9)
        ax.set_xlabel(r"Observed natural positive count $S_n=s$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(metric_df["positive_count"].unique()))
        ax.set_title("Rare small-count strata are most affected")
    axes[0].legend(fontsize=9.2, title=None, loc="upper right")
    c, d = prior
    fig.suptitle(
        rf"Small-count strata at $n={n}$ under $\theta \sim \operatorname{{Beta}}({c:g},{d:g})$",
        y=1.03,
    )
    save_figure(fig, output_dir, "figure5_balancing_small_count")


def plot_ranking_counterexample(output_dir: Path, ranking_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6), sharey=True)
    prompt_positions = {"Full prompt": 0, "Balanced prompt": 1}
    query_style = {
        "x1": ("#4C78A8", "o"),
        "x2": ("#E45756", "s"),
    }
    for ax, family in zip(axes, ["A", "B"]):
        family_df = ranking_df[ranking_df["family"] == family].copy()
        for query, (color, marker) in query_style.items():
            query_df = family_df[family_df["query"] == query].copy()
            x_values = [prompt_positions[prompt] for prompt in query_df["prompt"]]
            ax.plot(
                x_values,
                query_df["score"],
                color=color,
                linewidth=2.0,
                marker=marker,
                markersize=7.0,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.4,
                label=rf"${query}$",
            )
        if family == "A":
            rank_text = "\n".join(
                [
                    r"Full prompt: $x_1 \succ x_2$",
                    r"Balanced prompt: $x_2 \succ x_1$",
                    r"Order flips under balancing",
                ]
            )
        else:
            rank_text = "\n".join(
                [
                    r"Full prompt: $x_2 \succ x_1$",
                    r"Balanced prompt: $x_2 \succ x_1$",
                    r"No rank reversal",
                ]
            )
        ax.text(
            0.04,
            0.08,
            rank_text,
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#d0d0d0"},
        )
        ax.set_xticks([0, 1], ["Full prompt", "Balanced prompt"])
        ax.set_yscale("log", base=10)
        ax.set_xlabel("Prompt type")
        title_suffix = " (reversal)" if family == "A" else " (no reversal)"
        ax.set_title(rf"Realized family $F={family}$" + title_suffix)
    axes[0].set_ylabel(r"Predicted score $q(1 \mid x)$")
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=7.0,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.4,
            label=rf"${query}$",
        )
        for query, (color, marker) in query_style.items()
    ]
    axes[0].legend(handles=legend_handles, fontsize=9.5, title="Query", loc="upper left")
    save_figure(fig, output_dir, "figure6_balancing_ranking_reversal")


def build_prevalence_metrics(
    prior: tuple[float, float],
    support_sizes: list[int],
    token_model: NoisyLogitTokenModel,
    noisy_num_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    noisy_metrics = noisy_prevalence_token_metrics(
        prior=prior,
        token_model=token_model,
        num_samples=noisy_num_samples,
        rng=rng,
    )

    rows: list[dict[str, float | int | str]] = []
    for support_size in support_sizes:
        exact_metrics = {
            "Natural prompt": natural_expected_metrics(prior=prior, n=support_size),
            "Balanced 1:1 prompt": balanced_expected_metrics(prior=prior),
            "Balanced + true prevalence token": true_prevalence_token_metrics(prior=prior),
            "Balanced + noisy prevalence token": noisy_metrics,
        }
        for predictor, metric_dict in exact_metrics.items():
            for metric_name, value in metric_dict.items():
                rows.append(
                    {
                        "support_size": int(support_size),
                        "predictor": predictor,
                        "metric": metric_name,
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def build_calibration_table(
    prior: tuple[float, float],
    support_sizes: list[int],
    token_model: NoisyLogitTokenModel,
    num_samples: int,
    num_bins: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    calibration_sizes = sorted({support_sizes[min(1, len(support_sizes) - 1)], support_sizes[-1]})
    calibration_samples = pd.concat(
        [
            sample_prevalence_predictions(
                prior=prior,
                n=support_size,
                token_model=token_model,
                num_samples=num_samples,
                rng=rng,
            )
            for support_size in calibration_sizes
        ],
        ignore_index=True,
    )
    return prevalence_calibration_table(calibration_samples, num_bins=num_bins)


def build_summary_table(
    metrics_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    small_count_df: pd.DataFrame,
    small_count_excess_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    max_support = int(metrics_df["support_size"].max())
    metrics_at_max = metrics_df[metrics_df["support_size"] == max_support].copy()
    for row in metrics_at_max.itertuples():
        rows.append(
            {
                "experiment": "exp5_prevalence_curves",
                "slice": f"n={row.support_size}",
                "predictor": row.predictor,
                "metric": row.metric,
                "value": float(row.value),
            }
        )

    calibration_summary_rows: list[dict[str, float | int | str]] = []
    for (support_size, predictor), frame in calibration_df.groupby(["support_size", "predictor"], sort=False):
        calibration_summary_rows.append(
            {
                "support_size": int(support_size),
                "predictor": predictor,
                "weighted_abs_gap": float(np.average(frame["abs_gap"], weights=frame["count"])),
            }
        )
    calibration_summary = pd.DataFrame(calibration_summary_rows)
    for row in calibration_summary.itertuples():
        rows.append(
            {
                "experiment": "exp5_prevalence_calibration",
                "slice": f"n={row.support_size}",
                "predictor": row.predictor,
                "metric": "weighted_abs_gap",
                "value": float(row.weighted_abs_gap),
            }
        )

    for row in small_count_df.itertuples():
        rows.append(
            {
                "experiment": "exp6_small_count",
                "slice": f"s={row.positive_count}",
                "predictor": row.predictor,
                "metric": row.metric,
                "value": float(row.value),
            }
        )
    for row in small_count_excess_df.itertuples():
        if row.predictor == "Natural prompt":
            continue
        rows.append(
            {
                "experiment": "exp6_small_count_excess",
                "slice": f"s={row.positive_count}",
                "predictor": row.predictor,
                "metric": f"excess_{row.metric}",
                "value": float(row.excess_over_natural),
            }
        )

    order_rows = ranking_df[["family", "prompt", "rank_order", "order_correct_for_family"]].drop_duplicates()
    for row in order_rows.itertuples():
        rows.append(
            {
                "experiment": "exp7_ranking_reversal",
                "slice": f"family={row.family}",
                "predictor": row.prompt,
                "metric": "order_correct_for_family",
                "value": int(row.order_correct_for_family),
            }
        )
    rows.append(
        {
            "experiment": "exp7_ranking_reversal",
            "slice": "family=A",
            "predictor": "Balanced prompt",
            "metric": "family_a_balanced_log_loss_gap",
            "value": family_a_balanced_log_loss_gap(),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    prior = (args.c, args.d)
    rng = np.random.default_rng(args.seed)
    token_model = NoisyLogitTokenModel.build(prior=prior, sigma=args.noisy_token_sigma)

    prevalence_metrics_df = build_prevalence_metrics(
        prior=prior,
        support_sizes=args.support_sizes,
        token_model=token_model,
        noisy_num_samples=args.num_token_samples,
        rng=rng,
    )
    calibration_df = build_calibration_table(
        prior=prior,
        support_sizes=args.support_sizes,
        token_model=token_model,
        num_samples=args.calibration_samples,
        num_bins=args.calibration_bins,
        rng=rng,
    )
    small_count_df = conditional_small_count_table(
        prior=prior,
        n=args.small_count_n,
        s_values=list(range(args.max_small_count + 1)),
        token_model=token_model,
        noisy_num_samples=args.num_token_samples,
        rng=rng,
    )
    small_count_excess_df = build_small_count_excess_table(small_count_df)
    ranking_df = ranking_counterexample_table()
    summary_df = build_summary_table(
        metrics_df=prevalence_metrics_df,
        calibration_df=calibration_df,
        small_count_df=small_count_df,
        small_count_excess_df=small_count_excess_df,
        ranking_df=ranking_df,
    )

    prevalence_metrics_df.to_csv(output_dir / "experiment5_balancing_prevalence_curves.csv", index=False)
    calibration_df.to_csv(output_dir / "experiment5_balancing_prevalence_calibration.csv", index=False)
    small_count_df.to_csv(output_dir / "experiment6_balancing_small_count.csv", index=False)
    small_count_excess_df.to_csv(output_dir / "experiment6_balancing_small_count_excess.csv", index=False)
    ranking_df.to_csv(output_dir / "experiment7_balancing_ranking.csv", index=False)
    summary_df.to_csv(output_dir / "summary_imbalance_oracle.csv", index=False)

    plot_prevalence_curves(output_dir=output_dir, metrics_df=prevalence_metrics_df, prior=prior)
    plot_small_count(output_dir=output_dir, small_count_df=small_count_df, prior=prior, n=args.small_count_n)
    plot_ranking_counterexample(output_dir=output_dir, ranking_df=ranking_df)

    mean_prevalence = beta_mean(*prior)
    print(
        "Imbalance oracle experiments completed. "
        f"Results written to {output_dir} "
        f"(mean prevalence={mean_prevalence:.4f})."
    )


if __name__ == "__main__":
    main()
