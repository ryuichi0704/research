from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from k1_experiment import (
    DATASET_PATTERNS,
    DatasetBundle,
    ExperimentConfig,
    TwoLayerK1Net,
    alignment_statistics,
    barrier_profile,
    build_dataset_bundle,
    choose_device,
    config_from_settings,
    distribution_on_dataset,
    evaluate_model,
    gaussian_nll_from_eta,
    interpolate_models,
    optimal_transport_matching,
    permute_hidden_units,
    plot_barrier_curves,
    plot_parameter_distribution,
    plot_predictive_fit,
    plot_training_history,
    resolve_settings,
    save_model_checkpoint,
    train_model,
    true_mean,
    true_std,
)


def build_sweep_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Width sweep comparing barrier bounds and observed barriers.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="YAML config file. CLI arguments override YAML values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results_width_sweep"),
        help="Directory where width-sweep outputs are written.",
    )
    parser.add_argument(
        "--parameterization",
        choices=["natural", "meanvar"],
        default="meanvar",
        help="Width sweep supports both the natural and mean/variance parameterizations.",
    )
    parser.add_argument(
        "--width-exponents",
        type=int,
        nargs="+",
        default=[9, 10, 11, 12],
        help="Sweep widths 2^k for the listed exponents.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Number of model seeds to train per width. Consecutive pairs are evaluated.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First model seed used in the width sweep.",
    )
    parser.add_argument(
        "--max-parallel-workers",
        type=int,
        default=1,
        help="Number of seed-pair jobs to run concurrently.",
    )
    parser.add_argument(
        "--time-grid-points",
        type=int,
        default=401,
        help="Number of interpolation points used for dense barrier and exact-modulus certificates.",
    )
    parser.add_argument("--variance-min", type=float, default=None)
    parser.add_argument("--variance-max", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--precision-activation", choices=["softplus", "exp"], default=None)
    parser.add_argument("--dataset-pattern", choices=list(DATASET_PATTERNS), default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--evaluation-probe-points", type=int, default=None)
    parser.add_argument("--plot-points", type=int, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--learning-rate-min", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--barrier-points", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--seed-a", type=int, default=None)
    parser.add_argument("--seed-b", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    return parser


def raw_precision(model: TwoLayerK1Net, raw_scale: torch.Tensor) -> torch.Tensor:
    if model.precision_activation == "exp":
        return torch.exp(raw_scale) + model.lambda_min
    return F.softplus(raw_scale) + model.lambda_min


def raw_variance(model: TwoLayerK1Net, raw_scale: torch.Tensor) -> torch.Tensor:
    if model.precision_activation == "exp":
        return torch.exp(raw_scale) + model.variance_floor
    return F.softplus(raw_scale) + model.variance_floor


def normalized_conditional_moments(
    x_normalized: torch.Tensor,
    dataset_bundle: DatasetBundle,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_raw = x_normalized.detach().cpu().numpy() * dataset_bundle.x_std + dataset_bundle.x_mean
    mean_raw = true_mean(x_raw, dataset_bundle.dataset_pattern)
    std_raw = true_std(x_raw, dataset_bundle.dataset_pattern)

    mean_norm = (mean_raw - dataset_bundle.y_mean) / dataset_bundle.y_std
    var_norm = (std_raw / dataset_bundle.y_std) ** 2
    second_moment_norm = mean_norm**2 + var_norm

    mean_tensor = torch.from_numpy(mean_norm.astype(np.float32)).to(device)
    second_moment_tensor = torch.from_numpy(second_moment_norm.astype(np.float32)).to(device)
    return mean_tensor, second_moment_tensor


def conditional_risk_from_raw(
    model: TwoLayerK1Net,
    raw_mean: torch.Tensor,
    raw_scale: torch.Tensor,
    conditional_mean: torch.Tensor,
    conditional_second_moment: torch.Tensor,
) -> torch.Tensor:
    if model.parameterization == "natural":
        precision = raw_precision(model, raw_scale)
        return (
            raw_mean.square() / (4.0 * precision)
            + 0.5 * torch.log(math.pi / precision)
            - raw_mean * conditional_mean
            + precision * conditional_second_moment
        )
    variance = raw_variance(model, raw_scale)
    return (
        0.5 * torch.log(2.0 * math.pi * variance)
        + (raw_mean.square() - 2.0 * raw_mean * conditional_mean + conditional_second_moment)
        / (2.0 * variance)
    )


def omega_u_timewise(
    model: TwoLayerK1Net,
    u_hat: torch.Tensor,
    delta_u: torch.Tensor,
    conditional_mean: torch.Tensor,
    scale_low: torch.Tensor,
    scale_high: torch.Tensor,
) -> torch.Tensor:
    def evaluate(scale_value: torch.Tensor) -> torch.Tensor:
        if model.parameterization == "natural":
            return (
                delta_u * torch.abs(u_hat / (2.0 * scale_value) - conditional_mean)
                + delta_u.square() / (4.0 * scale_value)
            )
        return (
            delta_u * torch.abs(u_hat - conditional_mean) / scale_value
            + delta_u.square() / (2.0 * scale_value)
        )

    return torch.maximum(evaluate(scale_low), evaluate(scale_high))


def omega_s_timewise(
    model: TwoLayerK1Net,
    u_hat: torch.Tensor,
    conditional_second_moment: torch.Tensor,
    conditional_mean: torch.Tensor,
    scale_hat: torch.Tensor,
    scale_low: torch.Tensor,
    scale_high: torch.Tensor,
) -> torch.Tensor:
    def evaluate(scale_value: torch.Tensor) -> torch.Tensor:
        if model.parameterization == "natural":
            return (
                conditional_second_moment * (scale_value - scale_hat)
                + (u_hat.square() / 4.0) * (1.0 / scale_value - 1.0 / scale_hat)
                + 0.5 * torch.log(scale_hat / scale_value)
            )
        centered_second = u_hat.square() - 2.0 * u_hat * conditional_mean + conditional_second_moment
        return (
            0.5 * torch.log(scale_value / scale_hat)
            + 0.5 * centered_second * (1.0 / scale_value - 1.0 / scale_hat)
        )

    return torch.clamp(torch.maximum(evaluate(scale_low), evaluate(scale_high)), min=0.0)


@torch.no_grad()
def exact_modulus_statistics(
    model_a: TwoLayerK1Net,
    model_b: TwoLayerK1Net,
    dataset_bundle: DatasetBundle,
    device: torch.device,
    time_grid_points: int,
) -> dict[str, object]:
    if model_a.parameterization != model_b.parameterization:
        raise ValueError("exact_modulus_statistics requires matching parameterizations.")

    probe_x = dataset_bundle.evaluation_probe_x.to(device)
    test_x, test_y = dataset_bundle.test.tensors
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    conditional_mean, conditional_second_moment = normalized_conditional_moments(
        probe_x,
        dataset_bundle,
        device,
    )

    dist_a_probe = distribution_on_dataset(model_a, probe_x, device)
    dist_b_probe = distribution_on_dataset(model_b, probe_x, device)
    dist_a_test = distribution_on_dataset(model_a, test_x, device)
    dist_b_test = distribution_on_dataset(model_b, test_x, device)

    risk_a = conditional_risk_from_raw(
        model_a,
        dist_a_probe["raw"][:, :1],
        dist_a_probe["raw"][:, 1:2],
        conditional_mean,
        conditional_second_moment,
    )
    risk_b = conditional_risk_from_raw(
        model_b,
        dist_b_probe["raw"][:, :1],
        dist_b_probe["raw"][:, 1:2],
        conditional_mean,
        conditional_second_moment,
    )

    endpoint_a = float(gaussian_nll_from_eta(dist_a_test["eta1"], dist_a_test["eta2"], test_y).mean().item())
    endpoint_b = float(gaussian_nll_from_eta(dist_b_test["eta1"], dist_b_test["eta2"], test_y).mean().item())

    ts = np.linspace(0.0, 1.0, time_grid_points)
    delta_u_path = torch.zeros_like(conditional_mean)
    delta_s_path = torch.zeros_like(conditional_mean)
    raw_hat_history: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    term_u_max = 0.0
    term_s_max = 0.0
    term_j_max = 0.0
    total_timewise_max = 0.0
    dense_barrier_max = -float("inf")
    dense_barrier_argmax_t = 0.0

    for t in ts:
        model_t = interpolate_models(model_a, model_b, float(t), device)
        dist_t_probe = distribution_on_dataset(model_t, probe_x, device)
        dist_t_test = distribution_on_dataset(model_t, test_x, device)

        raw_hat_probe = (1.0 - t) * dist_a_probe["raw"] + t * dist_b_probe["raw"]
        u_hat = raw_hat_probe[:, :1]
        s_hat = raw_hat_probe[:, 1:2]

        delta_u = torch.abs(dist_t_probe["raw"][:, :1] - u_hat)
        delta_s = torch.abs(dist_t_probe["raw"][:, 1:2] - s_hat)
        delta_u_path = torch.maximum(delta_u_path, delta_u)
        delta_s_path = torch.maximum(delta_s_path, delta_s)

        if model_a.parameterization == "natural":
            scale_hat = raw_precision(model_a, s_hat)
            scale_low = raw_precision(model_a, s_hat - delta_s)
            scale_high = raw_precision(model_a, s_hat + delta_s)
        else:
            scale_hat = raw_variance(model_a, s_hat)
            scale_low = raw_variance(model_a, s_hat - delta_s)
            scale_high = raw_variance(model_a, s_hat + delta_s)

        omega_u = omega_u_timewise(model_a, u_hat, delta_u, conditional_mean, scale_low, scale_high)
        omega_s = omega_s_timewise(
            model_a,
            u_hat,
            conditional_second_moment,
            conditional_mean,
            scale_hat,
            scale_low,
            scale_high,
        )
        risk_hat = conditional_risk_from_raw(model_a, u_hat, s_hat, conditional_mean, conditional_second_moment)
        j_t = torch.clamp(risk_hat - ((1.0 - t) * risk_a + t * risk_b), min=0.0)

        term_u = float(omega_u.mean().item())
        term_s = float(omega_s.mean().item())
        term_j = float(j_t.mean().item())
        total_timewise = float((omega_u + omega_s + j_t).mean().item())

        term_u_max = max(term_u_max, term_u)
        term_s_max = max(term_s_max, term_s)
        term_j_max = max(term_j_max, term_j)
        total_timewise_max = max(total_timewise_max, total_timewise)

        nll_t = float(gaussian_nll_from_eta(dist_t_test["eta1"], dist_t_test["eta2"], test_y).mean().item())
        barrier_t = nll_t - ((1.0 - float(t)) * endpoint_a + float(t) * endpoint_b)
        if barrier_t > dense_barrier_max:
            dense_barrier_max = barrier_t
            dense_barrier_argmax_t = float(t)

        raw_hat_history.append((u_hat.detach().cpu(), s_hat.detach().cpu(), j_t.detach().cpu()))

    path_u_max = 0.0
    path_s_max = 0.0
    for u_hat_cpu, s_hat_cpu, j_t_cpu in raw_hat_history:
        u_hat = u_hat_cpu.to(device)
        s_hat = s_hat_cpu.to(device)
        if model_a.parameterization == "natural":
            scale_hat = raw_precision(model_a, s_hat)
            scale_low = raw_precision(model_a, s_hat - delta_s_path)
            scale_high = raw_precision(model_a, s_hat + delta_s_path)
        else:
            scale_hat = raw_variance(model_a, s_hat)
            scale_low = raw_variance(model_a, s_hat - delta_s_path)
            scale_high = raw_variance(model_a, s_hat + delta_s_path)

        omega_u_env = omega_u_timewise(model_a, u_hat, delta_u_path, conditional_mean, scale_low, scale_high)
        omega_s_env = omega_s_timewise(
            model_a,
            u_hat,
            conditional_second_moment,
            conditional_mean,
            scale_hat,
            scale_low,
            scale_high,
        )

        path_u_max = max(path_u_max, float(omega_u_env.mean().item()))
        path_s_max = max(path_s_max, float(omega_s_env.mean().item()))

    return {
        "time_grid_points": time_grid_points,
        "dense_barrier": {
            "max_barrier": dense_barrier_max,
            "argmax_t": dense_barrier_argmax_t,
            "endpoint_losses": [endpoint_a, endpoint_b],
        },
        "timewise_exact_modulus": {
            "bound": term_u_max + term_s_max + term_j_max,
            "term_u_max": term_u_max,
            "term_s_max": term_s_max,
            "term_j_max": term_j_max,
            "max_total_over_t": total_timewise_max,
        },
        "path_envelope_exact_modulus": {
            "bound": path_u_max + path_s_max + term_j_max,
            "term_u_max": path_u_max,
            "term_s_max": path_s_max,
            "term_j_max": term_j_max,
        },
    }


def consecutive_seed_pairs(seed_start: int, n_seeds: int) -> tuple[list[int], list[tuple[int, int]]]:
    seeds = list(range(seed_start, seed_start + n_seeds))
    pairs = [(seeds[index], seeds[index + 1]) for index in range(0, len(seeds) - 1, 2)]
    return seeds, pairs


def _positive_for_log(values: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    return np.maximum(values, floor)


def plot_loss_barriers(
    barrier_curves_by_width: dict[int, list[dict[str, object]]],
    output_path: Path,
) -> None:
    widths = sorted(barrier_curves_by_width.keys())
    if not widths:
        return

    fig, axes = plt.subplots(1, len(widths), figsize=(3.5 * len(widths), 3.1), squeeze=False)

    for column, width in enumerate(widths):
        ax = axes[0, column]
        pair_curves = barrier_curves_by_width[width]
        naive_barriers: list[float] = []
        aligned_barriers: list[float] = []

        for pair in pair_curves:
            naive_losses = np.asarray(pair["naive"], dtype=float)
            aligned_losses = np.asarray(pair["aligned"], dtype=float)
            lambdas = np.linspace(0.0, 1.0, naive_losses.size)
            ax.plot(lambdas, naive_losses, "-", color="C2", alpha=0.25, linewidth=0.8)
            ax.plot(lambdas, aligned_losses, "-", color="C0", alpha=0.4, linewidth=1.0)
            naive_barriers.append(float(pair["naive_barrier"]))
            aligned_barriers.append(float(pair["aligned_barrier"]))

        ax.set_title(f"width={width}")
        ax.set_xlabel(r"$t$")
        if column == 0:
            ax.set_ylabel("Loss (NLL)")
        ax.text(
            0.98,
            0.98,
            (
                f"naive={np.mean(naive_barriers):.4f}\n"
                f"aligned={np.mean(aligned_barriers):.4f}±{np.std(aligned_barriers):.4f}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )

    fig.suptitle("Synthetic K=1 mean/variance barriers across seed pairs", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_barrier_scaling_summary(
    results: list[dict[str, object]],
    output_path: Path,
) -> None:
    if not results:
        return

    widths = np.array([entry["width"] for entry in results], dtype=float)
    aligned_means = np.array([entry["aligned_profile_barrier_mean"] for entry in results], dtype=float)
    aligned_stds = np.array([entry["aligned_profile_barrier_std"] for entry in results], dtype=float)
    naive_means = np.array([entry["naive_profile_barrier_mean"] for entry in results], dtype=float)

    aligned_plot = _positive_for_log(aligned_means)
    naive_plot = _positive_for_log(naive_means)
    ref_y = float(aligned_plot[0])
    ref_w = float(widths[0])
    slope_one = ref_y * (ref_w / widths)
    slope_sqrt = ref_y * np.sqrt(ref_w / widths)

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.8), constrained_layout=True)
    ax.plot(widths, slope_one, "--", color="grey", alpha=0.4, linewidth=0.9, label=r"$O(1/N)$")
    ax.plot(widths, slope_sqrt, ":", color="grey", alpha=0.4, linewidth=0.9, label=r"$O(1/\sqrt{N})$")
    ax.errorbar(
        widths,
        aligned_plot,
        yerr=np.minimum(aligned_stds, np.maximum(aligned_plot - 1e-8, 0.0)),
        fmt="o-",
        color="C0",
        markersize=4,
        linewidth=1.4,
        label="Aligned",
    )
    ax.plot(widths, naive_plot, "s--", color="C2", markersize=3.5, linewidth=1.0, alpha=0.7, label="Unaligned")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Width N")
    ax.set_ylabel("Barrier")
    ax.set_title("Barrier scaling on synthetic K=1")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_training_loss_summary(
    all_histories: list[tuple[int, int, list[float]]],
    output_path: Path,
) -> None:
    if not all_histories:
        return

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.0), constrained_layout=True)
    width_colors: dict[int, str] = {}
    for width, seed, losses in all_histories:
        if width not in width_colors:
            width_colors[width] = f"C{len(width_colors)}"
        label = f"w={width}" if seed == min(item[1] for item in all_histories if item[0] == width) else None
        ax.plot(losses, color=width_colors[width], alpha=0.35, linewidth=0.9, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train NLL")
    ax.set_yscale("symlog")
    ax.set_title("Training loss across widths and seeds")
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_primary_width_comparison(
    results: list[dict[str, object]],
    output_path: Path,
) -> None:
    widths = np.array([entry["width"] for entry in results], dtype=float)
    actual = np.array([entry["actual_barrier_dense_mean"] for entry in results], dtype=float)
    primary = np.array([entry["timewise_exact_modulus_bound_mean"] for entry in results], dtype=float)
    visible_values = np.concatenate([actual, primary])
    positive_visible = visible_values[visible_values > 0.0]
    guide_level = float(np.exp(np.mean(np.log(positive_visible)))) if positive_visible.size else 1.0
    width_mid = float(np.exp(np.mean(np.log(widths))))
    slope_minus_one_guide = guide_level * (width_mid / widths)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.5), constrained_layout=True)

    ax.loglog(widths, _positive_for_log(actual), marker="o", linewidth=2.5, label="Observed barrier")
    ax.loglog(widths, _positive_for_log(primary), marker="o", linewidth=2.5, label="Primary theorem bound")
    ax.loglog(
        widths,
        slope_minus_one_guide,
        linestyle="--",
        linewidth=1.8,
        color="black",
        label="Slope -1 guide (~1/N)",
    )
    ax.set_xlabel("Width N")
    ax.set_ylabel("NLL barrier / upper bound")
    ax.set_title("Observed vs primary theorem bound")
    ax.legend()

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return math.inf
    return numerator / denominator


def pair_output_dir(base_output_dir: Path, width: int, seed_a: int, seed_b: int) -> Path:
    return base_output_dir / f"width_{width}" / f"pair_{seed_a}_{seed_b}"


def build_pair_task(
    config_base: ExperimentConfig,
    output_dir: Path,
    width: int,
    seed_a: int,
    seed_b: int,
    time_grid_points: int,
) -> dict[str, object]:
    config_dict = {**asdict(config_base), "width": width, "seed_a": seed_a, "seed_b": seed_b}
    return {
        "config": config_dict,
        "output_dir": str(pair_output_dir(output_dir, width, seed_a, seed_b)),
        "time_grid_points": time_grid_points,
    }


def run_pair_job(task: dict[str, object]) -> dict[str, object]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    config = ExperimentConfig(**task["config"])
    output_dir = Path(task["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(config.device)
    dataset_bundle = build_dataset_bundle(config)

    model_a, history_a = train_model(
        config,
        dataset_bundle,
        config.seed_a,
        device,
        show_progress=False,
    )
    model_b, history_b = train_model(
        config,
        dataset_bundle,
        config.seed_b,
        device,
        show_progress=False,
    )

    matching = optimal_transport_matching(model_a, model_b)
    permutation = np.asarray(matching["permutation"], dtype=int)
    matched_b = permute_hidden_units(model_b, permutation, device)

    identity_profile = barrier_profile(
        model_a,
        model_b,
        dataset_bundle.test,
        config.barrier_points,
        device,
    )
    matched_profile = barrier_profile(
        model_a,
        matched_b,
        dataset_bundle.test,
        config.barrier_points,
        device,
    )
    matched_stats = alignment_statistics(
        model_a=model_a,
        model_b=matched_b,
        evaluation_probe_x=dataset_bundle.evaluation_probe_x,
        config=config,
        y_star=dataset_bundle.y_star,
        device=device,
    )
    exact_modulus = exact_modulus_statistics(
        model_a=model_a,
        model_b=matched_b,
        dataset_bundle=dataset_bundle,
        device=device,
        time_grid_points=int(task["time_grid_points"]),
    )
    test_metrics_a = evaluate_model(model_a, dataset_bundle.test, device)
    test_metrics_b = evaluate_model(matched_b, dataset_bundle.test, device)
    merged_model = interpolate_models(model_a, matched_b, 0.5, device)

    plot_parameter_distribution(
        model=model_a,
        output_path=output_dir / "model_a_parameters.png",
        title=(
            f"width={config.width}, pattern={config.dataset_pattern}, "
            f"seeds=({config.seed_a},{config.seed_b}): model A parameters"
        ),
    )
    plot_parameter_distribution(
        model=matched_b,
        output_path=output_dir / "model_b_matched_parameters.png",
        title=(
            f"width={config.width}, pattern={config.dataset_pattern}, "
            f"seeds=({config.seed_a},{config.seed_b}): matched model B parameters"
        ),
    )
    plot_barrier_curves(
        identity_profile=identity_profile,
        matched_profile=matched_profile,
        output_path=output_dir / "lmc_barrier.png",
        title=(
            f"width={config.width}, pattern={config.dataset_pattern}, "
            f"seeds=({config.seed_a},{config.seed_b}): interpolation barrier"
        ),
    )
    plot_predictive_fit(
        model_a=model_a,
        model_b=matched_b,
        merged_model=merged_model,
        dataset_bundle=dataset_bundle,
        device=device,
        output_path=output_dir / "predictive_fit.png",
        title=(
            f"width={config.width}, pattern={config.dataset_pattern}, "
            f"seeds=({config.seed_a},{config.seed_b}): predictive fit"
        ),
    )
    plot_training_history(
        history_a=history_a,
        history_b=history_b,
        output_path=output_dir / "training_history.png",
        title=(
            f"width={config.width}, pattern={config.dataset_pattern}, "
            f"seeds=({config.seed_a},{config.seed_b}): training curves"
        ),
    )
    np.save(output_dir / "matching_permutation.npy", permutation)
    save_model_checkpoint(
        model=model_a,
        output_path=output_dir / "model_a.pt",
        config=config,
        dataset_bundle=dataset_bundle,
        role="model_a",
        seed=config.seed_a,
        training_history=history_a,
    )
    save_model_checkpoint(
        model=model_b,
        output_path=output_dir / "model_b.pt",
        config=config,
        dataset_bundle=dataset_bundle,
        role="model_b",
        seed=config.seed_b,
        training_history=history_b,
    )
    save_model_checkpoint(
        model=matched_b,
        output_path=output_dir / "model_b_matched.pt",
        config=config,
        dataset_bundle=dataset_bundle,
        role="model_b_matched",
        seed=config.seed_b,
        training_history=history_b,
        extra_metadata={"matching_permutation": permutation.tolist()},
    )
    save_model_checkpoint(
        model=merged_model,
        output_path=output_dir / "merged_model.pt",
        config=config,
        dataset_bundle=dataset_bundle,
        role="merged_model",
        extra_metadata={
            "interpolation_t": 0.5,
            "sources": ["model_a", "model_b_matched"],
        },
    )

    pair_summary = {
        "seeds": [config.seed_a, config.seed_b],
        "matching": {
            "ot_w1_identity_order": matching["identity_w1"],
            "ot_w1_matched": matching["w1"],
            "permutation": permutation.tolist(),
        },
        "identity_barrier": identity_profile,
        "matched_barrier": matched_profile,
        "matched_alignment": matched_stats,
        "exact_modulus": exact_modulus,
        "test_metrics": {
            "model_a": test_metrics_a,
            "model_b_matched": test_metrics_b,
        },
        "training_history": {
            "model_a_train_nll": history_a["train_nll"],
            "model_b_train_nll": history_b["train_nll"],
        },
        "artifacts": {
            "model_a_parameters": "model_a_parameters.png",
            "model_a_checkpoint": "model_a.pt",
            "model_b_checkpoint": "model_b.pt",
            "model_b_matched_checkpoint": "model_b_matched.pt",
            "model_b_matched_parameters": "model_b_matched_parameters.png",
            "merged_model_checkpoint": "merged_model.pt",
            "lmc_barrier": "lmc_barrier.png",
            "predictive_fit": "predictive_fit.png",
            "training_history": "training_history.png",
            "matching_permutation": "matching_permutation.npy",
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(pair_summary, handle, indent=2)

    return {
        "width": config.width,
        "seeds": [config.seed_a, config.seed_b],
        "output_dir": str(output_dir),
        "pair_summary": pair_summary,
    }


def run_width_sweep(args: argparse.Namespace) -> dict[str, object]:
    settings = resolve_settings(args)
    config_base = config_from_settings(settings, args.parameterization)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds, seed_pairs = consecutive_seed_pairs(args.seed_start, args.n_seeds)
    if not seed_pairs:
        raise ValueError("width sweep requires at least two seeds.")
    if args.max_parallel_workers <= 0:
        raise ValueError("max_parallel_workers must be positive.")

    results: list[dict[str, object]] = []
    barrier_curves_by_width: dict[int, list[dict[str, object]]] = {}
    all_histories: list[tuple[int, int, list[float]]] = []
    widths = [2 ** exponent for exponent in args.width_exponents]

    for width in widths:
        (output_dir / f"width_{width}").mkdir(parents=True, exist_ok=True)

    tasks = [
        build_pair_task(
            config_base=config_base,
            output_dir=output_dir,
            width=width,
            seed_a=seed_a,
            seed_b=seed_b,
            time_grid_points=args.time_grid_points,
        )
        for width in widths
        for seed_a, seed_b in seed_pairs
    ]

    completed_jobs: list[dict[str, object]] = []
    if args.max_parallel_workers == 1:
        for task in tqdm(tasks, desc="Pair experiments", unit="pair"):
            completed_jobs.append(run_pair_job(task))
    else:
        spawn_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.max_parallel_workers,
            mp_context=spawn_context,
        ) as executor:
            future_to_task = {executor.submit(run_pair_job, task): task for task in tasks}
            for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Pair experiments", unit="pair"):
                completed_jobs.append(future.result())

    jobs_by_width: dict[int, list[dict[str, object]]] = {width: [] for width in widths}
    for job in completed_jobs:
        jobs_by_width[int(job["width"])].append(job)

    for width in widths:
        config = ExperimentConfig(**{**asdict(config_base), "width": width})
        run_dir = output_dir / f"width_{width}"

        width_jobs = sorted(jobs_by_width[width], key=lambda job: tuple(job["seeds"]))
        pair_summaries: list[dict[str, object]] = []
        pair_summary_paths: list[str] = []
        barrier_curves: list[dict[str, object]] = []
        baseline_bounds: list[float] = []
        timewise_bounds: list[float] = []
        path_bounds: list[float] = []
        dense_actuals: list[float] = []
        coarse_naive_barriers: list[float] = []
        coarse_aligned_barriers: list[float] = []
        matched_w1_values: list[float] = []
        endpoint_nll_means: list[float] = []

        for job in width_jobs:
            pair_summary = job["pair_summary"]
            seed_a, seed_b = pair_summary["seeds"]
            matched_stats = pair_summary["matched_alignment"]
            exact_modulus = pair_summary["exact_modulus"]
            test_metrics_a = pair_summary["test_metrics"]["model_a"]
            test_metrics_b = pair_summary["test_metrics"]["model_b_matched"]
            identity_profile = pair_summary["identity_barrier"]
            matched_profile = pair_summary["matched_barrier"]

            baseline_bound = float(
                matched_stats.get(
                    "k1_barrier_bound",
                    matched_stats.get("raw_k1_barrier_bound", math.nan),
                )
            )
            timewise_bound = float(exact_modulus["timewise_exact_modulus"]["bound"])
            path_bound = float(exact_modulus["path_envelope_exact_modulus"]["bound"])
            actual_barrier_dense = float(exact_modulus["dense_barrier"]["max_barrier"])

            pair_summaries.append(pair_summary)
            pair_summary_paths.append(str((Path(job["output_dir"]) / "summary.json").relative_to(output_dir)))
            all_histories.append((width, seed_a, pair_summary["training_history"]["model_a_train_nll"]))
            all_histories.append((width, seed_b, pair_summary["training_history"]["model_b_train_nll"]))
            barrier_curves.append(
                {
                    "pair": [seed_a, seed_b],
                    "naive": identity_profile["losses"],
                    "aligned": matched_profile["losses"],
                    "naive_barrier": identity_profile["max_barrier"],
                    "aligned_barrier": matched_profile["max_barrier"],
                }
            )
            baseline_bounds.append(baseline_bound)
            timewise_bounds.append(timewise_bound)
            path_bounds.append(path_bound)
            dense_actuals.append(actual_barrier_dense)
            coarse_naive_barriers.append(float(identity_profile["max_barrier"]))
            coarse_aligned_barriers.append(float(matched_profile["max_barrier"]))
            matched_w1_values.append(float(pair_summary["matching"]["ot_w1_matched"]))
            endpoint_nll_means.append(0.5 * (float(test_metrics_a["nll"]) + float(test_metrics_b["nll"])))

        barrier_curves_by_width[width] = barrier_curves

        aggregate_entry = {
            "width": width,
            "num_pairs": len(pair_summaries),
            "baseline_bound_mean": float(np.mean(baseline_bounds)),
            "baseline_bound_std": float(np.std(baseline_bounds)),
            "timewise_exact_modulus_bound_mean": float(np.mean(timewise_bounds)),
            "timewise_exact_modulus_bound_std": float(np.std(timewise_bounds)),
            "path_envelope_exact_modulus_bound_mean": float(np.mean(path_bounds)),
            "path_envelope_exact_modulus_bound_std": float(np.std(path_bounds)),
            "actual_barrier_dense_mean": float(np.mean(dense_actuals)),
            "actual_barrier_dense_std": float(np.std(dense_actuals)),
            "naive_profile_barrier_mean": float(np.mean(coarse_naive_barriers)),
            "naive_profile_barrier_std": float(np.std(coarse_naive_barriers)),
            "aligned_profile_barrier_mean": float(np.mean(coarse_aligned_barriers)),
            "aligned_profile_barrier_std": float(np.std(coarse_aligned_barriers)),
            "baseline_ratio_mean": safe_ratio(float(np.mean(baseline_bounds)), float(np.mean(dense_actuals))),
            "timewise_ratio_mean": safe_ratio(float(np.mean(timewise_bounds)), float(np.mean(dense_actuals))),
            "path_envelope_ratio_mean": safe_ratio(float(np.mean(path_bounds)), float(np.mean(dense_actuals))),
            "matched_w1_mean": float(np.mean(matched_w1_values)),
            "matched_w1_std": float(np.std(matched_w1_values)),
            "endpoint_test_nll_mean": float(np.mean(endpoint_nll_means)),
            "endpoint_test_nll_std": float(np.std(endpoint_nll_means)),
        }

        summary = {
            "config": asdict(config),
            "seeds": seeds,
            "seed_pairs": seed_pairs,
            "pair_summary_paths": pair_summary_paths,
            "pair_summaries": pair_summaries,
            "aggregate": aggregate_entry,
        }
        with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        results.append(aggregate_entry)

    plot_loss_barriers(barrier_curves_by_width, output_dir / "loss_barriers.png")
    plot_barrier_scaling_summary(results, output_dir / "barrier_scaling.png")
    plot_training_loss_summary(all_histories, output_dir / "loss_curves.png")
    plot_primary_width_comparison(results, output_dir / "width_sweep_primary_vs_observed.png")

    aggregate = {
        "base_config": asdict(config_base),
        "seeds": seeds,
        "seed_pairs": seed_pairs,
        "n_seeds": args.n_seeds,
        "max_parallel_workers": args.max_parallel_workers,
        "width_exponents": args.width_exponents,
        "widths": widths,
        "time_grid_points": args.time_grid_points,
        "results": results,
        "artifacts": {
            "loss_barriers": "loss_barriers.png",
            "barrier_scaling": "barrier_scaling.png",
            "loss_curves": "loss_curves.png",
            "primary_plot": "width_sweep_primary_vs_observed.png",
        },
    }
    with (output_dir / "width_sweep_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
    return aggregate


def main() -> None:
    parser = build_sweep_arg_parser()
    args = parser.parse_args()
    run_width_sweep(args)


if __name__ == "__main__":
    main()
