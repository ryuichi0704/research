from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from k1_experiment import (
    DatasetBundle,
    ExperimentConfig,
    TwoLayerK1Net,
    alignment_statistics,
    build_dataset_bundle,
    choose_device,
    config_from_settings,
    distribution_on_dataset,
    evaluate_model,
    gaussian_nll_from_eta,
    interpolate_models,
    optimal_transport_matching,
    permute_hidden_units,
    resolve_settings,
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
        choices=["natural"],
        default="natural",
        help="Width sweep currently supports the natural parameterization only.",
    )
    parser.add_argument(
        "--width-exponents",
        type=int,
        nargs="+",
        default=[8, 9, 10, 11, 12, 13],
        help="Sweep widths 2^k for the listed exponents.",
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
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--evaluation-probe-points", type=int, default=None)
    parser.add_argument("--plot-points", type=int, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
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


def normalized_conditional_moments(
    x_normalized: torch.Tensor,
    dataset_bundle: DatasetBundle,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_raw = x_normalized.detach().cpu().numpy() * dataset_bundle.x_std + dataset_bundle.x_mean
    mean_raw = true_mean(x_raw)
    std_raw = true_std(x_raw)

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
    precision = raw_precision(model, raw_scale)
    return (
        raw_mean.square() / (4.0 * precision)
        + 0.5 * torch.log(math.pi / precision)
        - raw_mean * conditional_mean
        + precision * conditional_second_moment
    )


def omega_u_timewise(
    u_hat: torch.Tensor,
    delta_u: torch.Tensor,
    conditional_mean: torch.Tensor,
    r_low: torch.Tensor,
    r_high: torch.Tensor,
) -> torch.Tensor:
    def evaluate(r_value: torch.Tensor) -> torch.Tensor:
        return delta_u * torch.abs(u_hat / (2.0 * r_value) - conditional_mean) + delta_u.square() / (4.0 * r_value)

    return torch.maximum(evaluate(r_low), evaluate(r_high))


def omega_s_timewise(
    u_hat: torch.Tensor,
    conditional_second_moment: torch.Tensor,
    r_hat: torch.Tensor,
    r_low: torch.Tensor,
    r_high: torch.Tensor,
) -> torch.Tensor:
    def evaluate(r_value: torch.Tensor) -> torch.Tensor:
        return (
            conditional_second_moment * (r_value - r_hat)
            + (u_hat.square() / 4.0) * (1.0 / r_value - 1.0 / r_hat)
            + 0.5 * torch.log(r_hat / r_value)
        )

    return torch.clamp(torch.maximum(evaluate(r_low), evaluate(r_high)), min=0.0)


@torch.no_grad()
def exact_modulus_statistics(
    model_a: TwoLayerK1Net,
    model_b: TwoLayerK1Net,
    dataset_bundle: DatasetBundle,
    device: torch.device,
    time_grid_points: int,
) -> dict[str, object]:
    if model_a.parameterization != "natural" or model_b.parameterization != "natural":
        raise ValueError("exact_modulus_statistics currently supports natural parameterization only.")

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

        r_hat = raw_precision(model_a, s_hat)
        r_low = raw_precision(model_a, s_hat - delta_s)
        r_high = raw_precision(model_a, s_hat + delta_s)

        omega_u = omega_u_timewise(u_hat, delta_u, conditional_mean, r_low, r_high)
        omega_s = omega_s_timewise(u_hat, conditional_second_moment, r_hat, r_low, r_high)
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
        r_hat = raw_precision(model_a, s_hat)
        r_low = raw_precision(model_a, s_hat - delta_s_path)
        r_high = raw_precision(model_a, s_hat + delta_s_path)

        omega_u_env = omega_u_timewise(u_hat, delta_u_path, conditional_mean, r_low, r_high)
        omega_s_env = omega_s_timewise(u_hat, conditional_second_moment, r_hat, r_low, r_high)

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


def plot_primary_width_comparison(
    results: list[dict[str, object]],
    output_path: Path,
) -> None:
    widths = np.array([entry["width"] for entry in results], dtype=float)
    actual = np.array([entry["actual_barrier_dense"] for entry in results], dtype=float)
    primary = np.array([entry["timewise_exact_modulus_bound"] for entry in results], dtype=float)
    visible_values = np.concatenate([actual, primary])
    positive_visible = visible_values[visible_values > 0.0]
    guide_level = float(np.exp(np.mean(np.log(positive_visible)))) if positive_visible.size else 1.0
    width_mid = float(np.exp(np.mean(np.log(widths))))
    slope_minus_one_guide = guide_level * (width_mid / widths)

    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.5), constrained_layout=True)

    ax.loglog(widths, actual, marker="o", linewidth=2.5, label="Observed barrier")
    ax.loglog(widths, primary, marker="o", linewidth=2.5, label="Primary theorem bound")
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


def run_width_sweep(args: argparse.Namespace) -> dict[str, object]:
    settings = resolve_settings(args)
    config_base = config_from_settings(settings, "natural")
    device = choose_device(config_base.device)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    widths = [2 ** exponent for exponent in args.width_exponents]

    for width in widths:
        config = ExperimentConfig(**{**asdict(config_base), "width": width})
        run_dir = output_dir / f"width_{width}"
        run_dir.mkdir(parents=True, exist_ok=True)

        dataset_bundle = build_dataset_bundle(config)
        model_a, history_a = train_model(config, dataset_bundle, config.seed_a, device)
        model_b, history_b = train_model(config, dataset_bundle, config.seed_b, device)

        matching = optimal_transport_matching(model_a, model_b)
        permutation = np.asarray(matching["permutation"], dtype=int)
        matched_b = permute_hidden_units(model_b, permutation, device)

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
            time_grid_points=args.time_grid_points,
        )

        summary = {
            "config": asdict(config),
            "matching": {
                "ot_w1_identity_order": matching["identity_w1"],
                "ot_w1_matched": matching["w1"],
            },
            "matched_alignment": matched_stats,
            "exact_modulus": exact_modulus,
            "test_metrics": {
                "model_a": evaluate_model(model_a, dataset_bundle.test, device),
                "model_b_matched": evaluate_model(matched_b, dataset_bundle.test, device),
            },
            "training_history": {
                "model_a_train_nll": history_a["train_nll"],
                "model_b_train_nll": history_b["train_nll"],
            },
        }
        with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        actual_barrier_dense = float(exact_modulus["dense_barrier"]["max_barrier"])
        baseline_bound = float(matched_stats["k1_barrier_bound"])
        timewise_bound = float(exact_modulus["timewise_exact_modulus"]["bound"])
        path_bound = float(exact_modulus["path_envelope_exact_modulus"]["bound"])

        results.append(
            {
                "width": width,
                "baseline_bound": baseline_bound,
                "timewise_exact_modulus_bound": timewise_bound,
                "path_envelope_exact_modulus_bound": path_bound,
                "actual_barrier_dense": actual_barrier_dense,
                "baseline_ratio": safe_ratio(baseline_bound, actual_barrier_dense),
                "timewise_ratio": safe_ratio(timewise_bound, actual_barrier_dense),
                "path_envelope_ratio": safe_ratio(path_bound, actual_barrier_dense),
                "matched_w1": float(matching["w1"]),
                "endpoint_test_nll_mean": 0.5
                * (
                    float(summary["test_metrics"]["model_a"]["nll"])
                    + float(summary["test_metrics"]["model_b_matched"]["nll"])
                ),
            }
        )

    plot_primary_width_comparison(results, output_dir / "width_sweep_primary_vs_observed.png")

    aggregate = {
        "base_config": asdict(config_base),
        "width_exponents": args.width_exponents,
        "widths": widths,
        "time_grid_points": args.time_grid_points,
        "results": results,
        "artifacts": {
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
