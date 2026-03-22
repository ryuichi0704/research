from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

Parameterization = Literal["natural", "meanvar"]
PrecisionActivation = Literal["softplus", "exp"]


@dataclass
class ExperimentConfig:
    parameterization: Parameterization
    width: int = 1024
    variance_min: float = 1e-3
    variance_max: float = 10.0
    train_size: int = 1000
    test_size: int = 1000
    evaluation_probe_points: int = 2048
    plot_points: int = 2048
    x_max: float = 1.05
    epochs: int = 10000
    batch_size: int = 1000
    learning_rate: float = 1e-1
    learning_rate_min: float = 1e-3
    weight_decay: float = 0.0
    precision_activation: PrecisionActivation = "exp"
    barrier_points: int = 25
    data_seed: int = 2026
    seed_a: int = 0
    seed_b: int = 1
    device: str = "auto"


@dataclass
class DatasetBundle:
    train: TensorDataset
    test: TensorDataset
    evaluation_probe_x: torch.Tensor
    plot_x: torch.Tensor
    y_star: float
    x_mean: float
    x_std: float
    y_mean: float
    y_std: float


@dataclass
class NaturalPathSnapshot:
    t: float
    barrier: float
    exact_convexity_gap: float
    exact_vector_gap: float
    exact_jensen_gap: float
    eta_gap_mean: float
    eta1_gap_mean: float
    eta2_gap_mean: float
    eta_gap_tilde_mean: float
    eta1_gap_tilde_mean: float
    eta2_gap_tilde_mean: float
    eta_jensen_gap_mean: float
    eta2_jensen_gap_mean: float
    raw_gap_sup: float
    raw_gap_mean: float
    raw_mean_gap_sup: float
    raw_scale_gap_sup: float
    raw_mean_gap_mean: float
    raw_scale_gap_mean: float
    local_m_eta: float
    local_lambda_min: float
    local_c_eta_y_star: float
    local_c_eta_y_empirical: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def true_mean(x: np.ndarray) -> np.ndarray:
    return 0.7 * np.sin(7.5 * x) + 0.5 * x


def true_std(x: np.ndarray) -> np.ndarray:
    return np.full_like(x, 0.1)


def sample_synthetic_dataset(
    num_samples: int,
    x_max: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-x_max, x_max, size=(num_samples, 1)).astype(np.float32)
    mu = true_mean(x)
    sigma = true_std(x)
    gaussian_noise = rng.normal(loc=0.0, scale=1.0, size=(num_samples, 1)).astype(np.float32)
    y = mu + sigma * gaussian_noise
    return x.astype(np.float32), y.astype(np.float32)


def build_dataset_bundle(config: ExperimentConfig) -> DatasetBundle:
    train_x, train_y = sample_synthetic_dataset(
        num_samples=config.train_size,
        x_max=config.x_max,
        seed=config.data_seed,
    )
    test_x, test_y = sample_synthetic_dataset(
        num_samples=config.test_size,
        x_max=config.x_max,
        seed=config.data_seed + 1,
    )

    x_mean = float(train_x.mean())
    x_std = float(train_x.std())
    y_mean = float(train_y.mean())
    y_std = float(train_y.std())

    train_x = (train_x - x_mean) / x_std
    train_y = (train_y - y_mean) / y_std
    test_x = (test_x - x_mean) / x_std
    test_y = (test_y - y_mean) / y_std

    evaluation_probe_x = np.linspace(
        -config.x_max,
        config.x_max,
        config.evaluation_probe_points,
        dtype=np.float32,
    )
    plot_x = np.linspace(
        -config.x_max,
        config.x_max,
        config.plot_points,
        dtype=np.float32,
    )
    evaluation_probe_x = (evaluation_probe_x - x_mean) / x_std
    plot_x = (plot_x - x_mean) / x_std

    normalized_train_y = train_y
    normalized_test_y = test_y
    y_star = float(
        max(
            np.max(np.abs(normalized_train_y)),
            np.max(np.abs(normalized_test_y)),
        )
    )

    return DatasetBundle(
        train=TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        test=TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y)),
        evaluation_probe_x=torch.from_numpy(evaluation_probe_x[:, None]),
        plot_x=torch.from_numpy(plot_x[:, None]),
        y_star=y_star,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
    )


class TwoLayerK1Net(nn.Module):
    def __init__(
        self,
        width: int,
        parameterization: Parameterization,
        variance_min: float,
        variance_max: float,
        precision_activation: PrecisionActivation = "softplus",
    ) -> None:
        super().__init__()
        self.width = width
        self.parameterization = parameterization
        self.variance_min = variance_min
        self.variance_max = variance_max
        self.precision_activation = precision_activation
        self.lambda_min = 1.0 / (2.0 * variance_max)
        self.variance_floor = variance_min
        self.fc1 = nn.Linear(1, width, bias=True)
        self.fc2 = nn.Linear(width, 2, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.fc1.bias, mean=0.0, std=0.5)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1.0)

    def hidden_activations(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc1(x))

    def raw_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.hidden_activations(x)) / self.width

    def distribution(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.raw_head(x)
        if self.precision_activation == "exp":
            activate = torch.exp
        else:
            activate = F.softplus
        if self.parameterization == "natural":
            eta1 = raw[:, :1]
            lambda_coord = activate(raw[:, 1:2]) + self.lambda_min
            eta2 = -lambda_coord
            mu = eta1 / (2.0 * lambda_coord)
            var = 1.0 / (2.0 * lambda_coord)
        else:
            mu = raw[:, :1]
            var = activate(raw[:, 1:2]) + self.variance_floor
            eta1 = mu / var
            eta2 = -0.5 / var
        return {
            "raw": raw,
            "mu": mu,
            "var": var,
            "eta1": eta1,
            "eta2": eta2,
        }

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(x)
        if self.parameterization == "natural":
            eta1 = dist["eta1"]
            eta2 = dist["eta2"]
            log_partition = -(eta1**2) / (4.0 * eta2) + 0.5 * torch.log((-math.pi) / eta2)
            nll = log_partition - eta1 * y - eta2 * (y**2)
        else:
            mu = dist["mu"]
            var = dist["var"]
            nll = 0.5 * (torch.log(2.0 * math.pi * var) + (y - mu) ** 2 / var)
        return nll.mean()


def build_model(config: ExperimentConfig) -> TwoLayerK1Net:
    return TwoLayerK1Net(
        width=config.width,
        parameterization=config.parameterization,
        variance_min=config.variance_min,
        variance_max=config.variance_max,
        precision_activation=config.precision_activation,
    )


@torch.no_grad()
def evaluate_model(
    model: TwoLayerK1Net,
    dataset: TensorDataset,
    device: torch.device,
    batch_size: int = 1024,
) -> dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_sq_error = 0.0
    total_count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        batch_size_actual = x.shape[0]
        loss = model.loss(x, y)
        mu = model.distribution(x)["mu"]
        total_loss += float(loss.item()) * batch_size_actual
        total_sq_error += float(torch.sum((mu - y) ** 2).item())
        total_count += batch_size_actual
    return {
        "nll": total_loss / total_count,
        "rmse": math.sqrt(total_sq_error / total_count),
    }


def train_model(
    config: ExperimentConfig,
    dataset_bundle: DatasetBundle,
    seed: int,
    device: torch.device,
    show_progress: bool = True,
    progress_desc: str = "Training",
) -> tuple[TwoLayerK1Net, dict[str, list[float]]]:
    set_seed(seed)
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = DataLoader(
        dataset_bundle.train,
        batch_size=config.batch_size,
        shuffle=True,
    )

    history = {"train_nll": []}

    epoch_iterator = range(1, config.epochs + 1)
    if show_progress:
        epoch_iterator = tqdm(epoch_iterator, desc=progress_desc, unit="epoch")

    for epoch in epoch_iterator:
        lr_t = config.learning_rate_min + 0.5 * (config.learning_rate - config.learning_rate_min) * (
            1.0 + math.cos(math.pi * epoch / config.epochs)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_t

        model.train()
        epoch_losses: list[float] = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(x_batch, y_batch)
            (loss * config.width).backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        history["train_nll"].append(float(np.mean(epoch_losses)))

    return model, history


def extract_neuron_table(model: TwoLayerK1Net) -> np.ndarray:
    first_weight = model.fc1.weight.detach().cpu().numpy()[:, 0]
    first_bias = model.fc1.bias.detach().cpu().numpy()
    second_weight = model.fc2.weight.detach().cpu().numpy()
    mean_component = second_weight[0]
    variance_component = second_weight[1]
    return np.stack(
        [mean_component, variance_component, first_weight, first_bias],
        axis=1,
    )


def optimal_transport_matching(
    model_a: TwoLayerK1Net,
    model_b: TwoLayerK1Net,
) -> dict[str, object]:
    table_a = extract_neuron_table(model_a)
    table_b = extract_neuron_table(model_b)
    cost_matrix = np.linalg.norm(table_a[:, None, :] - table_b[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    permutation = np.empty_like(col_ind)
    permutation[row_ind] = col_ind
    matched_cost = float(cost_matrix[row_ind, col_ind].mean())
    identity_cost = float(np.linalg.norm(table_a - table_b, axis=1).mean())
    return {
        "permutation": permutation,
        "w1": matched_cost,
        "identity_w1": identity_cost,
    }


def permute_hidden_units(
    model: TwoLayerK1Net,
    permutation: np.ndarray,
    device: torch.device,
) -> TwoLayerK1Net:
    perm = torch.as_tensor(permutation, dtype=torch.long, device=device)
    permuted_model = copy.deepcopy(model).to(device)
    with torch.no_grad():
        permuted_model.fc1.weight.copy_(model.fc1.weight[perm])
        permuted_model.fc1.bias.copy_(model.fc1.bias[perm])
        permuted_model.fc2.weight.copy_(model.fc2.weight[:, perm])
    return permuted_model


def interpolate_models(
    model_a: TwoLayerK1Net,
    model_b: TwoLayerK1Net,
    t: float,
    device: torch.device,
) -> TwoLayerK1Net:
    interpolated = copy.deepcopy(model_a).to(device)
    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()
    new_state_dict = {}
    for key, value_a in state_dict_a.items():
        value_b = state_dict_b[key]
        new_state_dict[key] = (1.0 - t) * value_a + t * value_b
    interpolated.load_state_dict(new_state_dict)
    return interpolated


def barrier_profile(
    model_a: TwoLayerK1Net,
    model_b: TwoLayerK1Net,
    dataset: TensorDataset,
    num_points: int,
    device: torch.device,
) -> dict[str, object]:
    metric_a = evaluate_model(model_a, dataset, device)
    metric_b = evaluate_model(model_b, dataset, device)
    endpoint_a = metric_a["nll"]
    endpoint_b = metric_b["nll"]

    ts = np.linspace(0.0, 1.0, num_points)
    losses: list[float] = []
    barriers: list[float] = []
    for t in ts:
        interpolated = interpolate_models(model_a, model_b, float(t), device)
        loss_t = evaluate_model(interpolated, dataset, device)["nll"]
        linear_reference = (1.0 - float(t)) * endpoint_a + float(t) * endpoint_b
        losses.append(loss_t)
        barriers.append(loss_t - linear_reference)

    max_index = int(np.argmax(barriers))
    return {
        "ts": ts.tolist(),
        "losses": losses,
        "barriers": barriers,
        "max_barrier": float(barriers[max_index]),
        "argmax_t": float(ts[max_index]),
        "endpoint_losses": [endpoint_a, endpoint_b],
    }


def c_eta(m_value: float, lambda_min: float, y_star: float) -> float:
    return (
        m_value / (2.0 * lambda_min)
        + y_star
        + (m_value**2) / (4.0 * lambda_min**2)
        + 1.0 / (2.0 * lambda_min)
        + y_star**2
    )


def effective_lambda_min(variance_max: float) -> float:
    return 1.0 / (2.0 * variance_max)


def gaussian_nll_from_eta(
    eta1: torch.Tensor,
    eta2: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    log_partition = -(eta1**2) / (4.0 * eta2) + 0.5 * torch.log((-math.pi) / eta2)
    return log_partition - eta1 * y - eta2 * (y**2)


def distribution_from_raw(
    model: TwoLayerK1Net,
    raw: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if model.precision_activation == "exp":
        activate = torch.exp
    else:
        activate = F.softplus

    if model.parameterization == "natural":
        eta1 = raw[:, :1]
        lambda_coord = activate(raw[:, 1:2]) + model.lambda_min
        eta2 = -lambda_coord
        mu = eta1 / (2.0 * lambda_coord)
        var = 1.0 / (2.0 * lambda_coord)
    else:
        mu = raw[:, :1]
        var = activate(raw[:, 1:2]) + model.variance_floor
        eta1 = mu / var
        eta2 = -0.5 / var
    return {
        "raw": raw,
        "mu": mu,
        "var": var,
        "eta1": eta1,
        "eta2": eta2,
    }


@torch.no_grad()
def distribution_on_dataset(
    model: TwoLayerK1Net,
    x: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    dist = model.distribution(x.to(device))
    return {key: value.detach() for key, value in dist.items()}


def path_grid_points(num_plot_points: int) -> int:
    return max(101, 4 * num_plot_points + 1)


def sigmoid_second_derivative_abs_max() -> float:
    # max_{z in R} |sigma''(z)| for the logistic sigmoid.
    return math.sqrt(3.0) / 18.0


@torch.no_grad()
def natural_barrier_certificates(
    model_a: TwoLayerK1Net,
    model_b: TwoLayerK1Net,
    dataset_bundle: DatasetBundle,
    alignment_stats: dict[str, float],
    barrier_profile_stats: dict[str, object],
    config: ExperimentConfig,
    device: torch.device,
    num_points: int,
) -> dict[str, object]:
    if config.parameterization != "natural":
        raise ValueError("natural_barrier_certificates requires natural parameterization.")

    test_x, test_y = dataset_bundle.test.tensors
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    probe_x = dataset_bundle.evaluation_probe_x.to(device)

    dist_a_probe = distribution_on_dataset(model_a, probe_x, device)
    dist_b_probe = distribution_on_dataset(model_b, probe_x, device)
    dist_a_test = distribution_on_dataset(model_a, test_x, device)
    dist_b_test = distribution_on_dataset(model_b, test_x, device)

    table_a = extract_neuron_table(model_a)
    table_b = extract_neuron_table(model_b)
    mean_delta_rms = float(np.sqrt(np.mean((table_a[:, 0] - table_b[:, 0]) ** 2)))
    variance_delta_rms = float(np.sqrt(np.mean((table_a[:, 1] - table_b[:, 1]) ** 2)))
    mean_weight_rms = float(max(np.sqrt(np.mean(table_a[:, 0] ** 2)), np.sqrt(np.mean(table_b[:, 0] ** 2))))
    variance_weight_rms = float(max(np.sqrt(np.mean(table_a[:, 1] ** 2)), np.sqrt(np.mean(table_b[:, 1] ** 2))))
    m_delta_v_infty = max(mean_delta_rms, variance_delta_rms)

    xi_delta_sq = np.sum((table_a[:, 2:] - table_b[:, 2:]) ** 2, axis=1)
    v_a_sq = np.sum(table_a[:, :2] ** 2, axis=1)
    v_b_sq = np.sum(table_b[:, :2] ** 2, axis=1)
    weighted_hidden_mismatch = float(np.mean(np.maximum(v_a_sq, v_b_sq) * xi_delta_sq))

    b_n = float(alignment_stats["B_N"])
    delta_s = float(alignment_stats["Delta_s_N"])
    m_v_infty = float(alignment_stats["M_V_infty"])
    y_star = float(alignment_stats["Y_star"])
    lambda_min_cfg = float(alignment_stats["effective_lambda_min"])
    empirical_y_max = float(torch.max(torch.abs(test_y)).item())

    x_probe_abs_max = float(torch.max(torch.abs(probe_x)).item())
    x_test_abs_max = float(torch.max(torch.abs(test_x)).item())
    x_abs_max = max(x_probe_abs_max, x_test_abs_max)
    k_phi = 1.0
    k_nabla_global = 0.25 * math.sqrt(x_abs_max**2 + 1.0)
    l_nabla_global = sigmoid_second_derivative_abs_max() * (x_abs_max**2 + 1.0)

    ts = np.linspace(0.0, 1.0, num_points)
    snapshots: list[NaturalPathSnapshot] = []

    path_m_eta = 0.0
    path_lambda_min = float("inf")
    max_eta_gap_mean = 0.0
    max_eta_gap_mean_local_y_star = 0.0
    max_eta_gap_mean_local_y_emp = 0.0
    max_exact_convexity_gap = -float("inf")
    max_exact_vector_gap = -float("inf")
    max_exact_jensen_gap = -float("inf")

    for t in ts:
        model_t = interpolate_models(model_a, model_b, float(t), device)
        dist_t_probe = distribution_on_dataset(model_t, probe_x, device)
        dist_t_test = distribution_on_dataset(model_t, test_x, device)

        raw_hat_probe = (1.0 - t) * dist_a_probe["raw"] + t * dist_b_probe["raw"]
        eta_hat_probe_1 = (1.0 - t) * dist_a_probe["eta1"] + t * dist_b_probe["eta1"]
        eta_hat_probe_2 = (1.0 - t) * dist_a_probe["eta2"] + t * dist_b_probe["eta2"]
        tilde_probe = distribution_from_raw(model_a, raw_hat_probe)

        raw_hat_test = (1.0 - t) * dist_a_test["raw"] + t * dist_b_test["raw"]
        eta_hat_test_1 = (1.0 - t) * dist_a_test["eta1"] + t * dist_b_test["eta1"]
        eta_hat_test_2 = (1.0 - t) * dist_a_test["eta2"] + t * dist_b_test["eta2"]
        tilde_test = distribution_from_raw(model_a, raw_hat_test)

        raw_gap_probe = torch.max(torch.abs(dist_t_probe["raw"] - raw_hat_probe), dim=1).values
        raw_gap_probe_mean = torch.abs(dist_t_probe["raw"][:, :1] - raw_hat_probe[:, :1])
        raw_gap_probe_scale = torch.abs(dist_t_probe["raw"][:, 1:2] - raw_hat_probe[:, 1:2])
        eta_gap_probe_1 = torch.abs(dist_t_probe["eta1"] - eta_hat_probe_1)
        eta_gap_probe_2 = torch.abs(dist_t_probe["eta2"] - eta_hat_probe_2)
        eta_gap_probe = torch.maximum(
            eta_gap_probe_1,
            eta_gap_probe_2,
        )
        eta_gap_tilde_probe_1 = torch.abs(dist_t_probe["eta1"] - tilde_probe["eta1"])
        eta_gap_tilde_probe_2 = torch.abs(dist_t_probe["eta2"] - tilde_probe["eta2"])
        eta_gap_tilde_probe = torch.maximum(
            eta_gap_tilde_probe_1,
            eta_gap_tilde_probe_2,
        )
        eta_jensen_gap_probe_2 = torch.abs(tilde_probe["eta2"] - eta_hat_probe_2)
        eta_jensen_gap_probe = torch.maximum(
            torch.abs(tilde_probe["eta1"] - eta_hat_probe_1),
            eta_jensen_gap_probe_2,
        )

        local_m_eta_t = float(
            max(
                torch.max(torch.abs(dist_t_probe["eta1"])).item(),
                torch.max(torch.abs(eta_hat_probe_1)).item(),
                torch.max(torch.abs(tilde_probe["eta1"])).item(),
            )
        )
        local_lambda_min_t = float(
            min(
                torch.min(-dist_t_probe["eta2"]).item(),
                torch.min(-eta_hat_probe_2).item(),
                torch.min(-tilde_probe["eta2"]).item(),
            )
        )

        c_eta_y_star_t = c_eta(local_m_eta_t, local_lambda_min_t, y_star)
        c_eta_y_emp_t = c_eta(local_m_eta_t, local_lambda_min_t, empirical_y_max)

        nll_t = gaussian_nll_from_eta(dist_t_test["eta1"], dist_t_test["eta2"], test_y).mean()
        nll_hat = gaussian_nll_from_eta(eta_hat_test_1, eta_hat_test_2, test_y).mean()
        nll_tilde = gaussian_nll_from_eta(tilde_test["eta1"], tilde_test["eta2"], test_y).mean()
        endpoint_a = float(barrier_profile_stats["endpoint_losses"][0])
        endpoint_b = float(barrier_profile_stats["endpoint_losses"][1])
        linear_reference = (1.0 - float(t)) * endpoint_a + float(t) * endpoint_b

        snapshot = NaturalPathSnapshot(
            t=float(t),
            barrier=float(nll_t.item() - linear_reference),
            exact_convexity_gap=float((nll_t - nll_hat).item()),
            exact_vector_gap=float((nll_t - nll_tilde).item()),
            exact_jensen_gap=float((nll_tilde - nll_hat).item()),
            eta_gap_mean=float(eta_gap_probe.mean().item()),
            eta1_gap_mean=float(eta_gap_probe_1.mean().item()),
            eta2_gap_mean=float(eta_gap_probe_2.mean().item()),
            eta_gap_tilde_mean=float(eta_gap_tilde_probe.mean().item()),
            eta1_gap_tilde_mean=float(eta_gap_tilde_probe_1.mean().item()),
            eta2_gap_tilde_mean=float(eta_gap_tilde_probe_2.mean().item()),
            eta_jensen_gap_mean=float(eta_jensen_gap_probe.mean().item()),
            eta2_jensen_gap_mean=float(eta_jensen_gap_probe_2.mean().item()),
            raw_gap_sup=float(raw_gap_probe.max().item()),
            raw_gap_mean=float(raw_gap_probe.mean().item()),
            raw_mean_gap_sup=float(raw_gap_probe_mean.max().item()),
            raw_scale_gap_sup=float(raw_gap_probe_scale.max().item()),
            raw_mean_gap_mean=float(raw_gap_probe_mean.mean().item()),
            raw_scale_gap_mean=float(raw_gap_probe_scale.mean().item()),
            local_m_eta=local_m_eta_t,
            local_lambda_min=local_lambda_min_t,
            local_c_eta_y_star=c_eta_y_star_t,
            local_c_eta_y_empirical=c_eta_y_emp_t,
        )
        snapshots.append(snapshot)

        path_m_eta = max(path_m_eta, local_m_eta_t)
        path_lambda_min = min(path_lambda_min, local_lambda_min_t)
        max_eta_gap_mean = max(max_eta_gap_mean, snapshot.eta_gap_mean)
        max_eta_gap_mean_local_y_star = max(
            max_eta_gap_mean_local_y_star,
            snapshot.local_c_eta_y_star * snapshot.eta_gap_mean,
        )
        max_eta_gap_mean_local_y_emp = max(
            max_eta_gap_mean_local_y_emp,
            snapshot.local_c_eta_y_empirical * snapshot.eta_gap_mean,
        )
        max_exact_convexity_gap = max(max_exact_convexity_gap, snapshot.exact_convexity_gap)
        max_exact_vector_gap = max(max_exact_vector_gap, snapshot.exact_vector_gap)
        max_exact_jensen_gap = max(max_exact_jensen_gap, snapshot.exact_jensen_gap)

    c_eta_global = c_eta(k_phi * m_v_infty, lambda_min_cfg, y_star)
    c_eta_path_y_star = c_eta(path_m_eta, path_lambda_min, y_star)
    c_eta_path_y_emp = c_eta(path_m_eta, path_lambda_min, empirical_y_max)
    c1_global = k_phi * m_v_infty / (2.0 * lambda_min_cfg) + y_star
    c2_global = (k_phi * m_v_infty) ** 2 / (4.0 * lambda_min_cfg**2) + 1.0 / (2.0 * lambda_min_cfg) + y_star**2
    c1_path_y_star = path_m_eta / (2.0 * path_lambda_min) + y_star
    c2_path_y_star = path_m_eta**2 / (4.0 * path_lambda_min**2) + 1.0 / (2.0 * path_lambda_min) + y_star**2
    c1_path_y_emp = path_m_eta / (2.0 * path_lambda_min) + empirical_y_max
    c2_path_y_emp = path_m_eta**2 / (4.0 * path_lambda_min**2) + 1.0 / (2.0 * path_lambda_min) + empirical_y_max**2

    theorem_transfer = 0.5 * k_nabla_global * m_v_infty * math.sqrt(max(b_n, 0.0))
    theorem_jensen = (1.0 / 32.0) * delta_s
    refined_transfer = (
        0.25 * k_nabla_global * m_delta_v_infty * math.sqrt(max(b_n, 0.0))
        + 0.125 * l_nabla_global * m_v_infty * b_n
    )
    theorem_transfer_mean = 0.5 * k_nabla_global * mean_weight_rms * math.sqrt(max(b_n, 0.0))
    theorem_transfer_scale = 0.5 * k_nabla_global * variance_weight_rms * math.sqrt(max(b_n, 0.0))
    refined_transfer_mean = (
        0.25 * k_nabla_global * mean_delta_rms * math.sqrt(max(b_n, 0.0))
        + 0.125 * l_nabla_global * mean_weight_rms * b_n
    )
    refined_transfer_scale = (
        0.25 * k_nabla_global * variance_delta_rms * math.sqrt(max(b_n, 0.0))
        + 0.125 * l_nabla_global * variance_weight_rms * b_n
    )

    observed_transfer = max(snapshot.raw_gap_sup for snapshot in snapshots)
    observed_transfer_mean = max(snapshot.raw_mean_gap_sup for snapshot in snapshots)
    observed_transfer_scale = max(snapshot.raw_scale_gap_sup for snapshot in snapshots)
    observed_eta_gap = max(snapshot.eta_gap_mean for snapshot in snapshots)
    observed_eta1_gap = max(snapshot.eta1_gap_mean for snapshot in snapshots)
    observed_eta2_gap = max(snapshot.eta2_gap_mean for snapshot in snapshots)
    observed_vector_gap = max(snapshot.eta_gap_tilde_mean for snapshot in snapshots)
    observed_vector_gap_mean = max(snapshot.eta1_gap_tilde_mean for snapshot in snapshots)
    observed_vector_gap_scale = max(snapshot.eta2_gap_tilde_mean for snapshot in snapshots)
    observed_jensen_gap = max(snapshot.eta_jensen_gap_mean for snapshot in snapshots)
    observed_jensen_gap_scale = max(snapshot.eta2_jensen_gap_mean for snapshot in snapshots)
    exact_total_gap = observed_vector_gap + observed_jensen_gap

    dense_barrier = max(snapshot.barrier for snapshot in snapshots)
    dense_t = max(snapshots, key=lambda item: item.barrier).t
    convexity_t = max(snapshots, key=lambda item: item.exact_convexity_gap).t

    return {
        "grid_points": num_points,
        "empirical_y_max_test": empirical_y_max,
        "K_nabla_global": k_nabla_global,
        "L_nabla_global": l_nabla_global,
        "M_a_infty": mean_weight_rms,
        "M_c_infty": variance_weight_rms,
        "M_Delta_V_infty": m_delta_v_infty,
        "M_Delta_a": mean_delta_rms,
        "M_Delta_c": variance_delta_rms,
        "weighted_hidden_mismatch": weighted_hidden_mismatch,
        "path_strip": {
            "max_abs_eta1": path_m_eta,
            "min_neg_eta2": path_lambda_min,
            "C_eta_y_star": c_eta_path_y_star,
            "C_eta_empirical_y": c_eta_path_y_emp,
        },
        "observed_gaps": {
            "raw_transfer_sup": observed_transfer,
            "raw_transfer_mean_sup": observed_transfer_mean,
            "raw_transfer_scale_sup": observed_transfer_scale,
            "eta_gap_mean": observed_eta_gap,
            "eta1_gap_mean": observed_eta1_gap,
            "eta2_gap_mean": observed_eta2_gap,
            "eta_vector_gap_mean": observed_vector_gap,
            "eta1_vector_gap_mean": observed_vector_gap_mean,
            "eta2_vector_gap_mean": observed_vector_gap_scale,
            "eta_jensen_gap_mean": observed_jensen_gap,
            "eta2_jensen_gap_mean": observed_jensen_gap_scale,
        },
        "bounds": {
            "theorem_global": c_eta_global * (theorem_transfer + theorem_jensen),
            "theorem_local_strip_y_star": c_eta_path_y_star * (theorem_transfer + theorem_jensen),
            "theorem_local_strip_empirical_y": c_eta_path_y_emp * (theorem_transfer + theorem_jensen),
            "refined_transfer_global": c_eta_global * (refined_transfer + theorem_jensen),
            "refined_transfer_local_strip_y_star": c_eta_path_y_star * (refined_transfer + theorem_jensen),
            "refined_transfer_local_strip_empirical_y": c_eta_path_y_emp * (refined_transfer + theorem_jensen),
            "coordinatewise_theorem_global": c1_global * theorem_transfer_mean + c2_global * (theorem_transfer_scale + theorem_jensen),
            "coordinatewise_theorem_local_strip_y_star": c1_path_y_star * theorem_transfer_mean + c2_path_y_star * (theorem_transfer_scale + theorem_jensen),
            "coordinatewise_theorem_local_strip_empirical_y": c1_path_y_emp * theorem_transfer_mean + c2_path_y_emp * (theorem_transfer_scale + theorem_jensen),
            "coordinatewise_refined_global": c1_global * refined_transfer_mean + c2_global * (refined_transfer_scale + theorem_jensen),
            "coordinatewise_refined_local_strip_y_star": c1_path_y_star * refined_transfer_mean + c2_path_y_star * (refined_transfer_scale + theorem_jensen),
            "coordinatewise_refined_local_strip_empirical_y": c1_path_y_emp * refined_transfer_mean + c2_path_y_emp * (refined_transfer_scale + theorem_jensen),
            "observed_gap_global_strip_y_star": c_eta_global * exact_total_gap,
            "observed_gap_local_strip_y_star": c_eta_path_y_star * exact_total_gap,
            "observed_gap_local_strip_empirical_y": c_eta_path_y_emp * exact_total_gap,
            "observed_eta_gap_local_timewise_y_star": max_eta_gap_mean_local_y_star,
            "observed_eta_gap_local_timewise_empirical_y": max_eta_gap_mean_local_y_emp,
            "exact_convexity_upper_bound": max_exact_convexity_gap,
            "exact_positive_vector_gap": max(0.0, max_exact_vector_gap),
            "exact_positive_jensen_gap": max(0.0, max_exact_jensen_gap),
        },
        "components": {
            "theorem_transfer": theorem_transfer,
            "theorem_jensen": theorem_jensen,
            "refined_transfer": refined_transfer,
            "theorem_transfer_mean": theorem_transfer_mean,
            "theorem_transfer_scale": theorem_transfer_scale,
            "refined_transfer_mean": refined_transfer_mean,
            "refined_transfer_scale": refined_transfer_scale,
            "C1_global_y_star": c1_global,
            "C2_global_y_star": c2_global,
            "C1_path_y_star": c1_path_y_star,
            "C2_path_y_star": c2_path_y_star,
        },
        "dense_grid_reference": {
            "max_barrier": dense_barrier,
            "argmax_t": dense_t,
            "max_exact_convexity_upper_bound": max_exact_convexity_gap,
            "argmax_exact_convexity_t": convexity_t,
        },
        "snapshots": [asdict(snapshot) for snapshot in snapshots],
    }


@torch.no_grad()
def alignment_statistics(
    model_a: TwoLayerK1Net,
    model_b: TwoLayerK1Net,
    evaluation_probe_x: torch.Tensor,
    config: ExperimentConfig,
    y_star: float,
    device: torch.device,
) -> dict[str, float]:
    table_a = extract_neuron_table(model_a)
    table_b = extract_neuron_table(model_b)

    xi_a = table_a[:, 2:]
    xi_b = table_b[:, 2:]
    mean_weights_a = table_a[:, 0]
    variance_weights_a = table_a[:, 1]
    mean_weights_b = table_b[:, 0]
    variance_weights_b = table_b[:, 1]

    evaluation_probe_x = evaluation_probe_x.to(device)
    raw_dist_a = model_a.distribution(evaluation_probe_x)["raw"].detach().cpu().numpy()
    raw_dist_b = model_b.distribution(evaluation_probe_x)["raw"].detach().cpu().numpy()
    raw_a = raw_dist_a[:, 1]
    raw_b = raw_dist_b[:, 1]

    b_n = float(np.mean(np.sum((xi_a - xi_b) ** 2, axis=1)))
    delta_s = float(np.mean((raw_a - raw_b) ** 2))
    m_v_infty = float(
        max(
            np.sqrt(np.mean(mean_weights_a**2)),
            np.sqrt(np.mean(variance_weights_a**2)),
            np.sqrt(np.mean(mean_weights_b**2)),
            np.sqrt(np.mean(variance_weights_b**2)),
        )
    )

    stats = {
        "B_N": b_n,
        "Delta_s_N": delta_s,
        "Delta_raw_N": float(np.mean(np.sum((raw_dist_a - raw_dist_b) ** 2, axis=1))),
        "M_V_infty": m_v_infty,
    }

    if config.parameterization == "natural":
        k_phi = 1.0
        k_nabla = 0.25 * math.sqrt(config.x_max**2 + 1.0)
        m_value = k_phi * m_v_infty
        theorem_bound = c_eta(m_value, effective_lambda_min(config.variance_max), y_star) * (
            0.5 * k_nabla * m_v_infty * math.sqrt(max(b_n, 0.0))
            + (1.0 / 32.0) * delta_s
        )
        stats["k1_barrier_bound"] = theorem_bound
        stats["K_phi"] = k_phi
        stats["K_nabla"] = k_nabla
        stats["Y_star"] = y_star
        stats["effective_lambda_min"] = effective_lambda_min(config.variance_max)
    elif config.precision_activation == "exp":
        k_phi = 1.0
        k_nabla = 0.25 * math.sqrt(config.x_max**2 + 1.0)
        m_value = k_phi * m_v_infty
        lambda_m = m_value + y_star
        c_raw = lambda_m / config.variance_min + 0.5 + (lambda_m**2) / (2.0 * config.variance_min)
        rho_raw = lambda_m / config.variance_min + (lambda_m**2) / (2.0 * config.variance_min)
        theorem_bound = (
            0.5 * c_raw * k_nabla * m_v_infty * math.sqrt(max(b_n, 0.0))
            + 0.125 * rho_raw * stats["Delta_raw_N"]
        )
        stats["raw_k1_barrier_bound"] = theorem_bound
        stats["K_phi"] = k_phi
        stats["K_nabla"] = k_nabla
        stats["Y_star"] = y_star
        stats["raw_C"] = c_raw
        stats["raw_rho"] = rho_raw
    return stats


def load_yaml_config(config_path: Path) -> dict[str, object]:
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping at top level: {config_path}")
    return loaded


def flatten_yaml_settings(config_data: dict[str, object]) -> dict[str, object]:
    section_mappings = {
        "experiment": {
            "parameterization": "parameterization",
            "output_dir": "output_dir",
            "device": "device",
        },
        "model": {
            "width": "width",
            "variance_min": "variance_min",
            "variance_max": "variance_max",
            "precision_activation": "precision_activation",
        },
        "data": {
            "train_size": "train_size",
            "test_size": "test_size",
            "x_max": "x_max",
        },
        "train": {
            "epochs": "epochs",
            "batch_size": "batch_size",
            "learning_rate": "learning_rate",
            "learning_rate_min": "learning_rate_min",
            "weight_decay": "weight_decay",
        },
        "evaluation": {
            "barrier_points": "barrier_points",
            "probe_points": "evaluation_probe_points",
        },
        "visualization": {
            "plot_points": "plot_points",
        },
        "seeds": {
            "data": "data_seed",
            "model_a": "seed_a",
            "model_b": "seed_b",
        },
    }
    canonical_keys = {
        "parameterization",
        "output_dir",
        "device",
        "width",
        "variance_min",
        "variance_max",
        "precision_activation",
        "train_size",
        "test_size",
        "x_max",
        "epochs",
        "batch_size",
        "learning_rate",
        "learning_rate_min",
        "weight_decay",
        "barrier_points",
        "evaluation_probe_points",
        "plot_points",
        "data_seed",
        "seed_a",
        "seed_b",
    }

    flattened: dict[str, object] = {}

    for key, value in config_data.items():
        if key in canonical_keys:
            flattened[key] = normalize_setting(key, value)
            continue
        if key in section_mappings:
            if not isinstance(value, dict):
                raise ValueError(f"Config section '{key}' must be a mapping.")
            mapping = section_mappings[key]
            unknown_subkeys = sorted(set(value) - set(mapping))
            if unknown_subkeys:
                unknown = ", ".join(unknown_subkeys)
                raise ValueError(f"Unknown keys in config section '{key}': {unknown}")
            for subkey, subvalue in value.items():
                canonical_key = mapping[subkey]
                flattened[canonical_key] = normalize_setting(canonical_key, subvalue)
            continue
        raise ValueError(f"Unknown config key: {key}")

    return flattened


def normalize_setting(key: str, value: object) -> object:
    if value is None:
        return None
    if key == "output_dir":
        return Path(value)
    return value


def resolve_settings(args: argparse.Namespace) -> dict[str, object]:
    defaults = {
        "parameterization": "meanvar",
        "output_dir": Path("results"),
        "width": 1024,
        "variance_min": 1e-3,
        "variance_max": 10.0,
        "precision_activation": "exp",
        "train_size": 1000,
        "test_size": 1000,
        "evaluation_probe_points": 2048,
        "plot_points": 2048,
        "x_max": 1.05,
        "epochs": 10000,
        "batch_size": 1000,
        "learning_rate": 1e-1,
        "learning_rate_min": 1e-3,
        "weight_decay": 0.0,
        "barrier_points": 25,
        "data_seed": 2026,
        "seed_a": 0,
        "seed_b": 1,
        "device": "auto",
    }
    settings = dict(defaults)

    if args.config is not None:
        yaml_settings = flatten_yaml_settings(load_yaml_config(args.config))
        for key, value in yaml_settings.items():
            settings[key] = normalize_setting(key, value)

    for key in defaults:
        value = getattr(args, key)
        if value is not None:
            settings[key] = normalize_setting(key, value)

    valid_parameterizations = {"natural", "meanvar", "both"}
    if settings["parameterization"] not in valid_parameterizations:
        choices = ", ".join(sorted(valid_parameterizations))
        raise ValueError(f"parameterization must be one of: {choices}")

    valid_devices = {"auto", "cpu", "cuda"}
    if settings["device"] not in valid_devices:
        choices = ", ".join(sorted(valid_devices))
        raise ValueError(f"device must be one of: {choices}")

    variance_min = float(settings["variance_min"])
    variance_max = float(settings["variance_max"])
    if variance_min <= 0.0:
        raise ValueError("variance_min must be positive.")
    if variance_max <= 0.0:
        raise ValueError("variance_max must be positive.")
    if variance_min >= variance_max:
        raise ValueError("variance_min must be smaller than variance_max.")

    return settings


def plot_parameter_distribution(
    model: TwoLayerK1Net,
    output_path: Path,
    title: str,
) -> None:
    first_weight = model.fc1.weight.detach().cpu().numpy()[:, 0]
    first_bias = model.fc1.bias.detach().cpu().numpy()
    second_weight = model.fc2.weight.detach().cpu().numpy()
    mean_component = second_weight[0]
    variance_component = second_weight[1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    vmin = float(first_bias.min())
    vmax = float(first_bias.max())

    scatter_mean = axes[0].scatter(
        mean_component,
        first_weight,
        c=first_bias,
        cmap="coolwarm",
        s=36,
        edgecolors="black",
        linewidths=0.3,
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_title("Mean Component")
    axes[0].set_xlabel("Second-layer weight")
    axes[0].set_ylabel("First-layer weight")

    scatter_var = axes[1].scatter(
        variance_component,
        first_weight,
        c=first_bias,
        cmap="coolwarm",
        s=36,
        edgecolors="black",
        linewidths=0.3,
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title("Variance Component")
    axes[1].set_xlabel("Second-layer weight")
    axes[1].set_ylabel("First-layer weight")

    colorbar = fig.colorbar(scatter_var, ax=axes, shrink=0.95)
    colorbar.set_label("First-layer bias")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_barrier_curves(
    identity_profile: dict[str, object],
    matched_profile: dict[str, object],
    output_path: Path,
    title: str,
) -> None:
    ts = np.array(identity_profile["ts"], dtype=float)
    identity_losses = np.array(identity_profile["losses"], dtype=float)
    matched_losses = np.array(matched_profile["losses"], dtype=float)
    identity_barriers = np.array(identity_profile["barriers"], dtype=float)
    matched_barriers = np.array(matched_profile["barriers"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    axes[0].plot(ts, identity_losses, label="Identity matching", linewidth=2.0)
    axes[0].plot(ts, matched_losses, label="Matched", linewidth=2.0)
    axes[0].set_xlabel("Interpolation coefficient t")
    axes[0].set_ylabel("Test NLL")
    axes[0].set_title("Interpolation Loss")
    axes[0].legend()

    axes[1].plot(ts, identity_barriers, label="Identity matching", linewidth=2.0)
    axes[1].plot(ts, matched_barriers, label="Matched", linewidth=2.0)
    axes[1].axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xlabel("Interpolation coefficient t")
    axes[1].set_ylabel("Barrier")
    axes[1].set_title("LMC Barrier")
    axes[1].legend()

    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_predictive_fit(
    model_a: TwoLayerK1Net,
    model_b: TwoLayerK1Net,
    merged_model: TwoLayerK1Net,
    dataset_bundle: DatasetBundle,
    device: torch.device,
    output_path: Path,
    title: str,
) -> None:
    plot_x = dataset_bundle.plot_x.to(device)
    with torch.no_grad():
        dist_a = model_a.distribution(plot_x)
        dist_b = model_b.distribution(plot_x)
        dist_m = merged_model.distribution(plot_x)

    x_mean = dataset_bundle.x_mean
    x_std = dataset_bundle.x_std
    y_mean = dataset_bundle.y_mean
    y_std = dataset_bundle.y_std

    x_grid = plot_x[:, 0].detach().cpu().numpy() * x_std + x_mean
    mu_true = true_mean(x_grid[:, None]).reshape(-1)
    std_true = true_std(x_grid[:, None]).reshape(-1)

    models_info = [
        ("Model A", dist_a, "#1f77b4"),
        ("Model B (matched)", dist_b, "#d62728"),
        ("Merged (t=0.5)", dist_m, "#2ca02c"),
    ]

    train_x = dataset_bundle.train.tensors[0][:250, 0].numpy() * x_std + x_mean
    train_y = dataset_bundle.train.tensors[1][:250, 0].numpy() * y_std + y_mean

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), constrained_layout=True)

    for ax, (label, dist, color) in zip(axes, models_info):
        mu = dist["mu"][:, 0].detach().cpu().numpy() * y_std + y_mean
        std = np.sqrt(dist["var"][:, 0].detach().cpu().numpy()) * y_std

        ax.scatter(train_x, train_y, s=12, alpha=0.35, color="gray", label="Train samples")
        ax.fill_between(x_grid, mu_true - std_true, mu_true + std_true, color="black", alpha=0.10, label="True ±1σ")
        ax.plot(x_grid, mu_true, color="black", linewidth=2.0, label="True mean")
        ax.fill_between(x_grid, mu - std, mu + std, color=color, alpha=0.20, label=f"{label} ±1σ")
        ax.plot(x_grid, mu, color=color, linewidth=2.0, label=label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(label)
        ax.legend(loc="upper left", fontsize=7)

    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_training_history(
    history_a: dict[str, list[float]],
    history_b: dict[str, list[float]],
    output_path: Path,
    title: str,
) -> None:
    epochs_a = np.arange(1, len(history_a["train_nll"]) + 1)
    epochs_b = np.arange(1, len(history_b["train_nll"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)

    axes[0].plot(epochs_a, history_a["train_nll"], label="Train", linewidth=1.8)
    axes[0].set_title("Model A")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("NLL")
    axes[0].legend()

    axes[1].plot(epochs_b, history_b["train_nll"], label="Train", linewidth=1.8)
    axes[1].set_title("Model B")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("NLL")
    axes[1].legend()

    fig.suptitle(title)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_single_experiment(
    config: ExperimentConfig,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(config.device)
    dataset_bundle = build_dataset_bundle(config)

    model_a, history_a = train_model(config, dataset_bundle, config.seed_a, device)
    model_b, history_b = train_model(config, dataset_bundle, config.seed_b, device)

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

    identity_stats = alignment_statistics(
        model_a=model_a,
        model_b=model_b,
        evaluation_probe_x=dataset_bundle.evaluation_probe_x,
        config=config,
        y_star=dataset_bundle.y_star,
        device=device,
    )
    matched_stats = alignment_statistics(
        model_a=model_a,
        model_b=matched_b,
        evaluation_probe_x=dataset_bundle.evaluation_probe_x,
        config=config,
        y_star=dataset_bundle.y_star,
        device=device,
    )

    model_a_metrics = evaluate_model(model_a, dataset_bundle.test, device)
    model_b_metrics = evaluate_model(model_b, dataset_bundle.test, device)
    matched_b_metrics = evaluate_model(matched_b, dataset_bundle.test, device)

    identity_certificates: dict[str, object] | None = None
    matched_certificates: dict[str, object] | None = None
    if config.parameterization == "natural":
        certificate_points = path_grid_points(config.barrier_points)
        identity_certificates = natural_barrier_certificates(
            model_a=model_a,
            model_b=model_b,
            dataset_bundle=dataset_bundle,
            alignment_stats=identity_stats,
            barrier_profile_stats=identity_profile,
            config=config,
            device=device,
            num_points=certificate_points,
        )
        matched_certificates = natural_barrier_certificates(
            model_a=model_a,
            model_b=matched_b,
            dataset_bundle=dataset_bundle,
            alignment_stats=matched_stats,
            barrier_profile_stats=matched_profile,
            config=config,
            device=device,
            num_points=certificate_points,
        )

    plot_parameter_distribution(
        model=model_a,
        output_path=output_dir / "model_a_parameters.png",
        title=f"{config.parameterization}: model A parameters",
    )
    plot_parameter_distribution(
        model=matched_b,
        output_path=output_dir / "model_b_matched_parameters.png",
        title=f"{config.parameterization}: matched model B parameters",
    )
    plot_barrier_curves(
        identity_profile=identity_profile,
        matched_profile=matched_profile,
        output_path=output_dir / "lmc_barrier.png",
        title=f"{config.parameterization}: interpolation barrier",
    )
    merged_model = interpolate_models(model_a, matched_b, 0.5, device)
    plot_predictive_fit(
        model_a=model_a,
        model_b=matched_b,
        merged_model=merged_model,
        dataset_bundle=dataset_bundle,
        device=device,
        output_path=output_dir / "predictive_fit.png",
        title=f"{config.parameterization}: predictive fit",
    )
    plot_training_history(
        history_a=history_a,
        history_b=history_b,
        output_path=output_dir / "training_history.png",
        title=f"{config.parameterization}: training curves",
    )

    np.save(output_dir / "matching_permutation.npy", permutation)

    summary = {
        "config": asdict(config),
        "device": str(device),
        "effective_parameters": {
            "shared_variance_min": config.variance_min,
            "shared_variance_max": config.variance_max,
            "meanvar_variance_floor": config.variance_min,
            "natural_lambda_min": effective_lambda_min(config.variance_max),
        },
        "test_metrics": {
            "model_a": model_a_metrics,
            "model_b": model_b_metrics,
            "model_b_matched": matched_b_metrics,
        },
        "matching": {
            "ot_w1_identity_order": matching["identity_w1"],
            "ot_w1_matched": matching["w1"],
        },
        "identity_alignment": identity_stats,
        "matched_alignment": matched_stats,
        "identity_barrier": identity_profile,
        "matched_barrier": matched_profile,
        "identity_certificates": identity_certificates,
        "matched_certificates": matched_certificates,
        "artifacts": {
            "model_a_parameters": "model_a_parameters.png",
            "model_b_matched_parameters": "model_b_matched_parameters.png",
            "lmc_barrier": "lmc_barrier.png",
            "predictive_fit": "predictive_fit.png",
            "training_history": "training_history.png",
            "matching_permutation": "matching_permutation.npy",
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthetic LMC experiment inspired by paper/k1_raw_lmc.tex.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="YAML config file. CLI arguments override YAML values.",
    )
    parser.add_argument(
        "--parameterization",
        choices=["natural", "meanvar", "both"],
        default=None,
        help="Predict either Gaussian natural parameters or mean/variance parameters.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where figures and metrics are written.",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--variance-min", type=float, default=None)
    parser.add_argument("--variance-max", type=float, default=None)
    parser.add_argument("--precision-activation", choices=["softplus", "exp"], default=None)
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


def config_from_settings(settings: dict[str, object], parameterization: Parameterization) -> ExperimentConfig:
    return ExperimentConfig(
        parameterization=parameterization,
        width=int(settings["width"]),
        variance_min=float(settings["variance_min"]),
        variance_max=float(settings["variance_max"]),
        precision_activation=str(settings["precision_activation"]),
        train_size=int(settings["train_size"]),
        test_size=int(settings["test_size"]),
        evaluation_probe_points=int(settings["evaluation_probe_points"]),
        plot_points=int(settings["plot_points"]),
        x_max=float(settings["x_max"]),
        epochs=int(settings["epochs"]),
        batch_size=int(settings["batch_size"]),
        learning_rate=float(settings["learning_rate"]),
        learning_rate_min=float(settings["learning_rate_min"]),
        weight_decay=float(settings["weight_decay"]),
        barrier_points=int(settings["barrier_points"]),
        data_seed=int(settings["data_seed"]),
        seed_a=int(settings["seed_a"]),
        seed_b=int(settings["seed_b"]),
        device=str(settings["device"]),
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    settings = resolve_settings(args)

    parameterizations: list[Parameterization]
    parameterization_setting = str(settings["parameterization"])
    output_dir = Path(settings["output_dir"])
    if parameterization_setting == "both":
        parameterizations = ["natural", "meanvar"]
    else:
        parameterizations = [parameterization_setting]

    all_summaries: dict[str, object] = {}
    for parameterization in parameterizations:
        config = config_from_settings(settings, parameterization)
        run_output_dir = output_dir / parameterization
        summary = run_single_experiment(config=config, output_dir=run_output_dir)
        all_summaries[parameterization] = summary

    if len(all_summaries) > 1:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "comparison.json").open("w", encoding="utf-8") as handle:
            json.dump(all_summaries, handle, indent=2)


if __name__ == "__main__":
    main()
