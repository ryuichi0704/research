"""Model merging utilities: linear merge and Fisher-weighted merge.

Linear merge (Theorem 1):
    theta_merge = theta_0 + sum_j alpha_j * Delta_j
    where Delta_j = theta_j - theta_0, alpha_j = w_j(D_n)

Fisher-weighted merge (Proposition B.3):
    theta* = (sum_j w_j F_j)^{-1} (sum_j w_j F_j theta_j)
    using diagonal Fisher approximation.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tfmplayground.model import NanoTabPFNModel


def compute_task_vectors(
    base_state: dict[str, torch.Tensor],
    specialist_states: list[dict[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    """Compute Delta_j = theta_j - theta_0 for each specialist."""
    deltas = []
    for spec_state in specialist_states:
        delta = {}
        for key in base_state:
            delta[key] = spec_state[key] - base_state[key]
        deltas.append(delta)
    return deltas


def linear_merge(
    base_state: dict[str, torch.Tensor],
    specialist_states: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    """Support-adaptive linear merge.

    theta_merge = theta_0 + sum_j alpha_j * (theta_j - theta_0)

    Since sum(alpha_j) = 1, this equals the weighted average of specialist checkpoints.
    """
    merged = {}
    for key in base_state:
        delta_sum = sum(
            w * (spec[key] - base_state[key])
            for w, spec in zip(weights, specialist_states)
        )
        merged[key] = base_state[key] + delta_sum
    return merged


def compute_diagonal_fisher(
    model: NanoTabPFNModel,
    data_loader: DataLoader,
    num_samples: int = 200,
    device: torch.device = None,
) -> dict[str, torch.Tensor]:
    """Compute diagonal Fisher information matrix via sampling.

    F_ii = E[(d log p / d theta_i)^2]

    Uses the model's own predictions as labels (empirical Fisher).
    """
    if device is None:
        device = next(model.parameters()).device

    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.eval()
    count = 0

    for batch in data_loader:
        if count >= num_samples:
            break

        x = batch["x"].to(device)
        y = batch["y"].to(device)
        sep = batch["single_eval_pos"]

        model.zero_grad()
        data = (x, y[:, :sep])
        output = model(data, single_eval_pos=sep)

        # Sample from model distribution for empirical Fisher
        targets = y[:, sep:]
        targets = targets.reshape(-1).long()
        output_flat = output.view(-1, output.shape[-1])
        loss = F.cross_entropy(output_flat, targets)
        loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data ** 2

        count += 1

    for n in fisher:
        fisher[n] /= max(count, 1)

    return fisher


def fisher_merge(
    specialist_states: list[dict[str, torch.Tensor]],
    fisher_matrices: list[dict[str, torch.Tensor]],
    weights: list[float],
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Fisher-weighted merge.

    theta* = (sum_j w_j F_j)^{-1} (sum_j w_j F_j theta_j)
    """
    keys = list(specialist_states[0].keys())
    merged = {}

    for key in keys:
        numerator = sum(
            w * f[key] * s[key]
            for w, f, s in zip(weights, fisher_matrices, specialist_states)
        )
        denominator = sum(
            w * f[key]
            for w, f in zip(weights, fisher_matrices)
        )
        merged[key] = numerator / (denominator + eps)

    return merged


def checkpoint_distance(state1: dict, state2: dict) -> float:
    """L2 distance between two state dicts."""
    dist_sq = 0.0
    for key in state1:
        diff = state1[key].float() - state2[key].float()
        dist_sq += (diff ** 2).sum().item()
    return dist_sq ** 0.5
