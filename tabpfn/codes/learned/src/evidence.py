"""Support evidence computation for specialist weighting.

Implements prequential log-loss scoring:
    S_j(D_n) = log(rho_j) - sum_{i=1}^{n} [-log q_{j,i-1}(z_i | d_{i-1})]

and the resulting posterior:
    w_j(D_n) = softmax(S_j(D_n))
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import logsumexp

from tfmplayground.model import NanoTabPFNModel


def compute_prequential_log_loss(
    model: NanoTabPFNModel,
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device = None,
) -> float:
    """Compute cumulative prequential log loss of a PFN on support data.

    For i = 1, ..., n:
        context = (X[:i-1], y[:i-1])
        query = X[i]
        loss_i = -log P(y_i | context, X_i)

    Args:
        model: A NanoTabPFNModel in eval mode.
        X: (n, d) feature tensor for support set.
        y: (n,) integer label tensor.
        device: Device to run on.

    Returns:
        Cumulative prequential log loss (float).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    n = X.shape[0]
    cum_loss = 0.0

    with torch.no_grad():
        for i in range(n):
            # Context: first i points; query: point i
            X_ctx = X[:i].unsqueeze(0).to(device)   # (1, i, d)
            y_ctx = y[:i].unsqueeze(0).to(device)    # (1, i)
            X_q = X[i : i + 1].unsqueeze(0).to(device)  # (1, 1, d)

            if i == 0:
                # No context: concatenate query only, eval_pos=0
                logits = model((X_q, y_ctx[:, :0]), single_eval_pos=0)
            else:
                logits = model(X_ctx, y_ctx, X_q)

            # logits: (1, 1, num_classes)
            probs = F.softmax(logits[0, 0], dim=-1)
            true_label = int(y[i].item())
            if true_label < probs.shape[0]:
                p = probs[true_label].item()
            else:
                p = 1e-10  # unseen class fallback
            cum_loss += -np.log(max(p, 1e-10))

    return cum_loss


def compute_specialist_weights(
    specialists: list[NanoTabPFNModel],
    X: torch.Tensor,
    y: torch.Tensor,
    prior_weights: list[float] | None = None,
    device: torch.device = None,
) -> np.ndarray:
    """Compute posterior weights over specialists given support data.

    w_j(D_n) = softmax(log(rho_j) - prequential_loss_j)

    Args:
        specialists: List of specialist models.
        X: (n, d) support features.
        y: (n,) support labels.
        prior_weights: Prior rho_j for each specialist. Uniform if None.
        device: Device.

    Returns:
        Array of posterior weights summing to 1.
    """
    M = len(specialists)
    if prior_weights is None:
        prior_weights = [1.0 / M] * M

    scores = np.zeros(M)
    for j, (model, rho_j) in enumerate(zip(specialists, prior_weights)):
        pll = compute_prequential_log_loss(model, X, y, device=device)
        scores[j] = np.log(max(rho_j, 1e-30)) - pll

    # Numerically stable softmax
    weights = np.exp(scores - logsumexp(scores))
    return weights


def compute_evidence_curve(
    specialists: list[NanoTabPFNModel],
    X: torch.Tensor,
    y: torch.Tensor,
    prior_weights: list[float] | None = None,
    device: torch.device = None,
    step: int = 1,
) -> list[np.ndarray]:
    """Compute specialist weights at each prefix length.

    Returns a list of weight arrays, one per prefix length n = step, 2*step, ..., N.
    Useful for plotting evidence concentration curves.
    """
    N = X.shape[0]
    curves = []
    for n in range(step, N + 1, step):
        w = compute_specialist_weights(
            specialists, X[:n], y[:n], prior_weights, device
        )
        curves.append(w)
    return curves
