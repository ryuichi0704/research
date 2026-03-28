"""
Collapse experiment for the FULL BatchEnsemble mean-field model (eq.2).

    f_α(x) = (1/M) Σ_m (a_m ⊙ r_α^(2)) s_{α,m}^(2) σ( r_{α,m}^(1) w_m^T (s_α^(1) ⊙ x) )

All parameters are trainable.
Only the boundary fast weight initialization (s_α^(1), r_α^(2)) is varied across regimes.

Experiment 1: ε=0 → collapse (Thm 3.1), ε>0 → diversity.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Data: d_x=1, d_y=1, complex smooth 1D regression with a data gap
# ---------------------------------------------------------------------------
D_X = 1
D_Y = 1
N_TEST = 500


def true_fn(x):
    """Smooth 1D target fittable by 2-layer tanh with bias."""
    return (
        torch.sin(2 * np.pi * x)
        + 0.5 * torch.sin(4 * np.pi * x)
        + 0.3 * torch.cos(6 * np.pi * x)
    )


def make_train_data(n=300, noise_std=0.1, gap=(0.35, 0.65), seed=SEED):
    """Training data with a gap — no observations in [gap[0], gap[1]]."""
    torch.manual_seed(seed)
    x = torch.rand(n * 2, 1, device=DEVICE)  # oversample, then filter
    mask = (x.squeeze() < gap[0]) | (x.squeeze() > gap[1])
    x = x[mask][:n]
    y = true_fn(x) + noise_std * torch.randn_like(x)
    return x, y


x_train, y_train = make_train_data()
x_test = torch.linspace(0, 1, N_TEST, device=DEVICE).unsqueeze(1)
y_test_true = true_fn(x_test)
N_TRAIN = x_train.shape[0]


# ---------------------------------------------------------------------------
# Full BatchEnsemble model (eq.2)
# ---------------------------------------------------------------------------
class FullBatchEnsemble(nn.Module):
    """
    f_α(x) = (1/M) Σ_m (a_m ⊙ r_α^(2)) s_{α,m}^(2) σ( r_{α,m}^(1) w_m^T (s_α^(1) ⊙ x) )

    Trainable parameters:
        a  : (M, d_y)   shared second-layer weights
        w  : (M, d_x)   shared first-layer weights
        s2 : (K, M)     neuron-wise output fast weights
        r1 : (K, M)     neuron-wise input fast weights
        s1 : (K, d_x)   boundary input fast weights
        r2 : (K, d_y)   boundary output fast weights
    """

    def __init__(self, d_x, d_y, M, K):
        super().__init__()
        self.d_x, self.d_y, self.M, self.K = d_x, d_y, M, K
        # Mean-field parameterization: a ~ N(0,1), w ~ N(0,1), b1 ~ N(0,1)
        # Output has explicit 1/M factor → O(1/√M) at init
        # Gradient is O(1/M) per neuron → need lr = M * base_lr
        self.a = nn.Parameter(torch.randn(M, d_y))
        self.w = nn.Parameter(torch.randn(M, d_x))
        self.b1 = nn.Parameter(torch.randn(M))
        self.b2 = nn.Parameter(torch.zeros(d_y))   # output bias (finite-dim)
        # Neuron-wise fast weights
        self.s2 = nn.Parameter(torch.sign(torch.randn(K, M)))
        self.r1 = nn.Parameter(torch.sign(torch.randn(K, M)))
        # Boundary fast weights
        self.s1 = nn.Parameter(torch.ones(K, d_x))
        self.r2 = nn.Parameter(torch.ones(K, d_y))

    def forward(self, x):
        # f_α(x) = b2 + (1/M) Σ_m (a_m ⊙ r2_α) s2_αm σ( r1_αm (w_m^T (s1_α ⊙ x) + b1_m) )
        sx = self.s1.unsqueeze(1) * x.unsqueeze(0)          # (K, n, d_x)
        pre = torch.einsum("kni,mi->knm", sx, self.w)       # (K, n, M)
        pre = pre + self.b1                                   # (K, n, M)
        pre = self.r1.unsqueeze(1) * pre                     # (K, n, M)
        h = torch.tanh(pre)                                  # (K, n, M)
        h = self.s2.unsqueeze(1) * h                         # (K, n, M)
        ar = self.a.unsqueeze(0) * self.r2.unsqueeze(1)      # (K, M, d_y)
        out = torch.einsum("knm,kmd->knd", h, ar) / self.M  # ← 1/M
        return out + self.b2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_disagreement(model, x):
    with torch.no_grad():
        preds = model(x)
        f_bar = preds.mean(dim=0, keepdim=True)
        return ((preds - f_bar) ** 2).mean().item()


def init_model(M, K, eps_boundary, d_x=D_X, d_y=D_Y,
               backbone_seed=SEED, neuronwise_seed=SEED+100, boundary_seed=SEED+999):
    torch.manual_seed(backbone_seed)
    model = FullBatchEnsemble(d_x, d_y, M, K)

    torch.manual_seed(neuronwise_seed)
    model.s2.data = torch.sign(torch.randn(K, M))
    model.r1.data = torch.sign(torch.randn(K, M))
    model.s2.data[model.s2.data == 0] = 1.0
    model.r1.data[model.r1.data == 0] = 1.0

    torch.manual_seed(boundary_seed)
    if eps_boundary > 0:
        model.s1.data = torch.ones(K, d_x) + eps_boundary * torch.randn(K, d_x)
        model.r2.data = torch.ones(K, d_y) + eps_boundary * torch.randn(K, d_y)
    else:
        model.s1.data = torch.ones(K, d_x)
        model.r2.data = torch.ones(K, d_y)
    return model.to(DEVICE)


def train_be(model, x, y, base_lr, n_steps, record_every=25):
    """
    Mean-field training with SGD.
    Backbone + neuron-wise params: gradient is O(1/M), so lr = M * base_lr.
    Boundary + output bias: gradient is O(1), so lr = base_lr.
    """
    M = model.M
    opt = torch.optim.SGD([
        {"params": [model.a, model.w, model.b1, model.s2, model.r1], "lr": M * base_lr},
        {"params": [model.s1, model.r2, model.b2], "lr": base_lr},
    ], momentum=0.9)
    hist = {"step": [], "loss": [], "disagreement": [], "max_disagree": 0.0}
    for step in range(n_steps):
        opt.zero_grad()
        preds = model(x)
        loss = ((preds - y.unsqueeze(0)) ** 2).sum(dim=(1, 2)).sum() / x.shape[0]
        loss.backward()
        opt.step()
        if step % record_every == 0 or step == n_steps - 1:
            d = compute_disagreement(model, x)
            hist["step"].append(step)
            hist["loss"].append(loss.item())
            hist["disagreement"].append(d)
            hist["max_disagree"] = max(hist["max_disagree"], d)
    return hist


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
M = 1024
K = 10
LR = 2e-4   # base_lr; backbone gets M * LR ≈ 0.2
N_STEPS = 15000
RECORD_EVERY = 75

REGIMES = [
    ("Symmetric (ε=0)", 0.0),
    ("ε = 0.1", 0.1),
    ("ε = 0.5", 0.5),
    ("ε = 2.0", 2.0),
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print("=" * 60)
print("Experiment 1: Full BatchEnsemble — Collapse vs Diversity")
print("=" * 60)
print(f"  d_x={D_X}, d_y={D_Y}, M={M}, K={K}")
print(f"  N_train={N_TRAIN}, SGD(mom=0.9) base_lr={LR}, backbone_lr={M*LR:.1f}, {N_STEPS} steps")
print()

results = {}
for label, eps in REGIMES:
    print(f"  {label}")
    model = init_model(M, K, eps)
    d0 = compute_disagreement(model, x_train)
    print(f"    init disagreement: {d0:.2e}")
    hist = train_be(model, x_train, y_train, LR, N_STEPS, RECORD_EVERY)

    with torch.no_grad():
        preds_test = model(x_test).squeeze(-1).cpu().numpy()    # (K, n_test)

    results[label] = {
        "eps": eps, "history": hist,
        "preds_test": preds_test,
    }
    per_head_mse = hist["loss"][-1] / K
    print(f"    final loss(sum)={hist['loss'][-1]:.4f}  per-head MSE={per_head_mse:.4f}  "
          f"disagree={hist['disagreement'][-1]:.2e}  "
          f"max_disagree={hist['max_disagree']:.2e}")
    print()

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
print("Generating figure...")

fig = plt.figure(figsize=(18, 14))
gs = GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.30)

colors = plt.cm.tab10(np.arange(K))
x_np = x_test.squeeze().cpu().numpy()
y_true_np = y_test_true.squeeze().cpu().numpy()

# --- Row 1: Per-head function plots ---
x_train_np = x_train.squeeze().cpu().numpy()
y_train_np = y_train.squeeze().cpu().numpy()

for i, (label, eps) in enumerate(REGIMES):
    ax = fig.add_subplot(gs[0, i])
    # Data gap shading
    ax.axvspan(0.35, 0.65, alpha=0.08, color="red", label="no data")
    ax.plot(x_np, y_true_np, "k--", alpha=0.4, lw=1, label="truth")
    if i == 0:
        ax.scatter(x_train_np, y_train_np, s=3, c="gray", alpha=0.3, label="train", zorder=0)
    preds = results[label]["preds_test"]
    for a in range(K):
        ax.plot(x_np, preds[a], color=colors[a], lw=1.2, alpha=0.8, label=f"head {a+1}")
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.set_xlabel("x")
    if i == 0:
        ax.set_ylabel("f(x)")
    if i == 0:
        ax.legend(fontsize=6, loc="lower left")

# --- Row 2: Head deviations f_α − f̄ ---
for i, (label, eps) in enumerate(REGIMES):
    ax = fig.add_subplot(gs[1, i])
    ax.axvspan(0.35, 0.65, alpha=0.08, color="red")
    preds = results[label]["preds_test"]
    f_bar = preds.mean(axis=0)
    for a in range(K):
        ax.plot(x_np, preds[a] - f_bar, color=colors[a], lw=1.2, alpha=0.8)
    ax.axhline(0, color="gray", ls=":", lw=0.5)
    ax.set_xlabel("x")
    if i == 0:
        ax.set_ylabel("f_α(x) − f̄(x)")
    ymax = max(np.abs(preds - f_bar).max(), 1e-4) * 1.2
    ax.set_ylim(-ymax, ymax)
    ax.set_title(f"Head deviations ({label})", fontsize=10, fontweight="bold")

# --- Row 3 left: Disagreement dynamics ---
ax = fig.add_subplot(gs[2, :2])
for label, eps in REGIMES:
    h = results[label]["history"]
    d = np.array(h["disagreement"])
    ax.semilogy(h["step"], np.maximum(d, 1e-18), lw=2, label=label)
ax.set_xlabel("Training step")
ax.set_ylabel("Head disagreement (log)")
ax.set_title("Disagreement dynamics (Thm 3.1)", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(1e-10, 1e1)

# --- Row 3 right: Per-point ensemble spread ---
ax = fig.add_subplot(gs[2, 2:])
for label, eps in REGIMES:
    preds = results[label]["preds_test"]
    spread = preds.std(axis=0)
    ax.plot(x_np, spread, lw=1.2, alpha=0.8, label=label)
ax.set_xlabel("x")
ax.set_ylabel("std across heads")
ax.set_title("Per-point ensemble spread", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.set_yscale("log")

fig.suptitle(
    "Full BatchEnsemble (eq.2): Collapse vs Diversity — Boundary Init Only Varied\n"
    f"d_x={D_X}, d_y={D_Y}, M={M}, K={K}, tanh, "
    f"neuron-wise: iid random sign, SGD(momentum=0.9) base_lr={LR}, {N_STEPS} steps\n"
    "Mean-field parameterization: 1/M normalization, backbone lr = M × base_lr",
    fontsize=12, fontweight="bold", y=1.01
)

out = Path(__file__).parent / "collapse_experiment.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)

print("\nDone.")
