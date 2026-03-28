"""
Width sweep experiment: M = 2^10, 2^11, 2^12, 2^13, 2^14.

For each width, train the full BatchEnsemble model (mean-field parameterization)
across ε regimes and measure collapse.

Models are saved to collapse_models/ for reproducibility.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import json

# Import model and helpers from the main experiment
from collapse_experiment import (
    FullBatchEnsemble, init_model, train_be, compute_disagreement,
    true_fn, make_train_data, SEED, DEVICE, D_X, D_Y,
)

# ---------------------------------------------------------------------------
# Data (same as main experiment)
# ---------------------------------------------------------------------------
x_train, y_train = make_train_data()
N_TEST = 500
x_test = torch.linspace(0, 1, N_TEST, device=DEVICE).unsqueeze(1)
y_test_true = true_fn(x_test)
N_TRAIN = x_train.shape[0]

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
WIDTHS = [2**10, 2**11, 2**12, 2**13, 2**14]
K = 10
BASE_LR = 2e-4
N_STEPS = 15000
RECORD_EVERY = 75

REGIMES = [
    ("eps=0.0", 0.0),
    ("eps=0.1", 0.1),
    ("eps=0.5", 0.5),
    ("eps=2.0", 2.0),
]

# Output directory
OUT_DIR = Path(__file__).parent / "collapse_models"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------
print("=" * 60)
print("Width sweep: M = " + ", ".join(str(m) for m in WIDTHS))
print(f"K={K}, base_lr={BASE_LR}, N_STEPS={N_STEPS}")
print("=" * 60)

# results[M][eps_label] = {history, preds_test, disagreement_test}
all_results = {}

for M in WIDTHS:
    print(f"\n{'─'*40}")
    print(f"  M = {M}  (backbone_lr = {M * BASE_LR:.2f})")
    print(f"{'─'*40}")
    all_results[M] = {}

    for label, eps in REGIMES:
        print(f"    {label} ...", end=" ", flush=True)
        model = init_model(M, K, eps)
        hist = train_be(model, x_train, y_train, BASE_LR, N_STEPS, RECORD_EVERY)

        with torch.no_grad():
            preds_test = model(x_test).squeeze(-1).cpu().numpy()   # (K, N_TEST)
            disagree_test = compute_disagreement(model, x_test)

        per_head_mse = hist["loss"][-1] / K

        all_results[M][label] = {
            "eps": eps,
            "history": hist,
            "preds_test": preds_test,
            "disagree_test": disagree_test,
            "per_head_mse": per_head_mse,
        }

        # Save model
        model_path = OUT_DIR / f"model_M{M}_{label}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "M": M, "K": K, "eps": eps,
            "base_lr": BASE_LR, "n_steps": N_STEPS,
            "history": hist,
        }, model_path)

        print(f"MSE={per_head_mse:.5f}  disagree(train)={hist['disagreement'][-1]:.2e}  "
              f"disagree(test)={disagree_test:.2e}")

# Save summary as JSON (no numpy arrays)
summary = {}
for M in WIDTHS:
    summary[str(M)] = {}
    for label, eps in REGIMES:
        r = all_results[M][label]
        summary[str(M)][label] = {
            "eps": eps,
            "per_head_mse": r["per_head_mse"],
            "final_disagree_train": r["history"]["disagreement"][-1],
            "final_disagree_test": r["disagree_test"],
            "max_disagree": r["history"]["max_disagree"],
        }
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to {OUT_DIR / 'summary.json'}")

# ---------------------------------------------------------------------------
# Figure 1: Per-width panels (one row per width, columns = ε regimes)
# ---------------------------------------------------------------------------
print("\nGenerating figures...")

x_np = x_test.squeeze().cpu().numpy()
y_true_np = y_test_true.squeeze().cpu().numpy()
colors = plt.cm.tab10(np.arange(K))

fig, axes = plt.subplots(len(WIDTHS), len(REGIMES), figsize=(18, 4 * len(WIDTHS)))

for row, M in enumerate(WIDTHS):
    for col, (label, eps) in enumerate(REGIMES):
        ax = axes[row, col]
        ax.axvspan(0.35, 0.65, alpha=0.08, color="red")
        ax.plot(x_np, y_true_np, "k--", alpha=0.4, lw=1)
        preds = all_results[M][label]["preds_test"]
        for a in range(K):
            ax.plot(x_np, preds[a], color=colors[a], lw=0.8, alpha=0.7)
        mse = all_results[M][label]["per_head_mse"]
        dt = all_results[M][label]["disagree_test"]
        ax.set_title(f"M={M}, {label}\nMSE={mse:.4f}, d(test)={dt:.1e}",
                     fontsize=9, fontweight="bold")
        if col == 0:
            ax.set_ylabel("f(x)")
        ax.set_xlabel("x")
        ax.set_ylim(-2.0, 2.0)

fig.suptitle(
    "Width Sweep: Collapse under Mean-Field Parameterization\n"
    f"K={K}, SGD(mom=0.9), base_lr={BASE_LR}, {N_STEPS} steps",
    fontsize=13, fontweight="bold"
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out1 = Path(__file__).parent / "collapse_width_sweep_functions.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
print(f"  Saved: {out1}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: Disagreement vs width (summary plot)
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Left: final test disagreement vs M
ax = axes2[0]
for label, eps in REGIMES:
    disagrees = [all_results[M][label]["disagree_test"] for M in WIDTHS]
    ax.loglog(WIDTHS, disagrees, "o-", lw=2, ms=6, label=label)
ax.set_xlabel("Width M")
ax.set_ylabel("Final test disagreement")
ax.set_title("Test disagreement vs width", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: per-head MSE vs M
ax = axes2[1]
for label, eps in REGIMES:
    mses = [all_results[M][label]["per_head_mse"] for M in WIDTHS]
    ax.semilogx(WIDTHS, mses, "o-", lw=2, ms=6, label=label)
ax.axhline(0.01, color="gray", ls=":", lw=1, label="noise floor")
ax.set_xlabel("Width M")
ax.set_ylabel("Per-head MSE")
ax.set_title("Fit quality vs width", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig2.suptitle(
    "Width Dependence Summary\n"
    f"K={K}, mean-field param, SGD(mom=0.9), base_lr={BASE_LR}, {N_STEPS} steps",
    fontsize=12, fontweight="bold"
)
fig2.tight_layout(rect=[0, 0, 1, 0.92])
out2 = Path(__file__).parent / "collapse_width_sweep_summary.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"  Saved: {out2}")
plt.close(fig2)

print("\nDone.")
