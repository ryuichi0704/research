#!/usr/bin/env python3
"""Appendix corroboration experiment using TFM-Playground's NanoTabPFN model."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "learned_tfm"
DEFAULT_MPLCONFIGDIR = SCRIPT_DIR / "results" / ".mplconfig"
DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")


TFM_INSTALL_MESSAGE = """
TFM-Playground is not installed.

Install it from the official repository:
  https://github.com/automl/TFM-Playground/

Typical workflow:
  git clone https://github.com/automl/TFM-Playground.git
  cd TFM-Playground
  pip install -e .

Then rerun:
  python codes/run_learned_tfm.py
""".strip()

try:
    from tfmplayground.model import NanoTabPFNModel
except Exception as exc:  # pragma: no cover - optional dependency guard
    raise SystemExit(f"{TFM_INSTALL_MESSAGE}\n\nOriginal import error: {exc}") from exc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except Exception as exc:  # pragma: no cover - import guard
    raise SystemExit(f"PyTorch is required for this script.\n{exc}") from exc

try:  # pragma: no cover - optional helper
    from tfmplayground.utils import get_default_device  # type: ignore
except Exception:  # pragma: no cover - optional helper
    get_default_device = None


@dataclass(frozen=True)
class MixtureSpec:
    name: str
    components: tuple[tuple[float, float], ...]
    weights: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--steps-per-epoch", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-len", type=int, default=50)
    parser.add_argument("--num-eval-tasks", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2402)
    return parser.parse_args()


def configure_plotting() -> None:
    sns.set_theme(style="whitegrid", context="talk")


def get_device() -> torch.device:
    if get_default_device is not None:
        device = get_default_device()
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    weight_array = np.asarray(weights, dtype=float)
    weight_array = weight_array / np.sum(weight_array)
    return weight_array


def sample_task_batch(
    rng: np.random.Generator,
    mixture: MixtureSpec,
    batch_size: int,
    context_len: int,
    query_len: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a batch of synthetic Bernoulli tasks with nuisance features."""

    weights = normalize_weights(mixture.weights)
    component_indices = rng.choice(len(mixture.components), size=batch_size, p=weights)
    a = np.array([mixture.components[idx][0] for idx in component_indices], dtype=float)
    b = np.array([mixture.components[idx][1] for idx in component_indices], dtype=float)
    theta = rng.beta(a, b).astype(np.float32)

    total_len = context_len + query_len
    x1 = np.ones((batch_size, total_len, 1), dtype=np.float32)
    x2 = rng.normal(size=(batch_size, total_len, 1)).astype(np.float32)
    features = np.concatenate([x1, x2], axis=2)
    labels = rng.binomial(1, theta[:, None], size=(batch_size, total_len)).astype(np.int64)

    x_train = features[:, :context_len, :]
    y_train = labels[:, :context_len]
    x_test = features[:, context_len:, :]
    y_test = labels[:, context_len:]
    s_train = np.sum(y_train, axis=1)
    return x_train, y_train, x_test, y_test, s_train


def build_model(args: argparse.Namespace, num_outputs: int = 2) -> NanoTabPFNModel:
    return NanoTabPFNModel(
        num_attention_heads=args.heads,
        embedding_size=args.embedding_size,
        mlp_hidden_size=args.hidden_size,
        num_layers=args.layers,
        num_outputs=num_outputs,
    )


def run_model(
    model: NanoTabPFNModel,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    x_all = np.concatenate([x_train, x_test], axis=1)
    x_tensor = torch.tensor(x_all, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    return model((x_tensor, y_tensor), single_eval_pos=x_train.shape[1])


def train_checkpoint(
    model: NanoTabPFNModel,
    mixture: MixtureSpec,
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> pd.DataFrame:
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    rows: list[dict[str, float | int | str]] = []
    total_steps = 0
    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        for _ in range(args.steps_per_epoch):
            x_train, y_train, x_test, y_test, _ = sample_task_batch(
                rng=rng,
                mixture=mixture,
                batch_size=args.batch_size,
                context_len=args.context_len,
            )
            logits = run_model(model=model, x_train=x_train, y_train=y_train, x_test=x_test, device=device)
            logits = logits.squeeze(1)
            targets = torch.tensor(y_test[:, 0], dtype=torch.long, device=device)
            loss = criterion(logits, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu().item()))
            total_steps += 1

        rows.append(
            {
                "checkpoint": mixture.name,
                "epoch": epoch,
                "step": total_steps,
                "mean_loss": float(np.mean(epoch_losses)),
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def evaluate_checkpoint(
    model: NanoTabPFNModel,
    mixture_name: str,
    deployment_prior: tuple[float, float],
    prefix_lengths: Sequence[int],
    num_eval_tasks: int,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    deployment_mixture = MixtureSpec(
        name="deployment",
        components=(deployment_prior,),
        weights=(1.0,),
    )
    model.eval()
    logloss_rows: list[dict[str, float | int | str]] = []
    profile_rows: list[dict[str, float | int | str]] = []

    for n in prefix_lengths:
        x_train, y_train, x_test, y_test, s_train = sample_task_batch(
            rng=rng,
            mixture=deployment_mixture,
            batch_size=num_eval_tasks,
            context_len=int(n),
        )
        logits = run_model(model=model, x_train=x_train, y_train=y_train, x_test=x_test, device=device)
        probs = torch.softmax(logits.squeeze(1), dim=-1)[:, 1].detach().cpu().numpy()
        probs = np.clip(probs, 1e-8, 1.0 - 1e-8)
        y_true = y_test[:, 0].astype(np.int64)
        mean_log_loss = float(np.mean(-(y_true * np.log(probs) + (1 - y_true) * np.log(1.0 - probs))))
        logloss_rows.append(
            {
                "checkpoint": mixture_name,
                "n": int(n),
                "mean_log_loss": mean_log_loss,
            }
        )

        for s in (0, 1, 2):
            mask = s_train == s
            if not np.any(mask):
                continue
            profile_rows.append(
                {
                    "checkpoint": mixture_name,
                    "n": int(n),
                    "s": s,
                    "count": int(np.sum(mask)),
                    "mean_predicted_probability": float(np.mean(probs[mask])),
                }
            )

    return pd.DataFrame(logloss_rows), pd.DataFrame(profile_rows)


def save_checkpoint(
    model: NanoTabPFNModel,
    args: argparse.Namespace,
    mixture: MixtureSpec,
    output_dir: Path,
) -> Path:
    checkpoint_path = output_dir / f"{mixture.name}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "architecture": {
                "num_attention_heads": args.heads,
                "embedding_size": args.embedding_size,
                "mlp_hidden_size": args.hidden_size,
                "num_layers": args.layers,
                "num_outputs": 2,
            },
            "mixture_name": mixture.name,
            "components": mixture.components,
            "weights": mixture.weights,
        },
        checkpoint_path,
    )
    return checkpoint_path


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_results(logloss_df: pd.DataFrame, profile_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.5))

    sns.lineplot(
        data=logloss_df,
        x="n",
        y="mean_log_loss",
        hue="checkpoint",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_xscale("log", base=2)
    axes[0].set_title("Learned corroboration: mean test log loss")
    axes[0].set_xlabel("Context length n")
    axes[0].set_ylabel("Mean test log loss")

    profile_df = profile_df.copy()
    profile_df["series"] = profile_df.apply(
        lambda row: f"{row['checkpoint']} | s={int(row['s'])}",
        axis=1,
    )
    sns.lineplot(
        data=profile_df,
        x="n",
        y="mean_predicted_probability",
        hue="series",
        marker="o",
        ax=axes[1],
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_title("Learned corroboration: small-count predictions")
    axes[1].set_xlabel("Context length n")
    axes[1].set_ylabel(r"Mean predicted probability $\hat q(1\mid S_n=s)$")
    axes[1].legend(title="Checkpoint | count", fontsize=8, title_fontsize=10)

    save_figure(fig, output_dir, "figure_appendix_learned_boundary_corrob")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device()

    collapsed_broad = MixtureSpec(
        name="collapsed_broad",
        components=((0.5, 3.0), (0.5, 6.0), (0.2, 3.0), (0.2, 6.0)),
        weights=(1.0, 1.0, 1.0, 1.0),
    )
    boundary_bucket_05 = MixtureSpec(
        name="boundary_bucket_05",
        components=((0.5, 3.0), (0.5, 6.0)),
        weights=(1.0, 1.0),
    )
    mixtures = [collapsed_broad, boundary_bucket_05]
    deployment_prior = (0.5, 4.0)
    prefix_lengths = [4, 8, 16, 32, 50]

    training_logs: list[pd.DataFrame] = []
    evaluation_logs: list[pd.DataFrame] = []
    evaluation_profiles: list[pd.DataFrame] = []

    for idx, mixture in enumerate(mixtures):
        train_rng = np.random.default_rng(args.seed + 101 * (idx + 1))
        eval_rng = np.random.default_rng(args.seed + 1001 * (idx + 1))

        model = build_model(args=args)
        train_log = train_checkpoint(
            model=model,
            mixture=mixture,
            args=args,
            device=device,
            rng=train_rng,
        )
        training_logs.append(train_log)
        save_checkpoint(model=model, args=args, mixture=mixture, output_dir=output_dir)

        logloss_df, profile_df = evaluate_checkpoint(
            model=model,
            mixture_name=mixture.name,
            deployment_prior=deployment_prior,
            prefix_lengths=prefix_lengths,
            num_eval_tasks=args.num_eval_tasks,
            device=device,
            rng=eval_rng,
        )
        evaluation_logs.append(logloss_df)
        evaluation_profiles.append(profile_df)

    training_log_df = pd.concat(training_logs, ignore_index=True)
    evaluation_log_df = pd.concat(evaluation_logs, ignore_index=True)
    evaluation_profile_df = pd.concat(evaluation_profiles, ignore_index=True)

    training_log_df.to_csv(output_dir / "learned_training_log.csv", index=False)
    evaluation_log_df.to_csv(output_dir / "learned_boundary_logloss.csv", index=False)
    evaluation_profile_df.to_csv(output_dir / "learned_boundary_profiles.csv", index=False)
    plot_results(evaluation_log_df, evaluation_profile_df, output_dir=output_dir)

    print(f"Learned TFM corroboration written to {output_dir}")


if __name__ == "__main__":
    main()
