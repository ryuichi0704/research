"""Stage 2: Fine-tune specialist PFNs from base checkpoint on individual prior families."""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch import nn

from tfmplayground.callbacks import ConsoleLoggerCallback
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors import PriorDumpDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed


class SpecialistCallback(ConsoleLoggerCallback):
    """Logs training progress and tracks distance from base checkpoint."""

    def __init__(self, base_state_dict: dict):
        self.base_state_dict = {k: v.cpu() for k, v in base_state_dict.items()}

    def on_epoch_end(self, epoch, epoch_time, loss, model, **kwargs):
        # Compute L2 distance from base
        delta_norm_sq = 0.0
        for name, param in model.named_parameters():
            if name in self.base_state_dict:
                delta = param.cpu().detach() - self.base_state_dict[name]
                delta_norm_sq += (delta ** 2).sum().item()
        delta_norm = delta_norm_sq ** 0.5
        print(
            f"epoch {epoch:5d} | time {epoch_time:5.2f}s | loss {loss:5.4f} | "
            f"||theta - theta_0|| = {delta_norm:.4f}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune specialist PFN from base checkpoint")
    parser.add_argument("--base_checkpoint", type=str, required=True, help="Path to base checkpoint")
    parser.add_argument("--prior_data", type=str, required=True, help="Path to family-specific HDF5")
    parser.add_argument("--family_name", type=str, required=True, choices=["scm", "tree", "nn"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5, help="Low LR to stay near base")
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_randomness_seed(args.seed)
    device = get_default_device()

    # Load base checkpoint
    ckpt = torch.load(args.base_checkpoint, map_location=device)
    arch = ckpt["architecture"]

    model = NanoTabPFNModel(
        embedding_size=arch["embedding_size"],
        num_attention_heads=arch["num_attention_heads"],
        mlp_hidden_size=arch["mlp_hidden_size"],
        num_layers=arch["num_layers"],
        num_outputs=arch["num_outputs"],
    )
    model.load_state_dict(ckpt["model"])
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Ensure steps divisible by accumulate
    steps = (args.steps // args.accumulate) * args.accumulate

    prior = PriorDumpDataLoader(
        filename=args.prior_data,
        num_steps=steps,
        batch_size=args.batchsize,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()
    run_name = f"specialist_{args.family_name}"

    callbacks = [SpecialistCallback(base_state)]

    print(f"Fine-tuning {args.family_name} specialist from base checkpoint")
    print(f"  {steps} steps/epoch, {args.epochs} epochs, lr={args.lr}")

    trained_model, loss = train(
        model=model,
        prior=prior,
        criterion=criterion,
        epochs=args.epochs,
        accumulate_gradients=args.accumulate,
        lr=args.lr,
        device=device,
        callbacks=callbacks,
        run_name=run_name,
    )


if __name__ == "__main__":
    main()
