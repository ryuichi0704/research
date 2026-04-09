"""Stage 1: Train broad base PFN on heterogeneous prior mix (SCM + Tree + NN)."""

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

from src.data_loader import InterleavedPriorDataLoader


def main():
    parser = argparse.ArgumentParser(description="Train broad base PFN on mixed priors")
    parser.add_argument("--scm_data", type=str, required=True, help="Path to SCM HDF5")
    parser.add_argument("--tree_data", type=str, required=True, help="Path to Tree HDF5")
    parser.add_argument("--nn_data", type=str, required=True, help="Path to NN HDF5")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100, help="Steps per epoch per loader")
    parser.add_argument("--batchsize", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--embedding_size", type=int, default=192)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="base_broad")
    args = parser.parse_args()

    set_randomness_seed(args.seed)
    device = get_default_device()

    # Each loader contributes steps_per_loader batches; total = 3 * steps
    steps_per_loader = args.steps
    loaders = []
    for path in [args.scm_data, args.tree_data, args.nn_data]:
        loaders.append(PriorDumpDataLoader(
            filename=path,
            num_steps=steps_per_loader,
            batch_size=args.batchsize,
            device=device,
        ))

    # Total steps per epoch = 3 * steps_per_loader (round-robin)
    total_steps = 3 * steps_per_loader
    # Ensure divisible by accumulate
    total_steps = (total_steps // args.accumulate) * args.accumulate

    prior = InterleavedPriorDataLoader(loaders, num_steps=total_steps)

    model = NanoTabPFNModel(
        embedding_size=args.embedding_size,
        num_attention_heads=args.num_heads,
        mlp_hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_outputs=int(prior.max_num_classes),
    )

    criterion = nn.CrossEntropyLoss()
    callbacks = [ConsoleLoggerCallback()]

    print(f"Training broad base PFN: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  {total_steps} steps/epoch, {args.epochs} epochs, lr={args.lr}")
    print(f"  Data: SCM={args.scm_data}, Tree={args.tree_data}, NN={args.nn_data}")

    trained_model, loss = train(
        model=model,
        prior=prior,
        criterion=criterion,
        epochs=args.epochs,
        accumulate_gradients=args.accumulate,
        lr=args.lr,
        device=device,
        callbacks=callbacks,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
