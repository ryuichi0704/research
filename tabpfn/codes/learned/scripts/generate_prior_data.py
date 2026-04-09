"""Generate prior data with parallel support via TabICL's n_jobs.

Usage:
    # Tree (parallel, 16 workers):
    python scripts/generate_prior_data.py --prior_type tree_scm --n_jobs 16 --num_batches 100 --batch_size 512 --save_path data/tree_large.h5

    # SCM:
    python scripts/generate_prior_data.py --prior_type mlp_scm --n_jobs 16 --num_batches 100 --batch_size 512 --save_path data/scm_large.h5

    # NN (tabpfn, no n_jobs):
    python scripts/generate_prior_data.py --prior_type nn --num_batches 100 --batch_size 8 --save_path data/nn_large.h5
"""

import argparse
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import h5py
from tqdm import tqdm


def generate_tabicl(args):
    """Generate data using TabICL's PriorDataset with n_jobs parallelism."""
    from tabicl.prior.dataset import PriorDataset

    ds = PriorDataset(
        batch_size=args.batch_size,
        batch_size_per_gp=args.batch_size,
        min_features=args.min_features,
        max_features=args.max_features,
        max_classes=args.max_classes,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        prior_type=args.prior_type,
        n_jobs=args.n_jobs,
        num_threads_per_generate=1,
        device="cpu",
    )

    with h5py.File(args.save_path, "w") as f:
        f.create_dataset("problem_type", data="classification", dtype=h5py.string_dtype())
        f.create_dataset("max_num_classes", data=np.array((args.max_classes,)), chunks=(1,))
        f.create_dataset("original_batch_size", data=np.array((args.batch_size,)), chunks=(1,))

        X_ds = f.create_dataset(
            "X", shape=(0, args.max_seq_len, args.max_features),
            maxshape=(None, args.max_seq_len, args.max_features),
            chunks=(min(args.batch_size, 64), args.max_seq_len, args.max_features),
            compression="lzf",
        )
        y_ds = f.create_dataset(
            "y", shape=(0, args.max_seq_len),
            maxshape=(None, args.max_seq_len),
            chunks=(min(args.batch_size, 64), args.max_seq_len),
        )
        nf_ds = f.create_dataset(
            "num_features", shape=(0,), maxshape=(None,), dtype="i4",
            chunks=(min(args.batch_size, 64),),
        )
        nd_ds = f.create_dataset(
            "num_datapoints", shape=(0,), maxshape=(None,), dtype="i4",
            chunks=(min(args.batch_size, 64),),
        )
        sep_ds = f.create_dataset(
            "single_eval_pos", shape=(0,), maxshape=(None,), dtype="i4",
            chunks=(min(args.batch_size, 64),),
        )

        total_datasets = 0
        for batch_idx in tqdm(range(args.num_batches), desc=f"Generating {args.prior_type}"):
            X, y, d, seq_lens, train_sizes = next(ds)
            # X: (B, T, H) or nested tensor
            # Convert to numpy, pad to max dims
            X_np = X.cpu().numpy()
            y_np = y.cpu().numpy()
            d_np = d.cpu().numpy()
            seq_np = seq_lens.cpu().numpy()
            tr_np = train_sizes.cpu().numpy()

            B, T, H = X_np.shape
            # Pad to max_seq_len x max_features
            X_padded = np.zeros((B, args.max_seq_len, args.max_features), dtype=np.float32)
            X_padded[:, :T, :H] = X_np
            y_padded = np.zeros((B, args.max_seq_len), dtype=np.float32)
            y_padded[:, :T] = y_np

            n = X_ds.shape[0]
            X_ds.resize(n + B, axis=0)
            X_ds[n:] = X_padded
            y_ds.resize(n + B, axis=0)
            y_ds[n:] = y_padded
            nf_ds.resize(n + B, axis=0)
            nf_ds[n:] = d_np
            nd_ds.resize(n + B, axis=0)
            nd_ds[n:] = seq_np
            sep_ds.resize(n + B, axis=0)
            sep_ds[n:] = tr_np

            total_datasets += B

    print(f"Done: {total_datasets} datasets saved to {args.save_path}")


def generate_tabpfn(args):
    """Generate data using tabpfn-v1-prior (no n_jobs, but fast)."""
    from tfmplayground.priors.dataloader import TabPFNPriorDataLoader  # noqa: F811
    from tfmplayground.priors.utils import build_tabpfn_prior, dump_prior_to_h5

    tabpfn_config = build_tabpfn_prior("mlp", args.max_classes)
    loader = TabPFNPriorDataLoader(
        prior_type="mlp",
        num_steps=args.num_batches,
        batch_size=args.batch_size,
        num_datapoints_max=args.max_seq_len,
        num_features=args.max_features,
        device=torch.device("cpu"),
        **tabpfn_config,
    )
    dump_prior_to_h5(loader, args.max_classes, args.batch_size, args.save_path,
                     "classification", args.max_seq_len, args.max_features)
    print(f"Done: {args.num_batches * args.batch_size} datasets saved to {args.save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate prior data with parallel support")
    parser.add_argument("--prior_type", required=True,
                        choices=["mlp_scm", "tree_scm", "nn"],
                        help="mlp_scm/tree_scm use TabICL with n_jobs; nn uses tabpfn-v1-prior")
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_jobs", type=int, default=16, help="Parallel workers (TabICL only)")
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--min_seq_len", type=int, default=None)
    parser.add_argument("--min_features", type=int, default=1)
    parser.add_argument("--max_features", type=int, default=20)
    parser.add_argument("--max_classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    start = time.time()

    if args.prior_type == "nn":
        generate_tabpfn(args)
    else:
        generate_tabicl(args)

    elapsed = time.time() - start
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
