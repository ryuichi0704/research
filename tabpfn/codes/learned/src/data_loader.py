"""Interleaved data loader for broad base training across multiple prior families."""

import random
from torch.utils.data import DataLoader
from tfmplayground.priors import PriorDumpDataLoader


class InterleavedPriorDataLoader(DataLoader):
    """Round-robin across multiple PriorDumpDataLoaders.

    Each iteration yields batches cycling through the given loaders,
    so every epoch sees data from all prior families.
    """

    def __init__(self, loaders: list[PriorDumpDataLoader], num_steps: int):
        self.loaders = loaders
        self.num_steps = num_steps
        # inherit from first loader for compatibility
        self.max_num_classes = loaders[0].max_num_classes
        self.problem_type = loaders[0].problem_type

    def __iter__(self):
        iters = [iter(loader) for loader in self.loaders]
        for step in range(self.num_steps):
            it = iters[step % len(iters)]
            try:
                yield next(it)
            except StopIteration:
                # re-create iterator if exhausted
                idx = step % len(iters)
                iters[idx] = iter(self.loaders[idx])
                yield next(iters[idx])

    def __len__(self):
        return self.num_steps
