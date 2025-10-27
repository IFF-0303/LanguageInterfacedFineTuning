"""Sampling utilities for distributed instruction fine-tuning workflows."""

from __future__ import annotations

import math
from typing import Iterator

import torch
from torch.utils.data import Sampler


class DistributedWeightedRandomSampler(Sampler[int]):
    """Weighted random sampling that stays in sync across distributed workers.

    PyTorch's :class:`~torch.utils.data.WeightedRandomSampler` cannot be used as-is
    with ``DistributedDataParallel`` because each worker would draw an independent
    sample set, leading to duplicate work and desynchronised epoch lengths.  This
    sampler instead deterministically materialises a shared pool of weighted
    samples and then assigns an equal strided partition to each process.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        *,
        num_replicas: int,
        rank: int,
        replacement: bool = True,
        seed: int = 0,
    ) -> None:
        if weights.ndim != 1:
            raise ValueError("weights must be a 1D tensor of per-sample weights")
        if num_replicas <= 0:
            raise ValueError("num_replicas must be a positive integer")
        if not (0 <= rank < num_replicas):
            raise ValueError("rank must be in the range [0, num_replicas)")

        self.weights = weights.to(dtype=torch.float32)
        if torch.any(self.weights < 0):
            raise ValueError("weights must be non-negative")
        if torch.sum(self.weights) == 0:
            raise ValueError("at least one weight must be positive")

        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0

        # Ensure each worker receives an identical number of samples.
        self.total_size = int(math.ceil(len(self.weights) / num_replicas)) * num_replicas
        self.num_samples = self.total_size // num_replicas

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        # Materialise a shared set of weighted samples and take a strided slice
        # corresponding to the current worker.
        sample_indices = torch.multinomial(
            self.weights,
            self.total_size,
            self.replacement,
            generator=generator,
        )
        indices = sample_indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices.tolist())

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Alter the sampling seed so each epoch draws a fresh partition."""

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch
