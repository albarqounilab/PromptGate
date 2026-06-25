"""Helper utilities for data loading and class-count analysis."""

import random

import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler


def get_extended_class_counts(dataset, indices, num_id_classes):
    """Count label occurrences for a subset of a dataset.

    Iterates over *indices* and tallies how many samples have each label.
    Also computes the *purity* — the fraction of indices that are ID
    (i.e. have label in ``[0, num_id_classes)``).

    Args:
        dataset: A PyTorch dataset whose ``__getitem__`` returns
            ``(index, meta_dict)``. Alternatively, datasets with a
            ``labels`` attribute are accessed directly for speed.
        indices: Iterable of integer indices into *dataset*.
        num_id_classes: Number of in-distribution classes. Labels
            ``>= num_id_classes`` are treated as OOD.

    Returns:
        Tuple ``(counts, purity)``:
        - ``counts``: ``dict[label_int, int]`` — frequency of each label.
        - ``purity``: ``float`` — fraction of *indices* that are ID.
    """
    counts = {}
    id_count = 0
    total = len(indices)

    if total == 0:
        return counts, 0.0

    has_direct_labels = hasattr(dataset, 'labels')

    for idx in indices:
        if has_direct_labels:
            lbl = dataset.labels[idx]
        else:
            _, meta = dataset[idx]
            lbl = meta.get('original_label', -1)
            if hasattr(lbl, 'item'):
                lbl = lbl.item()

        counts[lbl] = counts.get(lbl, 0) + 1

        if 0 <= lbl < num_id_classes:
            id_count += 1

    purity = id_count / total
    return counts, purity


def make_worker_init_fn(args):
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
        np.random.seed(args.seed + worker_id)
    return worker_init_fn


def make_subset_sampler(args, indices):
    """Return a SubsetRandomSampler, seeded when VLM filtering is active.

    Seeding ensures that even random sampling is reproducible across runs
    when the VLM filter is on (important for fair ablations).

    Args:
        args: Experiment args namespace. Uses ``args.seed`` and ``args.vlm_filter``.
        indices: List of dataset indices to sample from.

    Returns:
        ``torch.utils.data.sampler.SubsetRandomSampler``.
    """
    if args.vlm_filter:
        return SubsetRandomSampler(indices, generator=torch.Generator().manual_seed(args.seed))
    return SubsetRandomSampler(indices)


def loader_worker_init(args, num_workers):
    """Return a DataLoader worker-init function when determinism is required.

    Workers are seeded with ``args.seed + worker_id`` when VLM filtering is
    active or for zero-worker FedEMBED loaders, ensuring reproducible
    data ordering across runs.

    Args:
        args: Experiment args namespace.
        num_workers: Number of DataLoader workers.

    Returns:
        A ``worker_init_fn`` callable, or ``None`` if no seeding is needed.
    """
    if args.vlm_filter:
        return make_worker_init_fn(args)
    if args.dataset == 'FedEMBED' and num_workers == 0:
        return make_worker_init_fn(args)
    return None


def remaining_unlabeled(args, pool, used):
    """Return indices in *pool* that are not in *used*.

    When VLM filtering is active the returned list is sorted to keep
    index order deterministic (important for mask alignment).

    Args:
        args: Experiment args namespace. Uses ``args.vlm_filter``.
        pool: Set of all available dataset indices.
        used: Set of already-queried or discarded indices.

    Returns:
        List of remaining unlabeled indices.
    """
    out = list(pool - used)
    return sorted(out) if args.vlm_filter else out
