import torch
import numpy as np
import pandas as pd


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for reproducibility across numpy, random, and torch."""
    import random as _random
    import torch.backends.cudnn as cudnn

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True


def cnt_sample_num(labeled_loader, num_classes):
    """Count per-class sample totals in a labeled DataLoader.

    Iterates over the loader once and tallies how many samples belong to
    each class index. Used to derive class-frequency weights for loss
    balancing during local training.

    Args:
        labeled_loader: A DataLoader whose items are ``(index, meta_dict)``
            tuples, where ``meta_dict['label']`` is the class label.
        num_classes: Number of ID classes (OOD labels are ignored).

    Returns:
        torch.Tensor of shape ``(num_classes,)`` on CUDA with per-class counts.
    """
    num = torch.zeros(num_classes).cuda()
    for _, (_, data) in enumerate(labeled_loader):
        label = data['label']
        num += torch.tensor([(label == i).sum() for i in range(num_classes)]).cuda()

    return num


def get_class_counts(dataset, indices, num_classes):
    """
    Returns an array of shape [num_classes] containing the count of each ground-truth label.
    """
    counts = np.zeros(num_classes, dtype=int)

    # Fast path for FedISIC (pandas based)
    if hasattr(dataset, "data_list") and isinstance(dataset.data_list, pd.DataFrame):
        subset_df = dataset.data_list.iloc[indices]
        # label is at -4 based on your dataset structure
        labels = subset_df.iloc[:, -4].values.astype(int)
    else:
        # Fallback: Loop (slower but works for any dataset)
        labels = []
        for idx in indices:
            _, sample = dataset[idx]
            # Handle both tensor and int labels
            lbl = sample.get("original_label", sample["label"])
            if isinstance(lbl, torch.Tensor):
                lbl = lbl.item()
            labels.append(lbl)

    # Count frequencies
    unique, u_counts = np.unique(labels, return_counts=True)
    for cls, count in zip(unique, u_counts):
        if cls < num_classes:
            counts[cls] = count

    return counts