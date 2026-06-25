"""
OpenPath* — Centroid-based ID Filtering + KMeans++ Diverse Selection.

Faithful reimplementation of OpenPath's core AL strategy:
  1. Extract task model embeddings for labeled + unlabeled samples
  2. Compute class centroids from labeled embeddings
  3. Cosine distance to nearest centroid → filter top id_ratio% as ID
  4. KMeans++ on filtered candidates → select diverse query batch

Reference: OpenPath (Huang et al., 2023) — adapted for federated setting.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from data.sampler import SubsetSequentialSampler


@torch.no_grad()
def _extract_embeddings(model, dataset, indices, batch_size=64, num_workers=4):
    """Extract task model embeddings for given indices."""
    if len(indices) == 0:
        return torch.empty(0, device='cuda'), torch.empty(0, dtype=torch.long)
    
    loader = DataLoader(
        dataset, batch_size=batch_size,
        sampler=SubsetSequentialSampler(indices),
        num_workers=num_workers, pin_memory=(num_workers > 0)
    )
    
    model.eval()
    all_feats = []
    all_labels = []
    
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            _, meta = batch
            imgs = meta['image'].cuda()
            lbls = meta.get('original_label', torch.zeros(imgs.shape[0], dtype=torch.long))
        elif isinstance(batch, dict):
            imgs = batch['image'].cuda()
            lbls = batch.get('original_label', torch.zeros(imgs.shape[0], dtype=torch.long))
        else:
            imgs = batch[0].cuda()
            lbls = torch.zeros(imgs.shape[0], dtype=torch.long)
        
        outputs = model(imgs)
        embeddings = outputs[2]  # feature embeddings
        all_feats.append(embeddings)
        if hasattr(lbls, 'cuda'):
            all_labels.append(lbls)
        else:
            all_labels.append(torch.tensor(lbls))
    
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


def centroid_id_filter(model, dataset, labeled_set, unlabeled_set, 
                       id_ratio=0.6, num_classes=8, batch_size=64,
                       dataset_name='FedISIC'):
    """
    OpenPath-style centroid-based ID filtering.
    
    1. Extract embeddings from labeled + unlabeled sets
    2. Compute class centroids (mean embedding per class)
    3. For each unlabeled sample: min cosine distance to any centroid
    4. Return top id_ratio% closest as ID candidates
    
    Args:
        model: Task model (EfficientNetB0)
        dataset: Training dataset
        labeled_set: List of labeled indices
        unlabeled_set: List of unlabeled indices
        id_ratio: Fraction of unlabeled pool to keep as ID candidates
        num_classes: Number of ID classes
        batch_size: Batch size for extraction
    
    Returns:
        filtered_indices: Indices (from unlabeled_set) that pass ID filter
        filtered_local_indices: Local indices (0..N-1) into unlabeled_set
        features: Embeddings for ALL unlabeled samples (for KMeans++)
    """
    # 1. Extract labeled embeddings
    _nw = 0 if dataset_name == 'FedEMBED' else 4
    lab_feats, lab_labels = _extract_embeddings(model, dataset, labeled_set, batch_size, num_workers=_nw)
    
    # 2. Compute class centroids
    centroids = []
    for c in range(num_classes):
        mask = lab_labels == c
        if mask.sum() > 0:
            centroids.append(F.normalize(lab_feats[mask].mean(dim=0, keepdim=True), dim=-1))
        else:
            # No samples for this class — skip
            centroids.append(torch.zeros(1, lab_feats.shape[1], device=lab_feats.device))
    
    centroids = torch.cat(centroids, dim=0)  # (num_classes, D)
    
    # 3. Extract unlabeled embeddings
    unlab_feats, _ = _extract_embeddings(model, dataset, unlabeled_set, batch_size, num_workers=_nw)
    
    if len(unlab_feats) == 0:
        return [], [], torch.empty(0)
    
    # 4. Cosine distance to nearest centroid
    unlab_normed = F.normalize(unlab_feats, dim=-1)
    centroids_normed = F.normalize(centroids, dim=-1)
    
    # (N, num_classes) cosine similarity
    sim = unlab_normed @ centroids_normed.T
    min_distance = 1.0 - sim.max(dim=1).values  # min cosine distance = 1 - max cosine similarity
    
    # 5. Keep top id_ratio% closest (smallest distance)
    n_keep = max(1, int(len(unlabeled_set) * id_ratio))
    _, top_indices = min_distance.topk(n_keep, largest=False)  # smallest distances
    
    filtered_local = top_indices.cpu().numpy()
    ul_arr = np.array(unlabeled_set)
    filtered_global = ul_arr[filtered_local].tolist()
    
    return filtered_global, filtered_local, unlab_feats


def kmeans_plus_plus_select(features, n_select, seed=None):
    """
    KMeans++ initialization to select diverse samples.
    
    Standard KMeans++ algorithm: iteratively select points that are 
    maximally distant from already-selected points, with probability 
    proportional to squared distance.
    
    Args:
        features: (N, D) tensor of feature vectors
        n_select: Number of samples to select
        seed: Random seed for reproducibility
    
    Returns:
        np.array: Indices of selected samples (into features array)
    """
    if n_select >= len(features):
        return np.arange(len(features))
    
    features_np = features.cpu().numpy() if isinstance(features, torch.Tensor) else features
    features_np = features_np / (np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-10)
    
    rng = np.random.RandomState(seed)
    N = len(features_np)
    
    # First center: random
    selected = [rng.randint(N)]
    
    for _ in range(1, n_select):
        # Compute min squared distance from each point to nearest selected center
        centers = features_np[selected]  # (k, D)
        # Cosine distance
        sims = features_np @ centers.T  # (N, k)
        min_dist = (1.0 - sims.max(axis=1)) ** 2  # squared cosine distance
        
        # Zero out already selected
        min_dist[selected] = 0.0
        
        # Sample proportional to distance
        probs = min_dist / (min_dist.sum() + 1e-10)
        next_idx = rng.choice(N, p=probs)
        selected.append(next_idx)
    
    return np.array(selected)


def openpath_query(model, data_unlabeled, unlabeled_set, query_num,
                   labeled_set=None, dataset_source=None,
                   id_ratio=0.6, num_classes=8, seed=None,
                   dataset_name='FedISIC'):
    """
    Full OpenPath* query: centroid filter → KMeans++ selection.
    
    Args:
        model: Task model (EfficientNetB0)
        data_unlabeled: Unlabeled dataset (for feature extraction)
        unlabeled_set: List of unlabeled indices
        query_num: Number of samples to select
        labeled_set: List of labeled indices (for centroid computation)
        dataset_source: Full training dataset (for label access)
        id_ratio: Fraction of pool to keep as ID candidates
        num_classes: Number of ID classes
        seed: Random seed
    
    Returns:
        chosen_indices: Local indices (into unlabeled_set) of selected samples
    """
    if labeled_set is None or len(labeled_set) == 0:
        # Round 0: no centroids possible, fall back to random
        indices = list(range(len(unlabeled_set)))
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        return indices[:query_num]
    
    # 1. Centroid ID filter
    _, filtered_local, all_features = centroid_id_filter(
        model, dataset_source, labeled_set, unlabeled_set,
        id_ratio=id_ratio, num_classes=num_classes,
        dataset_name=dataset_name
    )
    
    if len(filtered_local) == 0:
        # Fallback: random from full pool
        indices = list(range(len(unlabeled_set)))
        np.random.shuffle(indices)
        return indices[:query_num]
    
    # 2. KMeans++ on filtered candidates
    filtered_features = all_features[filtered_local]
    n_select = min(query_num, len(filtered_local))
    
    km_indices = kmeans_plus_plus_select(filtered_features, n_select, seed=seed)
    
    # Map back: km_indices → filtered_local → unlabeled_set local indices
    chosen_local = filtered_local[km_indices]
    
    return chosen_local.tolist()
