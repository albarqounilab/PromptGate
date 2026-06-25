"""Active-learning metric helpers (QP, AQR, purity).

This module provides small utilities to compute and update:

* **Query Precision (QP)** – fraction of queried samples that are ID.
* **Average Query Recall (AQR)** – fraction of all ID samples discovered so
  far by the AL strategy.
* Cumulative purity of the labeled pool.
"""

import numpy as np

# Processes a newly selected batch of data to see if the Active Learning strategy "guessed" correctly.
def update_cumulative_al_metrics(
    c_id,
    new_labeled_indices,
    local_data,
    args,
    local_sets,
    total_id_samples_per_client,
    next_global_total_labeled_id,
    next_global_total_dataset_id,
    next_qps,
):
    """Update per-round metrics for a client's newly queried batch.

    Given the indices queried in the current round, this function classifies
    them into ID vs OOD, computes the batch‑level Query Precision (QP), and
    accumulates global statistics needed to derive Average Query Recall (AQR)
    for the *next* round.

    Args:
        c_id: Client index.
        new_labeled_indices: List of dataset indices that were just queried.
        local_data: Per-client data dict; ``local_data['train'][c_id]`` is used
            to inspect labels / OOD flags.
        args: Global args namespace (contains ``num_classes``).
        local_sets: Dict containing current ``'labeled'`` / ``'discarded'``
            sets per client.
        total_id_samples_per_client: Array with the total number of ID samples
            per client in the full dataset.
        next_global_total_labeled_id: Running total of ID samples discovered
            across all clients (to be updated in place).
        next_global_total_dataset_id: Running total of ID samples present in
            all clients (to be updated in place).
        next_qps: List collecting per‑client QP values for this round.

    Returns:
        Tuple ``(valid_id_indices, ood_indices, batch_purity,
        next_global_total_labeled_id, next_global_total_dataset_id, next_qps)``.
    """
    valid_id_indices = []
    ood_indices = []
    _ds = local_data['train'][c_id]
    _has_labels = hasattr(_ds, 'labels')
    
    # Iterates over the newly labeled indices to determine if they are OOD or ID
    for idx in new_labeled_indices:
        if _has_labels:
            is_ood = 1 if _ds.labels[idx] >= args.num_classes else 0
        else:
            _, meta = _ds[idx]
            is_ood = meta.get('is_ood', 0)
        
        if is_ood == 1:
            ood_indices.append(idx)
        else:
            valid_id_indices.append(idx)

    # QP (Query Precision) – fraction of queried samples that are ID.
    batch_total = len(new_labeled_indices)
    batch_id = len(valid_id_indices)
    qp_val = batch_id / batch_total if batch_total > 0 else 0.0
    
    # AQR accumulation for the *next* round's state: how many ID samples have
    # been discovered so far out of all ID samples in the client's dataset.
    total_id_found_so_far = len(local_sets['labeled'][c_id]) + batch_id 
    total_id_in_dataset = int(total_id_samples_per_client[c_id])

    next_global_total_labeled_id += total_id_found_so_far
    next_global_total_dataset_id += total_id_in_dataset
    next_qps.append(qp_val)

    # Batch Purity
    batch_purity = batch_id / batch_total if batch_total > 0 else 0.0
    
    return (
        valid_id_indices,
        ood_indices,
        batch_purity,
        next_global_total_labeled_id,
        next_global_total_dataset_id,
        next_qps,
    )

# Computes the instantaneous cumulative totals for accurate logging (used to fix the AQR logging lag)
def get_current_round_global_metrics(local_sets, client_num):
    """Compute instantaneous global labeled counts and purity.

    This utility is used for logging without lag, by deriving the current
    cumulative purity directly from ``local_sets`` instead of relying on
    delayed accumulators.

    Args:
        local_sets: Dict containing per-client ``'labeled'`` and
            ``'discarded'`` index lists.
        client_num: Number of federated clients.

    Returns:
        Tuple ``(total_valid_id_now, total_selected_now, cumulative_purity)``.
    """

    total_valid_id_now = 0
    total_selected_now = 0
    for c_id in range(client_num):
        n_lab = len(local_sets['labeled'][c_id])
        n_disc = len(local_sets['discarded'][c_id])
        
        total_valid_id_now += n_lab
        total_selected_now += (n_lab + n_disc)

    # Computes the cumulative purity of the selected data
    cumulative_purity = total_valid_id_now / total_selected_now if total_selected_now > 0 else 0.0
    
    return total_valid_id_now, total_selected_now, cumulative_purity


def compute_round0_global_query_stats(local_sets, query_num, total_id_samples_per_client):
    """Compute Round‑0 global QP and AQR statistics.

    At the end of the warm‑up round, each client has:

    * queried ``query_num[c]`` samples, and
    * discovered ``len(local_sets['labeled'][c])`` ID samples.

    This function aggregates those counts into:

    * ``avg_qp`` – mean per‑client Query Precision at Round 0, and
    * ``avg_aqr`` – global Average Query Recall at Round 0.

    Args:
        local_sets: Dict with per-client ``'labeled'`` lists.
        query_num: Per-client initial query budgets.
        total_id_samples_per_client: Per-client total number of ID samples in
            the dataset.

    Returns:
        Tuple ``(avg_qp, avg_aqr)``.
    """
    qps = []
    global_total_labeled_id = 0
    global_total_dataset_id = 0

    client_num = len(local_sets["labeled"])

    for c in range(client_num):
        # Per-client Round‑0 counts:
        #   - n_labeled_id   : how many ID samples this client selected.
        #   - n_total_selected: how many samples it was allowed to query.
        #   - n_dataset_id   : how many ID samples exist in its full dataset.
        n_labeled_id = len(local_sets["labeled"][c])
        n_total_selected = int(query_num[c])
        n_dataset_id = int(total_id_samples_per_client[c])

        # Per-client Round‑0 QP: fraction of the initial budget that landed on
        # ID samples for this client.
        qr = n_labeled_id / n_total_selected if n_total_selected > 0 else 0.0
        qps.append(qr)

        # Accumulate ID counts across all clients so that we can compute a
        # single global AQR value at Round 0.
        global_total_labeled_id += n_labeled_id
        global_total_dataset_id += n_dataset_id

    # Average per-client QP across all clients at Round 0.
    avg_qp = float(np.mean(qps)) if len(qps) > 0 else 0.0

    # Global AQR at Round 0: total ID found / total ID present.
    # This is a *snapshot* used for reporting; subsequent rounds update their
    avg_aqr = (
        global_total_labeled_id / global_total_dataset_id
        if global_total_dataset_id > 0
        else 0.0
    )

    return avg_qp, avg_aqr
