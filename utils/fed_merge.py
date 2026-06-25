"""Federated model aggregation utilities.

Glossary
--------
FL  : Federated Learning
"""

import copy

# Private helpers
def _weight_state_dict(state_dict, weight):
    """Return a new state-dict with every tensor scaled by *weight*."""
    return {k: weight * v for k, v in state_dict.items()}


def _add_state_dicts(state_dict_a, state_dict_b):
    """Return a new state-dict that is the element-wise sum of two state-dicts.

    Both dicts must have identical keys.
    """
    return {k: state_dict_a[k] + state_dict_b[k] for k in state_dict_a}


# Public API
def fed_avg(global_model, local_models, client_weight):
    """Aggregate local models into *global_model* using weighted FedAvg.

    Each local model's state-dict is scaled by its client weight and summed.
    The result is loaded directly into *global_model* in-place.

    Args:
        global_model: The central model whose weights will be overwritten.
        local_models: List of client models.
        client_weight: 1-D array-like of per-client weights (must sum to 1).

    Returns:
        None — *global_model* is modified in-place.
    """
    new_model_dict = None

    for client_idx, model in enumerate(local_models):
        local_dict = model.state_dict()
        weighted = _weight_state_dict(local_dict, client_weight[client_idx])

        if new_model_dict is None:
            new_model_dict = weighted
        else:
            new_model_dict = _add_state_dicts(new_model_dict, weighted)

    global_model.load_state_dict(new_model_dict)


def fed_update(global_model, local_models):
    """Broadcast *global_model* weights to every client model.

    All local models are updated in-place with the current global state-dict.

    Args:
        global_model: The central model whose weights are broadcast.
        local_models: List of client models to overwrite.

    Returns:
        None — *local_models* are modified in-place.
    """
    global_dict = global_model.state_dict()
    for model in local_models:
        model.load_state_dict(global_dict)
