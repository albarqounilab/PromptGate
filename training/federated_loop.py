"""
training/federated_loop.py

Runs the inner federated-learning loop for one AL round.
----------
run_federated_loop(...)  -> (global_model, latest_global_metrics)
prepare_ood_loaders(...) -> list[DataLoader | None]
build_client_optimizers(...) -> (optimizers, schedulers)
reinitialize_or_warmstart_model(...) -> global_model
"""

import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from utils.fed_merge import fed_avg, fed_update

# Public: main federated training loop
def run_federated_loop(
    al_round_idx,
    args,
    client_num,
    global_model,
    local_models,
    local_data,
    local_sets,
    local_optimizers,
    local_schedulers,
    num_per_class,
    local_scalers,
    ood_loaders,
    train_fns,          
    exp_logger,
    test_detailed_2,
    client_query_stats,
):
    """
    Executes the inner Federated Learning (FL) loop for a single Active Learning round.

    This function coordinates the collaborative training process across multiple clients:
    1.  **Local Training**: Each client performs `args.max_round` iterations of local training
        on their acquired labeled dataset. The training function is dispatched based on the
        active learning method (e.g., standard FedAvg vs. OOD-aware methods like PAL).
    2.  **Aggregation (FedAvg)**: After each FL round, local model weights are aggregated
        into the `global_model` using weighted averaging based on the number of labeled
        samples each client possesses.
    3.  **Synchronization**: The updated global weights are then synchronized back to all
        `local_models` to start the next FL round from the same state.
    4.  **Evaluation**: Periodically (controlled by `args.display_freq`), the global model
        is evaluated on each client's test set to monitor performance progress.

    Args:
        al_round_idx: Current outer Active Learning round index.
        args: Configuration namespace containing `max_round`, `batch_size`, etc.
        global_model: The shared central model.
        local_models: List of client-specific models.
        local_sets: Dictionary tracking client data indices (used for sample-count weighting).
        train_fns: Dictionary of injected training functions (e.g., PAL trainer).
        exp_logger: Logger for tracking training loss and periodic test accuracy.

    Returns:
        Tuple[global_model, latest_global_metrics]: 
            The fully trained global model and a dictionary of final test metrics.
    """
    fl_pbar = tqdm(
        range(args.max_round),
        desc=f"AL-{al_round_idx + 1} Training",
        leave=False,
        position=1,
    )
    latest_global_metrics: dict = {}

    for round_idx in fl_pbar:
        round_loss = []

        #  Local training 
        for c_id in range(client_num):
            loss_val = _train_one_client(
                round_idx, c_id, al_round_idx, args,
                local_models[c_id],
                local_data["loaders"]["train"][c_id],
                local_optimizers[c_id],
                num_per_class[c_id],
                local_scalers[c_id],
                ood_loaders[c_id],
                train_fns,
            )
            # Step the local scheduler
            if local_schedulers[c_id] is not None:
                local_schedulers[c_id].step()

            round_loss.append(loss_val)
            exp_logger.log_train_metric(round_idx + 1, c_id, loss_val)

        #  FedAvg aggregation   
        train_sizes = [len(s) for s in local_sets["labeled"]]
        weights = np.array(train_sizes) / np.sum(train_sizes)
        fed_avg(global_model, local_models, weights)

        avg_loss = np.mean(round_loss)
        fl_pbar.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})
        exp_logger.log_train_metric(round_idx + 1, "Global", avg_loss)

        #  Periodic evaluation 
        if (round_idx + 1) % args.display_freq == 0 or (round_idx + 1) == args.max_round:
            latest_global_metrics = _evaluate_all_clients(
                round_idx, args, client_num, global_model,
                local_data, client_query_stats, exp_logger, test_detailed_2,
            )

        #  Sync aggregated weights back to local models (skip last round)     
        if round_idx < args.max_round - 1:
            fed_update(global_model, local_models)

    return global_model, latest_global_metrics


#   Public: setup helpers called once per AL round
def prepare_ood_loaders(args, client_num, local_data, local_sets, seed):
    """
    Builds per-client OOD DataLoaders for open-set methods (PAL, LfOSA).
    Returns a list of length client_num; entries are None when not needed.
    """
    ood_loaders = [None] * client_num
    if args.al_method not in ("LfOSA", "PAL"):
        return ood_loaders

    num_workers = 0 if args.dataset == "FedEMBED" else 4
    for c_id in range(client_num): # Iterate over the clients
        if local_sets["discarded"][c_id]:
            ood_loaders[c_id] = DataLoader( # Create a data loader for the discarded set
                local_data["train"][c_id],
                batch_size=args.batch_size, # Create a data loader for the discarded set on the batch size
                sampler=SubsetRandomSampler(
                    local_sets["discarded"][c_id],
                    generator=torch.Generator().manual_seed(seed), # Create a data loader for the discarded set on the generator
                ),
                num_workers=num_workers,
                pin_memory=(num_workers > 0),
            )
    return ood_loaders

# Build client optimizers and schedulers
def build_client_optimizers(args, client_num, local_models):
    """
    Constructs per-client optimizers and LR schedulers.

    Returns
    -------
    local_optimizers : list[Optimizer]
    local_schedulers : list[LRScheduler | None]
    """
    local_optimizers, local_schedulers = [], []

    for c_id in range(client_num):
        model = local_models[c_id]
        if args.dataset == "FedISIC":
            opt = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=5e-4)
            sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[50], gamma=0.1)
        elif args.dataset == "FedEMBED":
            opt = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-4)
            sch = None
        else:
            opt = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=5e-4)
            sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[50], gamma=0.1)

        local_optimizers.append(opt)
        local_schedulers.append(sch)

    return local_optimizers, local_schedulers


# Reinitialize or warmstart the model for the next AL round
def reinitialize_or_warmstart_model(al_round_idx, args, Model, global_model, local_models):
    """
    Either reinitialises the global model from scratch (default, FEAL protocol)
    or warm-starts from the previous round's weights (--no_reinit flag).

    Returns the (possibly new) global_model; local_models are synced in-place.
    """
    if al_round_idx == 0 or not args.no_reinit:
        global_model = Model(num_classes=args.num_classes).cuda()
        logging.info(
            f"[AL Round {al_round_idx + 1}] Model reinitialised from pretrained weights."
        )
    else:
        logging.info(
            f"[AL Round {al_round_idx + 1}] Warm-starting from previous round's model."
        )

    fed_update(global_model, local_models)
    return global_model


# Private helpers
# Train one client for one round
def _train_one_client(
    round_idx, c_id, al_round_idx, args,
    model, train_loader, optimizer, num_per_class_c, scaler, ood_loader,
    train_fns,
):
    """Dispatches to the correct training function based on args.al_method."""
    method = args.al_method

    if method == "LfOSA":
        return train_fns["train_lfosa"](
            round_idx, c_id, model, train_loader, optimizer, args,
            scaler=scaler, ood_dataloader=ood_loader, num_per_class=num_per_class_c,
        )
    if method == "PAL":
        return train_fns["train_pal"](
            round_idx, c_id, model, train_loader, optimizer, args,
            scaler=scaler, ood_dataloader=ood_loader, num_per_class=num_per_class_c,
        )

    # Standard cross-entropy training (all other methods)
    return train_fns["train"](
        round_idx, c_id, model, train_loader, optimizer,
        num_per_class_c, args, scaler=scaler,
    )


def _evaluate_all_clients(
    round_idx, args, client_num, global_model,
    local_data, client_query_stats, exp_logger, test_detailed_2,
):
    """Evaluates every client and logs results. Returns global-averaged metrics dict."""
    accum: dict = {}

    for c_id in range(client_num):
        metrics = test_detailed_2(
            dataset_name=args.dataset,
            model=global_model,
            test_loader=local_data["loaders"]["test"][c_id],
            num_classes=args.num_classes,
            device="cuda",
        )
        metrics["QR"]  = client_query_stats[c_id]["QR"]
        metrics["AQR"] = client_query_stats[c_id]["AQR"]

        exp_logger.log_test_metrics(round_idx + 1, c_id, metrics)
        for k, v in metrics.items():
            accum.setdefault(k, []).append(v)

    global_metrics = {k: np.mean(v) for k, v in accum.items()}
    exp_logger.log_test_metrics(round_idx + 1, "Global", global_metrics)
    return global_metrics