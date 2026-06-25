"""
Outer active-learning loop extracted from the experiment orchestrator.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_helpers import (
    get_extended_class_counts,
    loader_worker_init,
    make_subset_sampler,
    remaining_unlabeled,
)
from utils.cls.pal_helper import WNet
from utils.cls.selection_methods import query_samples
from utils.cls.test import test_detailed_2
from utils.cls.train_fedavg import train, train_lfosa, train_pal
from utils.fed_merge import fed_avg, fed_update
from utils.utils import cnt_sample_num
from utils.vlm_filter import FeatureCache, evaluate_vlm_breakdown, train_vlm_adapter


def run_active_learning_loop(
    *,
    args,
    Model,
    client_num: int,
    SUBSET: int,
    exp_logger,
    experiment_start_time: float,
    local_data: Dict[str, Any],
    local_sets: Dict[str, List[List[int]]],
    local_scalers: List[Any],
    local_vlm_cache: Dict[int, Any],
    local_vlm_features: Dict[int, Any],
    local_vlm_adapters: List[Any],
    local_vlm_optimizers: List[Any],
    train_slice_num: np.ndarray,
    query_num: np.ndarray,
    total_id_samples_per_client: np.ndarray,
    client_query_stats: Dict[int, Dict[str, float]],
    global_query_stats: Dict[str, float],
    get_model_probabilities: Callable[..., Any],
    get_extended_class_counts: Callable[..., Any],
    run_vlm_gated_warmup: Callable[..., Any],
    save_learnable_vectors: Callable[..., Any],
    vlm_feature_cache_root: Callable[..., str],
):
    """Run the full outer Active Learning loop over ``args.al_round`` rounds.

    Each AL round performs three phases:

    1. **FL training** — calls :func:`training.federated_loop.run_federated_loop`
       for ``args.max_round`` inner FL rounds of local training + FedAvg
       aggregation.
    2. **Querying** — calls ``query_samples()`` on each client's unlabeled pool
       and partitions the result into ID (→ ``labeled``) and OOD
       (→ ``discarded``) sets.
    3. **VLM adapter update** (when ``args.vlm_filter`` is active) — fine-tunes
       each client's prompt adapter on the newly labeled data, then aggregates
       the global context vectors via FedAvg.

    All results are logged via ``exp_logger`` at the end of each round.

    Args (keyword-only):
        args: Global experiment namespace from ``parse_args()``.
        Model: Callable (class) that returns a new task model when called as
            ``Model(num_classes=K)``.
        client_num: Number of federated clients.
        SUBSET: Total size of each client's unlabeled pool.
        exp_logger: :class:`exp_logging.experiment_logger.ExperimentLogger`
            instance used to write CSVs and console output.
        experiment_start_time: ``time.time()`` value from ``main()`` — used to
            log total wall-clock time.
        local_data: ``Dict[str, List[Dataset]]`` — keys ``'train'``,
            ``'unlabeled'``, ``'test'``; each a list of length ``client_num``.
        local_sets: ``Dict[str, List[List[int]]]`` — keys ``'labeled'``,
            ``'unlabeled'``, ``'discarded'``; each client's active index lists.
        local_scalers: ``List[GradScaler | None]`` — per-client AMP gradient
            scalers (``None`` when ``args.mixed_precision`` is ``False``).
        local_vlm_cache: ``Dict[int, Any]`` — per-client result dict returned
            by ``run_vlm_gated_warmup`` (contains embeddings, mask, scores).
        local_vlm_features: ``Dict[int, np.ndarray]`` — per-client BiomedCLIP
            embeddings, shape ``[N, 512]`` float32.
        local_vlm_adapters: ``List[nn.Module | None]`` — per-client prompt
            adapter; ``None`` until first VLM training round.
        local_vlm_optimizers: ``List[Optimizer | None]`` — per-client adapter
            optimizer; ``None`` until first VLM training round.
        train_slice_num: ``np.ndarray`` shape ``[client_num]`` int — number of
            samples in each client's full train split.
        query_num: ``np.ndarray`` shape ``[client_num]`` int — per-client query
            budget per AL round.
        total_id_samples_per_client: ``np.ndarray`` shape ``[client_num]``
            int — total number of ID samples in each client's train split.
        client_query_stats: ``Dict[int, Dict[str, float]]`` — per-client
            running QP / AQR accumulators; updated in-place each round.
        global_query_stats: ``Dict[str, float]`` — global QP / AQR state dict;
            updated in-place each round.
        get_model_probabilities: Callable wrapping the task-model inference
            pass.
        get_extended_class_counts: Callable returning ``(counts, purity)``
            for a subset of a dataset.
        run_vlm_gated_warmup: Callable that runs BiomedCLIP zero-shot warmup
            and returns the result cache dict.
        save_learnable_vectors: Callable that persists a trained adapter's
            context vectors to disk.
        vlm_feature_cache_root: Callable ``(args) -> str`` returning the root
            directory for BiomedCLIP feature caches.

    Returns:
        None — all outputs are written to disk via ``exp_logger``.
    """
    _vlm_emb_root = vlm_feature_cache_root(args)
    _num_id = int(getattr(args, 'num_id_classes', None) or getattr(args, 'num_classes', None) or 8)


    al_pbar = tqdm(range(args.al_round), desc="AL Rounds", position=0)

    local_models = []
    local_wnets = [] # For PAL
    local_wnet_optimizers = [] # Persistent Adam optimizers for PAL WNets
    for c_id in range(client_num):
        model = Model(num_classes=args.num_classes).cuda()
        local_models.append(model)

        if args.al_method == 'PAL':
            # Input dim is 2 * num_classes for OVA logits
            wnet = WNet(in_dim=2 * args.num_classes, hidden_dim=512).cuda()
            local_wnets.append(wnet)
            local_wnet_optimizers.append(torch.optim.Adam(wnet.parameters(), lr=6e-5))
        else:
            local_wnets.append(None)
            local_wnet_optimizers.append(None)
            
    
    for al_round_idx in al_pbar:
        round_start_time = time.time()
        args.current_al_round = al_round_idx
        
        # Reset VLM inference counter for this round
        setattr(get_model_probabilities, '_counter', {'calls': 0, 'samples': 0})
        exp_logger.create_al_round_directory(al_round_idx + 1)

        # Initialize Global Model
        global_model = Model(num_classes=args.num_classes).cuda()

        # syncing global and local
        fed_update(global_model, local_models)

        # 1. Initialization for FL Round
        local_optimizers = []
        local_schedulers = []
        num_per_class = []

        for c_id in range(client_num):
            if args.dataset == 'FedISIC':
                opt = torch.optim.Adam(local_models[c_id].parameters(), lr=args.base_lr, weight_decay=5e-4)
                sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[50], gamma=0.1)
            elif args.dataset == 'FedEMBED':
                opt = torch.optim.AdamW(local_models[c_id].parameters(), lr=args.base_lr, weight_decay=1e-4)
                # sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_round)
                sch = None
            else:
                opt = torch.optim.Adam(local_models[c_id].parameters(), lr=args.base_lr, weight_decay=5e-4)
                sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[50], gamma=0.1)
            local_optimizers.append(opt)
            local_schedulers.append(sch)
            num_per_class.append(cnt_sample_num(local_data['loaders']['train'][c_id], args.num_classes))

        # Sync Locals with Global (which holds previous round's knowledge)
        
        latest_global_metrics = {
            'Accuracy': 0.0, 'Balanced_Acc': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'F1_Score': 0.0, 'AUC': 0.0,
        }

        # PREPARE OOD LOADERS (Do this ONCE per AL round) 
        ood_loaders = [None] * client_num
        if args.al_method in ['LfOSA', 'PAL']:
            _nw = 0 if args.dataset == 'FedEMBED' else 4
            for c_id in range(client_num):
                if len(local_sets['discarded'][c_id]) > 0:
                    ood_loaders[c_id] = DataLoader(
                        local_data['train'][c_id], 
                        batch_size=args.batch_size, 
                        sampler=make_subset_sampler(args, local_sets['discarded'][c_id]),
                        num_workers=_nw, pin_memory=(_nw > 0),
                        worker_init_fn=loader_worker_init(args, _nw)
                    )

        # 2. Federated Learning Loop (Training & Testing)
        fl_pbar = tqdm(range(args.max_round), desc=f"AL-{al_round_idx+1} Training", leave=False, position=1)
        
        for round_idx in fl_pbar:
            round_loss = []
            
            for c_id in range(client_num):
                # TRAIN
                if args.al_method == 'LfOSA':
                    loss_val = train_lfosa(
                        round_idx, c_id, local_models[c_id], 
                        local_data['loaders']['train'][c_id], 
                        local_optimizers[c_id], 
                        args,
                        scaler=local_scalers[c_id],
                        ood_dataloader=ood_loaders[c_id]
                    )
                elif args.al_method == 'PAL':
                    loss_val = train_pal(
                        round_idx, c_id, local_models[c_id], 
                        local_data['loaders']['train'][c_id], 
                        local_optimizers[c_id], 
                        args,
                        scaler=local_scalers[c_id],
                        ood_dataloader=ood_loaders[c_id]
                    )
                else:
                    # Standard Training
                    loss_val = train(
                        round_idx, c_id, local_models[c_id], 
                        local_data['loaders']['train'][c_id], 
                        local_optimizers[c_id], 
                        num_per_class[c_id], 
                        args,
                        scaler=local_scalers[c_id] 
                    )
                # if args.dataset != 'FedEMBED':
                scheduler = local_schedulers[c_id]
                if scheduler is not None:
                    scheduler.step()
                round_loss.append(loss_val)
                exp_logger.log_train_metric(round_idx+1, c_id, loss_val)

            # Aggregation
            train_sizes = [len(s) for s in local_sets['labeled']]
            weights = np.array(train_sizes) / np.sum(train_sizes)
            fed_avg(global_model, local_models, weights)
            
            # Logging
            avg_loss = np.mean(round_loss)
            fl_pbar.set_postfix({'Avg Loss': f'{avg_loss:.4f}'})
            exp_logger.log_train_metric(round_idx+1, 'Global', avg_loss)

            # Testing
            if (round_idx + 1) % args.display_freq == 0 or (round_idx + 1) == args.max_round:
                global_metrics_accum = {}
                for c_id in range(client_num):
                    metrics = test_detailed_2(
                        dataset_name=args.dataset, 
                        model=global_model, 
                        test_loader=local_data['loaders']['test'][c_id], 
                        num_classes=args.num_classes,
                        device='cuda'
                    )
                    # Inject stored query metrics into the test metrics dict
                    metrics['QR'] = client_query_stats[c_id]['QR']
                    metrics['AQR'] = client_query_stats[c_id]['AQR']
                    exp_logger.log_test_metrics(round_idx+1, c_id, metrics)
                    for k, v in metrics.items():
                        global_metrics_accum.setdefault(k, []).append(v)
                    
                
                global_metrics = {k: np.mean(v) for k, v in global_metrics_accum.items()}
                exp_logger.log_test_metrics(round_idx+1, 'Global', global_metrics)
                al_pbar.set_description(f"AL Round {al_round_idx+1} | Acc: {global_metrics['Balanced_Acc']:.2%}")
                latest_global_metrics = global_metrics 

            if round_idx < args.max_round - 1:
                fed_update(global_model, local_models)
        

        #  3. Calculate Cumulative Stats & Log Global Summary (BEFORE QUERYING) 
        total_valid_id_now = 0
        total_selected_now = 0
        
        for c_id in range(client_num):
            n_lab = len(local_sets['labeled'][c_id])
            n_disc = len(local_sets['discarded'][c_id])
            
            total_valid_id_now += n_lab
            total_selected_now += (n_lab + n_disc)

        # Cumulative Purity = (Total Valid ID) / (Total Selected)
        cumulative_purity = total_valid_id_now / total_selected_now if total_selected_now > 0 else 0.0

        # CALCULATE DURATIONS HERE 
        current_time = time.time()
        round_train_duration = current_time - round_start_time
        total_elapsed = current_time - experiment_start_time
        
        print(f"\n[Timer] Round {al_round_idx+1} finished in {round_train_duration/60:.2f} mins.")

        # COMPUTE COST CSV 
        vlm_stats = getattr(get_model_probabilities, '_counter', {'calls': 0, 'samples': 0})
        vlm_params = 0
        adapter_type = 'none'
        if args.vlm_filter and args.vlm_dynamic and hasattr(args, '_local_vlm_adapters_ref'):
            # Count from first non-None adapter
            for _adp in args._local_vlm_adapters_ref:
                if _adp is not None:
                    vlm_params = sum(p.numel() for p in _adp.parameters() if p.requires_grad)
                    adapter_type = type(_adp).__name__
                    break
        
        cost_row = {
            'AL_Round': al_round_idx + 1,
            'Round_Time_Sec': round_train_duration,
            'VLM_Calls': vlm_stats['calls'],
            'VLM_Samples': vlm_stats['samples'],
            'VLM_Trainable_Params': vlm_params,
            'Adapter_Type': adapter_type,
            'AL_Method': args.al_method,
            'Consensus_Enabled': 0,
        }
        cost_csv = os.path.join(exp_logger.base_dir, 'compute_cost.csv')
        cost_df = pd.DataFrame([cost_row])
        if os.path.exists(cost_csv):
            cost_df.to_csv(cost_csv, mode='a', header=False, index=False)
        else:
            cost_df.to_csv(cost_csv, index=False)
        print(f"[Compute] VLM: {vlm_stats['calls']} calls, {vlm_stats['samples']} samples | "
              f"Adapter: {adapter_type} ({vlm_params:,} params)")

        exp_logger.log_global_summary(
            al_round=al_round_idx + 1,
            metrics_dict=latest_global_metrics,
            total_labeled=total_valid_id_now,  
            total_queries=total_selected_now,  
            id_purity=cumulative_purity,
            avg_qr=global_query_stats['Avg_QR'],   # Pass stored value
            avg_aqr=global_query_stats['Avg_AQR'], # Pass stored value
            round_train_time=round_train_duration,
            total_elapsed_time=total_elapsed   
        )
        
        if args.save_model_weights:
            torch.save(global_model.state_dict(), 
                    os.path.join(exp_logger.model_dir, f'AL{al_round_idx+1}_Global.pth'))

        # 4. Active Learning Query Strategy
        if al_round_idx < args.al_round - 1:
            print(f"\n[AL Round {al_round_idx+1}] Querying Samples for next round...")

            next_qrs = []
            next_global_total_labeled_id = 0
            next_global_total_dataset_id = 0
            # ENSURE CACHE IS LOADED FOR ALL CLIENTS BEFORE FILTERING
            if args.vlm_filter:
                for c_id in range(client_num):
                    if local_vlm_features.get(c_id) is None:
                        logging.info(f"[VLM] Cache miss for Client {c_id} (Pre-Filter). Computing embeddings...")
                        res = run_vlm_gated_warmup(
                            dataset=local_data['train'][c_id], 
                            dataset_name=args.dataset, 
                            device='cuda',
                            client_idx=c_id,
                            feature_cache_root=_vlm_emb_root,
                        )
                        if res.get('embeddings') is not None:
                            local_vlm_features[c_id] = res['embeddings']
                            # Save to disk
                            # save_path = os.path.join(exp_logger.base_dir, f"client_{c_id}_vlm_embeddings.npy")
                            # np.save(save_path, res['embeddings'])
                            from utils.vlm_filter import FeatureCache
                            fc = FeatureCache(args.dataset, c_id, cache_root=_vlm_emb_root)
                            emb = res['embeddings']
                            emb_np = emb if isinstance(emb, np.ndarray) else emb.numpy()
                            np.save(fc.npy_path, emb_np)
                            logging.info(f"  -> Saved VLM embeddings to {fc.npy_path}")


            for c_id in range(client_num):
                if args.save_model_weights:
                    # Save Local Model State
                    torch.save(local_models[c_id].state_dict(), 
                            os.path.join(exp_logger.model_dir, f'AL{al_round_idx+1}_Client{c_id}.pth'))
                
                curr_labeled = len(local_sets['labeled'][c_id])
                max_labeled = int(np.ceil(0.85 * train_slice_num[c_id]))
                
                if curr_labeled >= max_labeled: 
                    continue
                
                
                actual_query_num = min(query_num[c_id], max_labeled - curr_labeled)
                # DYNAMIC CONSENSUS UPDATE (Run only if Dynamic is ON)
                # FILTERING & POOL CREATION (Round > 0)
                current_unlabeled = local_sets['unlabeled'][c_id]
                
                # Default Init
                safe_indices = list(current_unlabeled)
                explore_indices = []

                if args.vlm_filter and args.vlm_dynamic and al_round_idx > 0:
                    
                    # A. GET PROBABILITIES 
                    # 1. Get VLM ID Probabilities (Sum < 1.0 if OOD)
                    vlm_probs = get_model_probabilities(
                        model=local_vlm_adapters[c_id],
                        dataset=local_data['train'][c_id],
                        indices=current_unlabeled,
                        is_vlm=True,
                        vlm_id_classes=getattr(args, 'num_classes', 8),
                        dataset_name=args.dataset,
                        device='cuda',
                        cached_features=local_vlm_features.get(c_id)
                    )

                    # B. APPLY STRATEGY 
                    # Default masks (all safe) — overridden by the strategy below
                    n_pool = len(current_unlabeled)
                    mask_safe = np.ones(n_pool, dtype=bool)
                    mask_explore = np.zeros(n_pool, dtype=bool)
                    mask_discard = np.zeros(n_pool, dtype=bool)
                    
                    if args.filter_strategy == 'vlm_only':
                        # STRATEGY: HIGHEST SCORE WINNER (Top-1)
                        print(f"[Client {c_id}] Strategy: VLM Only (Highest Class Winner)")
                        
                        # 1. Get FULL Probabilities (ID + OOD)
                        # Shape: [N, 15] (8 ID + 7 OOD)
                        vlm_full_probs = get_model_probabilities(
                            model=local_vlm_adapters[c_id],
                            dataset=local_data['train'][c_id],
                            indices=current_unlabeled,
                            is_vlm=True,
                            vlm_id_classes=getattr(args, 'num_classes', 8),
                            dataset_name=args.dataset,
                            device='cuda',
                            return_full=True,
                            cached_features=local_vlm_features.get(c_id)
                        )
                        
                        # 2. Find the Winner (Argmax)
                        # Returns index of the highest score (0 to 14)
                        winners = vlm_full_probs.argmax(dim=1).numpy()
                        
                        # 3. Is the winner an ID class?
                        # ID indices are 0 to 7. OOD indices are 8 to 14.
                        # If winner < num_classes, it is ID.
                        num_id = int(getattr(args, 'num_id_classes', None) or getattr(args, 'num_classes', None) or 8)
                        mask_safe = winners < num_id
                        
                        # Everything else (where winner >= 8) is OOD/Explore
                        mask_explore = ~mask_safe
                        mask_discard = np.zeros_like(mask_safe)

                    n_safe = mask_safe.sum()
                    n_explore = mask_explore.sum()
                    n_discard = mask_discard.sum()
                    
                    print(f"    -> Safe: {n_safe} | Explore: {n_explore} | Discard: {n_discard}")

                    safe_indices = [current_unlabeled[i] for i, m in enumerate(mask_safe) if m]
                    explore_indices = [current_unlabeled[i] for i, m in enumerate(mask_explore) if m]

                elif args.vlm_filter:
                     # Fallback Round 0 / Static
                     if local_vlm_cache.get(c_id) is not None:
                        mask = local_vlm_cache[c_id]
                        safe_indices = [i for i in current_unlabeled if mask[i]]
                        explore_indices = [i for i in current_unlabeled if not mask[i]]

                # STRATIFIED QUERY STRATEGY
                # Should I use the old Round 0 cache?
                if 'safe_indices' not in locals() or not args.vlm_dynamic:
                    current_unlabeled = local_sets['unlabeled'][c_id]
                    
                    # Define Pools using CACHE (Static Fallback) (safe_indices NOT exist yet? Is Dynamic Mode turned OFF)
                    if args.vlm_filter and local_vlm_cache.get(c_id) is not None:
                        mask = local_vlm_cache[c_id]
                        safe_indices = [i for i in current_unlabeled if mask[i]]
                        explore_indices = [i for i in current_unlabeled if not mask[i]]
                    else:
                        # Preserve the fresh data
                        safe_indices = list(current_unlabeled)
                        explore_indices = []

                # Allocate Budget
                # If explore_ratio is 0.0 (default), n_explore will be 0.
                n_explore = int(actual_query_num * args.explore_ratio)
                n_safe = actual_query_num - n_explore

                # Safety: Shift budget if a pool is empty
                if len(safe_indices) < n_safe:
                    n_explore += (n_safe - len(safe_indices))
                    n_safe = len(safe_indices)
                if len(explore_indices) < n_explore:
                    n_safe += (n_explore - len(explore_indices))
                    n_explore = len(explore_indices)
                
                if args.explore_ratio > 0:
                    print(f"[Client {c_id}] Stratified Query: {n_safe} Safe | {n_explore} Explore")

                # Query Helper Function
                def run_query_on_pool(candidate_pool, budget):
                    if budget <= 0 or len(candidate_pool) == 0: 
                        return [], []
                    
                    # Random Subset optimization for large pools
                    if len(candidate_pool) > SUBSET:
                         pool_subset = random.sample(candidate_pool, SUBSET)
                    else:
                         pool_subset = candidate_pool

                    # Entropy fallback for open-set methods in Round 0
                    # (specialized heads have no OOD data yet, so their scoring is unreliable)
                    effective_method = args.al_method
                    if al_round_idx == 0 and args.al_method in ['PAL', 'LfOSA']:
                        effective_method = 'Entropy'

                    rank_arg, scores = query_samples(
                        al_method=effective_method,
                        global_model=global_model,
                        local_model=local_models[c_id],
                        data_unlabeled=local_data['unlabeled'][c_id],
                        unlabeled_set=pool_subset, 
                        query_num=budget,
                        args=args,
                        labeled_set=local_sets['labeled'][c_id],
                        discarded_set=local_sets['discarded'][c_id],
                        dataset_source=local_data['train'][c_id],
                        wnet=local_wnets[c_id],
                        wnet_optimizer=local_wnet_optimizers[c_id]
                    )
                    
                    # Map back to global indices
                    selected_local = rank_arg[-budget:]
                    selected_scores = scores[-budget:]
                    pool_tensor = torch.tensor(pool_subset)
                    global_indices = list(pool_tensor[selected_local].numpy())
                    return global_indices, selected_scores

                # D. Execute Queries
                new_safe_indices, safe_scores = run_query_on_pool(safe_indices, n_safe)
                new_explore_indices, explore_scores = run_query_on_pool(explore_indices, n_explore)
                
                # E. Merge
                new_labeled_indices = new_safe_indices + new_explore_indices
                if len(new_labeled_indices) > 0:
                    # Concatenate scores for logging (handling potentially empty arrays)
                    parts = []
                    if len(safe_scores) > 0: parts.append(safe_scores)
                    if len(explore_scores) > 0: parts.append(explore_scores)
                    selected_scores = np.concatenate(parts) if parts else np.array([])
                else:
                    selected_scores = []

                # OOD FILTERING (The Oracle Step) 
                valid_id_indices = []
                ood_indices = []
                _ds = local_data['train'][c_id]
                _has_labels = hasattr(_ds, 'labels')
                
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
                
                # Calculate Metrics before list update 
                # QR / QP (Batch Precision)
                batch_total = len(new_labeled_indices)
                batch_id = len(valid_id_indices)
                qr_val = batch_id / batch_total if batch_total > 0 else 0.0

                # AQR (Accumulated Recall): labeled so far + this batch / total client ID pool
                total_id_found_so_far = len(local_sets['labeled'][c_id]) + batch_id
                total_id_in_dataset = total_id_samples_per_client[c_id]

                next_global_total_labeled_id += total_id_found_so_far
                next_global_total_dataset_id += total_id_in_dataset

                next_qrs.append(qr_val)

                # Batch Purity
                n_batch_total = len(new_labeled_indices)
                n_batch_id = len(valid_id_indices)
                batch_purity = n_batch_id / n_batch_total if n_batch_total > 0 else 0.0
                
                logging.info(f"Client {c_id} Query: {n_batch_total}. ID: {n_batch_id}, OOD: {len(ood_indices)}. Batch Purity: {batch_purity:.2%}")
                vlm_eval_snapshot = list(local_sets['unlabeled'][c_id])
                
                # Update Lists
                local_sets['labeled'][c_id].extend(valid_id_indices)
                local_sets['discarded'][c_id].extend(ood_indices)
                
                full_indices = set(range(len(local_data['train'][c_id])))
                all_used = set(local_sets['labeled'][c_id]) | set(local_sets['discarded'][c_id])
                local_sets['unlabeled'][c_id] = remaining_unlabeled(args, full_indices, all_used)

                # Logs
                # Log Selected Samples (CSV)
                exp_logger.log_selected_samples(c_id, new_labeled_indices, local_data['train'][c_id], selected_scores)
                
                # Query Set Stats
                # new_labeled_indices is the raw selection (Mixed ID + OOD)
                query_counts, query_pur = get_extended_class_counts(local_data['train'][c_id], new_labeled_indices, args.num_classes)
                exp_logger.log_client_state(c_id, "Query", query_counts, scores=selected_scores, purity=query_pur)

                # Labeled Stats (Cumulative: Valid ID + Discarded OOD)
                combined_labeled = local_sets['labeled'][c_id] + local_sets['discarded'][c_id]
                # lbl_counts, lbl_pur = get_extended_class_counts(local_data['train'][c_id], combined_labeled, args.num_classes)
                lbl_counts, lbl_pur = get_extended_class_counts(local_data['train'][c_id], combined_labeled, _num_id)
                exp_logger.log_client_state(c_id, "Labeled", lbl_counts, purity=lbl_pur)
                
                # Unlabeled Stats
                unlbl_counts, unlbl_pur = get_extended_class_counts(local_data['train'][c_id], local_sets['unlabeled'][c_id], args.num_classes)
                exp_logger.log_client_state(c_id, "Unlabeled", unlbl_counts, purity=unlbl_pur)

                # Re-init Loader
                _nw = 0 if args.dataset == 'FedEMBED' else 4
                local_data['loaders']['train'][c_id] = DataLoader(
                    dataset=local_data['train'][c_id],
                    batch_size=args.batch_size,
                    sampler=make_subset_sampler(args, local_sets['labeled'][c_id]),
                    num_workers=_nw, pin_memory=(_nw > 0),
                    worker_init_fn=loader_worker_init(args, _nw)
                )

                # VLM ADAPTER TRAIN & EVAL 
                if args.vlm_adapter is not None or args.vlm_eval:
                    from utils.vlm_filter import train_vlm_adapter, evaluate_vlm_breakdown
                    
                    # ROBUSTNESS CHECK: Lazy Loading
                    if local_vlm_features.get(c_id) is None:
                        logging.info(f"[VLM] Cache miss for Client {c_id}. Computing embeddings now...")
                        
                        res = run_vlm_gated_warmup(
                            dataset=local_data['train'][c_id], 
                            dataset_name=args.dataset, 
                            device='cuda',
                            client_idx=c_id,
                            feature_cache_root=_vlm_emb_root,
                        )
                        
                        if res.get('embeddings') is not None:
                            local_vlm_features[c_id] = res['embeddings']
                            # Save to disk
                            save_path = os.path.join(exp_logger.base_dir, f"client_{c_id}_vlm_embeddings.npy")
                            # np.save(save_path, res['embeddings'])
                            # logging.info(f"  -> Saved VLM embeddings to {save_path}")
                            from utils.vlm_filter import FeatureCache
                            fc = FeatureCache(args.dataset, c_id, cache_root=_vlm_emb_root)
                            emb = res['embeddings']
                            emb_np = emb if isinstance(emb, np.ndarray) else emb.numpy()
                            np.save(fc.npy_path, emb_np)
                            logging.info(f"  -> Saved VLM embeddings to {fc.npy_path}")


                    # Train Adapter
                    if args.vlm_adapter:
                        logging.info(f"[Client {c_id}] Updating VLM Adapter ({args.vlm_adapter})...")
                        
                        full_feats = local_vlm_features.get(c_id)
                        tr_feats, tr_lbls = None, None
                        if full_feats is not None:
                            if getattr(args, 'vlm_train_source', 'query') == 'query':
                                    candidate_pool = new_labeled_indices
                            else:
                                if args.only_id_coop:
                                    candidate_pool = local_sets['labeled'][c_id]
                                else:
                                    candidate_pool = local_sets['labeled'][c_id] + local_sets['discarded'][c_id]

                            vlm_indices = []
                            class_buckets = {}
                            
                            for idx in candidate_pool:
                                _, meta = local_data['train'][c_id][idx]
                                lbl = meta.get('original_label', -1)
                                if hasattr(lbl, 'item'): lbl = lbl.item()
                                
                                # group strictly by label to ensure diversity
                                if lbl not in class_buckets: class_buckets[lbl] = []
                                class_buckets[lbl].append(idx)

                            # Select up to args.coop_shots per class
                            for lbl, idx_list in class_buckets.items():
                                if args.coop_shots == -1 or len(idx_list) <= args.coop_shots:
                                    # Use all available if fewer than shots
                                    vlm_indices.extend(idx_list)
                                else:
                                    # Randomly sample
                                    selected = np.random.choice(idx_list, args.coop_shots, replace=False)
                                    vlm_indices.extend(selected)

                            # 1. Calculate Counts for all three sets
                            #    A. VLM Few-Shot Set (The specific subset chosen for adapter training)
                            vlm_dist, _ = get_extended_class_counts(local_data['train'][c_id], vlm_indices, args.num_classes)
                            
                            #    B. Current Query (Everything selected by AL this round)
                            query_dist, _ = get_extended_class_counts(local_data['train'][c_id], new_labeled_indices, args.num_classes)
                            
                            #    C. Accumulated Labeled (Total training set so far)
                            acc_dist, _ = get_extended_class_counts(local_data['train'][c_id], local_sets['labeled'][c_id], args.num_classes)

                            # 2. Prepare Data Rows
                            dist_rows = []
                            total_cols = args.num_classes + getattr(args, 'num_ood_classes', 1)
                            
                            # Row 1: VLM Few-Shot Selection
                            row_vlm = {"Set_Type": "VLM_FewShot_Selection", "Total_Count": len(vlm_indices)}
                            for i in range(total_cols): row_vlm[f"Class_{i}"] = vlm_dist.get(i, 0)
                            dist_rows.append(row_vlm)

                            # Row 2: Current Query
                            row_q = {"Set_Type": "Current_Query_Batch", "Total_Count": len(new_labeled_indices)}
                            for i in range(total_cols): row_q[f"Class_{i}"] = query_dist.get(i, 0)
                            dist_rows.append(row_q)

                            # Row 3: Accumulated Labeled
                            row_acc = {"Set_Type": "Accumulated_Labeled_Set", "Total_Count": len(local_sets['labeled'][c_id])}
                            for i in range(total_cols): row_acc[f"Class_{i}"] = acc_dist.get(i, 0)
                            dist_rows.append(row_acc)

                            # 3. Save to CSV
                            #    File: logs/.../AL_Round_X/client_Y_selection_stats.csv
                            dist_headers = ["Set_Type", "Total_Count"] + [f"Class_{i}" for i in range(total_cols)]
                            dist_csv_path = os.path.join(exp_logger.current_al_dir, f"client_{c_id}_selection_stats.csv")
                            pd.DataFrame(dist_rows).to_csv(dist_csv_path, columns=dist_headers, index=False)
                            
                            logging.info(f"[Client {c_id}] Saved selection stats to {dist_csv_path}")
                            
                            tr_feats = full_feats[vlm_indices]
                            lbls = []
                            for idx in vlm_indices:
                                _, meta = local_data['train'][c_id][idx]
                                lbl = meta.get('original_label', -1)
                                if hasattr(lbl, 'item'): lbl = lbl.item()
                                lbls.append(lbl)
                            tr_lbls = np.array(lbls)

                        # Unpack adapter AND history
                        local_vlm_adapters[c_id], local_vlm_optimizers[c_id], loss_hist = train_vlm_adapter(
                            model_type=args.vlm_adapter,
                            dataset_name=args.dataset,
                            train_loader=None, 
                            prev_adapter=local_vlm_adapters[c_id],
                            prev_optimizer=local_vlm_optimizers[c_id],
                            device='cuda',
                            cached_features=tr_feats,
                            cached_labels=tr_lbls,
                            args=args
                        )

                        if args.avoid_save_coop_vectors:
                            print("-> skipping saving coop vectors")
                        else:
                            # Save Learnable Vectors for t-SNE / Analysis
                            vec_dir = os.path.join(exp_logger.base_dir, "vectors")
                            os.makedirs(vec_dir, exist_ok=True)
                            
                            vec_path = os.path.join(vec_dir, f"vectors_client{c_id}_round{al_round_idx+1}.pth")
                            save_learnable_vectors(local_vlm_adapters[c_id], vec_path)
                            print(f"  -> Saved vectors to {vec_path}")
                        
                        
                        # Save Training Loss to CSV
                        if len(loss_hist) > 0:
                            df_loss = pd.DataFrame({
                                'Epoch': range(1, len(loss_hist) + 1),
                                'Loss': loss_hist
                            })

                            # Saves to: logs/.../AL_Round_X/Client_Y_CoOp_Loss.csv
                            loss_csv_path = os.path.join(exp_logger.current_al_dir, f"Client_{c_id}_CoOp_Loss.csv")
                            df_loss.to_csv(loss_csv_path, index=False)
                            logging.info(f"  -> Saved CoOp training loss to {loss_csv_path}")
                        

                    # 2. Evaluate (Only if requested)
                    if args.vlm_adapter is not None or args.vlm_eval:
                    
                        # [Evaluation Logic]
                        if args.vlm_eval:
                            # UNPACK TUPLE: metrics_breakdown (dict) and raw_br (dict of arrays)
                            metrics_breakdown, raw_br = evaluate_vlm_breakdown(
                                dataset_name=args.dataset,
                                vlm_adapter=local_vlm_adapters[c_id],
                                full_dataset=local_data['train'][c_id],
                                unlabeled_indices=vlm_eval_snapshot,
                                query_indices=new_labeled_indices,
                                labeled_indices=local_sets['labeled'][c_id] + local_sets['discarded'][c_id],
                                cached_features=local_vlm_features.get(c_id),
                                device='cuda',
                                args=args
                            )

                            exp_logger.log_vlm_metrics(al_round_idx+1, c_id, metrics_breakdown)

                            # PER-SAMPLE LOGGING (Every round for Dynamic/CoOp) 
                            if args.vlm_dynamic or args.vlm_adapter is not None:
                                # Get the filename/path from the data_list (Robust)
                                full_ds = local_data['train'][c_id]
                                paths = [full_ds.data_list.iloc[i, 0] for i in vlm_eval_snapshot]
                                
                                details_df = pd.DataFrame({
                                    'Dataset_Index': vlm_eval_snapshot,
                                    'Path': paths,
                                    'True_Label': raw_br['t'],
                                    'Pseudo_Label': raw_br['p'],
                                    'ID_Soft_Score': raw_br['ids'],
                                    'OOD_Soft_Score': raw_br['oods'],
                                    'Entropy': raw_br['ent']
                                })
                                
                                # Save the full breakdown for deep analysis
                                details_df.to_csv(os.path.join(exp_logger.current_al_dir, f"vlm_details_client{c_id}.csv"), index=False)
                                
                                # Save pool splits (ID Pool vs Explore Pool)
                                if args.vlm_filter and local_vlm_cache.get(c_id) is not None:
                                    current_mask = local_vlm_cache[c_id]
                                else:
                                    current_mask = np.ones(len(local_data['train'][c_id]), dtype=bool)

                                # Added bounds check to prevent IndexError if mask is smaller than dataset index
                                mask_len = len(current_mask)

                                # If index is within bounds, check mask. If out of bounds, default to False (Explore).
                                id_pool_df = details_df[details_df['Dataset_Index'].map(lambda x: current_mask[x] if x < mask_len else False)]

                                # If index is within bounds, check not mask. If out of bounds, default to True (Explore).
                                explore_df = details_df[details_df['Dataset_Index'].map(lambda x: (not current_mask[x]) if x < mask_len else True)]
                                # current_mask is the boolean array for the whole training set
                                id_pool_df.to_csv(os.path.join(exp_logger.current_al_dir, f"pool_id_client{c_id}.csv"), index=False)
                                explore_df.to_csv(os.path.join(exp_logger.current_al_dir, f"pool_explore_client{c_id}.csv"), index=False)
            
            # Update Global Averages for the NEXT Round's Log
            if next_qrs:
                global_query_stats['Avg_QR'] = np.mean(next_qrs)
                global_query_stats['Avg_AQR'] = (
                    next_global_total_labeled_id / next_global_total_dataset_id
                    if next_global_total_dataset_id > 0 else 0.0
                )

            # FEDERATED ADAPTER AGGREGATION BLOCK 
        # Triggered if we are using CoOp/CoCoOp/ResCoOp AND the federated flag is active
        is_coop = (args.vlm_adapter == 'CoOp_original' and args.coop_federated)
        is_cocoop = (args.vlm_adapter == 'CoCoOp' and args.coop_federated)
        is_rescoop = (args.vlm_adapter == 'ResCoOp' and args.coop_federated)

        if is_coop or is_cocoop or is_rescoop:
            print(f"\n[AL Round {al_round_idx+1}] >>> Performing Federated Aggregation (Type: {args.vlm_adapter})...")
            
            global_vectors_accum = None
            meta_net_accum = {}
            total_weight = 0
            
            # 1. Collect
            for c_id in range(client_num):
                adapter = local_vlm_adapters[c_id]
                if adapter is not None:
                    weight = len(local_sets['labeled'][c_id])
                    total_weight += weight

                    # A. Global Vectors
                    if hasattr(adapter, 'get_global_vectors'):
                        vec = adapter.get_global_vectors()
                        if vec is not None:
                            if global_vectors_accum is None:
                                global_vectors_accum = vec * weight
                            else:
                                global_vectors_accum += (vec * weight)
                    
                    # B. MetaNet (Only for CoCoOp)
                    if hasattr(adapter, 'meta_net'):
                        for name, param in adapter.meta_net.state_dict().items():
                            if name not in meta_net_accum:
                                meta_net_accum[name] = param.clone() * weight
                            else:
                                meta_net_accum[name] += param * weight

            # 2. Average & Broadcast
            if total_weight > 0:
                # Update Vectors
                if global_vectors_accum is not None:
                    avg_global_vectors = global_vectors_accum / total_weight
                    print(f"    -> Global Vector Norm: {avg_global_vectors.norm().item():.4f}")
                    for c_id in range(client_num):
                        if local_vlm_adapters[c_id] is not None:
                            local_vlm_adapters[c_id].load_global_vectors(avg_global_vectors)

                # Update MetaNet
                if meta_net_accum:
                    avg_meta_state = {k: v / total_weight for k, v in meta_net_accum.items()}
                    print(f"    -> MetaNet aggregated ({len(avg_meta_state)} params).")
                    for c_id in range(client_num):
                        if local_vlm_adapters[c_id] is not None:
                            local_vlm_adapters[c_id].meta_net.load_state_dict(avg_meta_state)

                print("    -> Aggregation complete.")
            else:
                print("    -> [Warn] Aggregation skipped (Weight is 0).")
        else:
            print(f"\n[AL Round {al_round_idx+1}] Final Round - Experiment Ends.")
    exp_logger.close()
    print("\n>>> Experiment Completed.")