"""
Main entry point for PromptGate experiments.

Orchestrates argument parsing, Round-0 warm-up, and delegates subsequent
active-learning rounds to :func:`training.active_learning_loop.run_active_learning_loop`.
"""

import logging
import os
import random
import time
import warnings

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
from data.dataset_generator import generate_dataset
from data.vlm_processor import get_model_probabilities
from exp_logging.experiment_logger import ExperimentLogger
from settings.config import METHOD_CONFIGS, parse_args
from training.active_learning_loop import run_active_learning_loop
from utils.utils import set_seed
from utils.vlm_filter import FeatureCache, run_vlm_gated_warmup, save_learnable_vectors
from visualization.visualizer import ExperimentVisualizer

logging.getLogger("open_clip").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def vlm_feature_cache_root(args):
    """Directory for VLM embedding caches: cache/<dataset>/<type_ood>/<ood>/."""
    override = getattr(args, "vlm_embedding_cache_dir", None)
    if override:
        return override
    type_ood = getattr(args, "type_ood", "FarOOD")
    ood = getattr(args, "ood", "50%")
    project_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(project_root, "cache", args.dataset, type_ood, ood)


def main():
    """Entry point for a single PromptGate experiment run.

    Parses CLI arguments, enforces VLM-filter configuration rules, loads
    per-client datasets, runs Round-0 (warmup) selection, then delegates
    the remaining AL rounds to :func:`training.active_learning_loop.run_active_learning_loop`.

    The function does not return; all results are persisted to disk via
    :class:`exp_logging.experiment_logger.ExperimentLogger`.
    """
    args = parse_args()
    experiment_start_time = time.time()  # Start global timer

    logging.getLogger("open_clip").setLevel(logging.ERROR)

    if args.vlm_adapter == 'CoOp_original' and args.coop_federated:
         args.coop_vectors = args.coop_global_vectors + args.coop_local_vectors
         print(f"[Federated CoOp] total vectors = {args.coop_vectors} "
             f"(global {args.coop_global_vectors} + local {args.coop_local_vectors})")

    #  RULE ENFORCEMENT 
    if args.coop_global_vectors == 0:
       args.vlm_adapter = 'CoOp_original'


    # RULE ENFORCEMENT 
    if not args.vlm_filter:
        # Rule 1: Coldstart
        print(">>> [Config] VLM Filter: OFF (Coldstart).")
        args.vlm_dynamic = False
        args.vlm_adapter = None
        
    else:
        if not args.vlm_dynamic:
            # Rule 2: Static
            if args.vlm_adapter is not None:
                raise ValueError("Static Mode (vlm_filter=True, vlm_dynamic=False) requires vlm_adapter=None. "
                                 "You cannot train an adapter if you are not using it dynamically.")
            print(">>> [Config] VLM Filter: STATIC (Round 0 Mask Locked).")
            
        else:
            # Rule 3: Dynamic
            if args.vlm_adapter is None:
                raise ValueError("Dynamic Mode (vlm_dynamic=True) requires a valid vlm_adapter "
                                 "(e.g., 'CoOp', 'MLP'). Cannot use None.")
            print(f">>> [Config] VLM Filter: DYNAMIC (Recalculating mask using {args.vlm_adapter}).")
    
    # CONFIGURATION MERGE 
    if args.al_method not in METHOD_CONFIGS:
        raise ValueError(f"Method {args.al_method} not found in METHOD_CONFIGS")
    
    method_config = METHOD_CONFIGS[args.al_method]
    for key, value in method_config.items():
        setattr(args, key, value)

    # SET NUM_CLASSES 
    if args.dataset == 'FedISIC':
        args.num_classes = 8
        client_num = 4
        args.num_ood_classes = 1
        SUBSET = 10000
        if args.al_method == 'PAL':
            from model.efficientnet_pal import EfficientNetB0 as Model
        elif args.al_method == 'LfOSA':
            from model.efficientnet_lfosa import EfficientNetB0 as Model
        else:
            from model.efficientnet import EfficientNetB0 as Model
    elif args.dataset == 'FedEMBED':
        args.num_classes = 4
        client_num = 4
        SUBSET = 10000
        from model.linearclassifier import LinearClassifier as Model
    else:
        print(f"[WARN] Unknown dataset {args.dataset}, defaulting to 8 classes.")

    set_seed(args.seed, args.deterministic)
    exp_logger = ExperimentLogger(args, method_config)
    
    # Initialize Data Structures
    local_data = {'train': [], 'unlabeled': [], 'test': [], 'loaders': {'train': [], 'test': []}}
    local_sets = {'labeled': [], 'unlabeled': [], 'discarded': []}
    local_models = []
    local_scalers = []
    local_wnets = []

    # VLM STATE 
    local_vlm_cache = {}      # Stores currently active mask (Safe Pool)
    local_frozen_masks = {}   # Stores the permanent Round 0 mask for Consensus
    local_vlm_features = {}   # Stores (N, 512) float32 array per client
    local_vlm_adapters = [None] * client_num
    args._local_vlm_adapters_ref = local_vlm_adapters  # for compute cost logging
    local_vlm_optimizers = [None] * client_num

    print(">>> Checking for pre-computed VLM feature cache...")
    from utils.vlm_filter import FeatureCache
    _vlm_emb_root = vlm_feature_cache_root(args)
    logging.info(f"  [FeatureCache] Root: {_vlm_emb_root}")
    for c_id in range(client_num):
        fc = FeatureCache(args.dataset, c_id, cache_root=_vlm_emb_root)
        if fc.exists():
            cached = fc.load()
            if cached is not None and cached.get('embeddings') is not None:
                local_vlm_features[c_id] = cached['embeddings']
                logging.info(f"  [FeatureCache] Client {c_id}: Loaded {cached['embeddings'].shape} embeddings from cache.")
            else:
                logging.info(f"  [FeatureCache] Client {c_id}: Cache exists but embeddings missing — will recompute if needed.")
        else:
            logging.info(f"  [FeatureCache] Client {c_id}: No cache found — will compute on first use if needed.")


    # Initialize Query Budget
    train_slice_num = np.zeros(client_num, dtype=int)
    query_num = np.zeros(client_num, dtype=int)

    # Trackers for Query Metrics
    total_id_samples_per_client = np.zeros(client_num, dtype=int)
    accumulated_relevant_found = np.zeros(client_num, dtype=int) 
    
    # Persistent storage for current metrics (Default to 0)
    client_query_stats = {c: {'QR': 0.0, 'AQR': 0.0} for c in range(client_num)}
    global_query_stats = {'Avg_QR': 0.0, 'Avg_AQR': 0.0}

    print(">>> Initializing Clients & Data...")
    print(f"Used csv -> {args.ood}")
    exp_logger.create_al_round_directory(0)
    for client_idx in range(client_num):
        # Generate Data
        d_train, d_unlab, d_test = generate_dataset(args.dataset, args.fl_method, client_idx, args)
        
        # Count Total ID Samples for AQR Denominator 
        if hasattr(d_train, 'labels'):
            # Fast path: direct label access (FedEMBED)
            total_relevant = sum(1 for lbl in d_train.labels if lbl < args.num_classes)
        else:
            total_relevant = 0
            for i in range(len(d_train)):
                _, meta = d_train[i]
                if meta.get('is_ood', 0) == 0: total_relevant += 1
        total_id_samples_per_client[client_idx] = total_relevant

        local_data['train'].append(d_train)
        local_data['unlabeled'].append(d_unlab)
        
        train_slice_num[client_idx] = len(d_train)
        
        # Calculate Budget
        if args.query_ratio > 0:
            query_num[client_idx] = int(len(d_train) * args.query_ratio)
        else:
            limit = int(np.ceil(0.85 * len(d_train)))
            query_num[client_idx] = min(args.budget, limit)

        # SELECTION STRATEGY (Warmup Round 0) 
        print(f"Client {client_idx}: Performing {args.warmup} selection...")
        curr_budget = query_num[client_idx]
        
        # 1. Force VLM scan if filter is ON, but respect specific strategy if provided
        if args.vlm_filter:
            if 'biomedclip' not in args.warmup:
                print(f"  [Info] --vlm_filter active: Defaulting warmup to 'biomedclip_random'")
                args.warmup = 'biomedclip_random'

        if args.vlm_filter or 'biomedclip' in args.warmup:
            # A. Run Inference
            res = run_vlm_gated_warmup(
                dataset=d_train, 
                dataset_name=args.dataset, 
                device='cuda',
                client_idx=client_idx,
                log_dir=exp_logger.base_dir,
                feature_cache_root=_vlm_emb_root,
            )
            
            # Save Embeddings to Cache and Disk
            if res.get('embeddings') is not None:
                local_vlm_features[client_idx] = res['embeddings']
                from utils.vlm_filter import FeatureCache
                fc = FeatureCache(args.dataset, client_idx, cache_root=_vlm_emb_root)
                emb = res['embeddings']
                emb_np = emb if isinstance(emb, np.ndarray) else emb.numpy()
                np.save(fc.npy_path, emb_np)
                logging.info(f"  -> Saved VLM embeddings to {fc.npy_path}")


            # Extract Data
            pool_candidates = np.array(res['pool']) 
            pool_scores = res['scores'][pool_candidates]
            pool_labels = res['pseudo_labels'][pool_candidates]
            
            # Save caches
            local_vlm_cache[client_idx] = res['mask']
            local_frozen_masks[client_idx] = res['mask']

            # SELECTION STRATEGIES
            if len(pool_candidates) < curr_budget:
                initial_candidates = pool_candidates.tolist()
            else:
                
                # 1. HIGHEST CONFIDENCE
                if 'highest' in args.warmup:
                    print("  -> Strategy: Highest ID Confidence")
                    sorted_indices = np.argsort(-pool_scores)
                    top_k_idx = sorted_indices[:curr_budget]
                    initial_candidates = pool_candidates[top_k_idx].tolist()

                # 2. STRATIFIED (Previously "Balanced Likelihood")
                # Balances Easy (High Conf) vs. Hard (Low Conf) samples
                elif 'stratified' in args.warmup:
                    print("  -> Strategy: Confidence Stratified (Likelihood Balance)")
                    sorted_indices = np.argsort(pool_scores)
                    sorted_pool = pool_candidates[sorted_indices]
                    
                    n_bins = 5
                    samples_per_bin = curr_budget // n_bins
                    initial_candidates = []
                    chunks = np.array_split(sorted_pool, n_bins)
                    
                    for chunk in chunks:
                        if len(chunk) > 0:
                            take_n = min(len(chunk), samples_per_bin)
                            selection = np.random.choice(chunk, take_n, replace=False)
                            initial_candidates.extend(selection)
                    
                    # Fill remainder randomly
                    if len(initial_candidates) < curr_budget:
                        remaining = sorted(list(set(pool_candidates) - set(initial_candidates)))
                        needed = curr_budget - len(initial_candidates)
                        initial_candidates.extend(remaining[:needed])

                # BALANCED (Class Balanced)
                # Balances Melanoma vs Nevus vs BCC etc. based on VLM prediction
                elif 'balanced' in args.warmup:
                    print("  -> Strategy: Class Balanced (Pseudo-Label Balance)")
                    unique_classes = np.unique(pool_labels)
                    n_classes = len(unique_classes)
                    target_per_class = curr_budget // n_classes
                    
                    initial_candidates = []
                    
                    # Iterate over each class the VLM found
                    for cls in unique_classes:
                        # Get all safe candidates belonging to this class
                        cls_indices = pool_candidates[pool_labels == cls]
                        
                        if len(cls_indices) > 0:
                            take_n = min(len(cls_indices), target_per_class)
                            selection = np.random.choice(cls_indices, take_n, replace=False)
                            initial_candidates.extend(selection)
                            
                    # Fill remainder randomly from the leftover pool
                    if len(initial_candidates) < curr_budget:
                        remaining = sorted(list(set(pool_candidates) - set(initial_candidates)))
                        needed = curr_budget - len(initial_candidates)
                        if len(remaining) >= needed:
                             initial_candidates.extend(np.random.choice(remaining, needed, replace=False))
                        else:
                             initial_candidates.extend(remaining)

                # 4. RANDOM (Default)
                else:
                    print("  -> Strategy: Random Sampling from Safe Pool")
                    temp_list = pool_candidates.tolist()
                    random.shuffle(temp_list)
                    initial_candidates = temp_list[:curr_budget]
            
            # Optional: Evaluation
            if args.vlm_eval:
                from utils.vlm_filter import evaluate_vlm_breakdown # Fixed import
                logging.info(f"  [Eval] Running Round 0 VLM Evaluation...")
                
                all_indices_set = set(range(len(d_train)))
                full_indices_list = list(range(len(d_train)))
                selected_set = set(initial_candidates)
                current_unlabeled = remaining_unlabeled(args, all_indices_set, selected_set)
                
                if args.vlm_filter:
                    current_mask = res['mask']
                else:
                    current_mask = np.ones(len(d_train), dtype=bool)

                metrics_0, raw_0 = evaluate_vlm_breakdown(
                    dataset_name=args.dataset,
                    vlm_adapter=None, 
                    full_dataset=d_train,
                    unlabeled_indices=full_indices_list,
                    query_indices=initial_candidates,
                    labeled_indices=initial_candidates,
                    device='cuda',
                    cached_features=res.get('embeddings'),
                    args=args
                )
                exp_logger.log_vlm_metrics(0, client_idx, metrics_0)
                
                # Robust Path logging for Round 0
                details_df = pd.DataFrame({
                    'Dataset_Index': full_indices_list,
                    'Path': [d_train.data_list.iloc[i, 0] for i in full_indices_list],
                    'True_Label': raw_0['t'],
                    'Pseudo_Label': raw_0['p'],
                    'ID_Soft_Score': raw_0['ids'],
                    'OOD_Soft_Score': raw_0['oods'],
                    'Entropy': raw_0['ent']
                })
                
                # Fixed typo: changed c_id to client_idx
                details_df.to_csv(os.path.join(exp_logger.current_al_dir, f"vlm_details_client{client_idx}.csv"), index=False)
                
                # Separate Pool Splits for Round 0
                mask = current_mask 
                id_pool_df = details_df[details_df['Dataset_Index'].map(lambda x: mask[x])]
                explore_df = details_df[details_df['Dataset_Index'].map(lambda x: not mask[x])]
                
                id_pool_df.to_csv(os.path.join(exp_logger.current_al_dir, f"pool_id_client{client_idx}.csv"), index=False)
                explore_df.to_csv(os.path.join(exp_logger.current_al_dir, f"pool_explore_client{client_idx}.csv"), index=False)
            
            # Force cleanup after Round 0
            torch.cuda.empty_cache()

        else:
            # Standard Random (No Filter)
            indices = list(range(len(d_train)))
            random.shuffle(indices)
            initial_candidates = indices[:curr_budget]
            local_vlm_cache[client_idx] = None
            
        # ASSIGNMENT & SAFETY CHECK (Ground Truth Filter) 
        valid_id_indices = []
        ood_indices = []
        _has_labels = hasattr(d_train, 'labels')
        
        for idx in initial_candidates:
            if _has_labels:
                is_ood = 1 if d_train.labels[idx] >= args.num_classes else 0
            else:
                _, sample_data = d_train[idx]
                is_ood = sample_data.get('is_ood', 0)
            
            if is_ood == 1:
                ood_indices.append(idx)
            else:
                valid_id_indices.append(idx)

        # 2. Update Accumulator & Calculate Round 0 Metrics
        accumulated_relevant_found[client_idx] += len(valid_id_indices)
        
        _qr = len(valid_id_indices) / len(initial_candidates) if len(initial_candidates) > 0 else 0.0
        _aqr = accumulated_relevant_found[client_idx] / total_id_samples_per_client[client_idx] if total_id_samples_per_client[client_idx] > 0 else 0.0
        
        # Save to persistent dict
        client_query_stats[client_idx]['QR'] = _qr
        client_query_stats[client_idx]['AQR'] = _aqr

        # 1. Labeled Set: Only Valid ID samples
        local_sets['labeled'].append(valid_id_indices)
        
        # 2. Discarded Set: OOD samples
        local_sets['discarded'].append(ood_indices)
        
        # 3. Unlabeled Set
        all_indices = set(range(len(d_train)))
        used_indices = set(initial_candidates)
        local_sets['unlabeled'].append(remaining_unlabeled(args, all_indices, used_indices))
        
        print(f"  -> Init Selected {len(initial_candidates)}. Valid ID: {len(valid_id_indices)}, OOD Discarded: {len(ood_indices)}")

        # Loaders
        if args.dataset == 'FedEMBED':
            train_loader = DataLoader(d_train, batch_size=args.batch_size, 
                                sampler=make_subset_sampler(args, local_sets['labeled'][client_idx]), 
                                num_workers=0, pin_memory=False, worker_init_fn=loader_worker_init(args, 0))
            test_loader = DataLoader(d_test, batch_size=args.batch_size, shuffle=False, 
                               num_workers=0, pin_memory=False, worker_init_fn=loader_worker_init(args, 0))
        else:
            train_loader = DataLoader(d_train, batch_size=args.batch_size, 
                                sampler=make_subset_sampler(args, local_sets['labeled'][client_idx]), 
                                num_workers=4, pin_memory=True, worker_init_fn=loader_worker_init(args, 4))
            test_loader = DataLoader(d_test, batch_size=args.batch_size, shuffle=False, 
                               num_workers=4, pin_memory=True, worker_init_fn=loader_worker_init(args, 4))
        
        local_data['loaders']['train'].append(train_loader)
        local_data['loaders']['test'].append(test_loader)

        # Model
        local_models.append(Model(num_classes=args.num_classes).cuda())
        local_scalers.append(torch.amp.GradScaler(enabled=args.mixed_precision))

    # Calculate Global Stats for Round 0 (Robust Length-Based) 
    qrs = []
    global_total_labeled_id = 0
    global_total_dataset_id = 0

    for c in range(client_num):
        # 1. Get Physical Counts
        n_labeled_id = len(local_sets['labeled'][c])      # Total ID found (Accumulated)
        n_total_selected = query_num[c]                   # Initial budget spent
        n_dataset_id = total_id_samples_per_client[c]     # Total ID in existence (Denominator)

        global_total_labeled_id += n_labeled_id
        global_total_dataset_id += n_dataset_id

        # 2. Calculate
        # QR / QP (Batch Precision) = ID Selected / Total Selected
        qr = n_labeled_id / n_total_selected if n_total_selected > 0 else 0.0
        qrs.append(qr)

    global_query_stats['Avg_QR'] = np.mean(qrs)
    global_query_stats['Avg_AQR'] = global_total_labeled_id / global_total_dataset_id if global_total_dataset_id > 0 else 0.0
    print(f">>> Round 0 Query Metrics: Avg_QR={global_query_stats['Avg_QR']:.4f}, Avg_AQR={global_query_stats['Avg_AQR']:.4f}")

    # LOG INITIAL STATE (Round 0) 
    print(">>> Logging Initial Data Distribution...")

    for c_id in range(client_num):
        true_len = len(local_data['train'][c_id])
        
        # 1. All Data (Stats for the entire local dataset)
        total_indices = list(range(true_len))
        all_counts, all_pur = get_extended_class_counts(local_data['train'][c_id], total_indices, args.num_classes)
        _num_id = getattr(args, 'num_id_classes', args.num_classes)
        all_counts, all_pur = get_extended_class_counts(local_data['train'][c_id], total_indices, _num_id)
        exp_logger.log_client_state(c_id, "All_Data", all_counts, purity=all_pur, true_total=true_len)

        # 2. Labeled (Effective Selection: Valid ID + Discarded OOD)
        # This combined set represents the total budget spent (e.g., 500 queries)
        combined_labeled = local_sets['labeled'][c_id] + local_sets['discarded'][c_id]
        lbl_counts, lbl_pur = get_extended_class_counts(local_data['train'][c_id], combined_labeled, args.num_classes)
        exp_logger.log_client_state(c_id, "Labeled", lbl_counts, purity=lbl_pur)
        
        # 3. Unlabeled (Remaining Pool)
        # unlbl_counts, unlbl_pur = get_extended_class_counts(local_data['train'][c_id], local_sets['unlabeled'][c_id], args.num_classes)
        unlbl_counts, unlbl_pur = get_extended_class_counts(local_data['train'][c_id], local_sets['unlabeled'][c_id], _num_id)
        exp_logger.log_client_state(c_id, "Unlabeled", unlbl_counts, purity=unlbl_pur)


    run_active_learning_loop(
        args=args,
        Model=Model,
        client_num=client_num,
        SUBSET=SUBSET,
        exp_logger=exp_logger,
        experiment_start_time=experiment_start_time,
        local_data=local_data,
        local_sets=local_sets,
        local_scalers=local_scalers,
        local_vlm_cache=local_vlm_cache,
        local_vlm_features=local_vlm_features,
        local_vlm_adapters=local_vlm_adapters,
        local_vlm_optimizers=local_vlm_optimizers,
        train_slice_num=train_slice_num,
        query_num=query_num,
        total_id_samples_per_client=total_id_samples_per_client,
        client_query_stats=client_query_stats,
        global_query_stats=global_query_stats,
        get_model_probabilities=get_model_probabilities,
        get_extended_class_counts=get_extended_class_counts,
        run_vlm_gated_warmup=run_vlm_gated_warmup,
        save_learnable_vectors=save_learnable_vectors,
        vlm_feature_cache_root=vlm_feature_cache_root,
    )

    print("\n>>> Starting Visualization...")
    visualizer = ExperimentVisualizer(exp_logger.base_dir, args.al_method)
    visualizer.generate_all(args.num_classes)
    print("\n>>> Plotting Completed.")


if __name__ == '__main__':
    main()
