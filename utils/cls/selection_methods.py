from utils.cls import entropy_helper, feal_helper, pal_helper, lfosa_helper, openpath_helper
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from data.sampler import SubsetSequentialSampler
import random
import numpy as np
from tqdm import tqdm
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import torch.distributions as dist
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import math
import copy
import higher

def query_samples(al_method, global_model, local_model, data_unlabeled, unlabeled_set, query_num, args, **kwargs):
    """
    Returns:
        indices (list/array): Indices relative to unlabeled_set, sorted by priority (Best first).
        scores (list/array): The score associated with each index.
    """
    if len(unlabeled_set) == 0: return [], np.array([])
    _nw = 0 if getattr(args, 'dataset', '') == 'FedEMBED' else 4
    unlabeled_loader = DataLoader(
        dataset=data_unlabeled,
        batch_size=args.batch_size,
        sampler=SubsetSequentialSampler(unlabeled_set), 
        num_workers=_nw, pin_memory=(_nw > 0)
    )

    if al_method == 'Random':
        indices = list(range(len(unlabeled_set)))
        random.shuffle(indices)
        scores = np.zeros(len(indices)) 
        return indices, scores

    elif al_method == 'Entropy':
        uncertainty = entropy_helper.get_entropy(global_model, unlabeled_loader)
        # Sort: [Low Entropy ... High Entropy]
        # Main loop picks last ones: rank_arg[-query_num:]
        # So high entropy (uncertain) will be selected.
        sorted_indices = torch.argsort(uncertainty, descending=False).cpu().numpy()
        sorted_scores = uncertainty[sorted_indices].cpu().numpy()
        return sorted_indices, sorted_scores

    elif al_method == 'FEAL':
        g_data, l_data, u_dis, l_features = feal_helper.fl_duc(global_model, local_model, unlabeled_loader)
        
        u_dis_norm = (u_dis - u_dis.min()) / (u_dis.max() - u_dis.min() + 1e-6)
        total_uncertainty = u_dis_norm * (g_data + l_data)
        
        # Sort Descending for relaxation input (Highest Uncertainty first)
        u_rank_arg = torch.argsort(total_uncertainty, descending=True)
        
        # Select best using Diversity Relaxation
        chosen_idx = feal_helper.relaxation(u_rank_arg, l_features, args, query_num)
        
        # Re-construct indices: [Remainder ... Chosen] so that query_samples logic selects last ones
        all_indices = set(range(len(unlabeled_set)))
        remain_idx = sorted(list(all_indices - set(chosen_idx)))
        
        # We put chosen at the end because main.py does: rank_arg[-query_num:]
        final_indices = remain_idx + chosen_idx
        
        chosen_scores = total_uncertainty[chosen_idx].cpu().numpy()
        remain_scores = total_uncertainty[remain_idx].cpu().numpy()
        final_scores = np.concatenate([remain_scores, chosen_scores])
        
        return final_indices, final_scores

    elif al_method == 'PAL':
        labeled_set = kwargs.get('labeled_set', [])
        discarded_set = kwargs.get('discarded_set', [])
        dataset_source = kwargs.get('dataset_source', None)
        wnet = kwargs.get('wnet', None)
        
        if wnet is None: raise ValueError("PAL requires 'wnet' kwarg")
        
        # Loaders
        labeled_loader = DataLoader(dataset_source, batch_size=args.batch_size, sampler=SubsetRandomSampler(labeled_set), num_workers=_nw)
        ood_loader = None
        if len(discarded_set) > 0:
            ood_loader = DataLoader(dataset_source, batch_size=args.batch_size, sampler=SubsetRandomSampler(discarded_set), num_workers=_nw)
            
        wnet_optimizer = kwargs.get('wnet_optimizer', None)
        
        # Meta-Update (pass epoch info for coef annealing)
        current_round = getattr(args, 'current_al_round', 0)
        total_epochs = getattr(args, 'max_round', 50)
        pal_helper.update_wnet(
            local_model, wnet, labeled_loader, ood_loader, 
            wnet_optimizer=wnet_optimizer,
            epoch=current_round, total_epochs=total_epochs
        )
        
        # Selection
        rank_arg, scores = pal_helper.pal_selection(
            local_model, wnet, unlabeled_loader, query_num,
            round_idx=current_round,
            total_rounds=args.al_round,
            num_classes=args.num_classes
        )
        return rank_arg, scores

    elif al_method == 'LfOSA':
        # 1. Compute GMM Scores
        scores = lfosa_helper.compute_lfosa_gmm_score(local_model, unlabeled_loader)
        
        # 2. Sort Ascending [Low Score ... High Score]
        # main.py selects LAST elements (Highest scores).
        sorted_indices = np.argsort(scores) 
        sorted_scores = scores[sorted_indices]
        
        return sorted_indices, sorted_scores

    elif al_method == 'OpenPath':
        labeled_set = kwargs.get('labeled_set', [])
        dataset_source = kwargs.get('dataset_source', None)
        
        if dataset_source is None: raise ValueError("OpenPath requires 'dataset_source'")
        
        chosen_local = openpath_helper.openpath_query(
            global_model, data_unlabeled, unlabeled_set, query_num,
            labeled_set=labeled_set,
            dataset_source=dataset_source,
            id_ratio=getattr(args, 'openpath_id_ratio', 0.6),
            num_classes=args.num_classes,
            seed=getattr(args, 'seed', 42),
            dataset_name=getattr(args, 'dataset', 'FedISIC')
        )
        
        # Format: [remainder ... chosen] for main.py's rank_arg[-budget:]
        all_indices = set(range(len(unlabeled_set)))
        chosen_set = set(chosen_local)
        remain_idx = sorted(list(all_indices - chosen_set))
        
        rank_arg = remain_idx + list(chosen_local)
        scores = np.zeros(len(rank_arg))
        
        return rank_arg, scores

    else:
        raise ValueError(f"Unknown AL Method: {al_method}")