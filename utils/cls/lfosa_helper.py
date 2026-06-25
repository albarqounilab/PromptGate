import torch
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
from data.sampler import SubsetSequentialSampler
from sklearn.mixture import GaussianMixture
from collections import defaultdict

def compute_lfosa_gmm_score(model, dataloader, device='cuda'):
    """
    Faithful LfOSA Selection:
    1. Filter samples predicted as ID (class < K) by the detection head.
    2. Fit 2-component GMM per class on Maximum Activation Values (RAW LOGITS).
    3. Score = 1 - Prob(Confident). (Higher Score = More Uncertain/Novel).
    """
    model.eval()
    K = model.K  # Number of known classes
    
    mavs = []       
    preds = []      
    global_idxs = [] 
    
    current_idx = 0
    
    # 1. Collect predictions
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict): x = batch['image']
            elif isinstance(batch, (list, tuple)): 
                x = batch[1]['image'] if isinstance(batch[1], dict) else batch[0]
            x = x.to(device)
            
            # Use Detect Head
            logits = model(x, detect=True)
            
            # Original LfOSA uses RAW LOGITS (max activation value), not softmax
            max_logits, predicted = logits.max(dim=1)
            
            for i in range(x.size(0)):
                pl = predicted[i].item()
                v_value = max_logits[i].item()
                
                # Only analyze samples predicted as ID (< K)
                # Samples predicted as OOD (class K) are ignored/discarded
                if pl < K:
                    mavs.append(v_value)
                    preds.append(pl)
                    global_idxs.append(current_idx)
                
                current_idx += 1

    N = current_idx
    final_scores = np.zeros(N, dtype=float) 
    
    # 2. Group by predicted class
    groups = defaultdict(list)
    for m, p, idx in zip(mavs, preds, global_idxs):
        groups[p].append((m, idx))
        
    # 3. Fit GMM per class
    for cls_k, pairs in groups.items():
        if len(pairs) < 2:
            # Not enough data to model distribution -> Assume Uncertain (High Score)
            for _, idx in pairs:
                final_scores[idx] = 1.0 
            continue
            
        arr = np.array([m for m, _ in pairs]).reshape(-1, 1)
        indices = [idx for _, idx in pairs]
        
        try:
            # Match original LfOSA GMM: max_iter=10, tol=1e-2, reg_covar=5e-4
            gmm = GaussianMixture(n_components=2, random_state=0, max_iter=10, tol=1e-2, reg_covar=5e-4).fit(arr)
            
            # The component with the higher mean represents "Confident" samples
            hi_comp_idx = np.argmax(gmm.means_.flatten())
            
            # Posterior prob of being in the "Confident" cluster
            posteriors = gmm.predict_proba(arr)
            prob_known = posteriors[:, hi_comp_idx]
            
            # We want to query UNCERTAIN samples (Low prob_known)
            scores = 1.0 - prob_known
            
            for score, idx in zip(scores, indices):
                final_scores[idx] = score
                
        except Exception as e:
            # Fallback if fit fails
            for _, idx in pairs:
                final_scores[idx] = 1.0

    return final_scores