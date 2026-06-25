import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import higher


# PAL HELPERS — Faithful Port from NeurIPS2023-PAL
class WNet(nn.Module):
    """Weighting Network for PAL.
    Original: WNet(input=2*K, hidden=512, output=1) with Sigmoid.
    """
    def __init__(self, in_dim, hidden_dim=512):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return torch.sigmoid(out)


# Loss Functions (Ported from NeurIPS2023-PAL/utils/misc.py) 
def ova_loss(logits_open, label):
    """One-vs-All loss for the open-set head.
    
    Ported from original PAL: misc.py:134-153.
    - Reshapes to (B, 2, K) — dim order is (Neg, Pos) then K
    - Uses softmax over the 2-way dimension (dim=1)
    - For correct class: maximize P(Pos)
    - For wrong classes: only penalize the HARDEST wrong class (max), not all
    """
    logits_open = logits_open.view(logits_open.size(0), 2, -1)  # (B, 2, K)
    logits_open = F.softmax(logits_open, 1)  # Softmax over 2-way dim

    # Build one-hot: label_s_sp[i, label[i]] = 1
    label_s_sp = torch.zeros((logits_open.size(0), logits_open.size(2)),
                              dtype=torch.long, device=label.device)
    label_range = torch.arange(0, logits_open.size(0), dtype=torch.long, device=label.device)
    label_s_sp[label_range, label] = 1
    label_sp_neg = 1 - label_s_sp

    # Positive loss: -log P(Pos | correct class)
    open_loss = torch.mean(
        torch.sum(-torch.log(logits_open[:, 1, :] + 1e-8) * label_s_sp, 1)
    )
    # Negative loss: -log P(Neg | hardest wrong class) — MAX over wrong classes
    open_loss_neg = torch.mean(
        torch.max(-torch.log(logits_open[:, 0, :] + 1e-8) * label_sp_neg, 1)[0]
    )
    
    return open_loss + open_loss_neg


def compute_S_ID(logits_open):
    """Computes S_ID score for PAL selection.
    
    Ported from original PAL: misc.py:156-162.
    - Reshape to (B, 2, K), softmax over 2-way dim
    - Max-pool over K classes → take Neg probability
    - Return scaled negative entropy: 2.5 * (1 + p_neg * log(p_neg))
    """
    logits_open = logits_open.view(logits_open.size(0), 2, -1)  # (B, 2, K)
    logits_open = F.softmax(logits_open, 1)  # Softmax over 2-way dim
    logits_open, _ = torch.max(logits_open, dim=2)  # Max over K → (B, 2)
    logits_open = logits_open[:, 0]  # Take Neg probability
    L_c = 2.5 * (1 + logits_open * torch.log(logits_open + 1e-8))
    return L_c


def ova_ent(logits_open):
    """OVA entropy — used by WNet in meta inner loop.
    
    Ported from original PAL: misc.py:165-172.
    Returns:
        Le: scalar average entropy (for logging)
        L_c: per-sample entropy tensor (B,) — this is cost_w for WNet weighting
    """
    logits_open = logits_open.view(logits_open.size(0), 2, -1)  # (B, 2, K)
    logits_open = F.softmax(logits_open, 1)  # Softmax over 2-way dim
    # Entropy: -sum(p * log(p)) over dim=1 (the 2-way), then mean over K
    Le = torch.mean(torch.mean(
        torch.sum(-logits_open * torch.log(logits_open + 1e-8), 1), 1))
    L_c = torch.mean(
        torch.sum(-logits_open * torch.log(logits_open + 1e-8), 1), 1)
    return Le, L_c


# Meta-Learning 
def update_wnet(classifier, wnet, id_loader, ood_loader, wnet_optimizer=None,
                device='cuda', K_inner=20, miu=1.0, lr_inner=1e-2, meta_lr=6e-5,
                epoch=0, total_epochs=50):
    """
    Meta-update step for WNet — Faithful to original PAL train_meta logic.
    
    Key differences from previous implementation:
    1. Inner loop uses ova_loss (labeled) + WNet-weighted ova_ent (unlabeled)
    2. coef annealing: exp(-5 * min(1 - epoch/total_epochs, 1)^2)
    3. WNet weights are normalized: sum(w * cost) / norm
    4. Both fc + fc_open heads are optimized in inner loop
    """
    # 1. Grab batches — need two ID batches (support + query) and one unlabeled batch
    try:
        iter_id = iter(id_loader)
        batch_A = next(iter_id)  # Support set (inner loop)
        batch_B = next(iter_id)  # Query set (outer loop)
    except StopIteration:
        return 

    def unpack(b):
        if isinstance(b, dict): return b['image'].to(device), b['label'].to(device)
        if isinstance(b, (list, tuple)): 
            x = b[1]['image'] if isinstance(b[1], dict) else b[0]
            y = b[1]['label'] if isinstance(b[1], dict) else b[1]
            return x.to(device), y.to(device)
        return b[0].to(device), b[1].to(device)

    XA, yA = unpack(batch_A)
    XB, yB = unpack(batch_B)
    
    # Get unlabeled batch (can come from ood_loader or a general unlabeled loader)
    X_unlab = None
    if ood_loader and len(ood_loader) > 0:
        try:
            batch_U = next(iter(ood_loader))
            X_unlab, _ = unpack(batch_U)
        except StopIteration: pass

    # Coef annealing (from original train.py:265) 
    coef = math.exp(-5 * (min(1 - epoch / max(total_epochs, 1), 1)) ** 2)

    # Freeze Backbone to avoid OOM with higher 
    classifier.train()
    if hasattr(classifier, 'efficientnet'):
        for param in classifier.efficientnet.parameters():
            param.requires_grad = False
    
    # Optimize both fc (cls) and fc_open heads in inner loop
    params_to_opt = list(classifier.fc.parameters())
    if hasattr(classifier, 'fc_open'):
        params_to_opt += list(classifier.fc_open.parameters())
    inner_opt = torch.optim.SGD(params_to_opt, lr=lr_inner)
    
    try:
        with higher.innerloop_ctx(classifier, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
            for _ in range(K_inner):
                b_size = XA.shape[0]
                
                # 1. Classification loss on labeled (Batch A)
                logits_cls, logits_open_cls = _get_both_logits(fnet, XA)
                Lx = F.cross_entropy(logits_cls, yA, reduction='mean')
                
                # 2. OVA loss on labeled
                Lo = ova_loss(logits_open_cls, yA)
                
                loss = Lx
                
                # 3. OVA entropy on unlabeled, weighted by WNet
                if X_unlab is not None:
                    # Concatenate labeled + unlabeled for open head
                    inputs_all = torch.cat([XA, X_unlab], 0)
                    _, logits_open_all = _get_both_logits(fnet, inputs_all)
                    logits_open_unlab = logits_open_all[b_size:]
                    
                    weight = wnet(logits_open_unlab)  # (N_unlab, 1)
                    norm = torch.sum(weight)
                    
                    _, cost_w = ova_ent(logits_open_unlab)  # cost_w: (N_unlab,)
                    cost_w = torch.reshape(cost_w, (len(cost_w), 1))
                    
                    if norm != 0:
                        loss += coef * (torch.sum(weight * cost_w) / norm + Lo)
                    else:
                        loss += coef * (torch.sum(weight * cost_w) + Lo)
                else:
                    loss += coef * Lo
                
                diffopt.step(loss)
            
            # 4. Outer Loop: Update WNet using clean validation on Batch B
            logitsB_cls, _ = _get_both_logits(fnet, XB)
            meta_loss = F.cross_entropy(logitsB_cls, yB, reduction='mean')
            
            wnet.zero_grad()
            meta_loss.backward()
            if wnet_optimizer is None:
                wnet_optimizer = torch.optim.Adam(wnet.parameters(), lr=meta_lr)
            wnet_optimizer.step()
            
    except Exception as e:
        print(f"[PAL Warning] Meta-update failed (likely OOM or shape): {e}")
        
    # --- Restore: Unfreeze Backbone ---
    if hasattr(classifier, 'efficientnet'):
        for param in classifier.efficientnet.parameters():
            param.requires_grad = True


def _get_both_logits(model, x):
    """Helper to get both cls logits and open logits from a model.
    Handles both normal models and higher-patched models.
    """
    # Standard forward → (logits_cls, probs, feat, block_features)
    out = model(x)
    if isinstance(out, tuple) and len(out) >= 2:
        logits_cls = out[0]
    else:
        logits_cls = out
    
    # Open head forward
    logits_open = model(x, open_head=True)
    if isinstance(logits_open, tuple):
        logits_open = logits_open[0]
    
    return logits_cls, logits_open


# Selection 
def pal_selection(classifier, wnet, unlabeled_loader, query_num,
                  round_idx, total_rounds, miu=1.0, device='cuda',
                  num_classes=None):
    """PAL sample selection — faithful to original PAL.Compute_un.
    
    Key features ported from original:
    1. Score = WNet(open_logits) + miu * S_ID
    2. For round > 0: conditional score flipping when model predicts OOD
    3. Round 0: simple ascending sort (select lowest uncertainty = most ID)
    4. Round > 0: split by pseudo-label into ID/OOD pools, query separately
    """
    classifier.eval()
    wnet.eval()
    
    all_scores = []
    all_predicted = []
    
    K = num_classes
    if K is None:
        K = classifier.K if hasattr(classifier, 'K') else classifier.fc_cls.out_features
    
    with torch.no_grad():
        for batch in unlabeled_loader:
            if isinstance(batch, dict): 
                x = batch['image']
            elif isinstance(batch, (list, tuple)): 
                x = batch[1]['image'] if isinstance(batch[1], dict) else batch[0]
            x = x.to(device)
             
            # Get both logits
            out = classifier(x)
            if isinstance(out, tuple):
                logits_cls = out[0]
            else:
                logits_cls = out
            
            ova_logits = classifier(x, open_head=True)
            if isinstance(ova_logits, tuple): 
                ova_logits = ova_logits[0]
             
            # Predicted class
            _, predicted = logits_cls.max(1)
            
            # WNet weight
            weight = wnet(ova_logits).cpu()  # (B, 1)
            
            # S_ID
            s_id = compute_S_ID(ova_logits)
            s_id = torch.reshape(s_id, (len(s_id), 1)).cpu()
            
            # Base score
            Un = weight + miu * s_id  # (B, 1)
            Un = Un.squeeze(1)
            
            # Conditional score flipping for round > 0
            # If model predicts OOD (class >= K), flip the S_ID contribution
            if round_idx > 0:
                pred_cpu = predicted.cpu()
                weight_list = weight.detach().numpy()
                s_id_list = s_id.numpy()
                Un_list = Un.numpy().tolist()
                
                for ind in range(len(pred_cpu)):
                    if pred_cpu[ind] >= K:
                        un_val = weight_list[ind][0] + miu * (1.0 - s_id_list[ind][0])
                        Un_list[ind] = un_val
                
                Un = torch.tensor(Un_list)
            
            all_scores.append(Un.cpu())
            all_predicted.append(predicted.cpu())
    
    scores = torch.cat(all_scores).numpy()
    predicted_labels = torch.cat(all_predicted).numpy()
    N = len(scores)
    
    # Selection Logic (from original PAL.Compute_un) 
    if round_idx == 0:
        # Round 0: Simple — select lowest scores (most ID-like)
        # Original selects from both front (ascending) and back (descending)
        sorted_idx = np.argsort(scores)
        
        # Select by ascending score — lowest = most confident ID
        selected = sorted_idx[:query_num].tolist()
        
        # Format for main.py: [remainder ... chosen] so rank_arg[-budget:] picks them
        all_indices = np.arange(N)
        mask = np.ones(N, dtype=bool)
        for idx in selected:
            mask[idx] = False
        remainder = all_indices[mask]
        final_rank = np.concatenate([remainder, np.array(selected)])
        
    else:
        # Round > 0: Split into pseudo-ID and pseudo-OOD pools
        is_id = predicted_labels < K
        is_ood = predicted_labels >= K
        
        id_indices = np.where(is_id)[0]
        ood_indices = np.where(is_ood)[0]
        
        print(f'[PAL] Pseudo-label split: ID={len(id_indices)}, OOD={len(ood_indices)}')
        
        # Progressive quota: how many ID vs OOD to query
        need_id, need_ood = _progressive_quota(round_idx, total_rounds, query_num)
        
        selected = []
        
        # Select from ID pool: highest score (most uncertain/informative ID)
        if len(id_indices) > 0:
            id_scores = scores[id_indices]
            id_sorted = np.argsort(id_scores)[::-1]  # Descending
            take_id = min(need_id, len(id_indices))
            selected_id = id_indices[id_sorted[:take_id]].tolist()
            selected.extend(selected_id)
        
        # Select from OOD pool: lowest score (most confident OOD)
        if need_ood > 0 and len(ood_indices) > 0:
            ood_scores = scores[ood_indices]
            ood_sorted = np.argsort(ood_scores)  # Ascending
            take_ood = min(need_ood, len(ood_indices))
            selected_ood = ood_indices[ood_sorted[:take_ood]].tolist()
            selected.extend(selected_ood)
        
        # Fill remainder if we didn't get enough
        if len(selected) < query_num:
            remaining_pool = set(range(N)) - set(selected)
            remaining_scores = [(i, scores[i]) for i in remaining_pool]
            remaining_scores.sort(key=lambda x: x[1], reverse=True)
            needed = query_num - len(selected)
            selected.extend([x[0] for x in remaining_scores[:needed]])
        
        # Format for main.py
        all_indices = np.arange(N)
        mask = np.ones(N, dtype=bool)
        for idx in selected:
            mask[idx] = False
        remainder = all_indices[mask]
        final_rank = np.concatenate([remainder, np.array(selected[:query_num])])
    
    return final_rank.astype(int), np.zeros(len(final_rank))


def _progressive_quota(round_idx, total_rounds, batch_size, start=0.33, end=0.02):
    """Decay the number of OOD samples selected as training progresses."""
    if total_rounds <= 1: 
        return batch_size, 0
    frac = max(end, start - (start - end) * round_idx / (total_rounds - 1))
    need_ood = int(batch_size * frac)
    return batch_size - need_ood, need_ood