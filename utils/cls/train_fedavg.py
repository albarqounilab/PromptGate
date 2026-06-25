import torch
import torch.nn as nn
from torch.amp import autocast
from utils.loss_func import EDL_Loss
from itertools import cycle
import numpy as np

import torch.nn.functional as F

def train(round_idx, client_idx, model, dataloader, optimizer, num_per_class, args, scaler=None):
    """
    Trains the model for one local epoch/round using Mixed Precision if enabled.
    """
    model.train()

    # Convert num_per_class to tensor on correct device
    if not isinstance(num_per_class, torch.Tensor):
        import numpy as np
        npc = torch.as_tensor(np.array(num_per_class), dtype=torch.float32, device="cuda")
    else:
        npc = num_per_class.to(dtype=torch.float32, device="cuda")


    # Define Criterion
    if args.dataset == 'FedISIC':
        if args.al_method == 'FEAL':
            prior = num_per_class / num_per_class.sum()
            criterion = EDL_Loss(prior=prior, kl_weight=args.kl_weight, annealing_step=args.annealing_step)
        else:
            weight = 2 - (num_per_class / num_per_class.sum())
            criterion = nn.CrossEntropyLoss(weight=weight,ignore_index=-1)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    if args.dataset == 'FedEMBED':
        # FedEMBED: Train on all 5 classes (0-3 density, 4 artefact) with class weighting
        # NO ignore_index - we explicitly train an artefact detector (5th class)
        if args.al_method == 'FEAL':
            prior = npc / npc.sum()
            criterion = EDL_Loss(prior=prior, kl_weight=args.kl_weight, annealing_step=args.annealing_step)
        else:
            weight = 2 - (npc / npc.sum())
            criterion = nn.CrossEntropyLoss(weight=weight)


    epoch_loss = []
    
    for batch_idx, (_, data) in enumerate(dataloader):
        optimizer.zero_grad()
        
        image = data['image'].cuda()
        label = data['label'].cuda()
        
        # MIXED PRECISION FORWARD PASS 
        # We wrap the forward pass and loss calculation in autocast.
        # This allows PyTorch to automatically switch to FP16 for compatible ops (convs, matmuls).
        with autocast('cuda', enabled=args.mixed_precision):
            outputs = model(image)
            logit = outputs[0]
            
            if args.al_method == 'FEAL':
                loss = criterion(logit, label, round_idx)
            else:
                loss = criterion(logit, label)
        
        # BACKWARD PASS 
        if args.mixed_precision and scaler is not None:
            # Scales loss to avoid underflow in FP16 gradients
            scaler.scale(loss).backward()
            
            # Unscales gradients and steps optimizer
            scaler.step(optimizer)
            
            # Updates the scale factor for next iteration
            scaler.update()
        else:
            # Standard FP32 backward
            loss.backward()
            optimizer.step()
        
        epoch_loss.append(loss.item())

    return sum(epoch_loss) / len(epoch_loss) if len(epoch_loss) > 0 else 0.0

def train_lfosa(round_idx, client_idx, model, dataloader, optimizer, args, scaler=None, ood_dataloader=None):
    model.train()
    
    criterion_cls = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_detect = nn.CrossEntropyLoss()

    epoch_loss = []
    
    # STRATEGY: Iterate enough to cover OOD data 
    # If ID data is small, cycle it to match OOD size (or a minimum number of steps)
    # This effectively trains the model for 'more epochs' on the ID data relative to the FL loop
    
    if ood_dataloader:
        len_id = len(dataloader)
        len_ood = len(ood_dataloader)
        num_steps = max(len_id, len_ood) # Train for whichever is longer
        
        id_iter = cycle(dataloader)
        ood_iter = cycle(ood_dataloader)
    else:
        num_steps = len(dataloader)
        id_iter = iter(dataloader)
        ood_iter = None

    for _ in range(num_steps):
        optimizer.zero_grad()
        
        # Get ID Batch
        try:
            _, data_id = next(id_iter)
        except StopIteration:
            id_iter = cycle(dataloader)
            _, data_id = next(id_iter)
            
        img_id = data_id['image'].cuda()
        lbl_id = data_id['label'].cuda()
        
        # Get OOD Batch
        img_ood = None
        lbl_ood = None
        if ood_iter:
            _, data_ood = next(ood_iter)
            img_ood = data_ood['image'].cuda()
            lbl_ood = torch.full((img_ood.size(0),), args.num_classes, device='cuda', dtype=torch.long)

        with autocast('cuda', enabled=args.mixed_precision):
            # A. Standard Cls (ID Only)
            logit_cls = model(img_id)[0]
            loss_cls = criterion_cls(logit_cls, lbl_id)
            
            # B. Detection Head (ID) -> Target: Original Label
            logit_det_id = model(img_id, detect=True)
            # Temperature scaling for ID data (known_T = 0.5) to sharpen predictions
            logit_det_id = logit_det_id / 0.5
            loss_det_id = criterion_detect(logit_det_id, lbl_id)
            
            total_loss = loss_cls + loss_det_id

            # C. Detection Head (OOD) -> Target: Class K
            if img_ood is not None:
                logit_det_ood = model(img_ood, detect=True)
                # Temperature scaling for OOD data (unknown_T = 2.0) to soften predictions
                logit_det_ood = logit_det_ood / 2.0
                loss_det_ood = criterion_detect(logit_det_ood, lbl_ood)
                total_loss += loss_det_ood

        if args.mixed_precision and scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        
        epoch_loss.append(total_loss.item())

    return sum(epoch_loss) / len(epoch_loss) if len(epoch_loss) > 0 else 0.0


def train_pal(round_idx, client_idx, model, dataloader, optimizer, args, scaler=None, ood_dataloader=None):
    """
    PAL Training — Faithful to original NeurIPS2023-PAL train_cls.
    
    Uses ova_loss (One-vs-All loss) on the open head instead of per-class
    binary CrossEntropyLoss. The loss is:
        loss = CE(cls_logits, labels) + ova_loss(open_logits, labels)
    
    For OOD data: uses ova_ent (OVA entropy) as regularization.
    """
    from utils.cls.pal_helper import ova_loss, ova_ent
    
    model.train()
    
    criterion_cls = nn.CrossEntropyLoss(ignore_index=-1)
    
    epoch_loss = []
    
    # --- Iterator Setup ---
    if ood_dataloader:
        len_id = len(dataloader)
        len_ood = len(ood_dataloader)
        num_steps = max(len_id, len_ood)
        id_iter = cycle(dataloader)
        ood_iter = cycle(ood_dataloader)
    else:
        num_steps = len(dataloader)
        id_iter = iter(dataloader)
        ood_iter = None
    
    for _ in range(num_steps):
        optimizer.zero_grad()
        
        # Get ID Batch
        try:
            _, data_id = next(id_iter)
        except StopIteration:
            id_iter = cycle(dataloader)
            _, data_id = next(id_iter)
            
        img_id = data_id['image'].cuda()
        lbl_id = data_id['label'].cuda()
        
        # Get OOD Batch
        img_ood = None
        if ood_iter:
            _, data_ood = next(ood_iter)
            img_ood = data_ood['image'].cuda()
        
        with autocast('cuda', enabled=args.mixed_precision):
            # A. Standard Classifier Loss (CE on cls head)
            logits_cls = model(img_id)[0]
            Lx = criterion_cls(logits_cls, lbl_id)
            
            # B. OVA Loss on Open Head (faithful to original PAL)
            out_open_id = model(img_id, open_head=True)  # (B, 2K)
            Lo = ova_loss(out_open_id, lbl_id)
            
            total_loss = Lx + Lo
            
            # C. OOD Regularization: OVA entropy on OOD data
            if img_ood is not None:
                out_open_ood = model(img_ood, open_head=True)
                Le_ood, _ = ova_ent(out_open_ood)
                total_loss += 0.5 * Le_ood
        
        # Backward
        if args.mixed_precision and scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()
        
        epoch_loss.append(total_loss.item())
    
    return sum(epoch_loss) / len(epoch_loss) if len(epoch_loss) > 0 else 0.0
