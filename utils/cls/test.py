import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, roc_auc_score
import torch.nn.functional as F

def test(dataset, model, dataloader, client_idx):
    model.eval()
    
    pred_list = []
    label_list = []
    
    with torch.no_grad():
        for _, (_, data) in enumerate(dataloader):
            image = data['image'].cuda()
            label = data['label'].cuda()
            
            # Forward pass
            logit = model(image)[0]
            pred = torch.argmax(logit, dim=1)
            
            pred_list.append(pred.cpu().numpy())
            label_list.append(label.cpu().numpy())

    # Concatenate all batches
    if len(pred_list) > 0:
        pred_list = np.concatenate(pred_list)
        label_list = np.concatenate(label_list)
    else:
        # Edge case: Empty dataloader
        return {
            'Accuracy': 0.0, 'Balanced_Acc': 0.0, 
            'Precision': 0.0, 'Recall': 0.0, 'F1_Score': 0.0
        }

    # Compute Metrics
    metrics = {
        'Accuracy': accuracy_score(label_list, pred_list),
        'Balanced_Acc': balanced_accuracy_score(label_list, pred_list),
        'Precision': precision_score(label_list, pred_list, average='macro', zero_division=0),
        'Recall': recall_score(label_list, pred_list, average='macro', zero_division=0),
        'F1_Score': f1_score(label_list, pred_list, average='macro', zero_division=0)
    }
    
    return metrics

def test_detailed(dataset_name, model, test_loader, num_classes=8, device='cuda'):
    """
    Evaluates the Task Model and returns Global + Per-Class Metrics.
    Now matches the original 'balanced_accuracy_score' logic.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                imgs = batch['image'].to(device)
                lbls = batch['label'].to(device)
            else:
                _, data = batch
                imgs = data['image'].to(device)
                lbls = data['label'].to(device)

            outputs = model(imgs)
            
            # Robust tuple unpacking
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(lbls.cpu().numpy())
            
    # 1. Global Metrics
    acc = accuracy_score(all_targets, all_preds)
    
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    
    g_prec, g_rec, g_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)

    metrics = {
        'Accuracy': acc,
        'Balanced_Acc': bal_acc,
        'Precision': g_prec,
        'Recall': g_rec,
        'F1_Score': g_f1
    }

    # 2. Per-Class Metrics
    prec_list, rec_list, f1_list, support_list = precision_recall_fscore_support(
        all_targets, all_preds, labels=range(num_classes), zero_division=0
    )
    
    for c in range(num_classes):
        metrics[f"Class_{c}_Count"] = support_list[c]
        metrics[f"Class_{c}_Prec"] = prec_list[c]
        metrics[f"Class_{c}_Rec"] = rec_list[c]
        metrics[f"Class_{c}_F1"] = f1_list[c]
        
    return metrics
    
def test_detailed_2(dataset_name, model, test_loader, num_classes=8, device='cuda'):
    model.eval()
    all_preds = []
    all_targets = []
    logits_list = [] # Collect logits instead of probs initially

    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                imgs = batch['image'].to(device)
                lbls = batch['label'].to(device)
            else:
                _, data = batch
                imgs = data['image'].to(device)
                lbls = data['label'].to(device)

            outputs = model(imgs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            # Save logits for manual softmax calculation later (to match reference precision)
            logits_list.append(logits.cpu().numpy())
            
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(lbls.cpu().numpy())

    # 1. Prepare Data
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_logits = np.concatenate(logits_list, axis=0)

    # 2. Manual Softmax (Matches reference implementation)
    z = all_logits - all_logits.max(axis=1, keepdims=True)
    all_probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)

    # 3. Standard Metrics
    acc = accuracy_score(all_targets, all_preds)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    g_prec, g_rec, g_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)

    # 4. AUC Calculation (Aligned with Reference)
    aucs = []
    # If it's binary, the reference loop still treats it as C classes (One-vs-Rest manually)
    # If you want strict adherence to the reference:
    for c in range(num_classes):
        # Create binary target for class c
        y_bin = (all_targets == c).astype(int)
        
        # Only calculate if both classes are present (0 and 1)
        # This prevents crashing if a class is missing in the test set
        if len(np.unique(y_bin)) == 2:
            try:
                score = roc_auc_score(y_bin, all_probs[:, c])
                aucs.append(score)
            except ValueError:
                pass
    
    # Macro Average
    auc_score = float(np.mean(aucs)) if aucs else 0.0

    metrics = {
        'Accuracy': acc,
        'Balanced_Acc': bal_acc,
        'Precision': g_prec,
        'Recall': g_rec,
        'F1_Score': g_f1,
        'AUC': auc_score
    }

    # 5. Per-Class Metrics
    prec_list, rec_list, f1_list, support_list = precision_recall_fscore_support(
        all_targets, all_preds, labels=range(num_classes), zero_division=0
    )
    
    for c in range(num_classes):
        metrics[f"Class_{c}_Count"] = support_list[c]
        metrics[f"Class_{c}_Prec"] = prec_list[c]
        metrics[f"Class_{c}_Rec"] = rec_list[c]
        metrics[f"Class_{c}_F1"] = f1_list[c]
        
    return metrics