import os

import numpy as np
import open_clip
import pandas as pd  
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score  
from torch.utils.data import DataLoader, Dataset

from data.sampler import SubsetSequentialSampler
from utils.vlm_filter import CLASSES, PROMPTS, get_train_prompts, BaseVLMWrapper


def get_model_probabilities(model, dataset, indices, device='cuda',
                            is_vlm=False, vlm_id_classes=8, dataset_name='FedISIC',
                            return_full=False, cached_features=None):
    """
    Returns the Probability Vector.
    If cached_features is provided (and is_vlm=True), skips image loading.
    """
    # Track VLM inference count for compute cost comparison
    if is_vlm and hasattr(get_model_probabilities, '_counter'):
        get_model_probabilities._counter['calls'] += 1
        get_model_probabilities._counter['samples'] += len(indices)

    # Helper to detect CoOp-style models
    def check_is_coop(m):
        t_str = str(type(m))
        return ('CoOp' in t_str) or hasattr(m, 'ctx_vectors') or hasattr(m, 'global_ctx')

    # Single place to prepare prompts & tokens (used by both fast & slow paths)
    def prepare_text_tokens():
        tokenizer = open_clip.get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        is_coop_local = check_is_coop(model)

        try:
            if is_coop_local and dataset_name in CLASSES:
                all_texts = get_train_prompts(dataset_name)
            else:
                p_data = PROMPTS[dataset_name]
                all_texts = p_data.get('ID', []) + p_data.get('OOD', [])
        except (KeyError, TypeError, AttributeError) as e:
            print(f"[WARNING] Prompt loading failed for '{dataset_name}': {e}")
            print("         → Using fallback generic medical prompts")
            all_texts = [f"A photo of a class {i} medical image." for i in range(vlm_id_classes)]

        if not all_texts:
            raise ValueError(f"No text prompts available for dataset '{dataset_name}'")

        tokens = tokenizer(all_texts).to(device)
        return tokens, is_coop_local

    # FAST PATH: Cached image features
    if is_vlm and cached_features is not None:
        tokens, is_coop = prepare_text_tokens()

        relevant_feats = torch.from_numpy(cached_features[indices]).to(device).float()

        with torch.no_grad():
            if hasattr(model, 'clip_model') and not is_coop:
                text_input = model.clip_model.encode_text(tokens)
                text_input /= text_input.norm(dim=-1, keepdim=True)
            else:
                text_input = tokens

            logits = model(relevant_feats, text_input)
            full_probs = torch.softmax(logits, dim=1)

            if return_full:
                return full_probs.cpu()
            else:
                return full_probs[:, :vlm_id_classes].cpu()

    # SLOW PATH: Images through DataLoader
    model.eval()
    loader = DataLoader(dataset, batch_size=64,
                        sampler=SubsetSequentialSampler(indices), num_workers=4)

    probs_list = []

    text_input = None
    if is_vlm:
        tokens, is_coop = prepare_text_tokens()

        with torch.no_grad():
            if hasattr(model, 'clip_model') and not is_coop:
                text_input = model.clip_model.encode_text(tokens)
                text_input /= text_input.norm(dim=-1, keepdim=True)
            else:
                text_input = tokens

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                imgs = batch['image'].to(device)
            elif isinstance(batch, (list, tuple)):
                if len(batch) >= 2 and isinstance(batch[1], dict):
                    imgs = batch[1]['image'].to(device)
                else:
                    imgs = batch[0].to(device)
            else:
                imgs = batch[0].to(device)

            if is_vlm:
                logits = model(imgs, text_input)
                full_probs = torch.softmax(logits, dim=1)
                id_probs = full_probs if return_full else full_probs[:, :vlm_id_classes]
            else:
                logits = model(imgs)
                id_probs = torch.softmax(logits, dim=1)

            probs_list.append(id_probs.cpu())

    if probs_list:
        return torch.cat(probs_list, dim=0)
    else:
        return torch.empty((0, vlm_id_classes), device='cpu')


class BenchmarkDataset(Dataset):
    def __init__(self, csv_file, root_dir=None, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = 'data/FedISIC_npy/{}.npy'.format(img_name)

        try:
            img_array = np.load(img_path)
            image = Image.fromarray(img_array.astype('uint8')).convert('RGB')
        except Exception as e:
            print(f"[Error] Could not load {img_path}: {e}")
            raise e

        if self.transform:
            image = self.transform(image)

        target = int(self.data.iloc[idx]['target'])
        return image, {'original_label': target, 'path': img_path, 'is_ood': 1 if target == 8 else 0}


def benchmark_vlm_performance(args, local_vlm_adapters, local_data, exp_logger):
    """
    Benchmarks VLM on the external stratified_poisoned_test.csv
    """
    print("\n>>> Starting VLM Benchmark on stratified_poisoned_test.csv ...")

    csv_path = "data/data_split/FedISIC/stratified_poisoned_test.csv"
    if not os.path.exists(csv_path):
        csv_path = "stratified_poisoned_test.csv"
        if not os.path.exists(csv_path):
            print(f"[Warn] Benchmark CSV {csv_path} not found. Skipping.")
            return

    base_vlm_instance = None
    results = []

    for c_id in range(4):
        full_df = pd.read_csv(csv_path)
        client_df = full_df[full_df['center'] == c_id]
        if client_df.empty:
            continue

        temp_csv = f"temp_bench_client_{c_id}.csv"
        client_df.to_csv(temp_csv, index=False)

        tr = None
        if hasattr(local_data['train'][0], 'transform'):
            tr = local_data['train'][0].transform
        elif hasattr(local_data['train'][0], 'dataset'):
            tr = local_data['train'][0].dataset.transform

        bench_ds = BenchmarkDataset(temp_csv, transform=tr)
        indices = list(range(len(bench_ds)))

        model_to_use = local_vlm_adapters[c_id]
        if model_to_use is None:
            if base_vlm_instance is None:
                print(f"[Benchmark] Client {c_id}: No adapter found. Initializing Base BiomedCLIP (Zero-Shot)...")
                base_vlm_instance = BaseVLMWrapper(device='cuda')
            model_to_use = base_vlm_instance

        try:
            probs = get_model_probabilities(
                model=model_to_use,
                dataset=bench_ds,
                indices=indices,
                is_vlm=True,
                vlm_id_classes=args.num_classes,
                dataset_name=args.dataset,
                device='cuda',
                return_full=True
            )

            preds = probs.argmax(dim=1).numpy()
            targets = client_df['target'].values

            id_mask = targets < args.num_classes
            acc_id = accuracy_score(targets[id_mask], preds[id_mask]) if id_mask.sum() > 0 else 0.0
            bal_acc_id = balanced_accuracy_score(targets[id_mask], preds[id_mask]) if id_mask.sum() > 0 else 0.0

            ood_mask = targets >= args.num_classes
            ood_acc = (preds[ood_mask] >= args.num_classes).mean() if ood_mask.sum() > 0 else 0.0

            pred_id_mask = preds < args.num_classes
            purity = (targets[pred_id_mask] < args.num_classes).mean() if pred_id_mask.sum() > 0 else 0.0

            results.append({
                'Client_ID': c_id, 'ID_Accuracy': acc_id, 'ID_Balanced_Acc': bal_acc_id,
                'ID_Purity_Precision': purity, 'OOD_Detection_Acc': ood_acc,
                'Total_ID_Samples': id_mask.sum(), 'Total_OOD_Samples': ood_mask.sum()
            })
        except Exception as e:
            print(f"[Error] Benchmarking Client {c_id} failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if os.path.exists(temp_csv):
                os.remove(temp_csv)

    if results:
        res_df = pd.DataFrame(results)
        save_path = os.path.join(exp_logger.base_dir, "test_vlm_performance.csv")
        res_df.to_csv(save_path, index=False)
        print(f">>> VLM Benchmark saved to: {save_path}")
        print(res_df)
