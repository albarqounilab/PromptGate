import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
from scipy import ndimage
import albumentations as A
from PIL import Image

import os
from glob import glob
import random
import numpy as np
import json
import pdb
import pandas as pd
import pickle

import torch.backends.cudnn as cudnn

# HELPER FOR FEDEMBED 
def subsample_by_patient(df, frac, seed=33, keep_artefact=True):
    if keep_artefact == False:
        df.drop(df[df['DENSITY'] >= 4].index, inplace=True)
    if frac >= 1.0: return df
    
    g = df.groupby("empi_anon").size()
    pats = g.index.to_list()
    rng = np.random.default_rng(seed)
    take = set(rng.choice(pats, size=max(1, int(len(pats)*frac)), replace=False))
    
    return df[df["empi_anon"].isin(take)].reset_index(drop=True)

# NEW DATASET CLASS 
class FedEMBED(Dataset):
    def __init__(self, fl_method='FedAvg', client_idx=None, mode='train', transform=None, ood=None, args=None):
        assert mode in ['train', 'test']
        
        # Clients defined in the reference repo
        self.client_name = ['Selenia Dimensions', 'Senograph 2000D ADS', 'Lorad Selenia', 'Clearview CSm']
        self.client_idx = client_idx if client_idx is not None else 0
        self.mode = mode
        self.transform = transform
        
        csv_root = "data/FedEMBED"
        feature_dir = "data/FedEMBED"
        feature_tag = "mammoclip"
        feature_suffix = "clahe"

        # Load CSV
        df = pd.read_csv(f"{csv_root}/{mode}.csv")
        
        # Clean Manufacturer Name
        df["ManufacturerModelName"] = df["ManufacturerModelName"].str.replace(r"^Senograph 2000D ADS_.*$", "Senograph 2000D ADS", regex=True)
        
        # Load Features
        with open(f"{feature_dir}/extracted-features-{feature_tag}-{mode}-{feature_suffix}.pkl", "rb") as f:
            feats = np.array(pickle.load(f))
        
        df["FEATURES"] = feats.tolist()
        df = df[df["Target"].notna()]
        
        # Filter by Client
        df = df[df["ManufacturerModelName"] == self.client_name[self.client_idx]].reset_index(drop=True)

        
        keep_artefact = False if mode == 'test' else True # Default logic from reference
        if ood == "ID":
             keep_artefact = False # Strict ID only

        df = subsample_by_patient(df, frac=1.0, keep_artefact=keep_artefact)

        self.features = np.stack(df["FEATURES"].to_numpy(), axis=0).astype(np.float32)
        self.image_paths = df['FPATH_PROC'].astype(str).tolist()
        
        # DENSITY: 0-3 are density classes (A-D), 4+ are artefacts/OOD
        density_vals = df["DENSITY"].astype(int).values
        
        self.labels = []
        for d in density_vals:
            if d >= 4:
                self.labels.append(4)  # Artefact/OOD class
            elif 0 <= d <= 3:
                self.labels.append(d)  # Density classes: 0,1,2,3
            else:
                raise ValueError(f"Unexpected DENSITY value: {d}")

        self.data_list = df # Keep ref for metadata

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        # 1. Get Feature Vector (1D Tensor)
        image_features = self.features[idx]
        
        # 2. Get Label
        label = self.labels[idx]
        
        # 3. Define OOD Logic
        # In FedEMBED: 0-3 are ID (Density), 4 is OOD (Artefact)
        is_ood = 1 if label >= 4 else 0
        
        # IMPORTANT: Keep the true label (including 4 for artefacts)
        # The old code trained a 5-class model, not with ignore_index
        train_label = label  # Keep all labels 0-4

        # Return dict matching FedISIC structure
        return idx, {
            'image': torch.from_numpy(image_features), # This is 1D (2048,) or (512,)
            'label': torch.tensor(train_label),
            'path': self.image_paths[idx],    # Critical for VLM to find the raw image
            'original_label': int(label),
            'is_ood': int(is_ood)
        }

# classification datasets
class FedISIC(Dataset):
    def __init__(self, fl_method='FedAvg', client_idx=None, mode='train', transform=None, ood=None, args=None):
        assert mode in ['train', 'test']

        self.num_classes = 8
        self.fl_method = fl_method
        self.client_name = ['client1', 'client2', 'client3', 'client4']
        self.client_idx = client_idx if client_idx is not None else 0
        self.mode = mode
        self.transform = transform

        # OOD split: in-distribution, or 50% Far-OOD (Fitzpatrick17k / DDI).
        if ood == "ID":
            df_path = 'data/data_split/FedISIC/train_test_split.csv'
        elif ood == "50%":
            df_path = 'data/data_split/FedISIC/realistic_train_test_split_far_ood_50.csv'
        else:
            raise ValueError(f"Unsupported --ood '{ood}' for FedISIC (use 'ID' or '50%').")

        df = pd.read_csv(df_path)
        self.data_list = df[(df['center']==self.client_idx) & (df['fold']==mode)]

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx:int):
        img_path = 'data/FedISIC_npy/{}.npy'.format(self.data_list.iloc[idx, 0])
        image = np.load(img_path)
        
        # Albumentations expects (224, 224, 3).
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)

        # OOD UPDATE START 
        try:
            label = self.data_list.iloc[idx]['target']
        except:
            label = self.data_list.iloc[idx, -4] 

        if 'is_ood' in self.data_list.columns:
            is_ood = self.data_list.iloc[idx]['is_ood']
        else:
            is_ood = 1 if label >= 8 else 0

        train_label = label if is_ood == 0 else -1 
        # OOD UPDATE END 

        if self.transform is not None:
            image = self.transform(image=image)['image']
        
        image = image.transpose(2, 0, 1).astype(np.float32)

        return idx, {
            'image': torch.from_numpy(image.copy()),
            'label': torch.tensor(train_label),
            'path': img_path,
            'original_label': int(label),
            'is_ood': int(is_ood)
        }


def generate_dataset(dataset, fl_method, client_idx, args):
    # General settings
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # If using multiple GPUs

    # Limit PyTorch threads
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    if dataset == 'FedISIC':
        from data.dataset_generator import FedISIC as Med_Dataset
        train_transform = A.Compose([
                                        A.Rotate(10),
                                        A.RandomBrightnessContrast(0.15, 0.1),
                                        A.Flip(p=0.5),
                                        A.CenterCrop(224,224),
                                        A.Normalize(always_apply=True)
                                    ])
        test_transform = A.Compose([
                                        A.CenterCrop(224,224),
                                        A.Normalize(always_apply=True)
                                    ])

    elif dataset == 'FedEMBED':
        from data.dataset_generator import FedEMBED as Med_Dataset
        # Features are pre-extracted, so transforms are None
        train_transform = None
        test_transform = None

    else:
        raise ValueError(f"Unsupported dataset '{dataset}' (expected 'FedISIC' or 'FedEMBED').")


    data_train = Med_Dataset(fl_method=fl_method, 
                                client_idx=client_idx,
                                mode='train',
                                transform=train_transform, ood = args.ood, args=args)
    
    data_unlabeled = Med_Dataset(fl_method=fl_method, 
                                    client_idx=client_idx,
                                    mode='train',
                                    transform=test_transform, ood = args.ood, args=args)  

    data_test = Med_Dataset(fl_method=fl_method,
                                client_idx=client_idx,
                                mode='test',
                                transform=test_transform, ood = args.ood, args=args)
                            
    return data_train, data_unlabeled, data_test