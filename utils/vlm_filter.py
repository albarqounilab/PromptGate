import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import open_clip
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm  
from PIL import Image
import os
import logging
from sklearn.metrics import precision_recall_fscore_support, accuracy_score  

# Suppress open_clip logs
logging.getLogger("open_clip").setLevel(logging.ERROR)
# CONFIGURATION 1: TRAINING CLASSES 
CLASSES = {
    'FedISIC': {
        0: "Melanoma",
        1: "Melanocytic nevus",
        2: "Basal cell carcinoma",
        3: "Actinic keratosis",
        4: "Benign keratosis",
        5: "Dermatofibroma",
        6: "Vascular lesion",
        7: "Squamous cell carcinoma",
        8: "A dermoscopy image of Unknown or artifact" 
    },
    'FedEMBED': {
        0: "Fatty",
        1: "Scattered fibroglandular",
        2: "Heterogeneously dense",
        3: "Extremely dense",
        4: "An image of Unknown or artifact"
    }
}

# --- CONFIGURATION 2: ZERO-SHOT PROMPTS ---
PROMPTS = {
    'FedISIC': {
        'ID': [
            "A dermoscopy image of Melanoma", 
            "A dermoscopy image of Melanocytic nevus",
            "A dermoscopy image of Basal cell carcinoma", 
            "A dermoscopy image of Actinic keratosis",
            "A dermoscopy image of Benign keratosis", 
            "A dermoscopy image of Dermatofibroma",
            "A dermoscopy image of Vascular lesion", 
            "A dermoscopy image of Squamous cell carcinoma"
        ],
        'OOD': [
            "An dermoscopy image of Unknown or artifact"
        ]
    },
    'FedEMBED': {
        'ID': [
            "A mammogram showing Fatty breast density (BI-RADS density A)",
            "A mammogram showing Scattered fibroglandular breast density (BI-RADS density B)",
            "A mammogram showing Heterogeneously dense breast density (BI-RADS density C)",
            "A mammogram showing Extremely dense breast density (BI-RADS density D)"
        ],
        'OOD': [
            "An image of a mammogram Unknown or artifact"
        ]
    }
}

def get_train_prompts(dataset_name):
    """Returns the condensed prompts for CoOp training."""
    if dataset_name in CLASSES:
        cls_dict = CLASSES[dataset_name]
        prompts = []
        for idx in sorted(cls_dict.keys()):
            class_name = cls_dict[idx]
            if dataset_name == 'FedISIC':
                if idx == 8: 
                    prompts.append(class_name) 
                else:
                    prompts.append(f"A dermoscopy image of {class_name}")
            elif dataset_name == 'FedEMBED':
                if idx == 4:
                    prompts.append(class_name)
                else:
                    prompts.append(f"A mammogram showing {class_name} breast density")
            else:
                prompts.append(f"An image of {class_name}")
        return prompts
    
    # Fallback
    return PROMPTS.get(dataset_name, {}).get('ID', []) + PROMPTS.get(dataset_name, {}).get('OOD', [])

# --- NEW VLM DATASET FOR FEDEMBED ---
class VLMDatasetEMBED(Dataset):
    def __init__(self, original_dataset, preprocess_fn):
        """
        Specialized Dataset for FedEMBED. 
        The original_dataset returns features, but here we load raw images 
        from disk for the VLM (CLIP/BiomedCLIP).
        """
        self.original_dataset = original_dataset
        self.preprocess = preprocess_fn
        
        self.img_root = "data/FedEMBED"

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # 1. Retrieve Metadata from the original dataset
        # FedEMBED.__getitem__ returns: idx, {'image': feature_tensor, 'path': str, ...}
        _, meta = self.original_dataset[idx]
        
        # 2. Get Path (Handle relative paths common in FedEMBED CSVs)
        rel_path = meta.get('path', '')
        
        # Construct absolute path
        if os.path.isabs(rel_path):
            img_path = rel_path
        else:
            img_path = os.path.join(self.img_root, rel_path)

        # 3. Load Raw Image
        try:
            image_pil = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback (black image) to prevent crashing if one file is missing
            # print(f"[VLMDatasetEMBED] Warning: Could not open {img_path}")
            print(f"[VLMDatasetEMBED] Warning: Could not open {img_path}: {e}")
            image_pil = Image.new('RGB', (224, 224), color='black')

        # 4. Apply CLIP/BiomedCLIP Preprocessing
        image_tensor = self.preprocess(image_pil)
        
        return idx, image_tensor

# DATASET & HELPERS 
class VLMDataset(Dataset):
    def __init__(self, original_dataset, preprocess_fn):
        self.original_dataset = original_dataset
        self.preprocess = preprocess_fn
        self.img_base = 'data/FedISIC_npy'

    def __len__(self):
        # FIX: Rely on the parent dataset's length. 
        # This prevents off-by-one errors if the parent is a Subset.
        return len(self.original_dataset)

    def _load_from_path(self, img_name):
        if os.path.isfile(img_name):
            img_path = img_name
        else:
            img_path = os.path.join(
                self.img_base,
                img_name if img_name.endswith('.npy') else f'{img_name}.npy',
            )
        image_np = np.load(img_path)

        if image_np.dtype != np.uint8 or image_np.max() < 10:
            img_min = image_np.min()
            img_max = image_np.max()
            if img_max > img_min:
                image_np = (image_np - img_min) / (img_max - img_min) * 255.0
            image_np = image_np.astype(np.uint8)
        return image_np

    def __getitem__(self, idx):
        # FIX: Retrieve metadata via the parent dataset to handle Subsets/indices correctly
        _, meta = self.original_dataset[idx]
        
        # Robustly extract the image path/name
        if isinstance(meta, dict) and 'path' in meta:
            img_name = meta['path']
        elif hasattr(self.original_dataset, 'data_list'):
             # Fallback: if it's a wrapper where data_list is exposed but metadata isn't full
             # Note: This branch might be risky if original_dataset is a Subset. 
             # The first branch (meta dict) is safer.
             img_name = self.original_dataset.data_list.iloc[idx, 0]
        else:
             img_name = "unknown"

        image_np = self._load_from_path(img_name)
        image_pil = Image.fromarray(image_np)
        image_tensor = self.preprocess(image_pil)
        return idx, image_tensor

def get_biomedclip_model(device):
    model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model.to(device).eval()
    return model, tokenizer, preprocess

class BaseVLMWrapper(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.clip_model, _, _ = get_biomedclip_model(device)
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, image_or_feats, text_tokens):
        if image_or_feats.dim() == 4:
            with torch.no_grad():
                img_feats = self.clip_model.encode_image(image_or_feats)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        else:
            img_feats = image_or_feats
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            
        with torch.no_grad():
            text_feats = self.clip_model.encode_text(text_tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            
        return 100.0 * img_feats @ text_feats.T

# ADAPTER DEFINITIONS 
class MetaNet(nn.Module):
    def __init__(self, vis_dim, ctx_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(vis_dim, vis_dim // 16),
            nn.ReLU(inplace=True),
            nn.Linear(vis_dim // 16, ctx_dim)
        )
    def forward(self, img_feats):
        return self.fc(img_feats)

class CoOpOriginal(nn.Module):
    def __init__(self, clip_model, num_ctx_vectors=16):
        super().__init__()
        self.clip_model = clip_model
        self.text_encoder = self.clip_model.text.transformer
        self.embeddings = self.text_encoder.embeddings.word_embeddings
        self.hidden_dim = self.text_encoder.config.hidden_size
        self.proj = self.clip_model.text.proj 
        self.ctx_vectors = nn.Parameter(torch.randn(num_ctx_vectors, self.hidden_dim))
        nn.init.normal_(self.ctx_vectors, std=0.02)
        for param in self.clip_model.parameters():
            param.requires_grad = False
    def encode_text(self, text_tokens):
        with torch.no_grad():
            token_embeds = self.embeddings(text_tokens)
        ctx = self.ctx_vectors.unsqueeze(0).expand(len(text_tokens), -1, -1)
        inputs_embeds = torch.cat([ctx, token_embeds], dim=1)
        out = self.text_encoder(inputs_embeds=inputs_embeds)
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            txt_hidden = out.pooler_output
        else:
            txt_hidden = out.last_hidden_state[:, 0, :]
        if self.proj is not None:
            text_feats = txt_hidden @ self.proj if not isinstance(self.proj, nn.Module) else self.proj(txt_hidden)
        else:
            text_feats = txt_hidden
        return text_feats / text_feats.norm(dim=-1, keepdim=True)
    def forward(self, image_or_feats, text_tokens):
        if image_or_feats.dim() == 4:
            with torch.no_grad():
                img_feats = self.clip_model.encode_image(image_or_feats)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        else:
            img_feats = image_or_feats
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        text_feats = self.encode_text(text_tokens)
        return 100.0 * img_feats @ text_feats.T

class FederatedCoOp(nn.Module):
    def __init__(self, clip_model, n_global=8, n_local=8, num_classes=8, 
                 fusion_strategy='concat', class_specific=False):
        super().__init__()
        self.clip_model = clip_model
        self.n_global = n_global
        self.n_local = n_local
        self.class_specific = class_specific
        self.num_classes = num_classes
        self.fusion_strategy = fusion_strategy
        self.text_encoder = self.clip_model.text.transformer
        self.embeddings = self.text_encoder.embeddings.word_embeddings
        self.hidden_dim = self.text_encoder.config.hidden_size
        self.proj = self.clip_model.text.proj 
        shape_g = (num_classes, n_global, self.hidden_dim) if class_specific else (n_global, self.hidden_dim)
        shape_l = (num_classes, n_local, self.hidden_dim) if class_specific else (n_local, self.hidden_dim)
        if n_global > 0:
            self.global_ctx = nn.Parameter(torch.randn(*shape_g))
            nn.init.normal_(self.global_ctx, std=0.02)
        else:
            self.register_parameter('global_ctx', None)
        if n_local > 0:
            self.local_ctx = nn.Parameter(torch.randn(*shape_l))
            nn.init.normal_(self.local_ctx, std=0.02)
        else:
            self.register_parameter('local_ctx', None)
        for param in self.clip_model.parameters():
            param.requires_grad = False
    def get_global_vectors(self):
        return self.global_ctx.detach().clone() if self.global_ctx is not None else None
    def load_global_vectors(self, new_vecs):
        if self.global_ctx is not None:
            with torch.no_grad():
                self.global_ctx.copy_(new_vecs)
    def get_context_vectors(self):
        vecs = []
        if self.global_ctx is not None: vecs.append(self.global_ctx)
        if self.local_ctx is not None: vecs.append(self.local_ctx)
        if not vecs: return None
        ctx = torch.cat(vecs, dim=1 if self.class_specific else 0)
        if not self.class_specific:
            ctx = ctx.unsqueeze(0).expand(self.num_classes, -1, -1)
        return ctx
    def encode_text(self, text_tokens):
        ctx = self.get_context_vectors()
        with torch.no_grad():
            token_embeds = self.embeddings(text_tokens)
        if ctx is not None:
            input_embeds = torch.cat([ctx, token_embeds], dim=1)
        else:
            input_embeds = token_embeds
        out = self.text_encoder(inputs_embeds=input_embeds)
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            txt_hidden = out.pooler_output
        else:
            txt_hidden = out.last_hidden_state[:, 0, :]
        if self.proj is not None:
            text_feats = txt_hidden @ self.proj if not isinstance(self.proj, nn.Module) else self.proj(txt_hidden)
        else:
            text_feats = txt_hidden
        return text_feats / text_feats.norm(dim=-1, keepdim=True)
    def forward(self, image_or_feats, text_tokens):
        if image_or_feats.dim() == 4:
            with torch.no_grad():
                img_feats = self.clip_model.encode_image(image_or_feats)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        else:
            img_feats = image_or_feats
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        text_feats = self.encode_text(text_tokens)
        return 100.0 * img_feats @ text_feats.T


class ResidualFederatedCoOp(nn.Module):
    """
    Residual Federated CoOp: shared prompt + gated local residual.
    
    ctx = shared + sigmoid(gate) * residual
    
    - shared_ctx:      Aggregated via FedAvg. Captures global medical knowledge.
    - local_residual:  Stays on-client, never aggregated. Initialized to ZERO.
    - gate:            Per-position learnable sigmoid gate. Initialized near 0.1.
    
    Only N prompt vectors enter the text encoder (not 2N like concatenation),
    keeping class tokens close to CLS position.
    """
    def __init__(self, clip_model, n_ctx=16, num_classes=8, class_specific=False):
        super().__init__()
        self.clip_model = clip_model
        self.n_ctx = n_ctx
        self.class_specific = class_specific
        self.num_classes = num_classes
        
        # Text encoder components (frozen)
        self.text_encoder = self.clip_model.text.transformer
        self.embeddings = self.text_encoder.embeddings.word_embeddings
        self.hidden_dim = self.text_encoder.config.hidden_size
        self.proj = self.clip_model.text.proj
        
        # --- Learnable Parameters ---
        shape = (num_classes, n_ctx, self.hidden_dim) if class_specific else (n_ctx, self.hidden_dim)
        
        # Shared prompt: aggregated via FedAvg (global knowledge)
        self.shared_ctx = nn.Parameter(torch.randn(*shape))
        nn.init.normal_(self.shared_ctx, std=0.02)
        
        # Local residual: stays on-client, small noise to break zero-gradient
        self.local_residual = nn.Parameter(torch.randn(*shape) * 0.01)
        
        # Channel-wise gate: sigmoid applied at runtime
        # Initialize to -1.0 → sigmoid(-1.0) ≈ 0.27 → trusts shared but warm enough for gradients
        gate_shape = (num_classes, n_ctx, self.hidden_dim) if class_specific else (n_ctx, self.hidden_dim)
        self.gate = nn.Parameter(torch.full(gate_shape, -1.0))
        
        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    # --- Compatibility API (same as FederatedCoOp) ---
    @property
    def global_ctx(self):
        """Alias for save_learnable_vectors compatibility."""
        return self.shared_ctx
    
    @property  
    def local_ctx(self):
        """Alias for save_learnable_vectors compatibility — exposes residual."""
        return self.local_residual
    
    def get_global_vectors(self):
        """Returns shared_ctx for FedAvg aggregation."""
        return self.shared_ctx.detach().clone()
    
    def load_global_vectors(self, new_vecs):
        """Loads aggregated shared_ctx. Residual and gate stay untouched."""
        with torch.no_grad():
            self.shared_ctx.copy_(new_vecs)
    
    def get_context_vectors(self):
        """Compute ctx = shared + sigmoid(gate) * residual."""
        g = torch.sigmoid(self.gate)  # (N, D) or (C, N, D) — channel-wise
        
        ctx = self.shared_ctx + g * self.local_residual
        
        if not self.class_specific:
            ctx = ctx.unsqueeze(0).expand(self.num_classes, -1, -1)
        
        return ctx
    
    def encode_text(self, text_tokens):
        ctx = self.get_context_vectors()
        with torch.no_grad():
            token_embeds = self.embeddings(text_tokens)
        if ctx is not None:
            input_embeds = torch.cat([ctx, token_embeds], dim=1)
        else:
            input_embeds = token_embeds
        out = self.text_encoder(inputs_embeds=input_embeds)
        if hasattr(out, 'pooler_output') and out.pooler_output is not None:
            txt_hidden = out.pooler_output
        else:
            txt_hidden = out.last_hidden_state[:, 0, :]
        if self.proj is not None:
            text_feats = txt_hidden @ self.proj if not isinstance(self.proj, nn.Module) else self.proj(txt_hidden)
        else:
            text_feats = txt_hidden
        return text_feats / text_feats.norm(dim=-1, keepdim=True)
    
    def forward(self, image_or_feats, text_tokens):
        if image_or_feats.dim() == 4:
            with torch.no_grad():
                img_feats = self.clip_model.encode_image(image_or_feats)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        else:
            img_feats = image_or_feats
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        text_feats = self.encode_text(text_tokens)
        return 100.0 * img_feats @ text_feats.T
    
    def get_gate_stats(self):
        """Returns gate values for logging/monitoring."""
        g = torch.sigmoid(self.gate).detach().cpu()
        return {'mean': g.mean().item(), 'min': g.min().item(), 'max': g.max().item(), 'std': g.std().item()}


class FederatedCoCoOp(FederatedCoOp):
    def __init__(self, clip_model, n_global=8, n_local=8, fusion_strategy='concat', ensemble_alpha=0.5):
        super().__init__(clip_model, n_global=n_global, n_local=n_local, 
                         num_classes=8, fusion_strategy=fusion_strategy, class_specific=False)
        self.meta_net = MetaNet(512, self.hidden_dim)
        self.ensemble_alpha = ensemble_alpha
    def forward(self, image_or_feats, text_tokens):
        if image_or_feats.dim() == 4:
            with torch.no_grad():
                img_feats = self.clip_model.encode_image(image_or_feats)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        else:
            img_feats = image_or_feats
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        ctx_bias = self.meta_net(img_feats)
        ctx_bias = ctx_bias.unsqueeze(1)
        def get_conditioned_text(vectors):
            with torch.no_grad():
                token_embeds = self.embeddings(text_tokens)
            batch_size = img_feats.shape[0]
            n_cls = token_embeds.shape[0]
            base_ctx = vectors.unsqueeze(0).expand(batch_size, -1, -1)
            conditioned_ctx = base_ctx + ctx_bias 
            conditioned_ctx = conditioned_ctx.unsqueeze(1).expand(-1, n_cls, -1, -1)
            token_embeds = token_embeds.unsqueeze(0).expand(batch_size, -1, -1, -1)
            full_input = torch.cat([conditioned_ctx, token_embeds], dim=2)
            full_input = full_input.reshape(batch_size * n_cls, -1, self.hidden_dim)
            out = self.text_encoder(inputs_embeds=full_input)
            if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                txt = out.pooler_output
            else:
                txt = out.last_hidden_state[:, 0, :]
            if self.proj is not None:
                txt = txt @ self.proj if not isinstance(self.proj, nn.Module) else self.proj(txt)
            txt = txt / txt.norm(dim=-1, keepdim=True)
            return txt.view(batch_size, n_cls, -1)
        if self.fusion_strategy == 'concat':
            vecs = []
            if self.global_ctx is not None: vecs.append(self.global_ctx)
            if self.local_ctx is not None: vecs.append(self.local_ctx)
            combined = torch.cat(vecs, dim=0) if vecs else None
            if combined is None: return 100.0 * img_feats @ self.encode_text(text_tokens).T
            txt_feats = get_conditioned_text(combined)
            return 100.0 * torch.einsum('bd, bcd -> bc', img_feats, txt_feats)
        elif self.fusion_strategy == 'ensemble':
            logits_g, logits_l = 0.0, 0.0
            if self.global_ctx is not None:
                txt_g = get_conditioned_text(self.global_ctx)
                logits_g = 100.0 * torch.einsum('bd, bcd -> bc', img_feats, txt_g)
            if self.local_ctx is not None:
                txt_l = get_conditioned_text(self.local_ctx)
                logits_l = 100.0 * torch.einsum('bd, bcd -> bc', img_feats, txt_l)
            return (self.ensemble_alpha * logits_g) + ((1 - self.ensemble_alpha) * logits_l)

def train_vlm_adapter(model_type, dataset_name, train_loader, prev_adapter=None, 
                      prev_optimizer=None,
                      device='cuda', coop_epochs=5, 
                      cached_features=None, cached_labels=None, args=None):
    use_cache = (cached_features is not None) 
    
    train_prompts = get_train_prompts(dataset_name)
    num_train_classes = len(train_prompts)
    
    if prev_adapter is None:
        clip_model, tokenizer, _ = get_biomedclip_model(device)
        
        n_glob = getattr(args, 'coop_global_vectors', 8) 
        n_loc  = getattr(args, 'coop_local_vectors', 8)
        
        fusion = getattr(args, 'vlm_fusion_strategy', 'concat')
        is_csc = getattr(args, 'vlm_csc', False)
        alpha  = getattr(args, 'vlm_ensemble_alpha', 0.5)
        
        if model_type == 'CoCoOp':
             adapter = FederatedCoCoOp(clip_model, n_global=n_glob, n_local=n_loc, 
                                      fusion_strategy=fusion, ensemble_alpha=alpha).to(device)
        elif model_type == 'ResCoOp':
             n_total = getattr(args, 'coop_vectors', 16)
             adapter = ResidualFederatedCoOp(
                clip_model, n_ctx=n_total,
                num_classes=len(get_train_prompts(dataset_name)),
                class_specific=is_csc
             ).to(device)
             print(f"[VLM] ResCoOp: {n_total} vectors, CSC={is_csc}, "
                   f"gate init≈{torch.sigmoid(torch.tensor(-2.2)).item():.2f}")
        else:
             adapter = FederatedCoOp(
                clip_model, n_global=n_glob, n_local=n_loc, 
                num_classes=len(get_train_prompts(dataset_name)), 
                fusion_strategy=fusion, 
                class_specific=is_csc 
            ).to(device)
                                      
    else:
        adapter = prev_adapter
        tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    
    # if dataset_name == 'FedEMBED':
    #     num_id_classes = 4
    #     unk_idx = 4
    # else:
    #     num_id_classes = getattr(args, 'num_id_classes', args.num_classes)
    #     unk_idx = num_id_classes
    num_id_classes = getattr(args, 'num_id_classes', getattr(args, 'num_classes', 8)) if args is not None else 8
    unk_idx = num_id_classes


    tokens = tokenizer(train_prompts).to(device)
    print(f"[Train] Using {num_train_classes} prompts")

    if prev_optimizer is None:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, adapter.parameters()), 
                              lr=0.002, momentum=0.9, weight_decay=5e-4)
        print(f"[VLM] Initialized new SGD optimizer")
    else:
        optimizer = prev_optimizer
        print(f"[VLM] Reusing existing optimizer")

    criterion = nn.CrossEntropyLoss()
    adapter.train()

    pbar = tqdm(range(coop_epochs), desc=f"Train {model_type}", leave=False)
    loss_history = [] 

    force_raw = isinstance(adapter, FederatedCoCoOp)
    
    if use_cache and not force_raw:
        if isinstance(cached_features, np.ndarray): cached_features = torch.from_numpy(cached_features).float()
        if isinstance(cached_labels, np.ndarray): cached_labels = torch.from_numpy(cached_labels).long()
        
        train_dset = TensorDataset(cached_features, cached_labels)
        fast_loader = DataLoader(train_dset, batch_size=32, shuffle=True)
        
        for epoch in pbar:
            running_loss = 0.0
            num_batches = 0
            for feats, lbls in fast_loader:
                feats, lbls = feats.to(device), lbls.to(device)
                
                max_prompt_idx = num_train_classes - 1
                train_targets = lbls.clone()

                is_ood = lbls >= num_id_classes
                train_targets[is_ood] = unk_idx

                train_targets = torch.clamp(train_targets, 0, max_prompt_idx)

                optimizer.zero_grad()
                logits = adapter(feats, tokens) 
                loss = criterion(logits, train_targets)
                
                # (orth_reg removed — counterproductive for residual learning)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1

            epoch_avg = running_loss / num_batches if num_batches > 0 else 0
            loss_history.append(epoch_avg)
            pbar.set_postfix({'Loss': f'{epoch_avg:.4f}'})

    elif train_loader is not None:
        for epoch in pbar:
            running_loss = 0.0
            num_batches = 0
            for batch in train_loader:
                if isinstance(batch, dict):
                    imgs = batch['image'].to(device)
                    lbls = batch['original_label'].to(device)
                else:
                    _, batch_data = batch
                    imgs = batch_data['image'].to(device)
                    lbls = batch_data['original_label'].to(device)
                
                remapped_lbls = lbls.clone()
                remapped_lbls[remapped_lbls >= num_id_classes] = unk_idx 
                remapped_lbls = torch.clamp(remapped_lbls, 0, num_train_classes - 1)

                optimizer.zero_grad()
                logits = adapter(imgs, tokens)
                loss = criterion(logits, remapped_lbls)

                # (orth_reg removed — counterproductive for residual learning)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_batches += 1

            epoch_avg = running_loss / num_batches if num_batches > 0 else 0
            loss_history.append(epoch_avg)
            pbar.set_postfix({'Loss': f'{epoch_avg:.4f}'})

    return adapter, optimizer, loss_history

def evaluate_vlm_breakdown(dataset_name, vlm_adapter, full_dataset, 
                           unlabeled_indices, query_indices, labeled_indices, 
                           device='cuda', cached_features=None, args=None):
    
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    if vlm_adapter is not None:
        eval_prompts = get_train_prompts(dataset_name)
        is_static_mode = False
    else:
        id_texts = PROMPTS.get(dataset_name, {}).get('ID', [])
        ood_texts = PROMPTS.get(dataset_name, {}).get('OOD', [])
        eval_prompts = id_texts + ood_texts
        is_static_mode = True
    
    tokens = tokenizer(eval_prompts).to(device)

    text_input = None
    is_dynamic_cocoop = isinstance(vlm_adapter, FederatedCoCoOp) if 'FederatedCoCoOp' in globals() else False
    
    if not is_dynamic_cocoop and vlm_adapter is not None and hasattr(vlm_adapter, 'encode_text'):
        text_input = vlm_adapter.encode_text(tokens)
    elif vlm_adapter is None:
        m, _, _ = get_biomedclip_model(device)
        with torch.no_grad():
            text_input = m.encode_text(tokens)
            text_input /= text_input.norm(dim=-1, keepdim=True)
        del m

    def get_preds_detailed(indices):
        if len(indices) == 0: 
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        
        # num_id = args.num_classes
        num_id = getattr(args, 'num_id_classes', getattr(args, 'num_classes', 8)) if args is not None else 8
        if dataset_name == 'FedEMBED':
            num_id = 4
        
        # --- FAST PATH (Cached Features) ---
        if cached_features is not None and text_input is not None:
            max_idx = len(cached_features) - 1
            valid_indices = [i for i in indices if i <= max_idx]
            
            if not valid_indices:
                 return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

            feats = torch.from_numpy(cached_features[valid_indices]).to(device).float()
            
            targets = []
            for idx in valid_indices:
                _, meta = full_dataset[idx]
                t = meta.get('original_label', -1)
                if hasattr(t, 'item'): t = t.item()
                targets.append(t)
            
            with torch.no_grad():
                logits = 100.0 * feats @ text_input.T
                sims = torch.softmax(logits, dim=1)
                
                if is_static_mode:
                    id_soft = sims[:, :num_id].sum(dim=1)
                    ood_soft = sims[:, num_id:].sum(dim=1)
                    preds = sims.argmax(dim=1)
                    preds[preds >= num_id] = num_id
                else:
                    id_soft = sims[:, :num_id].sum(dim=1)
                    ood_soft = sims[:, num_id] 
                    preds = sims.argmax(dim=1)
                
                binary_dist = torch.stack([id_soft, ood_soft], dim=1)
                entropy = -(binary_dist * torch.log(binary_dist + 1e-10)).sum(dim=1)
                
            return (preds.cpu().numpy(), np.array(targets), 
                    id_soft.cpu().numpy(), ood_soft.cpu().numpy(), entropy.cpu().numpy())
        
        # --- SLOW PATH (DataLoader) ---
        else:
            subset = torch.utils.data.Subset(full_dataset, indices)
            loader = DataLoader(subset, batch_size=32, num_workers=4, shuffle=False)
            all_preds, all_targets = [], []
            all_id_soft, all_ood_soft, all_entropy = [], [], []
            
            for _, batch in loader:
                if isinstance(batch, dict):
                    imgs = batch['image'].to(device)
                    lbls = batch.get('original_label', batch.get('label'))
                else:
                    _, batch_data = batch
                    imgs = batch_data['image'].to(device)
                    lbls = batch_data['original_label']
                
                with torch.no_grad():
                    if vlm_adapter: logits = vlm_adapter(imgs, tokens)
                    else:
                        assert text_input is not None, "text_input must not be None when vlm_adapter is None"
                        i = imgs / imgs.norm(dim=-1, keepdim=True)
                        logits = 100.0 * i @ text_input.T
                    
                    sims = torch.softmax(logits, dim=1)
                    
                    if is_static_mode:
                        id_soft = sims[:, :num_id].sum(dim=1)
                        ood_soft = sims[:, num_id:].sum(dim=1)
                        p = sims.argmax(dim=1)
                        p[p >= num_id] = num_id
                    else:
                        id_soft = sims[:, :num_id].sum(dim=1)
                        ood_soft = sims[:, num_id]
                        p = sims.argmax(dim=1)
                    
                    b_dist = torch.stack([id_soft, ood_soft], dim=1)
                    ent = -(b_dist * torch.log(b_dist + 1e-10)).sum(dim=1)
                    
                    all_preds.extend(p.cpu().numpy())
                    all_targets.extend(lbls.cpu().numpy() if isinstance(lbls, torch.Tensor) else np.array(lbls))
                    all_id_soft.extend(id_soft.cpu().numpy())
                    all_ood_soft.extend(ood_soft.cpu().numpy())
                    all_entropy.extend(ent.cpu().numpy())
                    
            return (np.array(all_preds), np.array(all_targets), 
                    np.array(all_id_soft), np.array(all_ood_soft), np.array(all_entropy))

    def calc(name, idxs):
        p, t, ids, oods, ent = get_preds_detailed(idxs)
        if len(p) == 0: return {}, None
        
        num_id = int(getattr(args, 'num_id_classes', getattr(args, 'num_classes', 8)) if args is not None else 8)
        is_gt_id = (t < num_id)
        is_gt_ood = (t >= num_id)
        
        acc_id = accuracy_score(t[is_gt_id], p[is_gt_id]) if is_gt_id.any() else 1.0
        acc_ood = (p[is_gt_ood] >= num_id).mean() if is_gt_ood.any() else 1.0
        purity = is_gt_id.mean()
        
        unique, counts = np.unique(t, return_counts=True)
        dist_str = " | ".join([f"{int(k)}:{v}" for k, v in zip(unique, counts)])
        
        res = {
            f"{name}_Count": len(idxs),
            f"{name}_Purity": purity,
            f"{name}_VLM_Acc_ID": acc_id,
            f"{name}_VLM_Acc_OOD": acc_ood,
            f"{name}_Dist": dist_str
        }
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            t, p, labels=range(num_id), average=None, zero_division=0
        )
        precision = np.atleast_1d(precision)
        recall = np.atleast_1d(recall)
        f1 = np.atleast_1d(f1)
        for i in range(num_id):
            res[f"{name}_Prec_Class_{i}"] = precision[i]
            res[f"{name}_Rec_Class_{i}"] = recall[i]
            res[f"{name}_F1_Class_{i}"] = f1[i]
            res[f"{name}_Count_Class_{i}"] = (t == i).sum()

        raw_data = {'p': p, 't': t, 'ids': ids, 'oods': oods, 'ent': ent}
        return res, raw_data

    # --- 1. CALCULATE RAW METRICS ---
    metrics = {}
    
    # A. Unlabeled
    m_unlabeled, unlabeled_raw = calc("Unlabeled", unlabeled_indices)
    metrics.update(m_unlabeled)
    
    # B. Splits (Pool ID / Explore)
    preds = unlabeled_raw['p']
    valid_unlabeled_indices = unlabeled_indices
    # Safety align indices with predictions (if cached)
    if len(preds) < len(unlabeled_indices):
         max_idx = len(cached_features) - 1 if cached_features is not None else float('inf')
         valid_unlabeled_indices = [i for i in unlabeled_indices if i <= max_idx]
    
    if len(preds) == 0: is_predicted_id = []
    else: is_predicted_id = (preds < (args.num_classes if args is not None else 8))
    
    pool_id_indices = [idx for idx, is_id in zip(valid_unlabeled_indices, is_predicted_id) if is_id]
    pool_explore_indices = [idx for idx, is_id in zip(valid_unlabeled_indices, is_predicted_id) if not is_id]
    
    m_pool_id, _ = calc("Pool_ID", pool_id_indices)
    metrics.update(m_pool_id)
    
    m_pool_explore, _ = calc("Pool_Explore", pool_explore_indices)
    metrics.update(m_pool_explore)
    
    # C. Query & Labeled
    m_query, _ = calc("Query", query_indices)
    metrics.update(m_query)
    
    m_labeled, _ = calc("Labeled", labeled_indices)
    metrics.update(m_labeled)

    # --- 2. DEFINE TOTALS FOR RECALL ---
    
    # Total IDs currently in the Unlabeled Set
    if unlabeled_raw is not None and len(unlabeled_raw['t']) > 0:
        total_ids_in_unlabeled = float((unlabeled_raw['t'] < (args.num_classes if args is not None else 8)).sum())
    else:
        total_ids_in_unlabeled = 0.0

    # Total IDs currently in the Labeled Set
    if 'Labeled_Count' in metrics:
        total_ids_in_labeled = metrics['Labeled_Count'] * metrics['Labeled_Purity']
    else:
        total_ids_in_labeled = 0.0

    # GLOBAL TOTAL IDs (Labeled + Unlabeled)
    total_ids_global = total_ids_in_labeled + total_ids_in_unlabeled

    # --- 3. COMPUTE DERIVED METRICS ---
    
    # Helper to apply correct denominator
    def compute_derived(pool_name, denominator, total_ids_in_set):
        purity = metrics.get(f'{pool_name}_Purity', 0.0)
        
        # ID Recall = (IDs in this set) / (Denominator)
        if denominator > 0:
            recall = total_ids_in_set / denominator
        else:
            recall = 0.0
            
        metrics[f'{pool_name}_ID_Recall'] = recall
        
        # F1 Score = 2 * (Purity * Recall) / (Purity + Recall)
        if (purity + recall) > 0:
            metrics[f'{pool_name}_F1'] = (2 * purity * recall) / (purity + recall)
        else:
            metrics[f'{pool_name}_F1'] = 0.0

    # A. Unlabeled Set Recall (Sensitivity of VLM on current pool)
    #    This is: (Predicted IDs that are True) / (Total True IDs in Pool)
    #    Already handled partly above, but let's standardize:
    #    Wait, standard "Unlabeled_ID_Recall" usually means: how many IDs did we *keep* in Pool_ID?
    if total_ids_in_unlabeled > 0:
        ids_in_pool_id = metrics.get('Pool_ID_Count', 0) * metrics.get('Pool_ID_Purity', 0.0)
        metrics['Unlabeled_ID_Recall'] = ids_in_pool_id / total_ids_in_unlabeled
    else:
        metrics['Unlabeled_ID_Recall'] = 0.0

    # B. Pool_ID & Pool_Explore (Relative to Unlabeled Set)
    ids_in_pool_id = metrics.get('Pool_ID_Count', 0) * metrics.get('Pool_ID_Purity', 0.0)
    compute_derived("Pool_ID", denominator=total_ids_in_unlabeled, total_ids_in_set=ids_in_pool_id)
    
    ids_in_pool_exp = metrics.get('Pool_Explore_Count', 0) * metrics.get('Pool_Explore_Purity', 0.0)
    compute_derived("Pool_Explore", denominator=total_ids_in_unlabeled, total_ids_in_set=ids_in_pool_exp)

    # C. Query (Relative to Unlabeled Set)
    #    "What fraction of the available IDs did we pick?"
    ids_in_query = metrics.get('Query_Count', 0) * metrics.get('Query_Purity', 0.0)
    compute_derived("Query", denominator=total_ids_in_unlabeled, total_ids_in_set=ids_in_query)

    # D. Labeled (Relative to GLOBAL TOTAL)
    #    "What fraction of ALL IDs have we collected?"
    compute_derived("Labeled", denominator=total_ids_global, total_ids_in_set=total_ids_in_labeled)

    return metrics, unlabeled_raw

class FeatureCache:
    def __init__(self, dataset_name, client_idx, cache_root=None):
        if cache_root is None:
            self.cache_dir = "./cache"
        else:
            self.cache_dir = cache_root

        base = f"{dataset_name}_client_{client_idx}_features_v3"
        self.filename    = f"{base}.pt"
        self.npy_filename = f"{base}.npy"
        self.filepath    = os.path.join(self.cache_dir, self.filename)
        self.npy_path    = os.path.join(self.cache_dir, self.npy_filename)

    def exists(self):
        return os.path.exists(self.filepath)

    def load(self):
        print(f"[FeatureCache] Loading cached features from {self.filepath}")
        try:
            # Add weights_only=False for PyTorch 2.6+ compatibility with numpy-heavy caches
            return torch.load(self.filepath, weights_only=False)
        except Exception as e:
            print(f"[FeatureCache] Failed to load {self.filepath}: {e}")
            return None

    def load_npy(self):
        """Load only the embeddings numpy array (faster, no torch needed)."""
        if not os.path.exists(self.npy_path):
            return None
        try:
            return np.load(self.npy_path)
        except Exception as e:
            print(f"[FeatureCache] Failed to load npy {self.npy_path}: {e}")
            return None

    def save(self, data):
        print(f"[FeatureCache] Saving features to {self.filepath}")
        os.makedirs(self.cache_dir, exist_ok=True)
        # 1. Save full result dict as .pt
        try:
            torch.save(data, self.filepath)
        except Exception as e:
            print(f"[FeatureCache] Failed to save {self.filepath}: {e}")
        # 2. Save embeddings-only as .npy (for easy numpy inspection)
        embeddings = data.get('embeddings') if isinstance(data, dict) else None
        if embeddings is not None:
            try:
                emb_np = embeddings if isinstance(embeddings, np.ndarray) else embeddings.numpy()
                np.save(self.npy_path, emb_np)
                print(f"[FeatureCache] Also saved embeddings as {self.npy_path}")
            except Exception as e:
                print(f"[FeatureCache] Failed to save npy {self.npy_path}: {e}")


def run_vlm_gated_warmup(dataset, dataset_name, device='cuda', client_idx=0, log_dir=None,
                          feature_cache_root=None):
    feature_cache = FeatureCache(dataset_name, client_idx, cache_root=feature_cache_root)
    
    if feature_cache.exists():
        cached_data = feature_cache.load()
        if cached_data is not None:
            return cached_data

    model, tokenizer, preprocess = get_biomedclip_model(device)
    
    id_texts = PROMPTS.get(dataset_name, {}).get('ID', [])
    ood_texts = PROMPTS.get(dataset_name, {}).get('OOD', [])
    all_texts = id_texts + ood_texts
    num_id = len(id_texts)
    
    with torch.no_grad():
        text_tokens = tokenizer(all_texts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    if dataset_name == 'FedEMBED':
        print(f"[VLM] Using specialized VLMDatasetEMBED for {dataset_name}")
        vlm_dataset = VLMDatasetEMBED(dataset, preprocess)
    else:
        # Default behavior (FedISIC, etc.)
        vlm_dataset = VLMDataset(dataset, preprocess)
    loader = DataLoader(vlm_dataset, batch_size=64, shuffle=False, num_workers=4)

    all_indices = []
    all_preds_is_id = []
    all_id_scores = [] 
    all_ood_scores = []
    all_entropies = []
    all_pseudo_labels = []
    all_img_feats = [] 

    with torch.no_grad():
        for batch_indices, imgs in tqdm(loader, desc="VLM Inference"):
            imgs = imgs.to(device)
            img_feats = model.encode_image(imgs)
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            all_img_feats.append(img_feats.cpu())

            sims = (100.0 * img_feats @ text_features.T).softmax(dim=-1)
            id_soft = sims[:, :num_id].sum(dim=1)
            ood_soft = sims[:, num_id:].sum(dim=1)

            binary_dist = torch.stack([id_soft, ood_soft], dim=1)
            entropy = -(binary_dist * torch.log(binary_dist + 1e-10)).sum(dim=1)

            preds = sims.argmax(dim=1) 
            
            preds[preds >= num_id] = num_id
            
            all_indices.extend(batch_indices.numpy())
            all_preds_is_id.extend((preds < num_id).cpu().numpy())
            all_id_scores.extend(id_soft.cpu().numpy())
            all_ood_scores.extend(ood_soft.cpu().numpy())
            all_entropies.extend(entropy.cpu().numpy())
            all_pseudo_labels.extend(preds.cpu().numpy())

    del model
    torch.cuda.empty_cache()

    full_embeddings = torch.cat(all_img_feats, dim=0).numpy() if all_img_feats else None

    indices_arr = np.array(all_indices)
    mask = np.array(all_preds_is_id)
    
    result = {
        'pool': indices_arr[mask].tolist(), 
        'mask': mask, 
        'scores': np.array(all_id_scores),
        'id_scores': np.array(all_id_scores),
        'ood_scores': np.array(all_ood_scores),
        'entropies': np.array(all_entropies),
        'pseudo_labels': np.array(all_pseudo_labels),
        'embeddings': full_embeddings 
    }
    feature_cache.save(result)

    return result



def save_learnable_vectors(adapter, save_path):
    """
    Saves ONLY the learnable prompt vectors as a lightweight PyTorch checkpoint.
    This allows for both t-SNE analysis AND reloading for inference.
    """
    data_to_save = {}
    
    # 1. Standard CoOp
    if hasattr(adapter, 'ctx_vectors') and adapter.ctx_vectors is not None:
        data_to_save['ctx_vectors'] = adapter.ctx_vectors.detach().cpu()

    # 2. Federated CoOp / CoCoOp (Global + Local)
    if hasattr(adapter, 'global_ctx') and adapter.global_ctx is not None:
        data_to_save['global_ctx'] = adapter.global_ctx.detach().cpu()
        
    if hasattr(adapter, 'local_ctx') and adapter.local_ctx is not None:
        data_to_save['local_ctx'] = adapter.local_ctx.detach().cpu()

    # 3. ResidualFederatedCoOp (shared + residual + gate)
    if hasattr(adapter, 'shared_ctx'):
        data_to_save['shared_ctx'] = adapter.shared_ctx.detach().cpu()
    if hasattr(adapter, 'local_residual'):
        data_to_save['local_residual'] = adapter.local_residual.detach().cpu()
    if hasattr(adapter, 'gate'):
        data_to_save['gate'] = adapter.gate.detach().cpu()
        gate_vals = torch.sigmoid(adapter.gate).detach().cpu()
        data_to_save['gate_sigmoid'] = gate_vals  # for easy analysis

    # Save as .pth (PyTorch format) but ONLY the vectors
    if data_to_save:
        torch.save(data_to_save, save_path)