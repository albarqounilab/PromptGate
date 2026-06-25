import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn

# LINEAR CLASSIFIER FOR FEDEMBED 
class LinearClassifier(nn.Module):
    def __init__(self, in_dim=2048, num_classes=5, dropout=0.2):
        super().__init__()
        self.dropout  = nn.Dropout(dropout)
        self.fc_cls   = nn.Linear(in_dim, num_classes)      # classifier
        self.fc_open  = wn(nn.Linear(in_dim, 2 * num_classes))  # PAL head
        self.fc_lfosa = nn.Linear(in_dim, num_classes + 1)      # LFOSA head
        self.fc = self.fc_cls

    @property
    def K(self):
        return self.fc_cls.out_features

    def forward(self, x, *, open_head=False, detect=False, embedding=False, return_feat=False):
        # x is the precomputed 2048-D feature vector
        feat = x.float()
        
        if open_head:
            out = self.fc_open(feat)
            return (out, feat) if return_feat else out

        if detect:
            out = self.fc_lfosa(feat)
            return (out, feat) if return_feat else out

        # cls path
        logits = self.fc_cls(self.dropout(feat))

        # IMPORTANT: expose a 4D block built from *feat*, not logits
        block = feat.unsqueeze(-1).unsqueeze(-1)  # (B, in_dim, 1, 1)
        
        # Match EfficientNet-like return: (logits, softmax, feat, block_list)
        return logits, F.softmax(logits, dim=-1), feat, [block]

    def forward_multi(self, x):
        feat = x.float()
        return feat, self.fc_cls(feat), self.fc_open(feat), self.fc_lfosa(feat)