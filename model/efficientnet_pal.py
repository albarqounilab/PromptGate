import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn
import os

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):
        super(EfficientNetB0, self).__init__()

        checkpoint_path = 'model/efficientnet_b0_ra-3dd342df.pth'
        if os.path.exists(checkpoint_path):
            self.efficientnet = timm.create_model('efficientnet_b0', pretrained=False, checkpoint_path=checkpoint_path)
        else:
            self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
        
        num_ftrs = self.efficientnet.classifier.in_features
        self.efficientnet.reset_classifier(0)
        
        # 2. Heads
        # Standard Classifier
        self.fc_cls = nn.Linear(num_ftrs, num_classes)
        
        # LfOSA Detection Head (K+1 classes)
        self.fc_lfosa = nn.Linear(num_ftrs, num_classes + 1)
        
        # PAL Open-Set Head (2*K classes) with Weight Norm
        self.fc_open = wn(nn.Linear(num_ftrs, num_classes * 2))

        # Alias
        self.fc = self.fc_cls

        self.block_features = []
        self.register_hooks()

    @property
    def K(self):
        return self.fc_cls.out_features

    def register_hooks(self):
        self.efficientnet.blocks[1].register_forward_hook(self.hook_block_forward)
        self.efficientnet.blocks[2].register_forward_hook(self.hook_block_forward)
        self.efficientnet.blocks[3].register_forward_hook(self.hook_block_forward)
        self.efficientnet.blocks[4].register_forward_hook(self.hook_block_forward)

    def hook_block_forward(self, module, input, output):
        self.block_features.append(output)

    def forward(self, x, embedding=False, detect=False, open_head=False):
        self.block_features = []
        
        if embedding:
            feat = x
        else:
            feat = self.efficientnet(x)

        # PAL Path: Returns (B, 2*K)
        if open_head:
            return self.fc_open(feat)

        # LfOSA Path
        if detect:
            return self.fc_lfosa(feat)

        # Standard Path
        out = self.fc_cls(feat)
        probs = F.softmax(out, dim=1)
        
        return out, probs, feat, self.block_features