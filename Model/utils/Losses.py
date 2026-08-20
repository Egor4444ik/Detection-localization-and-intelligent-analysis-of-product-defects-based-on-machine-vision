import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def dice_loss(pred, target, num_classes=6, smooth=1e-6):
    pred = pred.permute(0, 2, 1)
    pred_soft = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 2, 1).float()
    intersection = (pred_soft * target_one_hot).sum(dim=(1, 2))
    union = pred_soft.sum(dim=(1, 2)) + target_one_hot.sum(dim=(1, 2))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pts = None, labels = None, vertice = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pts = pts
        self.labels = labels
        self.vertice = vertice

    def forward(self, inputs, targets):
        inputs = inputs.permute(0, 2, 1)
        device = inputs.device

        pts = self.pts
        labels = self.labels 
        vertice = self.vertice

        object_pts_count = len(self.pts[labels==0])
        dent_pts_count = len(pts[labels==1])
        bump_pts_count = len(pts[labels==2])
        chip_pts_count = len(pts[labels==3])
        scratch_pts_count = len(pts[labels==4])
        local_deformation_pts_count = len(pts[labels==5])
        
        object_percent = object_pts_count/(vertice - object_pts_count) if (vertice != object_pts_count) else 1
        dent_pts_percent = dent_pts_count/(vertice - dent_pts_count) if (vertice != dent_pts_count) else 1
        bump_pts_percent = bump_pts_count/(vertice - bump_pts_count) if (vertice != bump_pts_count) else 1
        chip_pts_percent = chip_pts_count/(vertice - chip_pts_count) if (vertice != chip_pts_count) else 1
        scratch_pts_percent = scratch_pts_count/(vertice - scratch_pts_count) if (vertice != scratch_pts_count) else 1/6
        local_deformation_pts_percent = local_deformation_pts_count/(vertice - local_deformation_pts_count) if (vertice != local_deformation_pts_count) else 1/6
    
        object_weight = np.clip(1/(6*object_percent), 1/6, 5/6) if dent_pts_percent != 0 else 1/6
        dent_weight = np.clip(1/(6*dent_pts_percent), 1/6, 5/6) if dent_pts_percent != 0 else 1/6
        bump_weight = np.clip(1/(6*bump_pts_percent), 1/6, 5/6) if bump_pts_percent != 0 else 1/6
        chip_weight = np.clip(1/(6*chip_pts_percent), 1/6, 5/6) if chip_pts_percent != 0 else 1/6
        scratch_weight = np.clip(1/(6*scratch_pts_percent), 1/6, 5/6) if scratch_pts_percent != 0 else 1/6
        local_deformation_pts_weight = np.clip(1/(6*local_deformation_pts_percent), 1/6, 5/6) if local_deformation_pts_percent != 0 else 1/6
    
        class_weights = torch.tensor([object_weight, dent_weight, bump_weight, chip_weight, scratch_weight, local_deformation_pts_weight], dtype=torch.float32).to(device)

        ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()