import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, num_classes=6, smooth=1e-6):
    pred = pred.permute(0, 2, 1)
    pred_soft = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 2, 1).float()  # (B, C, M)
    intersection = (pred_soft * target_one_hot).sum(dim=(1, 2))
    union = pred_soft.sum(dim=(1, 2)) + target_one_hot.sum(dim=(1, 2))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, class_weights = [1.0, 100.0, 100.0, 100.0, 100.0, 100.0]):
        super().__init__()
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = inputs.permute(0, 2, 1)
        device = inputs.device
        class_weights = torch.tensor(self.class_weights).to(device)
        ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()