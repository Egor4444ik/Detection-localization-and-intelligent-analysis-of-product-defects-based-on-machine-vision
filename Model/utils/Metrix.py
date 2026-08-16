# utils.py
import numpy as np
import torch

def compute_iou(pred, target, num_classes):
    iou_list = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        iou_list.append(iou)
    mean_iou = np.mean(iou_list)
    return mean_iou, iou_list

def evaluate_model(model, dataloader, device, num_classes):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_iou = 0
    total_samples = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for pts, labels, cat in dataloader:
            pts, labels, cat = pts.to(device), labels.to(device), cat.to(device)
            logits = model(pts, cat)
            loss = criterion(logits.permute(0,2,1), labels)
            total_loss += loss.item() * pts.size(0)

            pred = torch.argmax(logits, dim=2)
            acc = (pred == labels).float().mean()
            total_acc += acc.item() * pts.size(0)

            iou, _ = compute_iou(pred.cpu().numpy(), labels.cpu().numpy(), num_classes)
            total_iou += iou * pts.size(0)
            total_samples += pts.size(0)

    return total_loss / total_samples, total_acc / total_samples, total_iou / total_samples