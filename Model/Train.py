import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .Model import RGCNN_Seg
from .Dataset.Dataset import ObjectsDataset
from .utils.Metrix import evaluate_model
import numpy as np

def train(input_3_dim_object_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    points = np.loadtxt(input_3_dim_object_path)
    if points.ndim == 1:
        points = points.reshape(-1, 3)
    print(f"Loaded {len(points)} points.")

    vertice = 2048
    F = [128, 512, 1024, 512, 128, 6]
    K = [6, 5, 3, 1, 1, 1]
    num_classes = 6
    num_categories = 1
    regularization = 1e-9

    model = RGCNN_Seg(vertice, F, K, num_classes, num_categories, regularization).to(device)

    dataset = ObjectsDataset(points, num_points=vertice, augment=True, category=0)
    val_dataset = ObjectsDataset(points, num_points=vertice, augment=False, category=0)

    train_loader = DataLoader(dataset, batch_size=26, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=26, shuffle=False, num_workers=4)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 50
    eval_freq = 30
    step = 0
    best_iou = 0.0

    for epoch in range(num_epochs):
        model.train()
        for pts, labels, cat in train_loader:
            pts, labels, cat = pts.to(device), labels.to(device), cat.to(device)
            optimizer.zero_grad()
            logits = model(pts, cat)
            loss = criterion(logits.permute(0, 2, 1), labels)

            reg_loss = sum(p.norm(2) for p in model.parameters()) * 1e-5
            total_loss = loss + reg_loss

            total_loss.backward()
            optimizer.step()

            step += 1

            if step % eval_freq == 0:
                val_loss, val_acc, val_iou = evaluate_model(model, val_loader, device, num_classes)
                print(f"Step {step}, Epoch {epoch+1:.2f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val mIoU: {val_iou:.4f}")
                if val_iou > best_iou:
                    best_iou = val_iou
                    torch.save(model.state_dict(), "best_model.pth")
                    print("  -> New best model saved.")

        print(f"Epoch {epoch+1} finished.")

    print(f"Training completed. Best mIoU: {best_iou:.4f}")

if __name__ == "__main__":
    train("teapot.txt")