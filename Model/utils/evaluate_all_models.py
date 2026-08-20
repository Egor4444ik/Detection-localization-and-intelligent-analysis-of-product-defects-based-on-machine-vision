import torch
import numpy as np
import os
from Model.Model import RGCNN_Seg
from Model.Dataset.Dataset import ObjectsDataset
from Model.utils.Metrix import evaluate_model
from torch.utils.data import DataLoader

def evaluate_models(model_dir, model_names, points_file, vertice=4096, batch_size=8, num_workers=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    points = np.loadtxt(points_file)
    val_dataset = ObjectsDataset(points, num_points=vertice, category=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    F = [128, 512, 1024, 512, 128, 70]
    K = [6, 5, 3, 1, 1, 1]
    num_classes = 6
    num_categories = 1
    regularization = 1e-9

    print(f"Using device: {device}")
    print(f"{'Model':<25} {'Val Loss':>10} {'Val Acc':>10} {'Val mIoU':>10}")
    print("-" * 60)

    for name in sorted(model_names):
        model_path = os.path.join(model_dir, name)
        if not os.path.exists(model_path):
            continue
        model = RGCNN_Seg(vertice, F, K, num_classes, num_categories, regularization).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only = True))
        model.eval()
        val_loss, val_acc, val_iou = evaluate_model(model, val_loader, device, num_classes)
        print(f"{name:<25} {val_loss:>10.4f} {val_acc:>10.4f} {val_iou:>10.4f}")

if __name__ == "__main__":
    model_dir = "."
    model_files = [f"best_model_{i}_epoch.pth" for i in range(1, 29)]
    existing_files = [f for f in model_files if os.path.exists(os.path.join(model_dir, f))]
    evaluate_models(model_dir, existing_files, "teapot.txt")