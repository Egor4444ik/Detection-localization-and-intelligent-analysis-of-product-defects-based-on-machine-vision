# UseModel.py
import torch
import numpy as np
import pyvista
from Model.Model import RGCNN_Seg
from .Dataset.CreateDefectDataset import DeformedObject

def load_model(model_path, vertice=2048, num_classes=6):
    model = RGCNN_Seg(
        vertice=vertice,
        F=[128, 512, 1024, 512, 128, 6],
        K=[6, 5, 3, 1, 1, 1],
        num_classes=num_classes,
        num_categories=1
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_cloud(model, points, cat=0):
    if len(points) >= 2048:
        idx = np.random.choice(len(points), 2048, replace=False)
    else:
        idx = np.random.choice(len(points), 2048, replace=True)
    selected = points[idx]
    pts = torch.from_numpy(selected).float().unsqueeze(0)
    cat_t = torch.tensor([cat], dtype=torch.long)

    with torch.no_grad():
        logits = model(pts, cat_t)
        probs = torch.softmax(logits, dim=2)
        probs_np = probs.squeeze(0).cpu().numpy()
        pred_labels = np.argmax(probs_np, axis=1)
    return pred_labels, selected, probs_np

def compute_metrics(pred_labels, true_labels, num_classes=6):
    pred = pred_labels.astype(np.int64)
    true = true_labels.astype(np.int64)
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(num_classes):
        for j in range(num_classes):
            conf[i, j] = np.sum((true == i) & (pred == j))
    acc_per_class = np.diag(conf) / (np.sum(conf, axis=1) + 1e-8)
    intersection = np.diag(conf)
    union = np.sum(conf, axis=1) + np.sum(conf, axis=0) - intersection
    iou_per_class = intersection / (union + 1e-8)
    overall_acc = np.sum(np.diag(conf)) / np.sum(conf)
    mean_iou = np.mean(iou_per_class)
    return conf, acc_per_class, iou_per_class, overall_acc, mean_iou

def print_statistics(name, true_labels, pred_labels, probs, num_classes=6):
    print(f"\n{'='*60}")
    print(f"Объект: {name}")
    print(f"Количество точек: {len(true_labels)}")
    
    true_counts = np.bincount(true_labels, minlength=num_classes)
    print("\nРаспределение истинных меток:")
    for c in range(num_classes):
        print(f"  Class {c}: {true_counts[c]} points ({100*true_counts[c]/len(true_labels):.2f}%)")
    
    pred_counts = np.bincount(pred_labels, minlength=num_classes)
    print("\nРаспределение предсказанных меток:")
    for c in range(num_classes):
        print(f"  Class {c}: {pred_counts[c]} points ({100*pred_counts[c]/len(pred_labels):.2f}%)")
    
    mean_probs = np.mean(probs, axis=0)
    print("\nСредние вероятности по классам:")
    for c in range(num_classes):
        print(f"  Class {c}: {mean_probs[c]:.4f}")
    
    conf, acc, iou, overall_acc, mean_iou = compute_metrics(pred_labels, true_labels, num_classes)
    print(f"\nOverall Accuracy: {overall_acc:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print("Per-class IoU:")
    for c in range(num_classes):
        print(f"  Class {c}: {iou[c]:.4f}")
    print("Per-class Accuracy:")
    for c in range(num_classes):
        print(f"  Class {c}: {acc[c]:.4f}")
    print(f"\nConfusion matrix (rows=true, cols=pred):")
    print(conf)

def plot_comparison(points, true_labels, pred_labels, title=""):
    plotter = pyvista.Plotter(shape=(1, 2))
    plotter.subplot(0, 0)
    plotter.add_text("Настоящие метки")
    plotter.add_points(points, scalars=true_labels, cmap=['blue','red','green','yellow','magenta','brown'], 
                       point_size=5, render_points_as_spheres=False)
    plotter.subplot(0, 1)
    plotter.add_text("Предсказанные метки")
    plotter.add_points(points, scalars=pred_labels, cmap=['blue','red','green','yellow','magenta','brown'],
                       point_size=5, render_points_as_spheres=False)
    plotter.show(title=title, interactive=True)

if __name__ == "__main__":
    from.Dataset.Augmentation import ObjectAugment
    input_file = "teapot.txt"
    model = load_model("best_model.pth", num_classes=6)
    original = np.loadtxt(input_file)
    min_pt = np.min(original, axis=0)
    max_pt = np.max(original, axis=0)
    scene_size = np.linalg.norm(max_pt - min_pt)

    base_obj = DeformedObject(original.copy())

    original_augmented = ObjectAugment(original.copy()).full_augment()
    results = [("WithoutDeffects", original, np.zeros(len(original_augmented), dtype=np.int64))]

    dent_obj = DeformedObject(original.copy())
    dent_obj.create_dent(radius=scene_size*0.04, depth=scene_size*0.01)
    dent_obj.full_augment()
    results.append(("Dent", dent_obj.points, dent_obj.labels))

    bump_obj = DeformedObject(original.copy())
    bump_obj.create_bump(radius=scene_size*0.04, height=scene_size*0.01)
    bump_obj.full_augment()
    results.append(("Bump", bump_obj.points, bump_obj.labels))

    chip_obj = DeformedObject(original.copy())
    chip_obj.create_chip(radius=scene_size*0.04, depth=scene_size*0.015)
    chip_obj.full_augment()
    results.append(("Chip", chip_obj.points, chip_obj.labels))

    scratch_obj = DeformedObject(original.copy())
    scratch_obj.create_scratch(length=scene_size*0.12, width=scene_size*0.012, depth=scene_size*0.006)
    scratch_obj.full_augment()
    results.append(("Scratch", scratch_obj.points, scratch_obj.labels))

    local_obj = DeformedObject(original.copy())
    local_obj.create_local_deformation(radius=scene_size*0.05, amplitude=scene_size*0.008)
    local_obj.full_augment()
    results.append(("LocalDeformation", local_obj.points, local_obj.labels))

    for name, pts, lbls in results:
        pred_labels, selected_pts, probs = predict_cloud(model, pts, cat=0)
        pts_tensor = torch.from_numpy(selected_pts).float().unsqueeze(0)
        cat_t = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            logits = model(pts_tensor, cat_t)
            probs = torch.softmax(logits, dim=2).squeeze(0).cpu().numpy()
            pred_labels = np.argmax(probs, axis=1)
        
        print_statistics(name, pred_labels, pred_labels, probs)
        
        plot_comparison(selected_pts, pred_labels, pred_labels, title=name)