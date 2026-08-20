import torch
import numpy as np
import pyvista
from Model.Model import RGCNN_Seg
from .Dataset.CreateDefectDataset import DeformedObject

def load_model(model_path, vertice=4096, num_classes=6):
    model = RGCNN_Seg(
        vertice=vertice,
        F=[128, 512, 1024, 512, 128, 70],
        K=[6, 5, 3, 1, 1, 1],
        num_classes=num_classes,
        num_categories=1
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict_cloud(model, points, cat=0, threshold=0.6):
    pts = torch.from_numpy(points).float().unsqueeze(0)
    cat_t = torch.tensor([cat], dtype=torch.long)

    with torch.no_grad():
        logits = model(pts, cat_t)
        probs = torch.softmax(logits, dim=2)       
        probs_np = probs.squeeze(0).cpu().numpy() 
        max_probs = np.max(probs_np, axis=1)       
        pred_labels = np.argmax(probs_np, axis=1)
        pred_labels[max_probs < threshold] = 0

    return pred_labels, points, probs_np

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

def plot_comparison(points, pred_labels, title=""):
    plotter = pyvista.Plotter()
    plotter.add_points(points, scalars=pred_labels, cmap=['blue','red','red','red','red','red'], 
                       point_size=8, render_points_as_spheres=False)
    plotter.show(title=title, interactive=True)

def compute_defect_metrics(points, labels, class_ids=[1, 2, 3, 4, 5]):
    metrics = {}
    for cls in class_ids:
        mask = (labels == cls)
        pts = points[mask]
        if len(pts) < 3:
            metrics[cls] = None
            continue

        centered = pts - pts.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        proj = centered @ eigvecs
        ranges = proj.max(axis=0) - proj.min(axis=0)
        sorted_ranges = np.sort(ranges)[::-1]

        metrics[cls] = {
            'depth': sorted_ranges[2],       
            'min_diameter': sorted_ranges[1],
            'max_diameter': sorted_ranges[0] 
        }

    print(metrics)

if __name__ == "__main__":
    from .Dataset.Augmentation import ObjectAugment
    input_file = "teapot.txt"
    model = load_model("best_model_28_epoch.pth", num_classes=6)
    original = np.loadtxt(input_file)
    indices = np.random.choice(len(original), model.vertice, replace=False)
    original = original[indices]
    min_pt = np.min(original, axis=0)
    max_pt = np.max(original, axis=0)
    scene_size = np.linalg.norm(max_pt - min_pt)

    base_obj = DeformedObject(original.copy())

    original_augmented = ObjectAugment(original.copy()).full_augment()
    results = [("WithoutDeffects", original, np.zeros(len(original), dtype=np.int64))]
    
    avg_spacing = 2*scene_size / (model.vertice ** (1/3))
    low_random_spacing, high_random_spacing = avg_spacing * 0.5, avg_spacing * 1
    radius = np.random.uniform(low_random_spacing, high_random_spacing)
    depth = np.random.uniform(low_random_spacing, high_random_spacing)
    height = np.random.uniform(low_random_spacing, high_random_spacing)
    length = np.random.uniform(low_random_spacing, high_random_spacing)
    width = np.random.uniform(low_random_spacing, high_random_spacing)
    amplitude = np.random.uniform(low_random_spacing, high_random_spacing)

    dent_obj = DeformedObject(original.copy())
    dent_obj.create_dent(radius=radius, depth=depth)
    dent_obj.full_augment()
    results.append(("Dent", dent_obj.points, dent_obj.labels))

    bump_obj = DeformedObject(original.copy())
    bump_obj.create_bump(radius=radius, height=height)
    bump_obj.full_augment()
    results.append(("Bump", bump_obj.points, bump_obj.labels))

    chip_obj = DeformedObject(original.copy())
    chip_obj.create_chip(radius=radius, depth=depth)
    chip_obj.full_augment()
    results.append(("Chip", chip_obj.points, chip_obj.labels))

    scratch_obj = DeformedObject(original.copy())
    scratch_obj.create_scratch(length=length, width=width, depth=depth)
    scratch_obj.full_augment()
    results.append(("Scratch", scratch_obj.points, scratch_obj.labels))

    local_obj = DeformedObject(original.copy())
    local_obj.create_local_deformation(radius=radius, amplitude=amplitude)
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
        compute_defect_metrics(selected_pts, pred_labels)
        
        plot_comparison(selected_pts, pred_labels, title=name)