# Dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from .CreateDefectDataset import DeformedObject

class ObjectsDataset(Dataset):
    def __init__(self, input_points, num_points=11755, category=0):
        self.num_points = num_points
        self.category = category
        self.original_points = input_points
        if len(self.original_points) >= num_points:
            indices = np.random.choice(len(self.original_points), num_points, replace=False)
            self.original_points = self.original_points[indices]
        else:
            repeat = num_points // len(self.original_points) + 1
            expanded = np.tile(self.original_points, (repeat, 1))
            noise = np.random.normal(0, 0.001, size=expanded.shape)
            expanded += noise
            self.original_points = expanded[:num_points]


    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        points = self.original_points.copy()
        obj = DeformedObject(points)
        points_aug, labels = obj.create_random_defects(
            min_defects=1,
            max_defects=10,
            seed=idx
        )
        labels = labels.astype(np.int64)
        pts_tensor = torch.from_numpy(points_aug).float()
        labels_tensor = torch.from_numpy(labels).long()
        cat_tensor = torch.tensor(self.category, dtype=torch.long)
        print(f"Уникальных меток: {np.unique(labels)}, в количестве этих классов: {np.bincount(labels)}")
        return pts_tensor, labels_tensor, cat_tensor