import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class CoarseBoxesDataset(Dataset):
    def __init__(self, image_dir: str, label_dir: str, coarse_box_dir: str):
        """
        Dataset that loads image, GT box, and precomputed YOLO coarse boxes.

        Args:
            image_dir: path to images (e.g., dataset/train/images)
            label_dir: path to labels (YOLO format)
            coarse_box_dir: path to saved YOLO coarse box .npy files
        """
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(".jpg") or f.endswith(".png")
        ])
        self.label_paths = [
            os.path.join(label_dir, os.path.splitext(os.path.basename(p))[0] + ".txt")
            for p in self.image_paths
        ]
        self.labels = [self._load_groundtruth(label_path) for label_path in self.label_paths]
        self.coarse_paths = [
            os.path.join(coarse_box_dir, os.path.splitext(os.path.basename(p))[0] + ".npy")
            for p in self.image_paths
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        # gt_boxes = self._load_groundtruth(self.label_paths[idx])  # [N, 4]
        gt_boxes = self.labels[idx]
        # Check if gt_boxes is inf
        if np.isinf(gt_boxes).any():
            print("infinite from here!")
            quit()
        coarse_boxes = np.load(self.coarse_paths[idx])            # [M, 5]

        return img, gt_boxes, coarse_boxes

    def _load_groundtruth(self, label_path: str) -> np.ndarray:
        boxes = []
        with open(label_path, "r") as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) != 5:
                    print(f"Skipping {label_path} because it has {len(tokens)} tokens")
                    continue
                _, x, y, w, h = map(float, tokens)
                boxes.append([x, y, w, h])
        return boxes
        # return np.array(boxes, dtype=np.float32)
