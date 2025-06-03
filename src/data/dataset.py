import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class DPOdataset(Dataset):
    def __init__(self, image_dir: str, label_dir: str, coarse_box_dir: str):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.coarse_box_dir = coarse_box_dir

        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith(".jpg") or f.endswith(".png")
        ])
        
        self.label_paths = [
            os.path.join(label_dir, os.path.splitext(os.path.basename(p))[0] + ".txt")
            for p in self.image_paths
        ]
        self.label_paths = [self._load_yolo_labels(p) for p in self.label_paths]
        self.coarse_paths = [
            os.path.join(coarse_box_dir, os.path.splitext(os.path.basename(p))[0] + ".npy")
            for p in self.image_paths
        ]
        self.max_gt = 0
        # filter out boxes with exact prediction.
        self.filtered = self.filter_box_and_match(gt=0.85)
        

    def print_max_gt(self):
        print("Max IoU with coarse boxes: ", self.max_gt)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if len(self.filtered[idx]) == 0:
            return None
        else:
            pairs = np.stack(self.filtered[idx], axis=0).astype(np.float32)
            gt_boxes = pairs[:,0]  # [N, 4]
            coarse_boxes = pairs[:,1]           # [M, 4]
            return img, gt_boxes, coarse_boxes

    def _load_yolo_labels(self, label_path: str) -> np.ndarray:
        boxes = []
        with open(label_path, "r") as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) < 5:
                    continue
                _, x, y, w, h = map(float, tokens)
                boxes.append([x, y, w, h])
        return np.array(boxes, dtype=np.float32)
    


    def filter_box_and_match(self, gt : float) -> np.ndarray: # get gt_boxes that is closet to the coarse box and make pairs
        # also filter out coarse boxes that have IoU with gt_boxes greater than gt
        filtered_pairs = []
        for idx in range(len(self.coarse_paths)):
            per_image_pairs = []
            coarse_boxes = np.load(self.coarse_paths[idx])
            for box in range(len(coarse_boxes)):
                coarse_box = self.normalize_box(coarse_boxes[box], img_size=Image.open(self.image_paths[idx]).size)
                gt_box = max(self.label_paths[idx], key  = lambda gt: compute_iou(coarse_box, gt))
                if compute_iou(coarse_box, gt_box) >= gt:
                    self.max_gt = max(self.max_gt, compute_iou(coarse_box, gt_box))
                    continue
                else :
                    per_image_pairs.append([gt_box, coarse_box])
            filtered_pairs.append(per_image_pairs)
        
        return filtered_pairs
    
    def normalize_box(self, box, img_size: tuple) -> np.ndarray:
        """
        Normalize a box in [x, y, w, h] format to [x, y, w, h] in range [0, 1].
        """
        x, y, w, h = box
        img_w, img_h = img_size
        return np.array([x / img_w, y / img_h, w / img_w, h / img_h], dtype=np.float32)


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute IoU between two boxes in [x, y, w, h] format.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    # Convert to [x0, y0, x1, y1]
    x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
    x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2
    x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
    x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2
    #Calculate intersection
    inter_x0 = max(x1_min, x2_min)
    inter_y0 = max(y1_min, y2_min)
    inter_x1 = min(x1_max, x2_max)
    inter_y1 = min(y1_max, y2_max)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    #Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0