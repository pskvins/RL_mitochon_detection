import ultralytics
import os
import numpy as np
from src.env.utils import compute_iou
import matplotlib.pyplot as plt
from PIL import Image
import sys
model = ultralytics.YOLO("yolov8n.pt")

# Read in the coarse boxes and labels
coarse_boxes_dir = "datasets/train/coarse_boxes_yolo_base"
labels_dir = "datasets/train/labels"
image_dir = "datasets/train/images"

coarse_boxes = {}
labels = {}
image_sizes = {}

# Read in the coarse boxes and labels
coarse_boxes_list = os.listdir(coarse_boxes_dir)
labels_list = os.listdir(labels_dir)

for coarse_box in coarse_boxes_list:
    # Name without .npy
    coarse_box_np = np.load(os.path.join(coarse_boxes_dir, coarse_box))
    # Sort the row by the last column
    coarse_box_np = coarse_box_np[np.argsort(coarse_box_np[:, -1])]
    coarse_boxes[coarse_box.split(".npy")[0]] = coarse_box_np

for image_file in os.listdir(image_dir):
    if image_file.endswith(".jpg"):
        image_sizes[image_file.split(".jpg")[0]] = Image.open(os.path.join(image_dir, image_file)).convert("RGB").size

for label_file in labels_list:
    # Name without .txt
    name = label_file.split(".txt")[0]
    label = []
    with open(os.path.join(labels_dir, label_file), "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 5:
                continue
            _, x, y, w, h = map(float, tokens)
            image_x, image_y = image_sizes[name]
            label.append([x * image_x, y * image_y, w * image_x, h * image_y])
    labels[name] = np.array(label)

def calculate_precision_recall_f1(pred_boxes, gt_boxes, iou_threshold=0.5, conf_threshold=0.3):
    """
    pred_boxes: np.ndarray of shape [N_pred, 5]
    gt_boxes: np.ndarray of shape [N_gt, 4]
    iou_threshold: float, IoU threshold to consider a prediction as a true positive
    conf_threshold: float, confidence threshold to filter predictions

    Returns: precision, recall, f1
    """
    # Filter predictions by confidence
    pred_boxes = pred_boxes[pred_boxes[:, -1] >= conf_threshold]
    matched_gt = set()
    tp = 0
    fp = 0

    for pred in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

# pred_boxes: np.ndarray of shape [N_pred, 5]
# gt_boxes: np.ndarray of shape [N_gt, 4]
precision_scores_list = []
recall_scores_list = []
f1_scores_list = []
for name in coarse_boxes.keys():
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for conf in range(3, 10, 1):
        conf = conf / 10
        pred_boxes = coarse_boxes[name]
        gt_boxes = labels[name]
        precision, recall, f1 = calculate_precision_recall_f1(pred_boxes, gt_boxes, iou_threshold=0.5, conf_threshold=conf)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    precision_scores_list.append(precision_scores)
    recall_scores_list.append(recall_scores)
    f1_scores_list.append(f1_scores)
precision_scores_list = np.mean(precision_scores_list, axis=0)
recall_scores_list = np.mean(recall_scores_list, axis=0)
f1_scores_list = np.mean(f1_scores_list, axis=0)

print(f"Precision: {precision_scores_list}")
print(f"Recall: {recall_scores_list}")
print(f"F1: {f1_scores_list}")

x_range = np.arange(0.3, 1.0, 0.1)

plt.plot(x_range, precision_scores_list)
plt.ylim(0, 1)
plt.savefig("precision_scores_base_train.png")
plt.close()

plt.plot(x_range, recall_scores_list)
plt.ylim(0, 1)
plt.savefig("recall_scores_base_train.png")
plt.close()

plt.plot(x_range, f1_scores_list)
plt.ylim(0, 1)
plt.savefig("f1_scores_base_train.png")
plt.close()