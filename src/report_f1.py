import ultralytics
import os
import numpy as np
from src.env.utils import compute_iou
import matplotlib.pyplot as plt
from PIL import Image
import sys
import pandas as pd
model = ultralytics.YOLO("yolov8n.pt")

# Read in the coarse boxes and labels
coarse_boxes_dir = "ddpg_runs/ddpg_exp1/boxes"
labels_dir = "datasets/valid/labels"
image_dir = "datasets/valid/images"

coarse_boxes = {}
labels = {}
image_sizes = {}

# Read in the coarse boxes and labels
coarse_boxes_list = os.listdir(coarse_boxes_dir)
coarse_boxes_list = [x for x in coarse_boxes_list if x.endswith(".npy")]
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

def calculate_real_precision_recall_f1(pred_boxes, gt_boxes, image_x, image_y, conf_threshold=0.3):
    """
    pred_boxes: np.ndarray of shape [N_pred, 5]
    gt_boxes: np.ndarray of shape [N_gt, 4]
    iou_threshold: float, IoU threshold to consider a prediction as a true positive
    conf_threshold: float, confidence threshold to filter predictions

    Returns: precision, recall, f1
    """
    # Filter predictions by confidence
    pred_boxes = pred_boxes[pred_boxes[:, -1] >= conf_threshold]
    gt_area = np.zeros((image_x, image_y)).astype(bool)
    for gt in gt_boxes:
        gt_area[int(gt[0]):int(gt[0])+int(gt[2]), int(gt[1]):int(gt[1])+int(gt[3])] = True
    pred_area = np.zeros((image_x, image_y)).astype(bool)
    for pred in pred_boxes:
        pred_area[int(pred[0]):int(pred[0])+int(pred[2]), int(pred[1]):int(pred[1])+int(pred[3])] = True

    # Compute IoU
    intersection = np.sum(pred_area & gt_area)

    precision = intersection / np.sum(pred_area) if np.sum(pred_area) > 0 else 0.0
    recall = intersection / np.sum(gt_area)
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

real_precision_scores_list = []
real_recall_scores_list = []
real_f1_scores_list = []
for name in coarse_boxes.keys():
    precision_scores = []
    recall_scores = []
    f1_scores = []
    for conf in range(3, 10, 1):
        conf = conf / 10
        pred_boxes = coarse_boxes[name]
        gt_boxes = labels[name]
        image_x, image_y = image_sizes[name]
        precision, recall, f1 = calculate_real_precision_recall_f1(pred_boxes, gt_boxes, image_x, image_y, conf_threshold=conf)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    real_precision_scores_list.append(precision_scores)
    real_recall_scores_list.append(recall_scores)
    real_f1_scores_list.append(f1_scores)
real_precision_scores_list = np.mean(real_precision_scores_list, axis=0)
real_recall_scores_list = np.mean(real_recall_scores_list, axis=0)
real_f1_scores_list = np.mean(real_f1_scores_list, axis=0)

print(f"Precision: {precision_scores_list}")
print(f"Recall: {recall_scores_list}")
print(f"F1: {f1_scores_list}")

print(f"Real Precision: {real_precision_scores_list}")
print(f"Real Recall: {real_recall_scores_list}")
print(f"Real F1: {real_f1_scores_list}")

x_range = np.arange(0.3, 1.0, 0.1)

plt.plot(x_range, precision_scores_list)
plt.ylim(0, 1)
plt.savefig(os.path.join(coarse_boxes_dir, "precision_scores_ddpg_val.png"))
plt.close()

plt.plot(x_range, recall_scores_list)
plt.ylim(0, 1)
plt.savefig(os.path.join(coarse_boxes_dir, "recall_scores_ddpg_val.png"))
plt.close()

plt.plot(x_range, f1_scores_list)
plt.ylim(0, 1)
plt.savefig(os.path.join(coarse_boxes_dir, "f1_scores_ddpg_val.png"))
plt.close()

plt.plot(x_range, real_precision_scores_list)
plt.ylim(0, 1)
plt.savefig(os.path.join(coarse_boxes_dir, "real_precision_scores_ddpg_val.png"))
plt.close()

plt.plot(x_range, real_recall_scores_list)
plt.ylim(0, 1)
plt.savefig(os.path.join(coarse_boxes_dir, "real_recall_scores_ddpg_val.png"))
plt.close()

plt.plot(x_range, real_f1_scores_list)
plt.ylim(0, 1)
plt.savefig(os.path.join(coarse_boxes_dir, "real_f1_scores_ddpg_val.png"))
plt.close()

# Save all the results in a csv file
results_df = pd.DataFrame({
    "precision": precision_scores_list,
    "recall": recall_scores_list,
    "f1": f1_scores_list,
    "real_precision": real_precision_scores_list,
    "real_recall": real_recall_scores_list,
    "real_f1": real_f1_scores_list
})
results_df.to_csv(os.path.join(coarse_boxes_dir, "results.csv"), index=False)