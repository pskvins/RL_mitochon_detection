import os
import yaml
import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score

from src.ddpg_model.agent import DDPGAgent
from src.env.utils import compute_iou, ResNet18FeatureExtractor
from src.env.box_env import BoxRefinementEnv
from src.data.coarse_boxes_loader import CoarseBoxesDataset
from src.data.generate_coarse_boxes import generate_coarse_boxes



# -----------------------
# CONFIG
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

#path configs
path_cfg = cfg["paths"]
image_dir = path_cfg["image_dir"]
label_dir = path_cfg["label_dir"]
coarse_dir = path_cfg["coarse_dir"]
yolo_path = path_cfg["yolo_path"]
pt_path = path_cfg['pt_path']

#agent
agent_cfg = cfg["agent"]
state_dim = agent_cfg["state_dim"]
action_dim = agent_cfg["action_dim"]
max_step = agent_cfg['max_step']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DDPGAgent(state_dim, action_dim, device)
agent.load(pt_path)
feature_extractor = ResNet18FeatureExtractor(device=device)

#save
save_cfg = cfg['save']
save_path = save_cfg['save_path']
save_figs = save_cfg["save_figs"]



if os.path.exists(coarse_dir) and \
    len(os.listdir(coarse_dir)) > 0:
    print("Using existing coarse boxes.")
else:
    print("No coarse boxes found. Running YOLO to generate them...")
    generate_coarse_boxes(
        image_dir=image_dir,
        output_dir=coarse_dir,
        model_path=yolo_path,
        conf_threshold=cfg.get('yolo_conf', 0.3)
    )
    
dataset = CoarseBoxesDataset(
    image_dir = image_dir,
    label_dir = label_dir,
    coarse_box_dir = coarse_dir
)



def compute_metrics(refined_boxes, gt_boxes, iou_threshold=0.5):
    """
    Funciton to compute metrics
    """
    matched_gt = set()
    tp, fp = 0, 0
    all_preds, all_gts, all_scores = [], [], []

    for r_box in refined_boxes:
        best_iou = 0
        best_idx = -1
        for i, g in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = compute_iou(r_box, g)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        all_scores.append(best_iou)
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_idx)
            all_preds.append(1)
        else:
            fp += 1
            all_preds.append(0)

    fn = len(gt_boxes) - len(matched_gt)
    all_gts.extend([1] * len(gt_boxes))
    all_preds.extend([0] * fn)
    all_scores.extend([0.0] * fn)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    ap_50 = precision
    return precision, recall, f1, ap_50, all_preds, all_gts, all_scores





os.makedirs(save_path, exist_ok=True)
metrics_list = []
all_preds_total, all_gts_total, all_scores_total = [], [], []
with torch.no_grad():
    for i in tqdm(range(len(dataset))):
        img, gt_boxes, coarse_boxes = dataset[i]
        img_w, img_h = img.size

        if np.max(gt_boxes) <= 1.0:
            gt_boxes = np.array([
                [x * img_w, y * img_h, w * img_w, h * img_h]
                for x, y, w, h in gt_boxes
            ], dtype=np.float32)

        refined_boxes = []

        for box in coarse_boxes:
            box = np.asarray(box, dtype=np.float32).reshape(4,)
            env = BoxRefinementEnv(
                image=img,
                gt_boxes=gt_boxes,
                initial_box=box,
                feature_extractor=feature_extractor,
                iou_fn=lambda a, b: 0.0 #dummy reward function
            )
            state = env.reset()
            for _ in range(max_step):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = agent.select_action(state_tensor, noise_std=0.0)
                action_np = action.squeeze(0).cpu().numpy()
                next_state, _, done, _ = env.step(action_np)
                state = next_state
                if done:
                    break
            refined_boxes.append(env.cur_box.copy())


        precision, recall, f1, ap_50, preds, gts, scores = compute_metrics(refined_boxes, gt_boxes)
        metrics_list.append((precision, recall, f1, ap_50))
        all_preds_total.extend(preds)
        all_gts_total.extend(gts)
        all_scores_total.extend(scores)

        if save_figs:
            vis_img = img.copy()
            draw = ImageDraw.Draw(vis_img)
            for box in coarse_boxes:
                x, y, w, h = box
                draw.rectangle([x-w/2, y-h/2, x+w/2, y+h/2], outline="blue", width=2)
            for box in refined_boxes:
                x, y, w, h = box
                draw.rectangle([x-w/2, y-h/2, x+w/2, y+h/2], outline="red", width=2)
            for box in gt_boxes:
                x, y, w, h = box
                draw.rectangle([x-w/2, y-h/2, x+w/2, y+h/2], outline="green", width=2)

            vis_img.save(os.path.join(save_path, f"/figs/refined_{i}.png"))

metrics_np = np.array(metrics_list)
avg_precision = np.mean(metrics_np[:, 0])
avg_recall = np.mean(metrics_np[:, 1])
avg_f1 = np.mean(metrics_np[:, 2])
avg_ap50 = np.mean(metrics_np[:, 3])

ap_full = average_precision_score(all_gts_total, all_scores_total)
prec_curve, recall_curve, _ = precision_recall_curve(all_gts_total, all_scores_total)

with open(os.path.join(save_path, "metrics.txt"), "w") as f:
    f.write(f"Precision: {avg_precision:.4f}\n")
    f.write(f"Recall: {avg_recall:.4f}\n")
    f.write(f"F1 Score: {avg_f1:.4f}\n")
    f.write(f"AP@0.5: {avg_ap50:.4f}\n")
    f.write(f"AP (avg precision full): {ap_full:.4f}\n")

# Confusion Matrix
cm = confusion_matrix(all_gts_total, all_preds_total, labels=[1, 0])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Positive (GT)", "Negative"])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
plt.close()

# PR Curve
plt.figure()
plt.plot(recall_curve, prec_curve, label=f"AP={ap_full:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.savefig(os.path.join(save_path, "pr_curve.png"))
plt.close()

print(f"Test result saved at:{save_path}")

