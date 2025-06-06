import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import yaml

import torch
from torch.utils.data import DataLoader
from src.data.dataset import DPOdataset
from src.data.image import FeatureExtractor, dpo_loss_fn, apply_deltas


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
model_path = path_cfg["model_path"]
learning_rate = float(path_cfg['learning_rate'])

ref_update_freq = cfg['ref_update_freq']
beta = cfg['beta']


#training loop 
train_cfg = cfg['train']
epochs = train_cfg['epochs']
conf_threshold = train_cfg['conf_threshold']

device = "cuda" if torch.cuda.is_available() else "cpu"

#save results
save_cfg = cfg["save"]
save_interval = save_cfg['save_interval']
checkpoint_dir = save_cfg["checkpoint_dir"]
save_best_only = save_cfg.get("save_best_only", False)

#save logs
log_cfg = cfg["log"]
use_tensorboard = log_cfg.get("use_tensorboard",True)
log_dir = log_cfg["log_dir"]
log_interval = log_cfg["log_interval"]

if use_tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_cfg["log_dir"])
else:
    writer = None


# -----------------------
# Generate coarse box .npy if missing
# -----------------------
if not os.path.exists(coarse_dir) or len(os.listdir(coarse_dir)) == 0:
    print("Coarse boxes not found. Running YOLO inference:")
    generate_coarse_boxes(
        image_dir=image_dir,
        output_dir=coarse_dir,
        model_path=model_path,
        conf_threshold=conf_threshold
    )
#-----------------------

dataset = DPOdataset(image_dir, label_dir, coarse_dir)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn = lambda x: x[0]
)

os.makedirs("checkpoints", exist_ok=True)

"""
Train_Loop
"""
import copy
# prepare policy and reference models
policy_model = FeatureExtractor(device=device)
reference_model = FeatureExtractor(device=device)
reference_model.load_state_dict(copy.deepcopy(policy_model.state_dict()))
reference_model.eval()

optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)

global_step = 0
for epoch in range(epochs): 
    print(f"[Epoch {epoch+1}/{epochs}]")
    policy_model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        if batch is None:
            continue
        img, gt_boxes, coarse_boxes = batch
        for box in range(len(coarse_boxes)):
            input_box = torch.tensor(coarse_boxes[box], dtype = torch.float32).to(device)
            gt_box = torch.tensor(gt_boxes[box], dtype=torch.float32).to(device)

            optimizer.zero_grad()

            policy_deltas = policy_model(img, input_box)
            policy_pred_boxes = apply_deltas(input_box, policy_deltas)
            
            with torch.no_grad():
                ref_deltas = reference_model(img, input_box)
                ref_pred_boxes = apply_deltas(input_box, ref_deltas)


            loss = dpo_loss_fn(
                policy_pred_boxes, 
                ref_pred_boxes, 
                gt_box, 
                input_box,
                beta=beta,
                box_loss_type='l1'
            )

            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch}. Skipping update.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            global_step += 1

        if global_step % ref_update_freq == 0:
            #print(f"Updating reference model at step {global_step}")
            reference_model.load_state_dict(copy.deepcopy(policy_model.state_dict()))
            reference_model.eval()
            
    

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} Summary: Avg Loss: {avg_loss:.4f}")
    writer.add_scalar("Loss/DPO", avg_loss, epoch + 1)
    # Save model checkpoint
    torch.save(policy_model.state_dict(), f"dpo_refiner_epoch_{epoch+1}.pth")
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


