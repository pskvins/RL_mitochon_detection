import os
import numpy as np
from PIL import Image
from tqdm.auto import tqdm  

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.ddpg_model.agent import DDPGAgent
from src.ddpg_model.replay_buffer import ReplayBuffer
from src.env.utils import compute_iou, YOLOv8FeatureExtractor

from src.env.box_env import BoxRefinementEnv
from src.data.coarse_boxes_loader import CoarseBoxesDataset
from src.data.generate_coarse_boxes import generate_coarse_boxes

import argparse
import yaml

# -----------------------
# CONFIG
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

# Path configs
path_cfg = cfg["paths"]
image_dir   = path_cfg["image_dir"]
label_dir   = path_cfg["label_dir"]
coarse_dir  = path_cfg["coarse_dir"]
model_path  = path_cfg["model_path"]

# Agent hyperparams
state_dim   = 4 + 195 # box(4) + YOLO feature(195)
action_dim  = 4          # dx, dy, dscale, p_term
agent_cfg   = cfg['agent']
actor_lr    = agent_cfg["actor_lr"]
critic_lr   = agent_cfg["critic_lr"]
gamma       = agent_cfg["gamma"]
tau         = agent_cfg['tau']

# Training loop hyperparams
train_cfg        = cfg['train']
epochs           = train_cfg['epochs']
steps_per_episode= train_cfg['steps_per_episode']
batch_size       = train_cfg['batch_size']
replay_start     = train_cfg['replay_start']
noise_std        = train_cfg['noise_std']
noise_decay      = train_cfg['noise_decay']
conf_threshold   = train_cfg['conf_threshold']

warmup_epochs    = train_cfg.get('warmup_epochs', 5) #Agent do only imitation learning during warmup_phase

device = "cuda" if torch.cuda.is_available() else "cpu"

# Save configs
save_cfg       = cfg["save"]
save_interval  = save_cfg['save_interval']
checkpoint_dir = save_cfg["checkpoint_dir"]
save_best_only = save_cfg.get("save_best_only", False)

# Logging configs
log_cfg        = cfg["log"]
use_tensorboard= log_cfg.get("use_tensorboard", True)
log_dir        = log_cfg["log_dir"]
log_interval   = log_cfg["log_interval"]

if use_tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir)
else:
    writer = None

# -----------------------
# Imitation helper
# -----------------------
def compute_imitation_action(cur_box: torch.Tensor, gt_box: np.ndarray, eps=1e-6):
    x_c, y_c, w, h = cur_box.tolist()
    x_g, y_g, w_g, h_g = gt_box.tolist()

    dx = (x_g - x_c) / (w + eps)
    dy = (y_g - y_c) / (h + eps)

    scale_cur = np.sqrt(w * h)
    scale_gt = np.sqrt(w_g * h_g)
    dscale = (scale_gt / (scale_cur + eps)) - 1.0

    p_term = 1.0 if compute_iou(cur_box.cpu().numpy(), gt_box) > 0.7 else 0.0
    return np.array([dx, dy, dscale, p_term], dtype=np.float32)

def select_best_gt(cur_box: np.ndarray, gt_boxes: np.ndarray, iou_threshold=0.1) -> np.ndarray:
    ious = np.array([compute_iou(cur_box, gt_box) for gt_box in gt_boxes])
    best_idx = np.argmax(ious)
    if ious[best_idx] < iou_threshold:
        return None
    return gt_boxes[best_idx]

# -----------------------
# Generate coarse boxes if missing
# -----------------------
if not os.path.exists(coarse_dir) or len(os.listdir(coarse_dir)) == 0:
    print("Coarse boxes not found. Running YOLO inference:")
    generate_coarse_boxes(
        image_dir=image_dir,
        output_dir=coarse_dir,
        model_path=model_path,
        conf_threshold=conf_threshold
    )

# -----------------------
# Dataset & modules
# -----------------------
dataset = CoarseBoxesDataset(image_dir, label_dir, coarse_dir)
feature_extractor = YOLOv8FeatureExtractor(model_path=model_path,device=device)
replay_buffer = ReplayBuffer(state_dim, action_dim)
agent = DDPGAgent(state_dim, action_dim, device=device)
dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, collate_fn=lambda x: x[0])

os.makedirs(checkpoint_dir, exist_ok=True)
best_reward = -float('inf')

# -----------------------
# Training Loop
# -----------------------
for epoch in range(epochs):
    print(f"[Epoch {epoch+1}/{epochs}]")
    episode_rewards = []
    noise_std = max(0.01, noise_decay * noise_std)
    imitation_phase = epoch < warmup_epochs

    for img, gt_boxes, coarse_boxes in tqdm(dataloader):
        img_w, img_h = img.size
        gt_boxes_px = np.array([
            [x * img_w, y * img_h, w * img_w, h * img_h]
            for x, y, w, h in gt_boxes
        ], dtype=np.float32)

        for box in coarse_boxes:
            box_px = np.asarray(box, dtype=np.float32).reshape(4,)
            env = BoxRefinementEnv(
                image=img,
                gt_boxes=gt_boxes_px,
                initial_box=box_px,
                feature_extractor=feature_extractor,
                iou_fn=compute_iou,
                max_steps=steps_per_episode,
                device=device
            )

            state = env.reset()
            total_reward = 0.0

            for t in range(steps_per_episode):
                state_tensor = state.unsqueeze(0).to(device)

                if imitation_phase:
                    gt_target = select_best_gt(env.cur_box.cpu().numpy(), gt_boxes_px)
                    if gt_target is None:
                        break
                    action_np = compute_imitation_action(env.cur_box, gt_target)

                    target_action = torch.tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)
                    pred_action = agent.actor(state_tensor)
                    imitation_loss = F.mse_loss(pred_action, target_action)
                    agent.actor_optimizer.zero_grad()
                    imitation_loss.backward()
                    agent.actor_optimizer.step()

                    # --- Critic도 업데이트 ---
                    reward = 1.0  # pseudo reward
                    next_state, _, done, _ = env.step(t + 1, action_np)
                    replay_buffer.add(
                        state.detach().cpu().numpy(),
                        action_np,
                        reward,
                        next_state.detach().cpu().numpy(),
                        done
                    )
                    actor_loss, critic_loss = agent.update(replay_buffer, batch_size=64)
                    action = target_action.detach()
                else:
                    with torch.no_grad():
                        action = agent.select_action(state_tensor, noise_std)
                    action_np = action.squeeze(0).cpu().numpy()
                    next_state, reward, done, _ = env.step(t + 1, action_np)
                    replay_buffer.add(
                        state.detach().cpu().numpy(),
                        action_np,
                        reward,
                        next_state.detach().cpu().numpy(),
                        done
                    )
                    if replay_buffer.size > replay_start:
                        actor_loss, critic_loss = agent.update(replay_buffer, batch_size=64)

                state = next_state
                total_reward += reward

                if done:
                    break

            episode_rewards.append(total_reward)

    avg_reward = np.mean(episode_rewards)
    print(f"[Epoch {epoch+1}] Avg Reward per box: {avg_reward:.4f}")

    if writer and (epoch + 1) % log_interval == 0:
        writer.add_scalar("Reward/Avg", avg_reward, epoch + 1)
        if not imitation_phase:
            writer.add_scalar("Loss/Actor", actor_loss, epoch + 1)
            writer.add_scalar("Loss/Critic", critic_loss, epoch + 1)

    if save_best_only:
        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save(os.path.join(checkpoint_dir, "ddpg_best.pt"))
            print(f"[Epoch {epoch+1}] Best model saved (reward {best_reward:.4f})")
    elif (epoch + 1) % save_interval == 0:
        save_path = os.path.join(checkpoint_dir, f"ddpg_epoch_{epoch+1}.pt")
        agent.save(save_path)
        print(f"[Epoch {epoch+1}] Model checkpoint saved.")
