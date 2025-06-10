import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
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

#path configs
path_cfg = cfg["paths"]
image_dir = path_cfg["image_dir"]
label_dir = path_cfg["label_dir"]
coarse_dir = path_cfg["coarse_dir"]
model_path = path_cfg["model_path"]

#agent
if "best" in model_path:
    state_dim = 4 + 1 + 195  # box(4) + confidence(1) + YOLO(195)
else:
    state_dim = 4 + 1 + 432  # box(4) + confidence(1) + YOLO(195)
action_dim = 4          # dx, dy, dscale, p_term
agent_cfg = cfg['agent']
actor_lr = agent_cfg["actor_lr"]
critic_lr = agent_cfg["critic_lr"]
gamma = agent_cfg["gamma"]
tau = agent_cfg['tau']


#training loop 
train_cfg = cfg['train']    
epochs = train_cfg['epochs']
steps_per_episode = train_cfg['steps_per_episode']
batch_size = train_cfg['batch_size']
replay_start = train_cfg['replay_start']
noise_std = train_cfg['noise_std']
# noise_decay = train_cfg['noise_decay']
conf_threshold = train_cfg['conf_threshold'] #thershold for filtering coarse boxes


device = "cuda" if torch.cuda.is_available() else "cpu"

#save results
save_cfg = cfg["save"]
save_interval = save_cfg['save_interval']
checkpoint_dir = save_cfg["checkpoint_dir"]
save_best_only = save_cfg.get("save_best_only", True)

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


# -----------------------
# Dataset and modules
# -----------------------
dataset = CoarseBoxesDataset(image_dir, label_dir, coarse_dir)
feature_extractor = YOLOv8FeatureExtractor(model_path=model_path, device=device)
replay_buffer = ReplayBuffer(state_dim, action_dim)
agent = DDPGAgent(state_dim, action_dim, device=device)
dataloader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=True,collate_fn=lambda x: x[0])
#-----------------------



# -----------------------
# Training Loop
# -----------------------
best_reward = -float('inf')
os.makedirs("checkpoints", exist_ok=True)
for epoch in range(epochs):
    print(f"[Epoch {epoch+1}/{epochs}]")
    episode_rewards = []
    # noise_std = max(0.01, noise_decay*noise_std) #noise decay
    noise_std = max(0.01, noise_std)
    
    for img, gt_boxes, coarse_boxes in tqdm(dataloader):
        for box in coarse_boxes:
            img_w, img_h = img.size
            image_array = np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
            # gt_boxes = np.array([
            #             [x * img_w, y * img_h, w * img_w, h * img_h]
            #             for x, y, w, h in gt_boxes], dtype=np.float32) #rescale
            gt_boxes_cp = gt_boxes.copy()
            gt_boxes_cp *= image_array
            box = np.asarray(box, dtype=np.float32).reshape(5,)
            # if box[-1] < 0.9: # Skip if confidence is high
            #     continue

            env = BoxRefinementEnv(
                image=img,
                gt_boxes=gt_boxes_cp,
                initial_box=box,
                feature_extractor=feature_extractor,
                iou_fn=compute_iou
            )

            state = env.reset()
            total_reward = 0

            for _ in range(steps_per_episode):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = agent.select_action(state_tensor, noise_std)
                action_np = action.squeeze(0).detach().cpu().numpy()

                next_state, reward, done, _ = env.step(action_np)
                total_reward += reward

                replay_buffer.add(state, action_np, reward, next_state, done)
                state = next_state

                if replay_buffer.size > 500:
                    actor_loss, critic_loss = agent.update(replay_buffer, batch_size=64)

                if done: #done if p_term > 0.5 or max steps hit
                    break

            episode_rewards.append(total_reward)

    avg_reward = np.mean(episode_rewards)
    print(f"[Epoch {epoch+1}] Avg Reward per box: {avg_reward:.4f}")
        
    # TensorBoard logging
    if writer and (epoch + 1) % log_cfg["log_interval"] == 0:
        writer.add_scalar("Reward/Avg", avg_reward, epoch + 1)
        if 'actor_loss' in locals():
            writer.add_scalar("Loss/Actor", actor_loss, epoch + 1)
            writer.add_scalar("Loss/Critic", critic_loss, epoch + 1)


    # Save model
    if save_best_only:
        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save(os.path.join(checkpoint_dir, "ddpg_best.pt"))
            print(f"[Epoch {epoch+1}] Best model saved with reward {best_reward:.4f}")
            
    elif (epoch + 1) % save_cfg["save_interval"] == 0:
        save_path = os.path.join(checkpoint_dir, f"ddpg_epoch_{epoch+1}.pt")
        agent.save(save_path)
        print(f"[Epoch {epoch+1}] Model checkpoint saved.")  