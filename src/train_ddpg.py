import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.ddpg_model.agent import DDPGAgent
from src.ddpg_model.replay_buffer import ReplayBuffer
from src.env.utils import compute_iou, ResNet18FeatureExtractor
#from src.env.utils import MobileNetFeatureExtractor
from src.env.box_env import BoxRefinementEnv
from src.data.corse_boxex_loader import CoarseBoxesDataset
from src.data.generate_coarse_boxes import generate_coarse_boxes

# -----------------------
# CONFIG
# -----------------------
image_dir = "datasets/train/images"
label_dir = "datasets/train/labels"
coarse_dir = "datasets/train/coarse_boxes_yolo"
model_path = "yolo_runs/train/finetune_exp1/weights/best.pt"

state_dim = 4 + 512  # box(4) + CNN(512)
action_dim = 4       # dx, dy, dscale, p_term

device = "cuda" if torch.cuda.is_available() else "cpu"
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
        conf_threshold=0.3
    )
#-----------------------


# -----------------------
# Dataset and modules
# -----------------------
dataset = CoarseBoxesDataset(image_dir, label_dir, coarse_dir)
feature_extractor = ResNet18FeatureExtractor(device=device)
#feature_extractor = MobileNetFeatureExtractor(device=device)

replay_buffer = ReplayBuffer(state_dim, action_dim)
agent = DDPGAgent(state_dim, action_dim, device=device)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True,collate_fn=lambda x: x[0])
#-----------------------


# -----------------------
# Training Loop
# -----------------------
epochs = 10
steps_per_episode = 3
os.makedirs("checkpoints", exist_ok=True)
for epoch in range(epochs):
    print(f"[Epoch {epoch+1}/{epochs}]")
    episode_rewards = []

    for img, gt_boxes, coarse_boxes in tqdm(dataloader):
        for box in coarse_boxes:
            box = np.asarray(box, dtype=np.float32).reshape(4,)
            env = BoxRefinementEnv(
                image=img,
                gt_boxes=gt_boxes,
                initial_box=box,
                feature_extractor=feature_extractor,
                iou_fn=compute_iou
            )

            state = env.reset()
            total_reward = 0

            for _ in range(steps_per_episode):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = agent.select_action(state_tensor, noise_std=0.1)
                action_np = action.squeeze(0).detach().cpu().numpy()

                next_state, reward, done, _ = env.step(action_np)
                total_reward += reward

                replay_buffer.add(state, action_np, reward, next_state, done)
                state = next_state

                if replay_buffer.size > 1000:
                    actor_loss, critic_loss = agent.update(replay_buffer, batch_size=64)

                if done:
                    break

            episode_rewards.append(total_reward)

    avg_reward = np.mean(episode_rewards)
    print(f"[Epoch {epoch+1}] Avg Reward per box: {avg_reward:.4f}")

    # Save model every 2 epochs
    if (epoch + 1) % 2 == 0:
        agent.save(f"checkpoints/ddpg_epoch_{epoch+1}.pt")