# train_ppo_improved.py
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import deque

import torch
from torch.utils.data import DataLoader

from ppo_model.agent import PPOAgent
from ppo_model.replay_buffer import PPORolloutBuffer
from env.utils import compute_iou, ResNet18FeatureExtractor

from env.box_env import BoxRefinementEnv
from data.coarse_boxes_loader import CoarseBoxesDataset
from data.generate_coarse_boxes import generate_coarse_boxes

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
image_dir = path_cfg["image_dir"]
label_dir = path_cfg["label_dir"]
coarse_dir = path_cfg["coarse_dir"]
model_path = path_cfg["model_path"]

# Agent configs
state_dim = 4 + 512  # box(4) + CNN(512)
action_dim = 4       # dx, dy, dscale, p_term
agent_cfg = cfg['agent']
actor_lr = agent_cfg["actor_lr"]
critic_lr = agent_cfg["critic_lr"]
gamma = agent_cfg["gamma"]
tau = agent_cfg['tau']

# PPO specific hyperparameters (add these to your config.yaml)
gae_lambda = agent_cfg.get("gae_lambda", 0.95)
clip_epsilon = agent_cfg.get("clip_epsilon", 0.2)
value_coef = agent_cfg.get("value_coef", 0.5)
entropy_coef = agent_cfg.get("entropy_coef", 0.01)
ppo_epochs = agent_cfg.get("ppo_epochs", 10)
max_grad_norm = agent_cfg.get("max_grad_norm", 0.5)

# Training configs
train_cfg = cfg['train']    
epochs = train_cfg['epochs']
steps_per_episode = train_cfg['steps_per_episode']
batch_size = train_cfg['batch_size']
replay_start = train_cfg['replay_start']
noise_std = train_cfg['noise_std']
noise_decay = train_cfg['noise_decay']
conf_threshold = train_cfg['conf_threshold']

# PPO specific configs
rollout_length = train_cfg.get('rollout_length', 256)  # Collect this many steps before update
n_rollout_boxes = train_cfg.get('n_rollout_boxes', 16)  # Process this many boxes per rollout

device = "cuda" if torch.cuda.is_available() else "cpu"

# Save configs
save_cfg = cfg["save"]
save_interval = save_cfg['save_interval']
checkpoint_dir = save_cfg["checkpoint_dir"]
save_best_only = save_cfg.get("save_best_only", False)

# Log configs
log_cfg = cfg["log"]
use_tensorboard = log_cfg.get("use_tensorboard", True)
log_dir = log_cfg["log_dir"]
log_interval = log_cfg["log_interval"]

if use_tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir)
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

# -----------------------
# Dataset and modules
# -----------------------
dataset = CoarseBoxesDataset(image_dir, label_dir, coarse_dir)
feature_extractor = ResNet18FeatureExtractor(device=device)

# Initialize PPO agent
agent = PPOAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    device=device,
    actor_lr=actor_lr,
    critic_lr=critic_lr,
    gamma=gamma,
    tau=tau,
    gae_lambda=gae_lambda,
    clip_epsilon=clip_epsilon,
    value_coef=value_coef,
    entropy_coef=entropy_coef,
    max_grad_norm=max_grad_norm,
    ppo_epochs=ppo_epochs,
    mini_batch_size=batch_size
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])

# -----------------------
# Helper Functions
# -----------------------
def collect_rollout(
    agent: PPOAgent,
    dataloader,
    feature_extractor,
    rollout_buffer: PPORolloutBuffer,
    n_steps: int,
    device: str
) -> float:
    """Collect rollout data across multiple boxes/images"""
    total_rewards = []
    steps_collected = 0
    
    data_iter = iter(dataloader)
    
    while steps_collected < n_steps:
        try:
            img, gt_boxes, coarse_boxes = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            img, gt_boxes, coarse_boxes = next(data_iter)
        
        if len(coarse_boxes) == 0:
            continue
            
        img_w, img_h = img.size
        gt_boxes = np.array([
            [x * img_w, y * img_h, w * img_w, h * img_h]
            for x, y, w, h in gt_boxes
        ], dtype=np.float32)
        
        # Process random box from this image
        box_idx = np.random.randint(len(coarse_boxes))
        box = np.asarray(coarse_boxes[box_idx], dtype=np.float32).reshape(4,)
        
        # Create environment
        env = BoxRefinementEnv(
            image=img,
            gt_boxes=gt_boxes,
            initial_box=box,
            feature_extractor=feature_extractor,
            iou_fn=compute_iou,
            max_steps=steps_per_episode,
            device=device
        )
        
        state = env.reset()
        episode_reward = 0
        
        for step in range(steps_per_episode):
            if steps_collected >= n_steps:
                break
                
            state_tensor = state.unsqueeze(0)
            
            # Get action and value
            with torch.no_grad():
                action_bounded, value, log_prob, action_unbounded = agent.select_action(state_tensor)
            
            action_np = action_bounded.squeeze(0).cpu().numpy()
            value_np = value.squeeze().item()
            log_prob_np = log_prob.item()
            
            # Step environment
            next_state, reward, done, info = env.step(action_np)
            episode_reward += reward
            
            # Add to rollout buffer
            rollout_buffer.add(
                state=state.cpu().numpy(),
                action=action_np,
                reward=reward,
                value=value_np,
                done=done,
                log_prob=log_prob_np
            )
            
            state = next_state
            steps_collected += 1
            
            if done:
                break
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards) if total_rewards else 0.0

# -----------------------
# Training Loop
# -----------------------
os.makedirs(checkpoint_dir, exist_ok=True)
best_reward = float('-inf')
global_step = 0
episode_rewards = deque(maxlen=100)

print(f"Starting PPO training on {device}")
print(f"Dataset size: {len(dataset)} images")
print(f"Rollout length: {rollout_length} steps")

for epoch in range(epochs):
    print(f"\n[Epoch {epoch+1}/{epochs}]")
    
    # Create rollout buffer for this epoch
    rollout_buffer = PPORolloutBuffer(
        buffer_size=rollout_length,
        state_dim=state_dim,
        action_dim=action_dim
    )
    
    # Collect rollouts
    avg_rollout_reward = collect_rollout(
        agent, dataloader, feature_extractor,
        rollout_buffer, rollout_length, device
    )
    episode_rewards.append(avg_rollout_reward)
    global_step += rollout_length
    
    # Get rollout data
    states, actions, rewards, values, dones, log_probs = rollout_buffer.get()
    
    # Convert to tensors and compute advantages
    states_tensor = torch.FloatTensor(states).to(device)
    actions_tensor = torch.FloatTensor(actions).to(device)
    rewards_tensor = torch.FloatTensor(rewards).to(device)
    values_tensor = torch.FloatTensor(values).to(device)
    dones_tensor = torch.FloatTensor(dones).to(device)
    log_probs_tensor = torch.FloatTensor(log_probs).to(device)
    
    # Compute final value for GAE
    with torch.no_grad():
        if dones[-1]:
            next_value = torch.zeros(1).to(device)
        else:
            # Get the last state that wasn't done
            last_state = torch.FloatTensor(states[-1]).unsqueeze(0).to(device)
            next_value = agent.critic(last_state).squeeze()
    
    # Compute returns and advantages
    advantages, returns = agent.compute_gae(
        rewards_tensor.unsqueeze(-1),
        values_tensor.unsqueeze(-1),
        dones_tensor.unsqueeze(-1),
        next_value
    )
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PPO update
    total_actor_loss = 0
    total_critic_loss = 0
    total_entropy = 0
    total_kl = 0
    n_updates = 0
    
    for ppo_epoch in range(ppo_epochs):
        # Shuffle data
        indices = torch.randperm(len(states_tensor))
        
        for start in range(0, len(states_tensor), batch_size):
            end = min(start + batch_size, len(states_tensor))
            batch_indices = indices[start:end]
            
            batch_states = states_tensor[batch_indices]
            batch_actions = actions_tensor[batch_indices]
            batch_log_probs_old = log_probs_tensor[batch_indices]
            batch_advantages = advantages[batch_indices].squeeze()
            batch_returns = returns[batch_indices].squeeze()
            
            # Evaluate actions
            log_probs, entropy, values = agent.evaluate_actions(batch_states, batch_actions)
            log_probs = log_probs.squeeze()
            entropy = entropy.squeeze()
            values = values.squeeze()
            
            # Compute ratio and losses
            ratio = torch.exp(log_probs - batch_log_probs_old)
            
            # Actor loss
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy.mean()
            
            # Critic loss
            critic_loss = value_coef * torch.nn.functional.mse_loss(values, batch_returns)
            
            # Total loss
            loss = actor_loss + critic_loss
            
            # Update networks
            agent.actor_optimizer.zero_grad()
            agent.critic_optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), max_grad_norm)
            
            agent.actor_optimizer.step()
            agent.critic_optimizer.step()
            
            # Track metrics
            with torch.no_grad():
                kl = (batch_log_probs_old - log_probs).mean()
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                total_kl += kl.item()
                n_updates += 1
        
        # Early stopping if KL too large
        if total_kl / n_updates > 0.02:
            print(f"  Early stopping at PPO epoch {ppo_epoch+1} due to large KL")
            break
    
    # Compute statistics
    avg_actor_loss = total_actor_loss / n_updates
    avg_critic_loss = total_critic_loss / n_updates
    avg_entropy = total_entropy / n_updates
    avg_kl = total_kl / n_updates
    avg_reward = np.mean(episode_rewards)
    
    print(f"[Epoch {epoch+1}] Rollout Reward: {avg_rollout_reward:.4f}, "
          f"Running Avg: {avg_reward:.4f}")
    print(f"  Actor Loss: {avg_actor_loss:.4f}, Critic Loss: {avg_critic_loss:.4f}")
    print(f"  Entropy: {avg_entropy:.4f}, KL: {avg_kl:.4f}")
    
    # TensorBoard logging
    if writer and (epoch + 1) % log_interval == 0:
        writer.add_scalar("Reward/Rollout", avg_rollout_reward, epoch + 1)
        writer.add_scalar("Reward/RunningAvg", avg_reward, epoch + 1)
        writer.add_scalar("Loss/Actor", avg_actor_loss, epoch + 1)
        writer.add_scalar("Loss/Critic", avg_critic_loss, epoch + 1)
        writer.add_scalar("Loss/Entropy", avg_entropy, epoch + 1)
        writer.add_scalar("Loss/KL", avg_kl, epoch + 1)
        writer.add_scalar("Training/GlobalStep", global_step, epoch + 1)
    
    # Save model
    if save_best_only:
        if avg_reward > best_reward:
            best_reward = avg_reward
            save_path = os.path.join(checkpoint_dir, "ppo_best.pt")
            agent.save(save_path)
            print(f"[Epoch {epoch+1}] Best model saved with reward {best_reward:.4f}")
    
    elif (epoch + 1) % save_interval == 0:
        save_path = os.path.join(checkpoint_dir, f"ppo_epoch_{epoch+1}.pt")
        agent.save(save_path)
        print(f"[Epoch {epoch+1}] Model checkpoint saved.")

print("\nTraining completed!")
if writer:
    writer.close()