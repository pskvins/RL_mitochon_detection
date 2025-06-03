# evaluate_ppo.py
import os
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
import json

from src.ppo_model.agent import PPOAgent
from src.env.utils import compute_iou, ResNet18FeatureExtractor, make_gif
from src.env.box_env import BoxRefinementEnv
from src.data.coarse_boxes_loader import CoarseBoxesDataset

import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
parser.add_argument('--output_dir', type=str, default='./evaluation_results_ppo', help='Output directory')
parser.add_argument('--visualize', action='store_true', help='Save visualizations')
parser.add_argument('--make_gif', action='store_true', help='Create GIF animations')
parser.add_argument('--max_images', type=int, default=None, help='Maximum images to evaluate')
parser.add_argument('--deterministic', action='store_true', help='Use deterministic policy')
args = parser.parse_args()

# Load config
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
state_dim = 4 + 512
action_dim = 4

# Load agent
agent_cfg = cfg['agent']
agent = PPOAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    device=device,
    actor_lr=agent_cfg["actor_lr"],
    critic_lr=agent_cfg["critic_lr"],
    gamma=agent_cfg["gamma"],
    tau=agent_cfg.get('tau', 0.005),
    gae_lambda=agent_cfg.get("gae_lambda", 0.95),
    clip_epsilon=agent_cfg.get("clip_epsilon", 0.2),
    value_coef=agent_cfg.get("value_coef", 0.5),
    entropy_coef=agent_cfg.get("entropy_coef", 0.01),
    max_grad_norm=agent_cfg.get("max_grad_norm", 0.5),
    ppo_epochs=agent_cfg.get("ppo_epochs", 10),
    mini_batch_size=cfg['train']['batch_size']
)

agent.load(args.checkpoint, map_location=device)
agent.actor.eval()
agent.critic.eval()

# Load dataset
path_cfg = cfg["paths"]
dataset = CoarseBoxesDataset(
    path_cfg["image_dir"],
    path_cfg["label_dir"],
    path_cfg["coarse_dir"]
)

# Feature extractor
feature_extractor = ResNet18FeatureExtractor(device=device)

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)
if args.visualize:
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
if args.make_gif:
    gif_dir = os.path.join(args.output_dir, 'gifs')
    os.makedirs(gif_dir, exist_ok=True)

# Evaluation metrics
all_metrics = {
    'initial_ious': [],
    'final_ious': [],
    'iou_improvements': [],
    'steps_taken': [],
    'success_rate': 0,  # Final IoU > threshold
    'improvement_rate': 0,  # Percentage of boxes improved
    'avg_improvement': 0
}

# IoU threshold for success
iou_threshold = 0.7
improvement_threshold = 0.05
max_steps = cfg['train']['steps_per_episode']

# Evaluate
print(f"Evaluating on {len(dataset)} images...")
print(f"Using {'deterministic' if args.deterministic else 'stochastic'} policy")
n_images = min(len(dataset), args.max_images) if args.max_images else len(dataset)

for idx in tqdm(range(n_images)):
    img, gt_boxes, coarse_boxes = dataset[idx]
    
    if len(coarse_boxes) == 0:
        continue
    
    img_w, img_h = img.size
    gt_boxes = np.array([
        [x * img_w, y * img_h, w * img_w, h * img_h]
        for x, y, w, h in gt_boxes
    ], dtype=np.float32)
    
    image_metrics = {
        'initial_ious': [],
        'final_ious': [],
        'refined_boxes': [],
        'trajectories': []
    }
    
    # For GIF creation
    refined_sequences = []
    
    # Visualize if requested
    if args.visualize:
        vis_img = img.copy()
        draw = ImageDraw.Draw(vis_img)
        
        # Draw ground truth boxes in green
        for gt_box in gt_boxes:
            x, y, w, h = gt_box
            draw.rectangle([x-w/2, y-h/2, x+w/2, y+h/2], outline='green', width=2)
    
    # Process each coarse box
    for box_idx, box in enumerate(coarse_boxes):
        box = np.asarray(box, dtype=np.float32).reshape(4,)
        
        # Create environment
        env = BoxRefinementEnv(
            image=img,
            gt_boxes=gt_boxes,
            initial_box=box,
            feature_extractor=feature_extractor,
            iou_fn=compute_iou,
            max_steps=max_steps,
            device=device
        )
        
        # Get initial IoU
        state = env.reset()
        initial_iou = env.initial_iou
        
        # Run refinement
        trajectory = []
        box_sequence = [box.tolist()]  # For GIF
        
        for step in range(max_steps):
            with torch.no_grad():
                # Use deterministic action if requested
                if args.deterministic:
                    # Use mean of the distribution
                    state_tensor = state.unsqueeze(0)
                    action_mean, _ = agent.actor(state_tensor)
                    
                    # Apply same transformations as in stochastic case
                    action = torch.zeros_like(action_mean)
                    action[:, :2] = torch.tanh(action_mean[:, :2]) * agent.actor.max_shift
                    action[:, 2] = torch.tanh(action_mean[:, 2]) * agent.actor.max_scale
                    action[:, 3] = torch.sigmoid(action_mean[:, 3])
                else:
                    # Use stochastic policy
                    action = agent.select_action(state.unsqueeze(0))
                
                # Also get value estimate
                value = agent.critic(state.unsqueeze(0))
            
            action_np = action.squeeze(0).cpu().numpy()
            next_state, reward, done, info = env.step(action_np)
            
            # Record trajectory
            trajectory.append({
                'step': step,
                'box': env.cur_box.cpu().numpy().tolist(),
                'action': action_np.tolist(),
                'reward': float(reward),
                'value': float(value.item()),
                'iou': float(info['iou'])
            })
            
            box_sequence.append(env.cur_box.cpu().numpy().tolist())
            
            state = next_state
            
            if done:
                break
        
        # Get final metrics
        final_iou = info['iou']
        final_box = env.cur_box.cpu().numpy()
        
        # Record metrics
        image_metrics['initial_ious'].append(initial_iou)
        image_metrics['final_ious'].append(final_iou)
        image_metrics['refined_boxes'].append(final_box.tolist())
        image_metrics['trajectories'].append(trajectory)
        
        all_metrics['initial_ious'].append(initial_iou)
        all_metrics['final_ious'].append(final_iou)
        all_metrics['iou_improvements'].append(final_iou - initial_iou)
        all_metrics['steps_taken'].append(len(trajectory))
        
        refined_sequences.append(box_sequence)
        
        # Visualize boxes
        if args.visualize:
            # Draw initial box in red (dashed)
            x, y, w, h = box
            draw.rectangle([x-w/2, y-h/2, x+w/2, y+h/2], outline='red', width=1)
            
            # Draw final box in blue
            x, y, w, h = final_box
            draw.rectangle([x-w/2, y-h/2, x+w/2, y+h/2], outline='blue', width=2)
            
            # Add text
            draw.text((x-w/2, y-h/2-15), 
                     f"IoU: {initial_iou:.3f}→{final_iou:.3f}", 
                     fill='blue')
    
    # Save visualization
    if args.visualize:
        vis_path = os.path.join(vis_dir, f'image_{idx:04d}.png')
        vis_img.save(vis_path)
    
    # Create GIF
    if args.make_gif and len(refined_sequences) > 0:
        gif_path = os.path.join(gif_dir, f'refinement_{idx:04d}.gif')
        make_gif(
            base_img=img,
            coarse_boxes=[cb for cb in coarse_boxes],