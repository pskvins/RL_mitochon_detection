# src/ppo_model/agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Dict, Optional

class Actor(nn.Module):
    """Actor network compatible with DDPG structure but for PPO"""
    def __init__(self, state_dim=516, max_shift=2.0, max_scale=0.2):
        super().__init__()
        
        self.max_shift = max_shift
        self.max_scale = max_scale
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=state_dim, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU()
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        
        # Mean and log_std for continuous actions
        self.action_mean = nn.Linear(128, 3)  # dx, dy, dscale
        self.action_log_std = nn.Parameter(torch.zeros(1, 3))
        
        self.term_mean = nn.Linear(128, 1)  # p_term
        self.term_log_std = nn.Parameter(torch.zeros(1, 1))
        
        # Initialize
        nn.init.constant_(self.term_mean.bias, -2.0)
        
    def forward(self, state):
        """
        Args:
            state: Tensor [B, 516]
        Returns:
            action_mean: [B, 4]
            action_log_std: [B, 4]
        """
        B = state.shape[0]
        x = state.view(B, 516, 1, 1)
        x = self.conv(x).view(B, -1)
        x = self.mlp(x)
        
        # Get means
        move_mean = self.action_mean(x)  # [B, 3]
        term_mean = self.term_mean(x)    # [B, 1]
        
        # Concatenate means
        means = torch.cat([move_mean, term_mean], dim=1)  # [B, 4]
        
        # Get log_stds
        move_log_std = self.action_log_std.expand(B, 3)
        term_log_std = self.term_log_std.expand(B, 1)
        log_stds = torch.cat([move_log_std, term_log_std], dim=1)  # [B, 4]
        
        return means, log_stds


class Critic(nn.Module):
    """Value network (critic) for PPO"""
    def __init__(self, state_dim=516):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=state_dim, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU()
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state):
        """
        Args:
            state: Tensor [B, 516]
        Returns:
            value: Tensor [B, 1]
        """
        B = state.shape[0]
        x = state.view(B, 516, 1, 1)
        x = self.conv(x).view(B, -1)
        value = self.mlp(x)
        return value


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,  # Keep for compatibility but not used in PPO
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        mini_batch_size: int = 64
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        
        # Networks
        self.actor = Actor(state_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
    def select_action(self, state: torch.Tensor, noise_std: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action - compatible with DDPG interface but uses stochastic policy
        
        Args:
            state: Tensor [1, state_dim] or [B, state_dim]
            noise_std: Ignored (kept for compatibility)
        
        Returns:
            action_bounded: Tensor [1, action_dim] or [B, action_dim]
            value: Tensor [1, 1] or [B, 1]
            log_prob: Tensor [1] or [B]
            action_unbounded: Tensor [1, action_dim] or [B, action_dim]
        """
        self.actor.eval()
        self.critic.eval()
        
        with torch.no_grad():
            # Get action distribution
            action_mean, action_log_std = self.actor(state.to(self.device))
            action_std = action_log_std.exp()
            
            # Sample from distribution
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            
            # Apply bounds
            action_bounded = torch.zeros_like(action)
            # dx, dy bounded by tanh * max_shift
            action_bounded[:, :2] = torch.tanh(action[:, :2]) * self.actor.max_shift
            # dscale bounded by tanh * max_scale  
            action_bounded[:, 2] = torch.tanh(action[:, 2]) * self.actor.max_scale
            # p_term bounded by sigmoid
            action_bounded[:, 3] = torch.sigmoid(action[:, 3])
            
            # Get value estimate
            value = self.critic(state.to(self.device))
            
            # Calculate log probability
            log_prob = dist.log_prob(action).sum(dim=-1)
            
        self.actor.train()
        self.critic.train()
        
        return action_bounded, value, log_prob, action
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy of given actions"""
        action_mean, action_log_std = self.actor(states)
        action_std = action_log_std.exp()
        
        # Inverse transform actions to unbounded space
        actions_unbounded = torch.zeros_like(actions)
        # Inverse of tanh * scale
        actions_unbounded[:, :2] = torch.atanh(
            torch.clamp(actions[:, :2] / self.actor.max_shift, -0.999, 0.999)
        )
        actions_unbounded[:, 2] = torch.atanh(
            torch.clamp(actions[:, 2] / self.actor.max_scale, -0.999, 0.999)
        )
        # Inverse of sigmoid
        actions_unbounded[:, 3] = torch.log(
            actions[:, 3] / (1 - actions[:, 3] + 1e-8)
        )
        
        # Compute log probabilities
        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions_unbounded).sum(dim=-1, keepdim=True)
        
        # Adjust for transformations
        # Jacobian for tanh
        log_probs -= torch.log(1 - (actions[:, :2] / self.actor.max_shift) ** 2 + 1e-8).sum(dim=-1, keepdim=True)
        log_probs -= torch.log(1 - (actions[:, 2:3] / self.actor.max_scale) ** 2 + 1e-8)
        # Jacobian for sigmoid
        log_probs -= torch.log(actions[:, 3:4] * (1 - actions[:, 3:4]) + 1e-8)
        
        # Entropy
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        # Value
        values = self.critic(states)
        
        return log_probs, entropy, values
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, replay_buffer, batch_size: int = 64):
        """
        PPO update - interface compatible with DDPG but uses PPO algorithm
        
        Args:
            replay_buffer: PPOReplayBuffer with rollout data
            batch_size: Mini-batch size for updates
        
        Returns:
            actor_loss, critic_loss (for compatibility)
        """
        # Get all data from buffer
        states, actions, rewards, next_states, dones, values, log_probs_old = replay_buffer.get_all()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)
        values = torch.FloatTensor(values).unsqueeze(-1).to(self.device)
        log_probs_old = torch.FloatTensor(log_probs_old).unsqueeze(-1).to(self.device)
        
        # Compute returns and advantages
        with torch.no_grad():
            next_value = self.critic(next_states[-1].unsqueeze(0))
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        total_actor_loss = 0
        total_critic_loss = 0
        n_updates = 0
        
        for _ in range(self.ppo_epochs):
            # Create mini-batches
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.mini_batch_size):
                end = min(start + self.mini_batch_size, len(states))
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                log_probs, entropy, values = self.evaluate_actions(batch_states, batch_actions)
                
                # Actor loss (PPO clip)
                ratio = torch.exp(log_probs - batch_log_probs_old)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                
                # Critic loss
                critic_loss = F.mse_loss(values, batch_returns)
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                n_updates += 1
        
        return total_actor_loss / n_updates, total_critic_loss / n_updates
    
    def save(self, path: str):
        """Save model - compatible with DDPG interface"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)
    
    def load(self, path: str, map_location=None):
        """Load model - compatible with DDPG interface"""
        checkpoint = torch.load(path, map_location=map_location)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])