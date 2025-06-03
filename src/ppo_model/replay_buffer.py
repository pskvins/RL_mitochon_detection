# src/ppo_model/replay_buffer.py
import numpy as np
import torch
from typing import Tuple

class ReplayBuffer:
    """
    PPO Replay Buffer - interface compatible with DDPG but stores trajectories
    """
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size
        
        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.values = []
        self.log_probs = []
        
        self.size = 0
        
    def add(self, state, action, reward, next_state, done, value=None, log_prob=None):
        """Add transition - extended with value and log_prob for PPO"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        
        # PPO specific
        if value is not None:
            self.values.append(value)
        else:
            self.values.append(0.0)
            
        if log_prob is not None:
            self.log_probs.append(log_prob)
        else:
            self.log_probs.append(0.0)
        
        self.size = min(self.size + 1, self.max_size)
        
        # Keep buffer size manageable
        if len(self.states) > self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            self.values.pop(0)
            self.log_probs.pop(0)
    
    def sample(self, batch_size: int, device: str = "cpu"):
        """Sample batch - kept for DDPG compatibility but not used in PPO"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        states = torch.FloatTensor([self.states[i] for i in indices]).to(device)
        actions = torch.FloatTensor([self.actions[i] for i in indices]).to(device)
        rewards = torch.FloatTensor([self.rewards[i] for i in indices]).to(device)
        next_states = torch.FloatTensor([self.next_states[i] for i in indices]).to(device)
        dones = torch.FloatTensor([self.dones[i] for i in indices]).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def get_all(self):
        """Get all stored data for PPO update"""
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.next_states),
            np.array(self.dones),
            np.array(self.values),
            np.array(self.log_probs)
        )
    
    def clear(self):
        """Clear buffer after PPO update"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.values = []
        self.log_probs = []
        self.size = 0


class PPORolloutBuffer:
    """
    Optimized rollout buffer specifically for PPO
    Stores complete episodes for on-policy updates
    """
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Pre-allocate arrays
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        
        self.pos = 0
        self.full = False
        
    def add(self, state, action, reward, value, done, log_prob):
        """Add single transition"""
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(action, np.ndarray) and action.ndim > 1:
            action = action.squeeze()
            
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.dones[self.pos] = done
        self.log_probs[self.pos] = log_prob
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
            
    def get(self):
        """Get all data and clear buffer"""
        if self.full:
            data = (
                self.states.copy(),
                self.actions.copy(),
                self.rewards.copy(),
                self.values.copy(),
                self.dones.copy(),
                self.log_probs.copy()
            )
        else:
            data = (
                self.states[:self.pos].copy(),
                self.actions[:self.pos].copy(),
                self.rewards[:self.pos].copy(),
                self.values[:self.pos].copy(),
                self.dones[:self.pos].copy(),
                self.log_probs[:self.pos].copy()
            )
        
        self.clear()
        return data
        
    def clear(self):
        """Reset buffer"""
        self.pos = 0
        self.full = False