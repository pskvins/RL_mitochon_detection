import numpy as np
import torch
from torch import Tensor
from typing import Tuple

class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 1000):
        """
        Replay buffer in DDPG algorithm for storing experience tuples (s, a, r, s', done).

        Args:
            state_dim: dimension of state which is concatenated vector of
                CNN feature, current position and size of box
            action_dim: dimension of the action vector
                default setting is 4, [dx, dy, scale, p_term]
            max_size: maximum number of tuples to store
        """
        self.max_size = max_size
        self.idx = 0  # pointer to the current index
        self.size = 0

        # pre-allocate memory for transitions
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            done: bool):
        """
        Add one transition to the buffer.
        Overwrites old data when buffer is full.

        Args:
            state (np.ndarray): Current state
            action (np.ndarray): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Episode done flag
        """

        #s, a, r, s', done
        self.state[self.idx] = state
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.next_state[self.idx] = next_state
        self.done[self.idx] = float(done)

        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size: int, device: str = "cpu") -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Sample a batch of transitions as torch.Tensor

        Args:
            batch_size (int): Number of samples to return
            device (str): device to return tensors on "cpu" or "cuda"

        Returns:
            Tuple of tensors: (state, action, reward, next_state, done)
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.tensor(self.state[idxs], dtype=torch.float32, device=device),
            torch.tensor(self.action[idxs], dtype=torch.float32, device=device),
            torch.tensor(self.reward[idxs], dtype=torch.float32, device=device),
            torch.tensor(self.next_state[idxs], dtype=torch.float32, device=device),
            torch.tensor(self.done[idxs], dtype=torch.float32, device=device)
        )
