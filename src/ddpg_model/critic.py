import torch
from torch import Tensor
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 4):
        """
        Critic network in DDPG algorithm

        Args:
            state_dim: dimension of state which is concatenated vector of
                CNN feature and current position 
            action_dim (int): dimension of the action vector
                default setting is 4, [dx, dy, scale, p_term]
        """
        super().__init__()
        self.input_dim = state_dim + action_dim  # e.g., 516 + 4 = 520

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU()
        )

        self.mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Q-value scalar
        )
    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        """
        Forward pass of the critic.

        Args:
            state: Tensor of shape [B, state_dim]
            action: Tensor of shape [B, action_dim]

        Returns:
            q_value: Tensor of shape [B, 1]
        """
        x = torch.cat([state, action], dim=1)  # [B, 520]
        B = x.size(0)
        x = x.view(B, self.input_dim, 1, 1)     # [B, 520, 1, 1]
        x = self.conv(x).view(B, -1)            # [B, 64]
        q = self.mlp(x)                         # [B, 1]
        return q
