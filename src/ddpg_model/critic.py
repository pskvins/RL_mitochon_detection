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
        input_dim = state_dim + action_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Q-value output
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
        x = torch.cat([state, action], dim=1)
        q_value = self.net(x)
        return q_value
