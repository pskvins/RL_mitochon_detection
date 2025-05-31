import torch
import torch.nn as nn

class Actor(nn.Module):
    """
    Actor network in DDPG algorithm

    Args:
        state_dim: dimension of state which is concatenated vector of
            CNN feature and current position of box
        max_shift: maximum pixel shift allowed in both x & y
        max_scale: maximum scale ratio allowed for zoom in/out
    """
    def __init__(self, state_dim, max_shift=2.0, max_scale=0.25):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=516, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU())

        self.action_head = nn.Linear(256, 3)  # dx, dy, dscale
        self.term_head = nn.Linear(256, 1)    # probability of termination
        nn.init.constant_(self.term_head.bias, -2.0) #low initial p_term

        self.max_shift = max_shift
        self.max_scale = max_scale

    def forward(self, state)-> torch.Tensor:
        """
        Forward pass of the actor.

        Args:
            state (torch.Tensor): Input state tensor of shape [B, state_dim]
                where state_dim is defined in __init__

        Returns:
            action(torch.Tensor): output action distribution of shape [B, 4]
                [dx, dy, scale, p_terminate]
        """
        B = state.shape[0]
        x = state.view(B, 516, 1, 1)  # [B, 516, 1, 1]
        x = self.conv(x).view(B, -1)  # [B, 64]
        x = self.mlp(x)

        move = torch.tanh(self.action_head(x))  # [B, 3]
        shift = move[:, :2] * self.max_shift
        scale = move[:, 2:3] * self.max_scale
        p_term = torch.sigmoid(self.term_head(x))
        action = torch.cat([shift, scale, p_term], dim=1)  # [B, 4]
        return action 
