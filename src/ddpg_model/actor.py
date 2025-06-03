import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim=516, max_shift=0.5, max_scale=0.2):
        super().__init__()
        self.max_shift = max_shift
        self.max_scale = max_scale
        self.state_dim = state_dim

        self.action_head = nn.Sequential(
            nn.Conv2d(state_dim, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # spatial 차원 줄이기
            nn.Flatten(),
            nn.Linear(256, 3)
        )

        self.term_head = nn.Sequential(
            nn.Conv2d(state_dim, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 1)
        )

        nn.init.constant_(self.term_head[-1].bias, -2.0)

    def forward(self, state):
        """
        Args:
            state: Tensor [B, C=state_dim, H, W]
        Returns:
            Tensor [B, 4] = dx, dy, dscale, p_term
        """
        move = torch.tanh(self.action_head(state))
        shift = move[:, :2] * self.max_shift
        scale = move[:, 2:3] * self.max_scale
        p_term = torch.sigmoid(self.term_head(state))
        return torch.cat([shift, scale, p_term], dim=1)
