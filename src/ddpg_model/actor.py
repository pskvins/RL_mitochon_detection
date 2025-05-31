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

        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,256)
        )

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
        out = self.net(state)

        move = torch.tanh(self.action_head(out))  # [dx, dy, dscale] ∈ [-1,1]
        shift = move[:, :2] * self.max_shift
        scale = move[:, 2:3] * self.max_scale
        #sigmoid for clipping the nn output to [0,1]
        p_term = torch.sigmoid(self.term_head(out)) 
        action = torch.cat([shift, scale, p_term], dim=1)

        return action 
