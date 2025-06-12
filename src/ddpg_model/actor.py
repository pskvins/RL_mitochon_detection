import torch
import torch.nn as nn
from ..env.utils import SwiGLU

class Actor(nn.Module):
    """
    Actor network in DDPG algorithm

    Args:
        state_dim: dimension of state which is concatenated vector of
            CNN feature and current position of box
        max_shift: maximum pixel shift allowed in both x & y
        max_scale: maximum scale ratio allowed for zoom in/out
    """
    def __init__(self, state_dim, max_shift=20.0, max_scale=0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.BatchNorm1d(256),
            # nn.ReLU(),
            SwiGLU(256),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            # nn.ReLU(),
            SwiGLU(256),
        )

        self.shift_head = nn.Linear(256, 2)   #predict dx, dy 
        self.scale_head = nn.Linear(256, 2)  #predict scale change 
        self.term_head = nn.Linear(256, 1)   #predict termination probability
        # self.delete_head = nn.Linear(256, 1)   #predict termination probability

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
        x = self.net(state)

        #tanh for clipping each neural net output to [-1,1]
        #then rescale to each range
        shift = torch.tanh(self.shift_head(x)) * self.max_shift       # [-max_shift, max_shift]
        scale = torch.tanh(self.scale_head(x)) * self.max_scale     # [-max_scale, max_scale]

        #sigmoid for clipping the nn output to [0,1]
        p_term = torch.sigmoid(self.term_head(x)) 
        # delete = torch.sigmoid(self.delete_head(x))

        action = torch.cat([shift, scale, p_term], dim=1)  # shape: [B, 4]
        return action 
