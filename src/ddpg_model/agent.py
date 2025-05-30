import torch
import torch.nn as nn
import torch.optim as optim
from .actor import Actor
from .critic import Critic
from .replay_buffer import ReplayBuffer

class DDPGAgent:
    def __init__(self, state_dim, action_dim,
        device = "cpu", actor_lr: float = 1e-4,
        critic_lr: float = 1e-3, gamma: float = 0.99, tau: float = 0.005):

        """
        Args: 
            state_dim: dimension of state which is concatenated vector of
                CNN feature and current position 
            action_dim (int): dimension of the action vector
                default setting is 4, s[dx, dy, scale, p_term]

            actor_lr, critic_lr: learning rate of actor and critic network
            gamma: discount factor in range (0,1]
            tau: soft update parameter
                ㄴ(Soft update: w <- tau * w + (1-tau)*w )
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau

        # Define Actor & Critic
        self.actor = Actor(state_dim).to(device)
        self.actor_target = Actor(state_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Define ptimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.loss_fn = nn.MSELoss()

    def select_action(self, state: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        """
        Args:
            state: Tensor [1, state_dim]
            noise_std: optional exploration noise

        Returns:
            action: Tensor [1, action_dim]
        """
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state.to(self.device))
        self.actor.train()

        if noise_std > 0:
            action += noise_std * torch.randn_like(action)
        return action.clamp(-1.0, 1.0)  # optional: clip action range

    def update(self, buffer: ReplayBuffer, batch_size: int = 64):
        """
        Sample from buffer and update actor/critic
        """
        s, a, r, s2, d = buffer.sample(batch_size, device=self.device)

        # --- Critic update ---
        with torch.no_grad():
            next_a = self.actor_target(s2)
            target_q = self.critic_target(s2, next_a)
            y = r + self.gamma * (1 - d) * target_q

        current_q = self.critic(s, a)
        critic_loss = self.loss_fn(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()    
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # --- Actor update ---
        pred_a = self.actor(s)
        actor_loss = -self.critic(s, pred_a).mean()

        if not torch.isfinite(actor_loss):
                print("Non-finite actor loss:", actor_loss.item())
                return None, None
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # --- Soft target update ---
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        return actor_loss.item(), critic_loss.item()

    def _soft_update(self, target_net: nn.Module, source_net: nn.Module):
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor'])
        self.critic_target.load_state_dict(checkpoint['critic'])