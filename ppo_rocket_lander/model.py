"""
The Actor-Critic network for PPO.

Two heads, each a small MLP:
  • actor  : state → MEAN of a Gaussian over the 2 continuous actions
  • critic : state → V(s), a single scalar "how good is this state" estimate

For continuous control the policy is a Normal distribution. The actor outputs
its MEAN; the spread (std) is a separate LEARNABLE parameter `log_std` that does
NOT depend on the state. This is the standard, simple PPO setup — the network
learns to be confident (small std) once it knows what to do.

Why Tanh instead of the DQN's ReLU?  Policy-gradient methods are sensitive to
the scale of activations; Tanh keeps things bounded and is the conventional
choice in PPO reference implementations.
"""
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = 8, action_dim: int = 2, hidden: int = 64):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),    nn.Tanh(),
            nn.Linear(hidden, action_dim),          # outputs the action MEAN
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),    nn.Tanh(),
            nn.Linear(hidden, 1),                   # outputs V(s)
        )

        # log(std), one per action dim. A Parameter => optimized by Adam.
        # Starting at 0 means std = exp(0) = 1.0 (lots of exploration at first).
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def get_dist(self, state):
        """Build the Gaussian policy distribution for a batch of states."""
        mean = self.actor(state)
        std  = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_value(self, state):
        """V(s), squeezed from shape (N, 1) to (N,)."""
        return self.critic(state).squeeze(-1)

    def evaluate(self, state, action):
        """
        Used during the PPO update. Given the states we visited and the actions
        we took, return:
          • log_prob : log π(a|s) under the CURRENT policy (summed over the 2 dims)
          • entropy  : how random the policy is (we bonus this to keep exploring)
          • value    : V(s) from the critic
        """
        dist     = self.get_dist(state)
        log_prob = dist.log_prob(action).sum(dim=-1)   # product of dims → sum of logs
        entropy  = dist.entropy().sum(dim=-1)
        value    = self.get_value(state)
        return log_prob, entropy, value
