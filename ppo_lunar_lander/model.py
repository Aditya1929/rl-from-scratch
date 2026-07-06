"""
The Actor-Critic network for PPO. (Essentially unchanged from the custom
rocket-lander version — the network doesn't know or care that the environment
is now Gymnasium. That's the point.)

Two heads, each a small MLP:
  • actor  : state → MEAN of a Gaussian over the 2 continuous actions
  • critic : state → V(s), a single "how good is this state" estimate

Because the actions are continuous, the policy is a Normal distribution. The
actor outputs its MEAN; the spread (std) is a separate LEARNABLE parameter
`log_std` that doesn't depend on the state. The network can shrink it to become
confident once it knows what to do.
"""
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim:    int   = 8,
        action_dim:   int   = 2,
        hidden:       int   = 64,
        log_std_init: float = -0.5,   # exp(-0.5) ≈ 0.61 initial std
    ):
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

        # log(std), one per action dim, optimized by Adam alongside the weights.
        # We start it BELOW zero (std ≈ 0.6 instead of 1.0). The actions live in
        # [-1, 1], so a std of 1.0 would push most samples past the edges and get
        # clipped — starting tighter makes early exploration cleaner and lets the
        # policy sharpen instead of staying permanently noisy.
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

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
          • log_prob : log π(a|s) under the CURRENT policy (summed over both dims)
          • entropy  : how random the policy is (we bonus this to keep exploring)
          • value    : V(s) from the critic
        """
        dist     = self.get_dist(state)
        log_prob = dist.log_prob(action).sum(dim=-1)   # joint log-prob over dims
        entropy  = dist.entropy().sum(dim=-1)
        value    = self.get_value(state)
        return log_prob, entropy, value
