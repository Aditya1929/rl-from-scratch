"""
PPO agent. (The same algorithm as the rocket-lander version — only a couple of
hyperparameters are retuned for LunarLander, noted inline.)

Contrast with the DQN agent in the Snake project:
  • DQN was OFF-policy: a 100k replay buffer, learn every step from old memories.
  • PPO is ON-policy: collect a fresh batch with the CURRENT policy, do a few
    epochs of updates on exactly that batch, then THROW IT AWAY.

The three ideas that make PPO work, all visible below:
  1. GAE — a low-variance estimate of "how much better was this action than
     average?"  (compute_gae)
  2. The CLIPPED surrogate objective — improve the policy, but don't let any one
     update move it too far.  (update)
  3. An entropy bonus — optional pressure to stay exploratory.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import ActorCritic


class RolloutBuffer:
    """Plain lists; on-policy data is collected, used once, then cleared."""
    def __init__(self):
        self.states, self.actions, self.logprobs = [], [], []
        self.rewards, self.dones, self.values     = [], [], []

    def add(self, s, a, lp, r, d, v):
        self.states.append(s);   self.actions.append(a); self.logprobs.append(lp)
        self.rewards.append(r);  self.dones.append(d);   self.values.append(v)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)


class PPOAgent:
    def __init__(
        self,
        state_dim:    int   = 8,
        action_dim:   int   = 2,
        lr:           float = 3e-4,
        gamma:        float = 0.99,   # discount: how far ahead we plan
        gae_lambda:   float = 0.95,   # GAE bias/variance knob (1=high var, 0=high bias)
        clip:         float = 0.20,   # the PPO clip range ε
        epochs:       int   = 10,     # passes over each rollout
        minibatch:    int   = 64,
        value_coef:   float = 0.50,   # weight of the critic loss
        entropy_coef: float = 0.0,    # exploration bonus. 0.0 lets the policy
                                      # confidently sharpen (LunarLander solves
                                      # fine without it); raise to ~0.01 if it
                                      # collapses to a bad habit too early.
        max_grad_norm:float = 0.50,
    ):
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip          = clip
        self.epochs        = epochs
        self.minibatch     = minibatch
        self.value_coef    = value_coef
        self.entropy_coef  = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net    = ActorCritic(state_dim, action_dim).to(self.device)
        self.optim  = optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    # ── Acting ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, state):
        """
        Sample an action from the current policy (stochastic — that randomness
        IS the exploration; there is no epsilon here). Returns the action plus
        the bookkeeping PPO needs later: the log-prob and the value estimate.
        """
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist   = self.net.get_dist(s)
        action = dist.sample()                      # may fall outside [-1, 1]
        logp   = dist.log_prob(action).sum(-1)      # log π(a|s)
        value  = self.net.get_value(s)
        # Store the RAW sampled action (and its log-prob). The env clips it to
        # the valid range; keeping the raw action makes the log-prob exact.
        return (action.squeeze(0).cpu().numpy(),
                float(logp.item()),
                float(value.item()))

    @torch.no_grad()
    def act_deterministic(self, state):
        """For play.py: take the policy MEAN, no sampling — the learned behavior."""
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.net.actor(s).squeeze(0).cpu().numpy()

    @torch.no_grad()
    def last_value(self, state):
        """Bootstrap value of the state AFTER the rollout, for GAE."""
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.net.get_value(s).item())

    # ── Advantage estimation ────────────────────────────────────────────────────

    def compute_gae(self, last_value):
        """
        Generalized Advantage Estimation. Walk BACKWARD through the rollout
        accumulating the TD-error δ_t = r_t + γ·V(s_{t+1}) − V(s_t):

            A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...

        λ trades bias vs variance. The (1 − done) mask cuts the chain at episode
        boundaries so reward never leaks across resets.
        Returns advantages and the value targets (returns = advantage + value).
        """
        rewards = self.buffer.rewards
        dones   = self.buffer.dones
        values  = self.buffer.values + [last_value]   # append bootstrap

        advantages = [0.0] * len(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask  = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae   = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae

        returns = [a + v for a, v in zip(advantages, values[:-1])]
        return advantages, returns

    # ── The PPO update ──────────────────────────────────────────────────────────

    def update(self, last_value):
        advantages, returns = self.compute_gae(last_value)

        states   = torch.as_tensor(np.array(self.buffer.states),  dtype=torch.float32, device=self.device)
        actions  = torch.as_tensor(np.array(self.buffer.actions), dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(self.buffer.logprobs,          dtype=torch.float32, device=self.device)
        returns  = torch.as_tensor(returns,                       dtype=torch.float32, device=self.device)
        advs     = torch.as_tensor(advantages,                    dtype=torch.float32, device=self.device)

        # Normalizing advantages stabilizes the gradient scale across updates.
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        n = len(states)
        idx = np.arange(n)

        # Several epochs of minibatch SGD over THIS rollout only.
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.minibatch):
                b = idx[start:start + self.minibatch]

                new_logp, entropy, value = self.net.evaluate(states[b], actions[b])

                # ── The clipped surrogate objective ──────────────────────────
                # ratio = π_new(a|s) / π_old(a|s).  In log space that's subtract
                # then exp. If the new policy makes a GOOD action (adv>0) much
                # more likely, ratio grows — the clip caps how much we reward
                # that in one update, preventing a destructive jump.
                ratio = torch.exp(new_logp - old_logp[b])
                surr1 = ratio * advs[b]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advs[b]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Critic regresses V(s) toward the GAE returns.
                value_loss = nn.functional.mse_loss(value, returns[b])

                # Entropy bonus (subtracted, since we MINIMIZE the total loss).
                entropy_loss = entropy.mean()

                loss = (policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy_loss)

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optim.step()

        self.buffer.clear()   # on-policy: discard the data we just trained on

    # ── Persistence ──────────────────────────────────────────────────────────────

    def save(self, path):
        torch.save({"net": self.net.state_dict(),
                    "optim": self.optim.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(ckpt["net"])
        self.optim.load_state_dict(ckpt["optim"])
