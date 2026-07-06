import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import QNet


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buf = deque(maxlen=capacity) # deque is a doubly-linked data structure and dynamically adds one item and removes the oldest to satisfy max capacity. 

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(a),
            torch.FloatTensor(r),
            torch.FloatTensor(np.array(s2)),
            torch.FloatTensor(d),
        )

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(
        self,
        state_dim:     int   = 11,
        action_dim:    int   = 3,
        lr:            float = 1e-3,
        gamma:         float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end:   float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size:    int   = 64,
        target_update: int   = 1000,
    ):
        self.action_dim    = action_dim # used to define the total number of possible actions. in this case its up, down, left
        # the three variables below are used to allow the snake to explore rather than exploit. 
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.train_steps   = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online = QNet(state_dim, action_dim).to(self.device) # the network that takes in the state values and outputs the Q-value of the outputs (these are the Q-values corresponding to the movements up, down, left, and right).
        self.target = QNet(state_dim, action_dim).to(self.device) # this is the network that the online network is trying to catch up to.
        # The target network is a snapshot of the online network at every 1000 steps so the target gives a good estimate for future state whilst the online keeps learning. 
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optim  = optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer()

    # ── Action selection ──────────────────────────────────────────────────────

    def act(self, state) -> int:
        if random.random() < self.epsilon: # if the random is below an epsilon value, then take a random action, essentially for exploration. 
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device) # converting the state into a FloatTensor and then unsqueezing for operations. 
            return self.online(s).argmax(dim=1).item() # passing it through the online network and retrieving the Q-value

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay) # decaying epsilon until it reaches the end. reducing randomness/exploration potential

    # ── Learning step ─────────────────────────────────────────────────────────

    def learn(self):
        if len(self.buffer) < self.batch_size: # make sure the buffer has at least 64 entries. only then start learning otherwise there isn't enough information to learn from. 
            return

        s, a, r, s2, done = self.buffer.sample(self.batch_size) # sample 64 states from the buffer
        s, a, r, s2, done = (t.to(self.device) for t in (s, a, r, s2, done))

        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1) # pass these states through an online network. results in output of [64, 3]. 64 states each with 3 Q-values.
        # however, it is important to note that the final column that we get is the list of Q-values corresponding to the action taken at each state. 

        with torch.no_grad():
            q_next = self.target(s2).max(1).values # this gives the best possible Q-value for the next state according to the target network
            q_tgt  = r + self.gamma * q_next * (1 - done) # this is where the done is important. if the snake dies and the episode ends, no future values are taken into account, just the current negative reward/punishment. 

        # in the above case, q_tgt is what the target network thinks the online should have predicted given the next state. the online network hence tries to catch up to this. 
        # now if the reward in the next state is high, the q_tgt will be high and so the q tries to catch up to that
        # this gets backpropogates through the previous states and the highest q-value action is taken
        # so not only is it taking the highest q-value action its trying to get the q-value to be as close to the target q-value as possible

        loss = nn.SmoothL1Loss()(q, q_tgt)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optim.step()

        self.train_steps += 1
        if self.train_steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "online":  self.online.state_dict(),
            "optim":   self.optim.state_dict()
            "epsilon": self.epsilon,
            "steps":   self.train_steps,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["online"])
        self.optim.load_state_dict(ckpt["optim"])
        self.epsilon     = ckpt["epsilon"]
        self.train_steps = ckpt["steps"]
