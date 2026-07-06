import torch.nn as nn


class QNet(nn.Module):
    """Two-hidden-layer MLP: Linear(11→256) → ReLU → Linear(256→256) → ReLU → Linear(256→3)."""

    def __init__(self, in_dim: int = 11, out_dim: int = 3, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)
