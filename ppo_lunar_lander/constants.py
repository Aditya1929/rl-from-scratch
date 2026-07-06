"""
Project-wide configuration for the PPO LunarLander.

Unlike the custom pygame version, we no longer hand-code gravity, thrust, or
colors — Gymnasium's LunarLander owns all the physics and rendering. So this
file shrinks to just the few knobs that are about the *experiment*, not the
game itself.
"""

# Which Gymnasium environment to learn.
#   "LunarLanderContinuous-v3" = 2 continuous actions (main engine + side thrusters).
#   (The plain "LunarLander-v3" is the DISCRETE 4-button version — not what we want,
#    since the point of PPO here is continuous control.)
ENV_ID = "LunarLanderContinuous-v3"

# Gymnasium's official "solved" bar: an average return of 200 over 100 episodes.
SOLVED_SCORE = 200.0

# Seed for the first reset, so runs are reproducible. Set to None for fully random.
SEED = 0
