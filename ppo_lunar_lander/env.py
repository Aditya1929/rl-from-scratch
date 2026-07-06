"""
A thin ADAPTER around Gymnasium's LunarLanderContinuous.

Why a wrapper at all?  Gymnasium's API is slightly richer than the simple
interface the rest of our code (copied in spirit from the DQN Snake project)
expects:

    Gymnasium                          Our simple interface
    ─────────────────────────────      ──────────────────────────────
    obs, info        = env.reset()     state        = env.reset()
    obs, reward,                       state, reward, done = env.step(a)
      terminated, truncated, info
      = env.step(a)

This class translates between them, so `model.py`, `agent.py`, `train.py`, and
`play.py` stay almost identical to the versions you already read — ONLY the
environment changed. That is the whole lesson: PPO doesn't care what it controls.

────────────────────────────────────────────────────────────────────────────
State — 8 floats (Gymnasium gives these to us, already sensibly scaled):
    [ x, y, vx, vy, angle, angular_vel, leg_left, leg_right ]
      x, y         : position; (0, 0) is the landing pad
      vx, vy       : linear velocity
      angle        : tilt (radians)
      angular_vel  : spin rate
      leg_left/right : 1.0 when that leg touches the ground, else 0.0

Action — 2 continuous floats in [-1, 1]:
      action[0]  main engine : <0 = off; 0→1 throttles the engine 50%→100%
      action[1]  side engines: < -0.5 fires left, > 0.5 fires right

Reward — GYM'S BUILT-IN shaping (the reason we switched):
      + moving toward the pad, slowing down, staying level, legs touching down
      - firing engines (fuel cost)
      +100 for landing, -100 for crashing
      A 100-episode average of +200 means "solved".
────────────────────────────────────────────────────────────────────────────
"""
import gymnasium as gym
import numpy as np

import constants as C


class LunarLanderEnv:
    def __init__(self, render: bool = False, seed: int | None = None):
        # render_mode="human" opens a window and auto-draws each frame;
        # None runs headless (fast — used for training).
        self.env = gym.make(C.ENV_ID, render_mode="human" if render else None)
        self.do_render = render
        self._seed = seed   # only used on the FIRST reset, then cleared

    def reset(self):
        obs, _info = self.env.reset(seed=self._seed)
        self._seed = None
        return obs.astype(np.float32)

    def step(self, action):
        # The policy is a Gaussian, so it can sample values outside the valid
        # box; clip them into [-1, 1] before handing the action to Gymnasium.
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        obs, reward, terminated, truncated, _info = self.env.step(action)

        # SUBTLETY worth understanding:
        #   terminated = the episode really ended (landed or crashed)
        #   truncated  = we just hit the 1000-step time limit
        # Strictly, GAE should only zero the future value on `terminated`
        # (a time-out isn't a real ending). We merge both into one `done` to
        # keep this simple, readable interface. LunarLander almost always
        # `terminated`s, so the bias is negligible — but now you know it's there.
        done = bool(terminated or truncated)
        return obs.astype(np.float32), float(reward), done

    def render(self):
        # In "human" mode Gymnasium already draws inside step()/reset(); this is
        # kept only so train.py/play.py can call env.render() like before.
        if self.do_render:
            self.env.render()

    def close(self):
        self.env.close()
