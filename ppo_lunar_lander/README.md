# PPO LunarLander (Gymnasium)

PPO from scratch, learning **`LunarLanderContinuous-v3`** — the standard
benchmark, with Gymnasium's well-tuned rewards. Same file layout and heavy
commenting as the DQN Snake project, so you can read PPO straight off the page.

## Why this version (vs the custom pygame one)

The custom rocket lander taught the mechanics, but its hand-written rewards
plateaued. Gymnasium's LunarLander gives you **rewards that are actually tuned
to be learnable**, plus a published "solved" bar (avg return ≥ 200 over 100
episodes) to measure against. Only `env.py` changed — the PPO code is the same.

## DQN (Snake) vs PPO (LunarLander)

| | DQN (Snake) | PPO (LunarLander) |
|---|---|---|
| Learns | `Q(s,a)` — value of each action | `π(a\|s)` — the policy directly |
| Actions | discrete (3) | **continuous (2)** ← the point |
| Exploration | ε-greedy (random with prob ε) | **stochastic policy** (sample a Gaussian) |
| Data | off-policy replay buffer | **on-policy** rollouts (use once, discard) |
| Stability trick | target network | **clipped objective** + GAE |

## Setup

```bash
pip install -r requirements.txt
# LunarLander needs the Box2D physics engine (included above). On Windows the
# wheel installs cleanly; if it ever fails, `pip install swig` first.
```

## Run

```bash
python train.py     # headless training; saves lunar_ppo_best/final.pth
python play.py --model lunar_ppo_best.pth   # watch it land (opens a window)
```

To watch it learn or see the live score curve, flip `render=True` / `plot=True`
in `train.py`'s `__main__`. Headless is many times faster.

## The control problem

State (8): `x, y, vx, vy, angle, angular_vel, left_leg, right_leg` (pad at origin).
Action (2 continuous, each in `[-1,1]`): `main engine` and `side thrusters`.
Reward: Gymnasium's shaping — toward-pad/upright/slow is good, fuel and crashing
are bad, +100 land / −100 crash. **Solved = mean return ≥ 200 over 100 episodes.**

## Where to read each PPO idea

1. **Stochastic policy** — `model.py: get_dist()`. Actor outputs a Gaussian
   *mean*; `log_std` is a learnable spread. `agent.py: select_action` samples it.
2. **GAE (advantage)** — `agent.py: compute_gae()`. Backward TD-error sweep;
   `gae_lambda` trades bias vs variance.
3. **Clipped objective** — `agent.py: update()`. `ratio = exp(new_logp −
   old_logp)`, then `min(ratio·A, clip(ratio, 1±ε)·A)`. *The* PPO line.
4. **Entropy bonus** — `agent.py: update()`. Default `0.0` here so the policy can
   sharpen; raise it if learning collapses early.
5. **Rewards** — now Gymnasium's, not ours. See `env.py`'s docstring.

### One Gymnasium subtlety (called out in `env.py`)

Gymnasium splits "episode ended" into `terminated` (really landed/crashed) and
`truncated` (hit the 1000-step limit). Strictly, GAE should only zero the future
on `terminated`. We merge them into one `done` for readability; the bias is tiny
because LunarLander almost always terminates. Worth knowing for when you read
production PPO code that keeps them separate.

## Knobs (all in `agent.py.__init__`)

- `entropy_coef` (0.0): raise to ~0.01 if it collapses to a bad habit early.
- `clip` (0.2): smaller = more conservative updates.
- `gae_lambda` (0.95) / `gamma` (0.99): credit-assignment horizon.
- `rollout_len` (in `train.py`): more steps per update = steadier, slower.

## Reading order

`constants.py` → `env.py` → `model.py` → `agent.py` → `train.py` → `play.py`.
