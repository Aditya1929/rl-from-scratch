# PPO Rocket Lander

A 2D rocket-landing environment + a from-scratch **PPO** agent, built in the same
hand-rolled style as the Snake DQN one folder up. This is a teaching project: the
code is heavily commented so you can read PPO straight off the page.

## Why this and not Snake/DQN?

| | DQN (Snake) | PPO (Lander) |
|---|---|---|
| Learns | `Q(s,a)` — value of each action | `π(a\|s)` — the policy directly |
| Actions | discrete (3) | **continuous (2)** ← the whole point |
| Exploration | ε-greedy (random with prob ε) | **stochastic policy** (sample from a Gaussian) |
| Data | off-policy replay buffer (reuse old) | **on-policy** rollouts (use once, discard) |
| Stability trick | target network | **clipped objective** + GAE |

DQN literally *cannot* output "thrust at 0.73". PPO can, because its policy is a
probability distribution the agent samples from.

## Run it

```bash
pip install -r requirements.txt

python train.py     # headless training (~minutes); saves lander_ppo_best/final.pth
python play.py --model lander_ppo_best.pth   # watch it land
```

To watch it learn (much slower), set `render=True` in `train.py`'s `__main__`.

## The control problem

State (8): `x, y, vx, vy, θ (tilt), ω (spin), left_leg, right_leg` — all relative to the pad.
Action (2 continuous): `main ∈ [0,1]` (throttle) and `torque ∈ [-1,1]` (spin).

The main engine fires along the rocket's heading, so to move sideways you must
**tilt, then thrust, then straighten** — the classic Lunar Lander challenge.

## Where to read each PPO idea in the code

1. **Stochastic policy** — `model.py: get_dist()`. The actor outputs a Gaussian
   *mean*; `log_std` is a learnable spread. `select_action` samples from it.
2. **GAE (advantage estimate)** — `agent.py: compute_gae()`. Backward pass over
   the rollout accumulating TD-errors; `gae_lambda` trades bias vs variance.
3. **Clipped surrogate objective** — `agent.py: update()`. `ratio = exp(new_logp
   - old_logp)`, then `min(ratio·A, clip(ratio,1±ε)·A)`. This is *the* PPO line.
4. **Entropy bonus** — `agent.py: update()`. Subtracted from the loss to keep the
   policy exploring instead of collapsing to deterministic too soon.
5. **Potential-based reward shaping** — `env.py: _potential()`. Dense per-step
   signal that makes the sparse ±100 landing reward learnable.

## Knobs worth turning (all in `agent.py.__init__`)

- `clip` (0.2): smaller = more conservative updates.
- `entropy_coef` (0.01): raise it if the policy collapses too early and stops improving.
- `gae_lambda` (0.95): toward 1.0 = lower bias/higher variance.
- `rollout_len` (in `train.py`): more steps per update = more stable, slower.

## Reading order

`constants.py` → `env.py` → `model.py` → `agent.py` → `train.py` → `play.py`.
