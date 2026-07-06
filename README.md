# RL from scratch

Three reinforcement-learning agents built by hand, with the algorithm written
out in the code rather than pulled from a library. Every file is heavily
commented so you can read the algorithm straight off the page. Same layout in
each folder (`env.py`, `model.py`, `agent.py`, `train.py`, `play.py`,
`constants.py`), so once you've read one, the others are easy.

## Projects

| Folder | Algorithm | Environment | Action space |
|---|---|---|---|
| [`snake_dqn`](snake_dqn) | **DQN** — learns `Q(s,a)`, ε-greedy, replay buffer, target network | Custom Snake | discrete |
| [`ppo_rocket_lander`](ppo_rocket_lander) | **PPO** — learns `π(a\|s)` directly, clipped objective + GAE | Custom 2D rocket lander | continuous |
| [`ppo_lunar_lander`](ppo_lunar_lander) | **PPO** — same agent, tuned rewards | Gymnasium `LunarLanderContinuous-v3` | continuous |

The progression is deliberate: **DQN → PPO** (value-based to policy-based,
discrete to continuous), then **custom lander → Gymnasium lander** (hand-written
rewards that plateaued → rewards actually tuned to be learnable, with a published
"solved" bar to measure against).

## DQN vs PPO at a glance

| | DQN (Snake) | PPO (Landers) |
|---|---|---|
| Learns | `Q(s,a)` — value of each action | `π(a\|s)` — the policy directly |
| Actions | discrete | **continuous** |
| Exploration | ε-greedy (random with prob ε) | **stochastic policy** (sample a Gaussian) |
| Data | off-policy replay buffer (reuse old) | **on-policy** rollouts (use once, discard) |
| Stability trick | target network | **clipped objective** + GAE |

## Running any project

```bash
cd <project_folder>
pip install -r requirements.txt

python train.py                 # train (headless); saves *_best.pth / *_final.pth
python play.py                  # watch the trained agent
```

Pre-trained weights (`*.pth`) are checked in, so `play.py` works right after
cloning — no need to train first.
