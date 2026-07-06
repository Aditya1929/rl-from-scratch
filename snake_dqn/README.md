# DQN Snake

Classic Snake with a **Deep Q-Network** agent, written from scratch and heavily
commented so you can read DQN straight off the page. This is the starting point
of the repo — the PPO landers one folder up build on the ideas here.

## The idea in one line

Learn `Q(s, a)` — the expected future reward of taking action `a` in state `s` —
then always act greedily on it. A neural net approximates `Q`; the tricks that
make it stable are a **replay buffer** and a **target network**.

## State, actions, rewards

The agent never sees pixels. `env.py` hands it an **11-value engineered state**:

| Group | Features |
|---|---|
| Danger | `danger_straight`, `danger_right`, `danger_left` (collision one step ahead) |
| Direction | `dir_left`, `dir_right`, `dir_up`, `dir_down` (one-hot) |
| Food | `food_left`, `food_right`, `food_up`, `food_down` (relative quadrant) |

- **Actions (3, relative):** `0` = go straight, `1` = turn right, `2` = turn left.
  Because turns are relative, reversing into itself is literally unrepresentable.
- **Rewards:** `+10` eat food, `-10` die, `+1` per surviving step.

## How it learns

| Piece | What it does | Where |
|---|---|---|
| `QNet` | MLP `11 → 256 → 256 → 3`, outputs a Q-value per action | `model.py` |
| ε-greedy | Acts randomly with prob ε (explore), else greedy (exploit); ε decays `1.0 → 0.01` | `agent.py:act` |
| Replay buffer | Stores 100k transitions, samples random batches of 64 to break correlation | `agent.py:ReplayBuffer` |
| Target network | A frozen copy of the online net, synced every 1000 steps, gives stable TD targets | `agent.py:learn` |
| TD update | `q_target = r + γ · maxₐ Q_target(s', a) · (1 − done)`, Huber loss, grad-clip | `agent.py:learn` |

## Run it

```bash
pip install -r requirements.txt

python train.py                          # train (~1000 episodes); set render=False for speed
python play.py                           # watch the best model play forever
python play.py --model snake_dqn_final.pth --fps 15   # final weights, slowed down
```

Pre-trained weights (`snake_dqn_best.pth`, `snake_dqn_final.pth`) are checked in,
so `play.py` works immediately after cloning.

## Where to go next

Snake is discrete-action and value-based. The next step in this repo swaps both:
**PPO** (`../ppo_rocket_lander`, `../ppo_lunar_lander`) learns the policy `π(a|s)`
directly and outputs **continuous** actions — see the comparison table in the
[top-level README](../README.md).
