"""
Load a trained policy and watch it land, forever.

Run:  python play.py
      python play.py --model lunar_ppo_best.pth

Like training's eval, this uses act_deterministic() — the policy MEAN, no
random sampling — so you see the behavior the agent actually learned.
A single-episode return ≥ 200 is a textbook-clean landing.
"""
import argparse

from env import LunarLanderEnv
from agent import PPOAgent
import constants as C


def play(model_path: str = "lunar_ppo_final.pth"):
    env   = LunarLanderEnv(render=True)
    agent = PPOAgent()
    agent.load(model_path)

    episode = 0
    returns = []
    print(f"Loaded {model_path}  |  close the window to quit")
    print("-" * 52)

    while True:
        state = env.reset()
        episode += 1
        ep_ret = 0.0

        while True:
            env.render()
            action = agent.act_deterministic(state)
            state, reward, done = env.step(action)
            ep_ret += reward
            if done:
                break

        returns.append(ep_ret)
        avg = sum(returns[-100:]) / len(returns[-100:])
        tag = "LANDED " if ep_ret >= C.SOLVED_SCORE else "        "
        print(f"  game {episode:4d} | return {ep_ret:8.1f} {tag}| avg-100 {avg:8.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lunar_ppo_final.pth")
    args = parser.parse_args()
    play(args.model)
