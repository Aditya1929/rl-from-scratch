"""
Load a trained policy and watch it try to land, forever.

Run:  python play.py
      python play.py --model lander_ppo_best.pth
      python play.py --fps 30          # slow it down

Unlike training, this uses act_deterministic() — the policy MEAN, no random
sampling — so you see the behavior the agent actually learned.
"""
import argparse

import pygame

from env import RocketLanderEnv
from agent import PPOAgent


def play(model_path: str = "lander_ppo_final.pth", fps: int = 50):
    env   = RocketLanderEnv(render=True, fps=fps)
    agent = PPOAgent()
    agent.load(model_path)

    episode = landings = 0
    print(f"Loaded {model_path}  |  close the window to quit")
    print("-" * 44)

    while True:
        state = env.reset()
        episode += 1

        while True:
            env.render()
            action = agent.act_deterministic(state)
            state, _, done = env.step(action)
            if done:
                break

        landed = env._is_soft_landing() and env.y <= 0.0
        landings += int(landed)
        result = "LANDED " if landed else "crashed"
        print(f"  game {episode:4d} | {result} | success rate {landings}/{episode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lander_ppo_final.pth")
    parser.add_argument("--fps",   type=int, default=50)
    args = parser.parse_args()
    play(args.model, args.fps)
