"""
Load the best saved model and watch it play forever.
Run:  python play.py
      python play.py --model snake_dqn_final.pth
      python play.py --fps 15   # slow it down to 15 fps
"""
import argparse
import pygame
from env import SnakeEnv
from agent import DQNAgent


def play(model_path: str = "snake_dqn_best.pth", fps: int = 30):
    env   = SnakeEnv(render=True, fps=fps)
    agent = DQNAgent()
    agent.load(model_path)
    agent.epsilon = 0.0   # pure exploitation — no random moves

    episode    = 0
    best_score = 0

    print(f"Loaded {model_path}  |  press the window's X to quit")
    print("-" * 40)

    while True:
        state = env.reset()
        episode += 1

        while True:
            env.render()
            action         = agent.act(state)
            state, _, done = env.step(action)
            if done:
                break

        score = env.score
        if score > best_score:
            best_score = score
        print(f"  game {episode:4d} | score {score:3d} | best {best_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="snake_dqn_final.pth")
    parser.add_argument("--fps",   type=int, default=30)
    args = parser.parse_args()
    play(args.model, args.fps)
