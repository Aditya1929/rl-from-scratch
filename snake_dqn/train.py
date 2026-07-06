import os

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pygame

from env import SnakeEnv
from agent import DQNAgent


def _update_plot(ax, scores, means):
    ax.cla()
    ax.set_title("DQN Snake — Training Scores")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    x = range(len(scores))
    ax.plot(x, scores, alpha=0.45, color="steelblue",  label="Score")
    ax.plot(x, means,  linewidth=2, color="darkorange", label="Mean-100")
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.pause(0.001)


def train(
    episodes:   int  = 1_000,
    render:     bool = True,
    max_steps:  int  = 2_000,
    save_dir:   str  = ".",
    plot_every: int  = 20,
):
    env   = SnakeEnv(render=render)
    agent = DQNAgent()

    scores, means = [], []
    best_score    = 0

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))

    print(f"Device: {agent.device}  |  Training for {episodes} episodes")
    print("-" * 60)

    for ep in range(1, episodes + 1):
        state = env.reset()

        for _ in range(max_steps):
            if render:
                env.render()

            # the loop that takes action, spits out the next state, reward, and done. the state gets added to buffer. the agent learns and the next state becomes the current state
            action          = agent.act(state)
            nxt, rew, done  = env.step(action)
            agent.buffer.push(state, action, rew, nxt, float(done))
            agent.learn()
            state = nxt

            if done:
                break

        agent.decay_epsilon()

        score = env.score
        scores.append(score)
        mean = float(np.mean(scores[-100:]))
        means.append(mean)

        if score > best_score:
            best_score = score
            path = os.path.join(save_dir, "snake_dqn_best.pth")
            agent.save(path)
            print(f"  *** New best: {best_score}  (ep {ep})  saved → {path}")

        if ep % plot_every == 0:
            _update_plot(ax, scores, means)

        if ep % 50 == 0 or ep == 1:
            print(
                f"  ep {ep:5d} | score {score:3d} | "
                f"mean-100 {mean:5.1f} | eps {agent.epsilon:.3f} | "
                f"best {best_score}"
            )

    final_path = os.path.join(save_dir, "snake_dqn_final.pth")
    agent.save(final_path)
    print("-" * 60)
    print(f"Done.  Best score: {best_score}  |  Final model → {final_path}")

    plt.ioff()
    _update_plot(ax, scores, means)
    plt.show()

    if render and pygame.get_init():
        pygame.quit()


if __name__ == "__main__":
    train(
        episodes   = 1_000,
        render     = True,    # False = headless (faster training)
        max_steps  = 2_000,
        save_dir   = ".",
        plot_every = 20,
    )
