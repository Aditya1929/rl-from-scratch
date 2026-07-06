"""
PPO training loop.

The on-policy rhythm (very different from the DQN's per-step learn()):

    repeat:
        1. COLLECT a fixed number of steps (ROLLOUT_LEN) using the current policy,
           crossing episode boundaries freely (GAE's done-mask handles resets).
        2. UPDATE: run several epochs of clipped PPO over that batch.
        3. throw the batch away and collect fresh data.

Run headless (render=False) for speed; flip render=True to watch it learn.
"""
import os

import numpy as np
import pygame

from env import RocketLanderEnv
from agent import PPOAgent


def _update_plot(plt, ax, returns, means):
    ax.cla()
    ax.set_title("PPO Rocket Lander — Episode Return")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.plot(returns, alpha=0.35, color="steelblue",  label="Return")
    ax.plot(means,   linewidth=2, color="darkorange", label="Mean-50")
    ax.axhline(100, color="green", linestyle="--", linewidth=1, label="soft-landing bonus")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.pause(0.001)


def train(
    total_updates: int  = 600,    # number of collect→update cycles
    rollout_len:   int  = 2048,   # steps collected before each update
    render:        bool = True,
    watch_fps:     int  = 240,    # frame cap while watching (higher = faster training)
    plot:          bool = True,   # live matplotlib score window
    save_dir:      str  = ".",
    plot_every:    int  = 5,
):
    env   = RocketLanderEnv(render=render, fps=watch_fps)
    agent = PPOAgent()

    ep_returns, means = [], []
    best_mean = -1e9

    plt = ax = None
    if plot:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 4))

    print(f"Device: {agent.device}  |  {total_updates} updates × {rollout_len} steps")
    print("-" * 64)

    state = env.reset()
    ep_ret = 0.0

    for update in range(1, total_updates + 1):
        # ── 1. Collect a rollout ────────────────────────────────────────────────
        for _ in range(rollout_len):
            if render:
                env.render()

            action, logp, value = agent.select_action(state)
            nxt, reward, done    = env.step(action)
            agent.buffer.add(state, action, logp, reward, float(done), value)

            state   = nxt
            ep_ret += reward

            if done:
                ep_returns.append(ep_ret)
                means.append(float(np.mean(ep_returns[-50:])))
                ep_ret = 0.0
                state  = env.reset()

        # ── 2. Update on exactly that batch ─────────────────────────────────────
        last_value = agent.last_value(state)   # bootstrap the unfinished episode
        agent.update(last_value)

        # ── 3. Logging / checkpointing ──────────────────────────────────────────
        recent_mean = means[-1] if means else float("nan")
        if means and recent_mean > best_mean:
            best_mean = recent_mean
            agent.save(os.path.join(save_dir, "lander_ppo_best.pth"))

        if plot and update % plot_every == 0 and ep_returns:
            _update_plot(plt, ax, ep_returns, means)

        if update % 10 == 0 or update == 1:
            print(f"  update {update:4d} | episodes {len(ep_returns):5d} | "
                  f"mean-50 return {recent_mean:8.1f} | best {best_mean:8.1f} | "
                  f"std {agent.net.log_std.exp().mean().item():.3f}")

    agent.save(os.path.join(save_dir, "lander_ppo_final.pth"))
    print("-" * 64)
    print(f"Done. Best mean-50 return: {best_mean:.1f}  |  saved lander_ppo_final.pth")

    if plot:
        plt.ioff()
        if ep_returns:
            _update_plot(plt, ax, ep_returns, means)
        plt.show()

    if render and pygame.get_init():
        pygame.quit()


if __name__ == "__main__":
    train(
        total_updates = 600,
        rollout_len   = 2048,
        render        = False,   # headless: no game window
        plot          = False,   # headless: no score plot
        save_dir      = ".",
        plot_every    = 5,
    )
