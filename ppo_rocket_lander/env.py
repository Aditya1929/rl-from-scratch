"""
2D Rocket Lander environment (pygame).

Same hand-rolled style as the Snake DQN env: reset() / step() / render(),
talking to the agent only through (state, reward, done).

State — 8 floats (mirrors the classic Lunar Lander layout):
    [ x, y, vx, vy, theta, omega, leg_left, leg_right ]
      x, y      : position relative to the pad (pad center is the origin)
      vx, vy    : linear velocity
      theta     : tilt angle in radians (0 = upright, + = tilted counter-clockwise)
      omega     : angular velocity
      leg_left  : 1.0 if the left  foot is touching the ground, else 0.0
      leg_right : 1.0 if the right foot is touching the ground, else 0.0

Action — 2 CONTINUOUS floats (this is the whole point of PPO):
    [ main, torque ]
      main   ∈ [0, 1]   throttle of the main engine (clipped by the agent)
      torque ∈ [-1, 1]  side-thruster torque: spins the rocket

Reward — potential-based shaping (see _potential) + fuel cost + terminal bonus:
      step:     (potential - prev_potential)  - fuel_used
      success:  +100   (soft landing on the pad, upright, slow)
      crash:    -100   (hit the ground hard / off-pad, or flew off-screen)
"""
import math
import random

import numpy as np
import pygame

import constants as C


class RocketLanderEnv:
    def __init__(self, render: bool = False, fps: int = C.FPS):
        self.do_render = render
        self.fps       = fps
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((C.WIDTH, C.HEIGHT))
            pygame.display.set_caption("Rocket Lander — PPO")
            self.font   = pygame.font.SysFont("Consolas", 16)
            self.ticker = pygame.time.Clock()
        self.reset()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self):
        self.x      = random.uniform(-C.START_X_JIT, C.START_X_JIT)
        self.y      = C.START_Y
        self.vx     = random.uniform(-C.START_V_JIT, C.START_V_JIT)
        self.vy     = random.uniform(-C.START_V_JIT, C.START_V_JIT)
        self.theta  = random.uniform(-C.START_A_JIT, C.START_A_JIT)
        self.omega  = 0.0
        self.steps  = 0
        self.last_main   = 0.0   # remembered only so render() can draw the flame
        self.last_torque = 0.0

        # Potential-based shaping needs the PREVIOUS potential to take a difference.
        self.prev_potential = self._potential()
        return self._state()

    def step(self, action):
        """action = [main, torque]. Returns (next_state, reward, done)."""
        self.steps += 1

        # 1) Clip the raw (possibly out-of-range) action the policy sampled.
        main   = float(np.clip(action[0],  0.0, 1.0))
        torque = float(np.clip(action[1], -1.0, 1.0))
        self.last_main, self.last_torque = main, torque

        # 2) Apply forces.  The main engine thrusts along the rocket's "up" axis,
        #    which is the upright direction (0, 1) rotated by theta:
        #        body_up = (-sin θ, cos θ)
        #    so when tilted, less of the thrust fights gravity — that is the
        #    coupling that makes this a real control problem.
        thrust = C.MAIN_POWER * main
        ax = -math.sin(self.theta) * thrust
        ay =  math.cos(self.theta) * thrust - C.GRAVITY
        ang_acc = C.TORQUE_POWER * torque

        # 3) Semi-implicit Euler integration.
        self.vx    += ax * C.DT
        self.vy    += ay * C.DT
        self.omega += ang_acc * C.DT
        self.x     += self.vx * C.DT
        self.y     += self.vy * C.DT
        self.theta += self.omega * C.DT

        # 4) Reward: potential difference (shaping) minus fuel burned.
        potential = self._potential()
        reward = potential - self.prev_potential
        self.prev_potential = potential
        reward -= 0.30 * main            # main engine is expensive
        reward -= 0.03 * abs(torque)     # side thrusters are cheap

        # 5) Termination.
        done = False
        if self.y <= 0.0:                       # touched the ground
            done = True
            if self._is_soft_landing():
                reward += 100.0
            else:
                reward -= 100.0
        elif abs(self.x) > C.OUT_X or self.y > C.OUT_Y:   # flew away
            done = True
            reward -= 100.0
        elif self.steps >= C.MAX_STEPS:         # ran out of time (truncated)
            done = True

        return self._state(), reward, done

    def render(self):
        if not self.do_render:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        s = self.screen
        s.fill(C.SKY)

        # Ground line + landing pad (drawn in screen space).
        ground_y = self._sy(0.0)
        pygame.draw.line(s, C.GREY, (0, ground_y), (C.WIDTH, ground_y), 2)
        pad_l = self._sx(-C.PAD_HALF_W)
        pad_r = self._sx( C.PAD_HALF_W)
        pad_color = C.PAD_OK
        pygame.draw.line(s, pad_color, (pad_l, ground_y), (pad_r, ground_y), 6)

        # Rocket: a triangle pointing "up" along its body axis, rotated by theta.
        body = [(0.0, 0.10), (-0.05, -0.05), (0.05, -0.05)]   # nose, left foot, right foot
        pts  = [self._to_screen(self._rotate(px, py)) for px, py in body]
        pygame.draw.polygon(s, C.ROCKET, pts)

        # Flame underneath, length proportional to throttle.
        if self.last_main > 0.02:
            flen = 0.04 + 0.12 * self.last_main
            flame = [(-0.03, -0.05), (0.03, -0.05), (0.0, -0.05 - flen)]
            fpts  = [self._to_screen(self._rotate(px, py)) for px, py in flame]
            pygame.draw.polygon(s, C.FLAME, fpts)

        hud = f"step {self.steps:3d}  vx {self.vx:+.2f}  vy {self.vy:+.2f}  " \
              f"θ {self.theta:+.2f}  ω {self.omega:+.2f}"
        s.blit(self.font.render(hud, True, C.WHITE), (8, 8))
        pygame.display.flip()
        self.ticker.tick(self.fps)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _state(self):
        legs = 1.0 if self.y <= C.LEG_HEIGHT else 0.0
        return np.array([
            self.x, self.y, self.vx, self.vy,
            self.theta, self.omega, legs, legs,
        ], dtype=np.float32)

    def _potential(self):
        """
        A 'closeness to a good landing' score.  Higher is better.  PPO never
        sees this directly — the per-step reward is the DIFFERENCE of this
        potential between consecutive states (potential-based shaping), which
        provably does not change the optimal policy but makes the gradient
        signal dense instead of only ±100 at the very end.
        """
        dist  = math.hypot(self.x, self.y)
        speed = math.hypot(self.vx, self.vy)
        return -100.0 * dist - 100.0 * speed - 100.0 * abs(self.theta)

    def _is_soft_landing(self):
        return (
            abs(self.x)     < C.PAD_HALF_W and
            abs(self.vx)    < C.LAND_VX    and
            abs(self.vy)    < C.LAND_VY    and
            abs(self.theta) < C.LAND_ANGLE and
            abs(self.omega) < C.LAND_OMEGA
        )

    # ── Coordinate transforms (world → screen) ─────────────────────────────────

    def _rotate(self, px, py):
        """Rotate a body-frame point by theta, then translate to world position."""
        c, sN = math.cos(self.theta), math.sin(self.theta)
        wx = self.x + (px * c - py * sN)
        wy = self.y + (px * sN + py * c)
        return wx, wy

    def _sx(self, wx):
        return int((wx + C.WORLD_X) / (2 * C.WORLD_X) * C.WIDTH)

    def _sy(self, wy):
        return int(C.HEIGHT - (wy / C.WORLD_Y) * C.HEIGHT)

    def _to_screen(self, pt):
        return (self._sx(pt[0]), self._sy(pt[1]))
