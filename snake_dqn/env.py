import random
import numpy as np
import pygame

from constants import (
    BLOCK, WIDTH, HEIGHT, GRID_W, GRID_H,
    BLACK, WHITE, GREEN, DARK_GREEN, RED,
    RIGHT, LEFT, UP, DOWN, CW,
)


class SnakeEnv:
    """
    Pygame Snake environment.

    State — 11 binary floats:
      [danger_straight, danger_right, danger_left,
       dir_left, dir_right, dir_up, dir_down,
       food_left, food_right, food_up, food_down]

    Actions: 0 = straight, 1 = turn_right, 2 = turn_left
    Rewards: +10 food, -10 death, +1 per survival step
    """

    def __init__(self, render: bool = False, fps: int = 60):
        self.do_render = render
        self.fps       = fps
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Snake — DQN")
            self.font   = pygame.font.SysFont("Arial", 18)
            self.ticker = pygame.time.Clock()
        self.reset()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self):
        cx, cy     = GRID_W // 2, GRID_H // 2
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.dir   = RIGHT
        self.score = 0
        self.steps = 0
        self._place_food()
        return self._state()

    def step(self, action: int):
        """Returns (next_state, reward, done)."""
        self.steps += 1

        idx = CW.index(self.dir)
        if action == 1:
            self.dir = CW[(idx + 1) % 4]
        elif action == 2:
            self.dir = CW[(idx - 1) % 4]

        hx, hy   = self.snake[0]
        dx, dy   = self.dir
        new_head = (hx + dx, hy + dy)
        self.snake.insert(0, new_head)

        if self._collides(new_head):
            self.snake.pop()
            return self._state(), -10.0, True

        if new_head == self.food:
            self.score += 1
            reward = 10.0
            self._place_food()
        else:
            self.snake.pop()
            reward = 1.0

        return self._state(), reward, False

    def render(self):
        if not self.do_render:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        screen = pygame.display.get_surface()
        screen.fill(BLACK)

        for i, (x, y) in enumerate(self.snake):
            color = DARK_GREEN if i == 0 else GREEN
            pygame.draw.rect(screen, color,
                             (x * BLOCK + 1, y * BLOCK + 1, BLOCK - 2, BLOCK - 2),
                             border_radius=3)

        fx, fy = self.food
        pygame.draw.circle(screen, RED,
                           (fx * BLOCK + BLOCK // 2, fy * BLOCK + BLOCK // 2),
                           BLOCK // 2 - 1)

        screen.blit(self.font.render(f"Score: {self.score}", True, WHITE), (6, 6))
        pygame.display.flip()
        self.ticker.tick(self.fps)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _place_food(self):
        occupied = set(self.snake)
        while True:
            pt = (random.randint(0, GRID_W - 1), random.randint(0, GRID_H - 1))
            if pt not in occupied:
                self.food = pt
                return

    def _collides(self, pt=None):
        if pt is None:
            pt = self.snake[0]
        x, y = pt
        return x < 0 or x >= GRID_W or y < 0 or y >= GRID_H or pt in self.snake[1:]

    def _state(self):
        hx, hy = self.snake[0]
        idx    = CW.index(self.dir)
        d_fwd  = CW[idx]
        d_rgt  = CW[(idx + 1) % 4]
        d_lft  = CW[(idx - 1) % 4]

        def ahead(d): return (hx + d[0], hy + d[1])

        fx, fy = self.food
        return np.array([
            self._collides(ahead(d_fwd)),
            self._collides(ahead(d_rgt)),
            self._collides(ahead(d_lft)),
            self.dir == LEFT,
            self.dir == RIGHT,
            self.dir == UP,
            self.dir == DOWN,
            fx < hx,
            fx > hx,
            fy < hy,
            fy > hy,
        ], dtype=np.float32)
