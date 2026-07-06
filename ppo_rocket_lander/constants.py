"""
World, physics, and rendering constants for the PPO Rocket Lander.

The world uses MATH coordinates: origin (0, 0) sits at the center of the
landing pad, x grows to the right, y grows UP. Rendering flips y so it looks
right on a pygame window (where pixel-y grows down).
"""

# ── Rendering ─────────────────────────────────────────────────────────────────
WIDTH  = 600
HEIGHT = 400
FPS    = 50

# World extents (in world units) that map onto the screen.
WORLD_X = 1.5    # screen shows x ∈ [-1.5, 1.5]
WORLD_Y = 2.0    # screen shows y ∈ [ 0.0, 2.0]

# Colors
BLACK   = (10,  10,  18)
WHITE   = (235, 235, 235)
GREY    = (90,  90,  100)
SKY     = (18,  18,  30)
FLAME   = (255, 170, 40)
ROCKET  = (210, 210, 220)
PAD_OK  = (60,  200, 120)
PAD_BAD = (200, 70,  70)

# ── Physics ───────────────────────────────────────────────────────────────────
DT           = 1.0 / FPS   # seconds per step
GRAVITY      = 0.30        # downward acceleration (units / s²)
MAIN_POWER   = 0.60        # max upward accel from the main engine (> GRAVITY so it CAN hover)
TORQUE_POWER = 3.0         # max angular accel from side thrusters (rad / s²)

# ── Start conditions (randomized each reset) ───────────────────────────────────
START_Y      = 1.30        # how high the rocket spawns
START_X_JIT  = 0.40        # horizontal spawn range: x ∈ [-0.4, 0.4]
START_V_JIT  = 0.10        # initial vx, vy ∈ [-0.1, 0.1]
START_A_JIT  = 0.10        # initial tilt θ ∈ [-0.1, 0.1] rad

# ── Landing pad + success criteria ─────────────────────────────────────────────
PAD_HALF_W   = 0.20        # pad spans x ∈ [-0.2, 0.2]
LEG_HEIGHT   = 0.06        # below this height the legs are "touching"

LAND_VX      = 0.25        # |vx| must be under this to count as a soft landing
LAND_VY      = 0.40        # |vy| likewise
LAND_ANGLE   = 0.25        # |θ|  must be near upright (rad)
LAND_OMEGA   = 0.50        # |ω|  must be nearly still

# ── Episode bounds ─────────────────────────────────────────────────────────────
MAX_STEPS    = 500
OUT_X        = 1.50        # |x| beyond this = flew off the screen → crash
OUT_Y        = 2.00        # y above this    = flew off the top    → crash
