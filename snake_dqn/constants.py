BLOCK  = 20
WIDTH  = 640
HEIGHT = 480
GRID_W = WIDTH  // BLOCK   # 32 cells
GRID_H = HEIGHT // BLOCK   # 24 cells

# Colors
BLACK      = (0,   0,   0)
WHITE      = (255, 255, 255)
GREEN      = (0,   200,   0)
DARK_GREEN = (0,   140,   0)
RED        = (220,  50,  50)

# Directions in clockwise order — index arithmetic gives relative turns
RIGHT = ( 1,  0)
DOWN  = ( 0,  1)
LEFT  = (-1,  0)
UP    = ( 0, -1)
CW    = [RIGHT, DOWN, LEFT, UP]
