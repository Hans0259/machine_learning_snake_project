import pygame
import random
import numpy as np
import sys
from collections import namedtuple

Position = namedtuple("Position", "x y")


class Direction:
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3


class Snake:
    def __init__(self, start_blocks, image, block_size):
        self.blocks = start_blocks[:]    
        self.head = self.blocks[0]
        self.dir = Direction.RIGHT
        self.image = image
        self.block = block_size

    def handle_action(self, action: int):
        dirs = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = dirs.index(self.dir)
        #turn right
        if action == 1:     
            self.dir = dirs[(idx + 1) % 4]
        #turn left
        elif action == 2:   
            self.dir = dirs[(idx - 1) % 4]
        #no action, keep straight
        action == 0

        self.move()

    def move(self):
        if self.dir == Direction.RIGHT:
            mv = (1, 0)
        elif self.dir == Direction.LEFT:
            mv = (-1, 0)
        elif self.dir == Direction.UP:
            mv = (0, -1)
        else:
            mv = (0, 1)

        new_head = Position(self.head.x + mv[0], self.head.y + mv[1])
        self.blocks.insert(0, new_head)
        self.head = new_head

    def draw(self, surface):
        for b in self.blocks:
            surface.blit(self.image, (b.x * self.block, b.y * self.block))


class Game:

    def __init__(self, width=320, height=320): #use a smaller size for CNN since it is far more harder to learn than the DQN
        pygame.init()

        self.width = width
        self.height = height
        self.block = 16  # size of block

        self.grid_w = width // self.block
        self.grid_h = height // self.block

        self.nC = 4      # channel
        self.nA = 3      # motion: forward / right / left

        self.surface = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake CNN")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)

        # load image
        self.img_snake = pygame.transform.scale(
            pygame.image.load("snake.png"), (self.block, self.block)
        )
        self.img_point = pygame.transform.scale(
            pygame.image.load("point.png"), (self.block, self.block)
        )
        self.img_wall = pygame.transform.scale(
            pygame.image.load("wall.png"), (self.block, self.block)
        )

        self.reset()

    def absolute_dist(p1: Position, p2: Position):
        return abs(p1.x - p2.x) + abs(p1.y - p2.y)

    def collide_wall(self, head: Position):
        return (
            head.x < 0
            or head.x >= self.grid_w
            or head.y < 0
            or head.y >= self.grid_h
        )

    def collide_self(snake: Snake):
        return snake.head in snake.blocks[1:]

    #initialize the game
    def reset(self):
        self.score = 0
        self.steps = 0

        #start point
        self.snake = Snake(
            [Position(5, 5), Position(4, 5)],
            self.img_snake,
            self.block,
        )

        self.point = Position(1, 1)
        self._place_point()

    def _place_point(self):
        x = random.randint(1, self.grid_w - 2)
        y = random.randint(1, self.grid_h - 2)
        pos = Position(x, y)
        if pos in self.snake.blocks:
            self._place_point()


    def play_step(self, action):
        reward = -0.01  # very little negative reward for not getting the point
        done = False

        # for exit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        #all the same logic as the DQN model
        old_dist = self.absolute_dist(self.snake.head, self.point)

        self.snake.handle_action(action)

        if self.collide_wall(self.snake.head) or self.collide_self(self.snake):
            reward -= 20.0
            done = True
            self._draw()
            self.clock.tick(30)
            return reward, done, self.score

        ate = False
        if self.snake.head == self.point:
            ate = True
            self.score += 1
            reward += 25.0 
            self._place_point()
        else:
            if len(self.snake.blocks) > 1:
                self.snake.blocks.pop()

        # 4. reward shaping
        new_dist = self.absolute_dist(self.snake.head, self.point)
        if not ate:
            if new_dist < old_dist:
                reward += 3.0   # get close
            else:
                reward -= 1.0   # get away

        max_d = self.grid_w + self.grid_h
        reward += (1.0 - new_dist / max_d) * 0.5  # more reward for getting closer

        # avoid generate a safe loop and not getting point
        self.steps += 1
        if self.steps > 200 + 20 * len(self.snake.blocks):
            reward -= 5.0
            done = True

        self._draw()
        self.clock.tick(60)

        return reward, done, self.score

    # CNN: 4 × H × W
    def get_grid_obs(self):
        H, W = self.grid_h, self.grid_w
        grid = np.zeros((self.nC, H, W), dtype=np.float32)

        # channel 0: wall
        grid[0, 0, :] = 1.0
        grid[0, H - 1, :] = 1.0
        grid[0, :, 0] = 1.0
        grid[0, :, W - 1] = 1.0

        #channel 1: point
        fx, fy = self.point.x, self.point.y
        sigma = 2.0
        max_d = float(H + W)

        for y in range(H):
            for x in range(W):
                dx = x - fx
                dy = y - fy
                d2 = dx * dx + dy * dy
                val = np.exp(-d2 / (2.0 * sigma * sigma))
                grid[1, y, x] = val

        if 0 <= fx < W and 0 <= fy < H:
            grid[1, fy, fx] = 1.0

        # channel 2: snake body
        for b in self.snake.blocks:
            if 0 <= b.x < W and 0 <= b.y < H:
                grid[2, b.y, b.x] = 1.0

        # channel 3: check position around the snake head
        hx, hy = self.snake.head
        danger = np.zeros((H, W), dtype=np.float32)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                x, y = hx + dx, hy + dy
                if 0 <= x < W and 0 <= y < H:
                    if (
                        x == 0 or x == W - 1 or y == 0 or y == H - 1
                        or Position(x, y) in self.snake.blocks[1:]
                    ):
                        danger[y, x] = 1.0
        grid[3] = danger

        return grid

    #render for every 50 or 100 game, to see result
    def render_one_episode(self, agent):
        self.reset()
        done = False

        while not done:
            pygame.event.pump()
            state = self.get_grid_obs()
            action = agent.get_action(state, explore=False)
            _, done, _ = self.play_step(action)

            self.clock.tick(10)#easier to see under slow-mo

    # same process as DQN to draw the UI
    def _draw(self):
        self.surface.fill((0, 0, 0))

        for x in range(self.grid_w):
            self.surface.blit(self.img_wall, (x * self.block, 0))
            self.surface.blit(self.img_wall, (x * self.block, (self.grid_h - 1) * self.block))
        for y in range(self.grid_h):
            self.surface.blit(self.img_wall, (0, y * self.block))
            self.surface.blit(self.img_wall, ((self.grid_w - 1) * self.block, y * self.block))

        self.surface.blit(
            self.img_point, (self.point.x * self.block, self.point.y * self.block)
        )

        self.snake.draw(self.surface)

        txt = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.surface.blit(txt, (5, 5))

        pygame.display.update()
