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
        # start_blocks: [Position, Position, ...]
        self.blocks = start_blocks[:]  # 深拷贝
        self.head = self.blocks[0]
        self.dir = Direction.RIGHT
        self.image = image
        self.block = block_size

    def handle_action(self, action: int):
        """
        action: 0 = forward, 1 = turn right, 2 = turn left
        """
        dirs = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = dirs.index(self.dir)

        if action == 1:      # 右转
            self.dir = dirs[(idx + 1) % 4]
        elif action == 2:    # 左转
            self.dir = dirs[(idx - 1) % 4]
        # action == 0 保持方向

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
            surface.blit(
                self.image,
                (b.x * self.block, b.y * self.block)
            )


class Game:
    def __init__(self, width=160, height=160):
        pygame.init()

        self.width = width
        self.height = height
        self.block = 16  # 一格大小

        self.grid_w = width // self.block
        self.grid_h = height // self.block

        self.nC = 4  # 通道数：墙 / 果实 / snake1 / snake2
        self.nA = 3  # 动作数：forward/right/left

        self.surface = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake CNN")
        self.clock = pygame.time.Clock()

        # 加载图片
        self.img_snake = pygame.transform.scale(
            pygame.image.load("snake.png"), (self.block, self.block)
        )
        self.img_food = pygame.transform.scale(
            pygame.image.load("point.png"), (self.block, self.block)
        )
        self.img_wall = pygame.transform.scale(
            pygame.image.load("wall.png"), (self.block, self.block)
        )

        # 暂时不用墙，留接口：0=空，1=墙
        self.wall = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)

        self.reset()

    # ------------------------------------------------------------------
    # 基本工具函数
    # ------------------------------------------------------------------
    def absolute_dist(self, p1: Position, p2: Position):
        # 曼哈顿距离
        return abs(p1.x - p2.x) + abs(p1.y - p2.y)

    def collide_wall(self, head: Position):
        return (
            head.x < 0
            or head.x >= self.grid_w
            or head.y < 0
            or head.y >= self.grid_h
        )

    def collide_self(self, snake: Snake):
        return snake.head in snake.blocks[1:]

    def collide_other(self, a: Snake, b: Snake):
        return a.head in b.blocks

    # ------------------------------------------------------------------
    # 初始化 & 食物
    # ------------------------------------------------------------------
    def reset(self):
        self.score = 0
        self.steps = 0

        # 两条蛇起始位置（保证在网格中间区域）
        self.snake = Snake(
            [Position(3, 3), Position(2, 3)],
            self.img_snake,
            self.block,
        )
        self.snake2 = Snake(
            [Position(self.grid_w - 4, self.grid_h - 4),
             Position(self.grid_w - 5, self.grid_h - 4)],
            self.img_snake,
            self.block,
        )

        # 食物位置
        self.food = Position(1, 1)
        self._place_food()

    def _place_food(self):
        while True:
            x = random.randint(1, self.grid_w - 2)
            y = random.randint(1, self.grid_h - 2)
            pos = Position(x, y)
            if (
                pos not in self.snake.blocks
                and pos not in self.snake2.blocks
            ):
                self.food = pos
                break

    # ------------------------------------------------------------------
    #  环境一步
    # ------------------------------------------------------------------
    def play_step(self, action1, action2):
        # 总 reward
        reward = 0.0
        done = False

        # 处理退出
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # ------- 1. 记录 snake1 移动前距离 -------
        old_dist1 = self.absolute_dist(self.snake.head, self.food)

        # ------- 2. 两条蛇移动 -------
        self.snake.handle_action(action1)
        if action2 is not None:
            self.snake2.handle_action(action2)

        # ------- 3. 碰撞检测（先看死没死）-------
        died = False
        if (
            self.collide_wall(self.snake.head)
            or self.collide_self(self.snake)
            or self.collide_other(self.snake, self.snake2)
        ):
            died = True

        if (
            self.collide_wall(self.snake2.head)
            or self.collide_self(self.snake2)
            or self.collide_other(self.snake2, self.snake)
        ):
            died = True

        if died:
            reward -= 30.0       # 死亡强负奖励
            return reward, True, self.score

        # ------- 4. 吃果实逻辑 -------
        ate1 = False
        if self.snake.head == self.food:
            ate1 = True
            self.score += 1
            reward += 20.0       # snake1 吃到果实
            self._place_food()
        else:
            if len(self.snake.blocks) > 1:
                self.snake.blocks.pop()

        # snake2 只是陪跑，也可以给一点团队奖励（弱一点）
        if self.snake2.head == self.food:
            self.score += 1
            reward += 10.0       # 团队吃到果实
            self._place_food()
        else:
            if len(self.snake2.blocks) > 1:
                self.snake2.blocks.pop()

        # ------- 5. reward shaping：只针对 snake1 -------
        # 注意：要用吃果实之前的 food 位置，所以放在吃完逻辑之后有点绕，
        # 简单起见，重新用当前位置估计一下也可以。
        new_dist1 = self.absolute_dist(self.snake.head, self.food)

        # 如果这一回合没直接吃到，就给“靠近 or 远离”的奖励
        if not ate1:
            if new_dist1 < old_dist1:
                reward += 1.0    # 靠近果实
            else:
                reward -= 0.5    # 远离果实

        # ------- 6. 步数上限，防止无聊兜圈子 -------
        self.steps += 1
        if self.steps > 200 + 20 * len(self.snake.blocks):
            reward -= 5.0
            return reward, True, self.score

        # ------- 7. 绘图 & 返回 -------
        self.draw()
        self.clock.tick(60)

        return reward, done, self.score


    # ------------------------------------------------------------------
    #  CNN 观察：4×H×W
    # ------------------------------------------------------------------
    def get_grid_obs(self, which=1):
        grid = np.zeros((4, self.grid_h, self.grid_w), dtype=np.float32)

        # 墙
        grid[0, :, :] = self.wall

        # 食物
        if 0 <= self.food.x < self.grid_w and 0 <= self.food.y < self.grid_h:
            grid[1, self.food.y, self.food.x] = 1.0

        # snake1
        for b in self.snake.blocks:
            if 0 <= b.x < self.grid_w and 0 <= b.y < self.grid_h:
                grid[2, b.y, b.x] = 1.0

        # snake2
        for b in self.snake2.blocks:
            if 0 <= b.x < self.grid_w and 0 <= b.y < self.grid_h:
                grid[3, b.y, b.x] = 1.0

        return grid

    # ------------------------------------------------------------------
    #  渲染一局（policy eval，用在 main.py 每20局可视化）
    # ------------------------------------------------------------------
    def render_one_episode(self, agent1, agent2):
        self.reset()
        done = False

        while not done:
            pygame.event.pump()

            s1 = self.get_grid_obs(which=1)
            s2 = self.get_grid_obs(which=2)

            a1 = agent1.get_action(s1, explore_ratio=0.0)
            a2 = agent2.get_action(s2, explore_ratio=0.0)

            _, done, _ = self.play_step(a1, a2)

            self.draw()
            self.clock.tick(10)

    # ------------------------------------------------------------------
    #  绘图
    # ------------------------------------------------------------------
    def draw(self):
        self.surface.fill((0, 0, 0))

        # 墙（现在全 0，如果以后要加迷宫可以在 self.wall 中设置为 1）
        for y in range(self.grid_h):
            for x in range(self.grid_w):
                if self.wall[y, x] == 1:
                    self.surface.blit(
                        self.img_wall, (x * self.block, y * self.block)
                    )

        # 食物
        self.surface.blit(
            self.img_food, (self.food.x * self.block, self.food.y * self.block)
        )

        # 蛇
        self.snake.draw(self.surface)
        self.snake2.draw(self.surface)

        pygame.display.update()
