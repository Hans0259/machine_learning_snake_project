import pygame
import random
from collections import namedtuple
from pygame.locals import K_RIGHT,K_LEFT,K_UP,K_DOWN,QUIT
import numpy as np
import sys

#represent the position in tuple
Position = namedtuple('Point', 'x, y')

#current direction of the snake
class Direction:
    right = 0
    left = 1
    up = 2
    down = 3

class Snake:
#load the image of the snake and crop into square image and shrink to proper size
    def load_tile(path, block_size):
        img = pygame.image.load(path).convert_alpha()
        w, h = img.get_size()

        # center-crop to square
        s = min(w, h)
        img = img.subsurface(pygame.Rect((w - s)//2, (h - s)//2, s, s)).copy()

        # trim transparent margins
        mask = pygame.mask.from_surface(img)
        rects = mask.get_bounding_rects()
        if rects:
            img = img.subsurface(rects[0]).copy()

        return pygame.transform.smoothscale(img, (block_size, block_size))
    
    def __init__(self, block_size, start_blocks = None, image_path='snake.png'):
        self.block_size = block_size

        # default restart position for the first snake is (10,15) and (9,15)
        if start_blocks is None:
            self.blocks = [Position(10,15), Position(9,15)]
        else:
        #use this for the second snake
            self.blocks = start_blocks
        
        self.head = self.blocks[0]

        #initialize the direction (right)
        self.current_direction = Direction.right

        self.image = Snake.load_tile('snake.png', block_size) 
    
    def move(self):
        
        # in pygame, the origin is top left, so up is -1, down is +1
        if (self.current_direction == Direction.right):
            movesize = (1, 0)
        elif (self.current_direction == Direction.left):
            movesize = (-1, 0)
        elif (self.current_direction == Direction.up):
            movesize = (0, -1)
        else:
            movesize = (0, 1)
        
        #relocate the head for each move
        new_head = Position(self.head.x + movesize[0], self.head.y + movesize[1])  
        self.blocks.insert(0,new_head)
        self.head = self.blocks[0]

    def handle_action(self, action):

        # the direction is handle by a clockwise loop,+1 is turning clockwise for 90deg, and -1 is turning counterclockwise for 90deg
        clock_wise = [Direction.right, Direction.down, Direction.left, Direction.up] #0,1,2,3 for right,down,left,up
        idx = clock_wise.index(self.current_direction)
        if action == 0:
            new_dir = clock_wise[idx] # no change
        elif action == 1:
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn
        elif action == 2:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn
        self.current_direction = new_dir
        self.move()

#draw
    def draw(self, surface):
        #locate all the part of snake and use the image to show in the UI
        for block in self.blocks:
            dest = pygame.Rect(
                block.x * self.block_size,
                block.y * self.block_size,
                self.block_size,
                self.block_size
            )
            surface.blit(self.image, dest)

class point:
    #set the point position, and load the png file to the UI
    def __init__(self,block_size):
        self.block_size = block_size
        self.image = pygame.image.load('point.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (block_size, block_size))
        self.position = Position(1, 1)     

    def draw(self,surface):
        rect = self.image.get_rect()
        rect.left = self.position.x * self.block_size
        rect.top = self.position.y * self.block_size
        surface.blit(self.image, rect)


class Wall:
    #upload the wall.png anf rescale
    def __init__(self,block_size):
        self.block_size = block_size
        self.map = self.load_map('map.txt')
        self.image = pygame.image.load('wall.png')
        self.image = pygame.transform.scale(self.image, (block_size, block_size))

#read the map.txt and know the map size
    def load_map(self,fileName):
        with open(fileName,'r') as map_file:
            content = map_file.readlines()
            content =  [list(line.strip()) for line in content]
        return content  

    def draw(self,surface):
        for row, line in enumerate(self.map):
            for col, char in enumerate(line):
                # in the map.txt file, 1 represent the wall boundary
                if char == '1':
                    position = (col*self.block_size,row*self.block_size)
                    surface.blit(self.image, position)     


class Game:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    def __init__(self,Width=640, Height=480): #size of the game
        pygame.init()
        self.block_size = 16
        self.Win_width , self.Win_height = (Width, Height)
        #proper area for the game
        self.Space_width = self.Win_width//self.block_size-2
        self.Space_height = self.Win_height//self.block_size-2
        self.surface = pygame.display.set_mode((self.Win_width, self.Win_height))
        self.Clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 32)

        self.snake = Snake(self.block_size)
        # new code for second snake
        # setup a different start position for the second snake, the first snake start position is (10, 15) and (9,15)
        snake2_start = [Position(30, 15), Position(29, 15)]
        self.snake2 = Snake(self.block_size, start_blocks = snake2_start)
        
        #set point and wall
        self.point = point(self.block_size)
        self.wall = Wall(self.block_size)
        
        self.nS = 24 #number of neuron, 20+4 for two snakes
        self.nA = 3 # 3 output neuron for 3 actions: left, right, forward
        self.reset()

    def reset(self):
        #initial game state after snake hit 
        self.score = 0
        
        #same starting point for both snakes
        self.snake =Snake(self.block_size)
        snake2_start = [Position(30, 15), Position(29, 15)]
        self.snake2 = Snake(self.block_size, start_blocks = snake2_start)

        self.position_point()
        self.total_step = 0
        self.reward = 0

    #randomly place the point(reward)
    def position_point(self):
        bx = random.randint(1, self.Space_width)
        by = random.randint(1, self.Space_height)
        self.point.position = Position(bx, by)
        if self.point.position in self.snake.blocks:
            self.position_point()

    # handle collision
    def point_collision(self, snake, is_player=False):
        head = snake.blocks[0]
        # if get the reward, then regenerate a new one, and add score
        if (head.x == self.point.position.x and head.y == self.point.position.y):
            self.position_point() 
            if is_player:
                self.score +=1
                self.reward = 30
            return True
        else:
            #if not get the reward, remain the same body length 
            if len(snake.blocks) > 1:
                snake.blocks.pop()
            return False

#check collision for snake 1
    def head_hit_body(self, position = None): 
        if position is None:
            position = self.snake.blocks[0]
        if position in self.snake.blocks[1:]:
            return True
        return False
#check collision for snake 2
    def snake2_head_hit_body(self, position = None): 
        if position is None:
            position = self.snake2.blocks[0]
        if position in self.snake2.blocks[1:]:
            return True
        return False

    def head_hit_wall(self, position=None):
        if position is None:
            position = self.snake.blocks[0]

        # Bounds check
        if position.y >= 29 or position.x >= 39 or position.y < 0 or position.x < 0:
            return True

        # Wall map check
        if self.wall.map[position.y][position.x] == '1':
            return True

        return False

    def snake2_head_hit_wall(self, position=None):
        if position is None:
            position = self.snake2.blocks[0]

        # Bounds check
        if position.y >= 29 or position.x >= 39 or position.y < 0 or position.x < 0:
            return True

        # Wall map check
        if self.wall.map[position.y][position.x] == '1':
            return True

        return False

    #show score
    def draw_data(self):
        text = "score: {0}".format(self.score)
        text_img = self.font.render(text, 1, Game.WHITE)
        text_rect = text_img.get_rect(centerx=self.surface.get_width()/2, top=32)
        self.surface.blit(text_img, text_rect)
    
    #draw the game UI
    def draw(self):
        self.surface.fill(Game.WHITE)
        self.wall.draw(self.surface)
        self.point.draw(self.surface)
        self.snake.draw(self.surface)
        self.snake2.draw(self.surface)# draw second snake
        self.draw_data()
        pygame.display.update()
    
    #check collision with each other
    def head_hit_snake2(self, position=None):
        if position is None:
            position = self.snake.blocks[0]
        if position in self.snake2.blocks:
            return True
        return False
    def snake2_head_hit_snake1(self, position=None):
        if position is None:
            position = self.snake2.blocks[0]
        if position in self.snake.blocks:
            return True
        return False

    # main loop 
    def play_step(self, action1, action2):
        game_over = False
        self.reward = -0.01

        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Update step
        self.total_step += 1

        # Move the snake based on action
        self.snake.handle_action(action1)
        self.snake2.handle_action(action2)
        
        # Check if the snake ate a point
        self.point_collision(self.snake, is_player=True)
        self.point_collision(self.snake2, is_player=True)
        # Check for collisions
        if (self.head_hit_wall() or
            self.head_hit_body() or
            self.head_hit_snake2() or# new collision with second snake
            self.snake2_head_hit_snake1() or # second snake collision with first snake
            self.snake2_head_hit_wall() or # second snake collision with wall
            self.snake2_head_hit_body() or # second snake collision with itself
            self.total_step > 100 * len(self.snake.blocks)): #false loop
            game_over = True
            self.reward = -1
            return self.reward, game_over, self.score

        # Draw everything
        self.draw()

        # Control game speed
        self.Clock.tick(60)

        # Return current reward, game status, and score
        return self.reward, game_over, self.score

    def get_state(self, state=1):        
        if state == 1:
            snake1 = self.snake
            snake2 = self.snake2
            body_hit_s1 = self.head_hit_body
            wall_hit_s1 = self.head_hit_wall
            current_dir = self.snake.current_direction
        else:
            snake1 = self.snake2
            snake2 = self.snake
            body_hit_s1 = self.snake2_head_hit_body
            wall_hit_s1 = self.snake2_head_hit_wall
            current_dir = self.snake2.current_direction            

        if not snake1.blocks:              # safety
            self.reset()
            return np.zeros(24, dtype=int)

        head = snake1.blocks[0]

        # check next postion around the snake head
        point_l = Position(head.x - 1, head.y)
        point_r = Position(head.x + 1, head.y)
        point_u = Position(head.x, head.y - 1)
        point_d = Position(head.x, head.y + 1)

        danger_lb_r = body_hit_s1(point_r)
        danger_lb_l = body_hit_s1(point_l)
        danger_lb_u = body_hit_s1(point_u)
        danger_lb_d = body_hit_s1(point_d)

        danger_lw_r = wall_hit_s1(point_r)
        danger_lw_l = wall_hit_s1(point_l)
        danger_lw_u = wall_hit_s1(point_u)
        danger_lw_d = wall_hit_s1(point_d)

        # check the area between head and edge, if block by body, then return value 1
        points_l = [Position(i, head.y) for i in range(1, head.x)]
        points_r = [Position(i, head.y) for i in range(head.x + 1, self.Space_width)]
        points_u = [Position(head.x, i) for i in range(1, head.y)]
        points_d = [Position(head.x, i) for i in range(head.y + 1, self.Space_height)]

        danger_b_l = np.any(np.array([body_hit_s1(point) for point in points_l]))
        danger_b_r = np.any(np.array([body_hit_s1(point) for point in points_r]))
        danger_b_u = np.any(np.array([body_hit_s1(point) for point in points_u]))
        danger_b_d = np.any(np.array([body_hit_s1(point) for point in points_d]))

        dir_l = current_dir == Direction.left
        dir_r = current_dir == Direction.right
        dir_u = current_dir == Direction.up
        dir_d = current_dir == Direction.down

        point_l_flag = self.point.position.x < head.x
        point_r_flag = self.point.position.x > head.x
        point_u_flag = self.point.position.y < head.y
        point_d_flag = self.point.position.y > head.y

        # check danger from second snake
        danger_lb_r2 = point_r in snake2.blocks
        danger_lb_l2 = point_l in snake2.blocks
        danger_lb_u2 = point_u in snake2.blocks
        danger_lb_d2 = point_d in snake2.blocks


        state = [
            danger_lb_r, danger_lb_l, danger_lb_u, danger_lb_d,
            danger_lw_r, danger_lw_l, danger_lw_u, danger_lw_d,
            danger_b_l, danger_b_r, danger_b_u, danger_b_d,
            dir_l, dir_r, dir_u, dir_d,
            point_l_flag, point_r_flag, point_u_flag, point_d_flag,
            danger_lb_r2, danger_lb_l2, danger_lb_u2, danger_lb_d2
        ]

        return np.array(state, dtype=int)
