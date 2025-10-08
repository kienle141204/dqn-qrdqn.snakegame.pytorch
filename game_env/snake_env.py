import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SnakeEnv(gym.Env):
    def __init__(self, grid_size=10):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        # State space: continuous vector
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4, self.grid_size, self.grid_size), dtype=np.float32
        )
        self.start_point = (self.grid_size//2, self.grid_size//2)
        self.max_steps = grid_size * grid_size * grid_size  # Giới hạn số bước
        self.reset()
    
    def reset(self):
        self.snake = [self.start_point]
        self.food = self._generate_food()
        self.current_direction = 1  # 0: up, 1: down, 2: left, 3: right
        self.done = False
        self.steps = 0
        self.points = 0
        return self._get_obs()

    def _generate_food(self):
        while True:
            food = (np.random.randint(0, self.grid_size), 
                   np.random.randint(0, self.grid_size))
            if food not in self.snake:
                return food

    def _get_obs(self):
        """Trả về trạng thái dạng lưới one-hot: (4, grid_size, grid_size)
        Kênh 0: đầu rắn
        Kênh 1: thân rắn
        Kênh 2: thức ăn
        Kênh 3: tường
        """
        grid = np.zeros((4, self.grid_size, self.grid_size), dtype=np.float32)

        # Kênh 0: đầu rắn
        head_x, head_y = self.snake[0]
        grid[0, head_x, head_y] = 1.0

        # Kênh 1: thân rắn (trừ đầu)
        for (x, y) in self.snake[1:]:
            grid[1, x, y] = 1.0

        # Kênh 2: thức ăn
        food_x, food_y = self.food
        grid[2, food_x, food_y] = 1.0

        # Kênh 3: tường 
        grid[3, 0, :] = 1.0
        grid[3, -1, :] = 1.0
        grid[3, :, 0] = 1.0
        grid[3, :, -1] = 1.0

        return grid

    
    def step(self, action):
        self.steps += 1
        head_x, head_y = self.snake[0]
        
        # Prevent 180-degree turns
        if (action == 0 and self.current_direction == 1) or \
           (action == 1 and self.current_direction == 0) or \
           (action == 2 and self.current_direction == 3) or \
           (action == 3 and self.current_direction == 2):
            action = self.current_direction  # Keep current direction
        
        self.current_direction = action
        
        # Move
        if action == 0:  # up
            head_x -= 1
        elif action == 1:  # down
            head_x += 1
        elif action == 2:  # left
            head_y -= 1
        else:  # right
            head_y += 1
        
        # Check collision
        if (head_x < 0 or head_x >= self.grid_size or 
            head_y < 0 or head_y >= self.grid_size or 
            (head_x, head_y) in self.snake):
            reward = -10
            self.done = True
        else:
            self.snake.insert(0, (head_x, head_y))
            
            # Check if ate food
            if (head_x, head_y) == self.food:
                reward = 10
                self.points += 1
                self.food = self._generate_food()
            else:
                reward = -0.01  # Small negative reward to encourage efficiency
                self.snake.pop()
        
        # Check if exceeded max steps
        if self.steps >= self.max_steps:
            self.done = True
            
        return self._get_obs(), reward, self.done, {}