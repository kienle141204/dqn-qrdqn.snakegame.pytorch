import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SnakeEnv(gym.Env):
    def __init__(self, grid_size=10):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(13,), dtype=np.float32
        )
         
        self.max_steps = grid_size * grid_size * 2  # Giới hạn số bước
        self.reset()
    
    def reset(self):
        start_x = np.random.randint(2, self.grid_size - 2)
        start_y = np.random.randint(2, self.grid_size - 2)
        self.snake = [(start_x, start_y)]

        self.food = self._generate_food()
        self.current_direction = 1  # 0: up, 1: down, 2: left, 3: right
        self.done = False
        self.steps = 0
        self.points = 0
        self.steps_since_food = 0
        return self._get_obs()

    def _generate_food(self):
        while True:
            food = (np.random.randint(0, self.grid_size), 
                   np.random.randint(0, self.grid_size))
            if food not in self.snake:
                return food

    def _get_obs(self):
        """Trả về vector observation tối ưu 13 chiều:
        - 4 giá trị: nguy hiểm ở 4 hướng (up, down, left, right)
        - 4 giá trị: hướng hiện tại (one-hot)
        - 4 giá trị: hướng tới thức ăn (up, down, left, right)
        - 1 giá trị: khoảng cách chuẩn hóa tới thức ăn
        """
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Kiểm tra nguy hiểm ở 4 hướng (va chạm tường hoặc thân)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        danger = []
        for dx, dy in directions:
            next_x, next_y = head_x + dx, head_y + dy
            is_danger = (next_x < 0 or next_x >= self.grid_size or 
                        next_y < 0 or next_y >= self.grid_size or 
                        (next_x, next_y) in self.snake)
            danger.append(1.0 if is_danger else 0.0)
        
        # Hướng hiện tại (one-hot encoding)
        direction_vec = [0.0] * 4
        direction_vec[self.current_direction] = 1.0
        
        # Hướng tới thức ăn (có thể có nhiều hướng = 1 nếu thức ăn ở góc chéo)
        food_direction = [
            1.0 if food_x < head_x else 0.0,  # food ở phía trên
            1.0 if food_x > head_x else 0.0,  # food ở phía dưới
            1.0 if food_y < head_y else 0.0,  # food ở bên trái
            1.0 if food_y > head_y else 0.0,  # food ở bên phải
        ]
        
        distance = (abs(head_x - food_x) + abs(head_y - food_y)) / (2 * self.grid_size)
        
        # Ghép tất cả thành vector 13 chiều
        obs = np.array(danger + direction_vec + food_direction + [distance], dtype=np.float32)
        return obs
    
    def step(self, action):
        self.steps += 1
        self.steps_since_food += 1
        head_x, head_y = self.snake[0]
        
        if (action == 0 and self.current_direction == 1) or \
           (action == 1 and self.current_direction == 0) or \
           (action == 2 and self.current_direction == 3) or \
           (action == 3 and self.current_direction == 2):
            action = self.current_direction  
        
        self.current_direction = action
        old_distance = abs(head_x - self.food[0]) + abs(head_y - self.food[1])
        
        if action == 0:  # up
            head_x -= 1
        elif action == 1:  # down
            head_x += 1
        elif action == 2:  # left
            head_y -= 1
        else:  # right
            head_y += 1

        new_distance = abs(head_x - self.food[0]) + abs(head_y - self.food[1])
        
        # Kiểm tra va chạm
        if (head_x < 0 or head_x >= self.grid_size or 
            head_y < 0 or head_y >= self.grid_size or 
            (head_x, head_y) in self.snake):
            reward = -10
            self.done = True
        else:
            self.snake.insert(0, (head_x, head_y))
            
            # Ăn được thức ăn
            if (head_x, head_y) == self.food:
                reward = 10
                self.points += 1
                self.food = self._generate_food()
                self.steps_since_food = 0
            else:
                if new_distance < old_distance:
                    reward = 0.1  
                else:
                    reward = -0.15  
                
                if self.steps_since_food > self.grid_size * 2:
                    reward -= 0.05
                
                self.snake.pop()
        
        if self.steps >= self.max_steps:
            reward -= 0.5  
            self.done = True
            
        return self._get_obs(), reward, self.done, {}