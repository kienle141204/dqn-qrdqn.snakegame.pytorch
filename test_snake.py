import pygame
import torch
import numpy as np
from network.cnn import CNNDQN
from network.mlp import MLP
from network.qr_dqn import QRDQN
from game_env.snake_env import SnakeEnv  # Import env của bạn
import sys
import time

class SnakeGameVisualizer:
    def __init__(self, grid_size=10, cell_size=40, model_path='logs/snake_cnn_dqn_best.pth'):
        self.model_path = f"logs/snake_cnn_dqn_best_{grid_size}.pth"
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window_size = grid_size * cell_size
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        
        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption('Snake AI - DQN Test')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 30)
        self.big_font = pygame.font.Font(None, 50)
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = MLP(13, 4).to(self.device)
        self.model = QRDQN(13, 4).to(self.device)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['policy_net'])
            self.model.eval()
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        # Environment
        self.env = SnakeEnv(grid_size=grid_size)
        
        # Stats
        self.total_games = 0
        self.total_score = 0
        self.best_score = 0
        self.current_score = 0
        
    # def select_action(self, state):
    #     """Chọn action từ model (không có epsilon)"""
    #     with torch.no_grad():
    #         state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    #         q_values = self.model(state_tensor)
    #         return q_values.max(1)[1].item()
    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            
            if hasattr(self.model, 'get_q_values'):
                q_values = self.model.get_q_values(state_tensor)
            else:
                q_values = self.model(state_tensor)
            
            if q_values.dim() > 1:
                return q_values[0].argmax().item()
            else:
                return q_values.argmax().item()
    
    def draw_snake(self):
        for x, y in self.env.snake:
            rect = pygame.Rect(
                y * self.cell_size,
                x * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            pygame.draw.rect(self.screen, self.GREEN, rect)
    
    def draw_food(self):
        """Vẽ thức ăn"""
        x, y = self.env.food
        pygame.draw.circle(
            self.screen, 
            self.RED, 
            (y * self.cell_size + self.cell_size // 2, 
             x * self.cell_size + self.cell_size // 2),
            self.cell_size // 3
        )
    
    def draw_score(self):
            score_text = self.font.render(f"Score: {self.current_score}", True, self.WHITE)  
            self.screen.blit(score_text, (10, 10)) 
    

    def run(self, fps=10):
        """Chạy game loop"""
        state = self.env.reset()
        running = True
        paused = False
        game_over = False
        current_fps = fps
        
        print("\n" + "="*60)
        print("SNAKE AI VISUAL TEST")
        print("="*60)
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused 
                    elif event.key == pygame.K_r:
                        state = self.env.reset()
                        game_over = False
                        self.current_score = 0
            
            # Update game state
            if not paused and not game_over:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                self.current_score = self.env.points
                
                if done:
                    game_over = True
                    self.total_games += 1
                    self.total_score += self.current_score
                    self.best_score = max(self.best_score, self.current_score)
                    print(f"Game {self.total_games}: Score = {self.current_score}, "
                          f"Length = {len(self.env.snake)}, Steps = {self.env.steps}")
            
            # Draw everything
            self.screen.fill(self.BLACK)
            self.draw_food()
            self.draw_snake()
            self.draw_score()
            
            pygame.display.flip()
            self.clock.tick(current_fps)
        
        pygame.quit()
        
        # Print final stats
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        print(f"Total Games: {self.total_games}")
        print(f"Average Score: {self.total_score / max(1, self.total_games):.2f}")
        print(f"Best Score: {self.best_score}")
        print("="*60)


def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description='Test Snake AI with visualization')
    parser.add_argument('--model', type=str, default='logs/snake_mlp_qrdqn_best_10.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--grid-size', type=int, default=10,
                       help='Grid size (default: 10)')
    parser.add_argument('--cell-size', type=int, default=40,
                       help='Cell size in pixels (default: 40)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Initial FPS (default: 10)')
    
    args = parser.parse_args()
    
    visualizer = SnakeGameVisualizer(
        grid_size=args.grid_size,
        cell_size=args.cell_size,
        model_path=args.model
    )
    visualizer.run(fps=args.fps)


if __name__ == "__main__":
    main()