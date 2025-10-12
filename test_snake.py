import pygame
import torch
import numpy as np
import sys
import argparse

from network.mlp import MLP
from network.qr_dqn import QRDQN
from game_env.snake_env import SnakeEnv 

class SnakeGameVisualizer:
    def __init__(self, args):
        self.grid_size = args.grid_size
        self.cell_size = args.cell_size
        self.window_size = self.grid_size * self.cell_size * 2
        self.agent_type = args.agent_type

        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption(f'Snake AI - {self.agent_type.upper()} Test')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 30)
        
        # load model 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Select model 
        if self.agent_type == 'dqn':
            self.model = MLP(13, 4).to(self.device)
            self.model_path = f"logs/snake_dqn_best_grid{self.grid_size}.pth"
        elif self.agent_type == 'qr-dqn':
            self.model = QRDQN(13, 4).to(self.device)
            self.model_path = f"logs/snake_qr-dqn_best_grid{self.grid_size}.pth"
        else:
            print(f"Error: Unknown agent type '{self.agent_type}'")
            sys.exit(1)
        
        # Load the trained weights
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['policy_net'])
            self.model.eval()
            print(f"Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            print(f"Error: Model file not found at {self.model_path}")
            print("Please make sure you have trained the model first and the path is correct.")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")
            sys.exit(1)
        
        self.env = SnakeEnv(grid_size=self.grid_size * 2)
        
        self.total_games = 0
        self.total_score = 0
        self.best_score = 0
        self.current_score = 0
        
    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if self.agent_type == 'dqn':
                q_values = self.model(state_tensor)
                return q_values.max(1)[1].item()
            elif self.agent_type == 'qr-dqn':
                quantiles = self.model(state_tensor)
                q_values = quantiles.mean(dim=2)
                return q_values.max(1)[1].item()

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
        state = self.env.reset()
        running = True
        paused = False
        game_over = False
        
        print("\n" + "="*60)
        print("SNAKE AI VISUAL TEST")
        print("Controls: [SPACE] to Pause/Resume, [R] to Reset, [Q] to Quit")
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
                    elif event.key == pygame.K_q:
                        running = False
            
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
            self.clock.tick(fps)
        
        pygame.quit()
        
        # Print final stats
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        if self.total_games > 0:
            print(f"Total Games Played: {self.total_games}")
            print(f"Average Score: {self.total_score / self.total_games:.2f}")
            print(f"Best Score: {self.best_score}")
        else:
            print("No games were completed.")
        print("="*60)


def main():
    """Main function to parse arguments and run the visualizer."""
    parser = argparse.ArgumentParser(description='Test a trained Snake AI model with visualization')
    
    parser.add_argument('--agent-type', type=str, default='dqn', choices=['dqn', 'qr-dqn'],
                        help='The type of agent model to test (dqn or qr-dqn). Default: dqn')
    parser.add_argument('--grid-size', type=int, default=10,
                        help='The grid size the model was trained on. Default: 10')
    parser.add_argument('--cell-size', type=int, default=40,
                        help='The size of each cell in pixels for display. Default: 40')
    parser.add_argument('--fps', type=int, default=15,
                        help='Frames per second for the visualization. Default: 15')
    
    args = parser.parse_args()
    
    visualizer = SnakeGameVisualizer(args)
    visualizer.run(fps=args.fps)


if __name__ == "__main__":
    main()