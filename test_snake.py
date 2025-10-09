import pygame
import torch
import numpy as np
from network.cnn import CNNDQN
from game_env.snake_env import SnakeEnv  # Import env của bạn
import sys
import time

class SnakeGameVisualizer:
    def __init__(self, grid_size=10, cell_size=40, model_path='logs/snake_cnn_dqn_best.pth'):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window_size = grid_size * cell_size
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 100, 255)
        self.DARK_GREEN = (0, 150, 0)
        self.YELLOW = (255, 255, 0)
        self.GRAY = (50, 50, 50)
        
        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size + 300, self.window_size))
        pygame.display.set_caption('Snake AI - DQN Test')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 30)
        self.big_font = pygame.font.Font(None, 50)
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNDQN(grid_size, action_size=4).to(self.device)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['policy_net'])
            self.model.eval()
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
        
        # Environment
        self.env = SnakeEnv(grid_size=grid_size)
        
        # Stats
        self.total_games = 0
        self.total_score = 0
        self.best_score = 0
        self.current_score = 0
        
    def select_action(self, state):
        """Chọn action từ model (không có epsilon)"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return q_values.max(1)[1].item()
    
    def draw_grid(self):
        """Vẽ lưới"""
        for x in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, self.GRAY, (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.screen, self.GRAY, (0, y), (self.window_size, y))
    
    def draw_snake(self):
        """Vẽ rắn"""
        for i, (x, y) in enumerate(self.env.snake):
            rect = pygame.Rect(
                y * self.cell_size + 2,
                x * self.cell_size + 2,
                self.cell_size - 4,
                self.cell_size - 4
            )
            if i == 0:  # Head
                pygame.draw.rect(self.screen, self.DARK_GREEN, rect)
                pygame.draw.rect(self.screen, self.GREEN, rect, 3)
                # Vẽ mắt
                eye_size = 4
                eye_offset = self.cell_size // 4
                left_eye = (y * self.cell_size + eye_offset, x * self.cell_size + eye_offset)
                right_eye = (y * self.cell_size + self.cell_size - eye_offset, x * self.cell_size + eye_offset)
                pygame.draw.circle(self.screen, self.YELLOW, left_eye, eye_size)
                pygame.draw.circle(self.screen, self.YELLOW, right_eye, eye_size)
            else:  # Body
                pygame.draw.rect(self.screen, self.GREEN, rect)
    
    def draw_food(self):
        """Vẽ thức ăn"""
        x, y = self.env.food
        rect = pygame.Rect(
            y * self.cell_size + 5,
            x * self.cell_size + 5,
            self.cell_size - 10,
            self.cell_size - 10
        )
        pygame.draw.circle(
            self.screen, 
            self.RED, 
            (y * self.cell_size + self.cell_size // 2, 
             x * self.cell_size + self.cell_size // 2),
            self.cell_size // 3
        )
    
    def draw_stats(self):
        """Vẽ thống kê bên phải"""
        stats_x = self.window_size + 10
        
        # Title
        title = self.big_font.render('STATS', True, self.YELLOW)
        self.screen.blit(title, (stats_x, 20))
        
        # Current game
        y_offset = 80
        texts = [
            f"Score: {self.current_score}",
            f"Length: {len(self.env.snake)}",
            f"Steps: {self.env.steps}",
            "",
            f"Games: {self.total_games}",
            f"Avg Score: {self.total_score / max(1, self.total_games):.1f}",
            f"Best: {self.best_score}",
            "",
            "Controls:",
            "SPACE: Pause",
            "R: Restart",
            "Q: Quit",
            "+/-: Speed"
        ]
        
        for i, text in enumerate(texts):
            if text == "":
                y_offset += 10
                continue
            color = self.YELLOW if i < 3 else self.WHITE
            rendered = self.font.render(text, True, color)
            self.screen.blit(rendered, (stats_x, y_offset + i * 30))
    
    def draw_game_over(self):
        """Vẽ màn hình game over"""
        overlay = pygame.Surface((self.window_size, self.window_size))
        overlay.set_alpha(180)
        overlay.fill(self.BLACK)
        self.screen.blit(overlay, (0, 0))
        
        game_over_text = self.big_font.render('GAME OVER', True, self.RED)
        score_text = self.font.render(f'Score: {self.current_score}', True, self.WHITE)
        restart_text = self.font.render('Press R to restart', True, self.YELLOW)
        
        self.screen.blit(game_over_text, 
                        (self.window_size // 2 - game_over_text.get_width() // 2, 
                         self.window_size // 2 - 50))
        self.screen.blit(score_text,
                        (self.window_size // 2 - score_text.get_width() // 2,
                         self.window_size // 2 + 10))
        self.screen.blit(restart_text,
                        (self.window_size // 2 - restart_text.get_width() // 2,
                         self.window_size // 2 + 50))
    
    def run(self, fps=10):
        """Chạy game loop"""
        state = self.env.reset()
        running = True
        paused = False
        game_over = False
        current_fps = fps
        
        print("\n" + "="*60)
        print("🎮 SNAKE AI VISUAL TEST")
        print("="*60)
        print("Controls:")
        print("  SPACE: Pause/Resume")
        print("  R: Restart game")
        print("  Q/ESC: Quit")
        print("  +: Increase speed")
        print("  -: Decrease speed")
        print("="*60 + "\n")
        
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
                    elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                        current_fps = min(60, current_fps + 2)
                        print(f"Speed: {current_fps} FPS")
                    elif event.key == pygame.K_MINUS:
                        current_fps = max(2, current_fps - 2)
                        print(f"Speed: {current_fps} FPS")
            
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
            self.draw_grid()
            self.draw_food()
            self.draw_snake()
            self.draw_stats()
            
            if game_over:
                self.draw_game_over()
            
            if paused:
                pause_text = self.big_font.render('PAUSED', True, self.YELLOW)
                self.screen.blit(pause_text,
                               (self.window_size // 2 - pause_text.get_width() // 2,
                                self.window_size // 2 - pause_text.get_height() // 2))
            
            pygame.display.flip()
            self.clock.tick(current_fps)
        
        pygame.quit()
        
        # Print final stats
        print("\n" + "="*60)
        print("📊 FINAL STATISTICS")
        print("="*60)
        print(f"Total Games: {self.total_games}")
        print(f"Average Score: {self.total_score / max(1, self.total_games):.2f}")
        print(f"Best Score: {self.best_score}")
        print("="*60)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Snake AI with visualization')
    parser.add_argument('--model', type=str, default='logs/snake_cnn_dqn_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--grid-size', type=int, default=10,
                       help='Grid size (default: 10)')
    parser.add_argument('--cell-size', type=int, default=40,
                       help='Cell size in pixels (default: 40)')
    parser.add_argument('--fps', type=int, default=10,
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