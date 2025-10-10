import torch
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import numpy as np
import argparse
import yaml 
import os
from agent.snake_agent import Agent 
from game_env.snake_env import SnakeEnv

def train(args):
    """
    Unified training function for both DQN and QR-DQN.
    """
    env = SnakeEnv(grid_size=args.grid_size)
    
    # Initialize the agent 
    agent = Agent(
        agent_type=args.agent_type,
        grid_size=args.grid_size,
        action_size=4,
        state_dim=13,
        num_quantiles=args.num_quantiles,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay
    )
    
    os.makedirs('logs', exist_ok=True)
    
    scores = []
    losses = []
    avg_scores = []
    best_avg_score = -float('inf') 
    
    # Fill replay buffer
    print("Pre-filling Replay Buffer...")
    state = env.reset()
    for _ in range(args.prefill_steps):
        action = random.randrange(4)
        next_state, reward, done, _ = env.step(action)
        agent.memory.push(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()
    
    # Training loop 
    print("\n" + "="*70)
    print(f"Starting training for Agent: {args.agent_type.upper()}")
    print(f"Parameters: {vars(args)}")
    print("="*70)
    
    for episode in range(args.episodes):
        state = env.reset()
        episode_loss = []
        
        while True:
            action = agent.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            
            for _ in range(args.train_steps_per_env_step):
                loss = agent.train(args.batch_size)
                if loss is not None:
                    episode_loss.append(loss)
            
            if done:
                break
        
        scores.append(env.points)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Update the target network
        if episode % args.target_update_freq == 0:
            agent.update_target_network()
        
        agent.update_epsilon()
        
        # Decay the learning rate
        if episode > 0 and episode % args.lr_decay_step == 0:
            agent.scheduler.step()
        
        # Calculate metrics
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        # Save the best model
        if avg_score > best_avg_score and len(scores) >= 100:
            best_avg_score = avg_score
            save_path = f'logs/snake_{args.agent_type}_best_grid{args.grid_size}.pth'
            agent.save(save_path)

        # Log progress periodically
        if episode % args.log_freq == 0:
            current_lr = agent.optimizer.param_groups[0]['lr']
            print(f"Ep {episode:4d}/{args.episodes} | "
                  f"Score: {env.points:3d} | "
                  f"Avg100: {avg_score:5.2f} | "
                  f"BestAvg: {best_avg_score:5.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Loss: {np.mean(episode_loss) if episode_loss else 0:.4f} | "
                  f"Buffer: {len(agent.memory)}")
        
        # Save a checkpoint
        if episode > 0 and episode % args.save_checkpoint_freq == 0:
            agent.save(f'logs/checkpoint_{args.agent_type}_ep{episode}_grid{args.grid_size}.pth')
            
    print("="*70)
    print("Training finished!")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Train a Snake RL Agent')

    # --- Optional: Path to a config file to override defaults ---
    parser.add_argument('--config', type=str, default=None, help='Path to the YAML config file (optional).')
    
    # --- General Configuration ---
    parser.add_argument('--agent-type', type=str, default='dqn', choices=['dqn', 'qr-dqn'], help='The type of agent to train.')
    parser.add_argument('--grid-size', type=int, default=10, help='Size of the game grid.')
    parser.add_argument('--episodes', type=int, default=3000, help='Number of episodes to train for.')

    # --- Agent & Hyperparameters Configuration ---
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor.')
    parser.add_argument('--epsilon-decay', type=float, default=0.995, help='Epsilon decay rate.')
    parser.add_argument('--num-quantiles', type=int, default=51, help='Number of quantiles for QR-DQN.')
    
    # --- Training Loop Configuration ---
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--target-update-freq', type=int, default=50, help='Target network update frequency (in episodes).')
    parser.add_argument('--lr-decay-step', type=int, default=500, help='Number of episodes before decaying the learning rate.')
    parser.add_argument('--prefill-steps', type=int, default=1000, help='Number of random steps to pre-fill the replay buffer.')
    parser.add_argument('--train-steps-per-env-step', type=int, default=2, help='Number of training calls per environment step.')
    
    # --- Logging & Saving Configuration ---
    parser.add_argument('--log-freq', type=int, default=50, help='Logging frequency (in episodes).')
    parser.add_argument('--save-checkpoint-freq', type=int, default=500, help='Checkpoint saving frequency (in episodes).')

    args = parser.parse_args()

    # If a config file is provided, load it and update defaults
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Override parser's default values with values from the config file
        # Use hyphens in yaml and underscores in code
        config_dict = {k.replace('-', '_'): v for k, v in config_dict.items()}
        parser.set_defaults(**config_dict)
        
        # Re-parse arguments to apply changes from the config file
        # and to allow command-line args to override the config file
        args = parser.parse_args()

    # Start training
    train(args)

if __name__ == "__main__":
    main()