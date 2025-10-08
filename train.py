import torch 
from torch import optim, nn
from network.mlp import MLP as DQN
from utils import ReplayBuffer
import random
import matplotlib.pyplot as plt
from agent.snake_agent import snake_Agent 
from game_env.snake_env import SnakeEnv
import numpy as np

def train_dqn(env, episodes=1000):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer(10000)

    gamma = 0.99
    batch_size = 64
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    update_target_every = 10

    rewards_history = []

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(500):
            # Epsilon-greedy policy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_net(torch.FloatTensor(state))
                    action = torch.argmax(q_values).item()

            next_state, reward, done, truncated, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) > batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                # Q-learning update
                q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    targets = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done or truncated:
                break

        # Cập nhật target network
        if ep % update_target_every == 0:
            target_net.load_state_dict(q_net.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)

        if (ep+1) % 100 == 0:
            print(f"Episode {ep+1}: reward={total_reward:.2f}, epsilon={epsilon:.3f}")
    torch.save(q_net.state_dict(), "logs/dqn_cartpole.pth")
    return q_net, rewards_history

def train_cnn_dqn(episodes=2000, grid_size=10, batch_size=64, target_update=10):
    env = SnakeEnv(grid_size=grid_size)
    agent = snake_Agent(grid_size=grid_size, action_size=4)
    
    scores = []
    losses = []
    avg_scores = []
    best_avg_score = 0
    
    print("\n" + "="*70)
    print("CNN-DQN Training Started")
    print("="*70)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_loss = []
        
        while True:
            action = agent.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            # Train
            loss = agent.train(batch_size)
            if loss is not None:
                episode_loss.append(loss)
            
            if done:
                break
        
        scores.append(env.points)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Update target network
        if episode % target_update == 0:
            agent.update_target_network()
        
        agent.update_epsilon()
        
        # Learning rate decay
        if episode % 50 == 0 and episode > 0:
            agent.scheduler.step()
        
        # Calculate metrics
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        avg_scores.append(avg_score)
        
        # Save best model
        if avg_score > best_avg_score and len(scores) >= 100:
            best_avg_score = avg_score
            agent.save('logs/snake_cnn_dqn_best.pth')
        
        # Logging
        if episode % 50 == 0:
            current_lr = agent.optimizer.param_groups[0]['lr']
            print(f"Ep {episode:4d}/{episodes} | "
                  f"Score: {env.points:3d} | "
                  f"Avg: {avg_score:6.2f} | "
                  f"Best: {best_avg_score:6.2f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Loss: {np.mean(episode_loss) if episode_loss else 0:.4f} | "
                  f"Buffer: {len(agent.memory)}")
        
        # Save checkpoint
        if episode % 500 == 0 and episode > 0:
            agent.save(f'logs/checkpoint_cnn_ep{episode}.pth')
    
    return agent, scores, losses, avg_scores


def plot_training_results(scores, avg_scores, losses):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Scores over time
    axes[0, 0].plot(scores, alpha=0.3, label='Episode Score', color='skyblue')
    axes[0, 0].plot(avg_scores, label='100-Episode Average', linewidth=2.5, color='darkblue')
    axes[0, 0].set_xlabel('Episode', fontsize=12)
    axes[0, 0].set_ylabel('Score', fontsize=12)
    axes[0, 0].set_title('Training Progress', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss over time
    if losses:
        axes[0, 1].plot(losses, alpha=0.7, color='coral', linewidth=1.5)
        axes[0, 1].set_xlabel('Episode', fontsize=12)
        axes[0, 1].set_ylabel('Loss', fontsize=12)
        axes[0, 1].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Score distribution
    axes[1, 0].hist(scores, bins=30, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    axes[1, 0].axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scores):.2f}')
    axes[1, 0].set_xlabel('Score', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Score Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Moving averages
    for window, color in [(10, 'lightblue'), (50, 'blue'), (100, 'darkblue')]:
        if len(scores) >= window:
            ma = [np.mean(scores[max(0, i-window):i+1]) for i in range(len(scores))]
            axes[1, 1].plot(ma, label=f'{window}-Episode MA', linewidth=2, color=color)
    axes[1, 1].set_xlabel('Episode', fontsize=12)
    axes[1, 1].set_ylabel('Average Score', fontsize=12)
    axes[1, 1].set_title('Moving Averages Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cnn_dqn_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Training results saved to 'cnn_dqn_training_results.png'")




if __name__ == "__main__":
    # import gymnasium as gym
    # env = gym.make("CartPole-v1")
    # trained_qnet, rewards = train_dqn(env)
    train_cnn_dqn()