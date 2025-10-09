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

def train_cnn_dqn(episodes=3000, grid_size=10, batch_size=64, target_update=50):
    env = SnakeEnv(grid_size=grid_size)
    agent = snake_Agent(grid_size=grid_size, action_size=4, 
                        lr=0.0005, gamma=0.99, epsilon_decay=0.997)  
    
    scores = []
    losses = []
    avg_scores = []
    best_avg_score = 0
    
    print("\n" + "="*70)
    print("Warm-up Phase: Collecting initial experiences...")
    print("="*70)
    state = env.reset()
    for _ in range(1000):
        action = random.randrange(4)
        next_state, reward, done, _ = env.step(action)
        agent.memory.push(state, action, reward, next_state, done)
        state = next_state if not done else env.reset()
    
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
            
            # Train multiple times per step for better sample efficiency
            for _ in range(2):
                loss = agent.train(batch_size)
                if loss is not None:
                    episode_loss.append(loss)
            
            if done:
                break
        
        scores.append(env.points)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Update target network more frequently
        if episode % target_update == 0:
            agent.update_target_network()
        
        agent.update_epsilon()
        
        # Learning rate decay
        if episode % 200 == 0 and episode > 0:
            agent.scheduler.step()
        
        # Calculate metrics
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        avg_scores.append(avg_score)
        
        # Save best model
        if avg_score > best_avg_score and len(scores) >= 50:
            best_avg_score = avg_score
            agent.save('logs/snake_cnn_dqn_best.pth')
        
        # Logging
        if episode % 50 == 0:
            current_lr = agent.optimizer.param_groups[0]['lr']
            recent_scores = scores[-50:] if len(scores) >= 50 else scores
            print(f"Ep {episode:4d}/{episodes} | "
                  f"Score: {env.points:3d} | "
                  f"Avg50: {np.mean(recent_scores):5.2f} | "
                  f"Avg100: {avg_score:5.2f} | "
                  f"Best: {best_avg_score:5.2f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Loss: {np.mean(episode_loss) if episode_loss else 0:.4f} | "
                  f"Buffer: {len(agent.memory)}")
        
        # Save checkpoint
        if episode % 500 == 0 and episode > 0:
            agent.save(f'logs/checkpoint_cnn_ep{episode}.pth')
    
    return agent, scores, losses, avg_scores




if __name__ == "__main__":
    # import gymnasium as gym
    # env = gym.make("CartPole-v1")
    # trained_qnet, rewards = train_dqn(env)
    train_cnn_dqn()