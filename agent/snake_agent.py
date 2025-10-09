import torch 
import torch.nn as nn
import torch.optim as optim
from utils import ReplayBuffer
import random
from network.cnn import CNNDQN
from network.mlp import MLP
from network.qr_dqn import QRDQN

class snake_Agent:
    def __init__(self, grid_size, action_size, lr=0.0005, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.grid_size = grid_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Main network and target network
        # self.policy_net = CNNDQN(self.grid_size, action_size=action_size).to(self.device)
        # self.target_net = CNNDQN(self.grid_size, action_size=action_size).to(self.device)
        self.policy_net = MLP(13, action_size).to(self.device)
        self.target_net = MLP(13, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(100000)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)
    
    def select_action(self, state,  training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) 
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


class snake_qrdqn_Agent(snake_Agent):
    def __init__(self, grid_size, action_size, num_quantiles=51, lr=0.0005, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        super(snake_qrdqn_Agent, self).__init__(grid_size, action_size, lr, gamma, 
                                                epsilon_start, epsilon_end, epsilon_decay)
        self.num_quantiles = num_quantiles
        
        # Override networks for QR-DQN
        self.policy_net = QRDQN(13, action_dim=action_size, num_quantiles=num_quantiles).to(self.device)
        self.target_net = QRDQN(13, action_dim=action_size, num_quantiles=num_quantiles).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(100000)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)
    
    def select_action(self, state, training=True):
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            
            if hasattr(self.policy_net, 'get_q_values'):
                q_values = self.policy_net.get_q_values(state_tensor)
            else:
                q_values = self.policy_net(state_tensor)
            
            if q_values.dim() > 1:
                return q_values[0].argmax().item()
            else:
                return q_values.argmax().item()
    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current quantile values
        current_quantiles = self.policy_net(states).gather(1, actions.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.num_quantiles)).squeeze(1)
        
        # Next quantile values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_actions = next_q_values.mean(2).max(1)[1]
            next_quantiles = next_q_values.gather(1, next_actions.unsqueeze(1).unsqueeze(1).expand(-1, -1, self.num_quantiles)).squeeze(1)
            target_quantiles = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_quantiles
        
        # Compute Huber loss for quantile
        td_errors = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
        huber_loss = torch.where(td_errors.abs() < 1.0, 0.5 * td_errors.pow(2), td_errors.abs() - 0.5)

        quantile_weights = torch.abs(torch.arange(0, self.num_quantiles, device=self.device).float() / self.num_quantiles - (td_errors.detach() < 0).float())
        loss = (quantile_weights.unsqueeze(1) * huber_loss).mean()

        # Compute Huber loss for quantile
        td_errors = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
        huber_loss = torch.where(td_errors.abs() < 1.0, 0.5 * td_errors.pow(2), td_errors.abs() - 0.5)

        # Fixed quantile weights calculation
        tau = (torch.arange(self.num_quantiles, device=self.device).float() + 0.5) / self.num_quantiles
        quantile_weights = torch.abs(tau.view(1, -1, 1) - (td_errors.detach() < 0).float())
        
        loss = (quantile_weights * huber_loss).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()
    
    def update_target_network(self):
        return super().update_target_network()
    
    def update_epsilon(self):
        return super().update_epsilon()
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
    
    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

        