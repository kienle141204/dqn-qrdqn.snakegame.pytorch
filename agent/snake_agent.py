import torch
import torch.nn as nn
import torch.optim as optim
import random
from utils import ReplayBuffer 
from network.cnn import CNNDQN 
from network.mlp import MLP      
from network.qr_dqn import QRDQN 

class Agent:
    """
    Một lớp Agent hợp nhất có thể hoạt động như DQN hoặc QR-DQN.
    
    Args:
        agent_type (str): Loại agent, có thể là 'dqn' hoặc 'qr-dqn'.
        grid_size (int): Kích thước của môi trường grid (dùng cho CNN).
        action_size (int): Số lượng hành động có thể có.
        state_dim (int): Kích thước của vector trạng thái (dùng cho MLP).
        num_quantiles (int): Số lượng quantile cho QR-DQN.
        lr (float): Tốc độ học (learning rate).
        gamma (float): Hệ số chiết khấu.
        epsilon_start (float): Giá trị epsilon ban đầu cho khám phá.
        epsilon_end (float): Giá trị epsilon cuối cùng.
        epsilon_decay (float): Hệ số suy giảm của epsilon.
    """
    def __init__(self, agent_type, grid_size, action_size, state_dim=13, num_quantiles=51, 
                 lr=0.0005, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        
        if agent_type not in ['dqn', 'qr-dqn']:
            raise ValueError("agent_type phải là 'dqn' hoặc 'qr-dqn'")
            
        self.agent_type = agent_type
        self.grid_size = grid_size
        self.action_size = action_size
        self.state_dim = state_dim
        self.num_quantiles = num_quantiles
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network 
        if self.agent_type == 'dqn':
            print("Agent DQN.")
            self.policy_net = MLP(self.state_dim, self.action_size).to(self.device)
            self.target_net = MLP(self.state_dim, self.action_size).to(self.device)
        elif self.agent_type == 'qr-dqn':
            print("Agent QR-DQN.")
            self.policy_net = QRDQN(self.state_dim, self.action_size, self.num_quantiles).to(self.device)
            self.target_net = QRDQN(self.state_dim, self.action_size, self.num_quantiles).to(self.device)
            
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(100000)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)
    
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Action 
            if self.agent_type == 'dqn':
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
            elif self.agent_type == 'qr-dqn':
                # Với QR-DQN, chúng ta lấy trung bình các quantile để có Q-value
                quantiles = self.policy_net(state_tensor)
                q_values = quantiles.mean(dim=2)
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
        
        # Loss
        if self.agent_type == 'dqn':
            current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_net(next_states).max(1)[0].detach()
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
            
        elif self.agent_type == 'qr-dqn':
            current_quantiles = self.policy_net(states)
            current_quantiles = current_quantiles.gather(1, actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.num_quantiles)).squeeze(1)

            with torch.no_grad():
                next_q_values_dist = self.target_net(next_states)
                next_q_values = next_q_values_dist.mean(2)
                next_actions = next_q_values.max(1)[1]
                
                next_quantiles = next_q_values_dist.gather(1, next_actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.num_quantiles)).squeeze(1)
                target_quantiles = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_quantiles

            td_errors = target_quantiles.unsqueeze(2) - current_quantiles.unsqueeze(1)
            huber_loss = torch.where(td_errors.abs() < 1.0, 0.5 * td_errors.pow(2), td_errors.abs() - 0.5)
            
            tau = (torch.arange(self.num_quantiles, device=self.device).float() + 0.5) / self.num_quantiles
            quantile_weights = torch.abs(tau.view(1, 1, -1) - (td_errors.detach() < 0).float())
            
            loss = (quantile_weights * huber_loss).mean()

        # Optimizer 
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

