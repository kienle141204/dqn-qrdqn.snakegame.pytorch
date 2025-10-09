import torch 
import torch.nn as nn

class QRDQN(nn.Module):
    def __init__(self, state_dim, action_dim, num_quantiles=51):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.action_dim = action_dim
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim * num_quantiles)
        )
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size = x.size(0)
        quantiles = self.fc(x)
        return quantiles.view(batch_size, self.action_dim, self.num_quantiles)
        
    def get_q_values(self, x):
        quantiles = self.forward(x)  
        q_values = quantiles.mean(dim=2)  
        return q_values