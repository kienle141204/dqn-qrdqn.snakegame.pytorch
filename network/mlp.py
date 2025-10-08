import torch 
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_size, hidden_layer=[512, 256, 128, 64, 32], action_size=4):
        super(DQN, self).__init__()
        layers = []
        for i in range(len(hidden_layer)):
            if i == 0:
                layers.append(nn.Linear(state_size, hidden_layer[i]))
            else:
                layers.append(nn.Linear(hidden_layer[i-1], hidden_layer[i]))
            layers.append(nn.ReLU())
        self.forward_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_layer[-1], action_size)
    
    def forward(self, x):
        x = self.forward_layers(x)
        return self.fc(x)