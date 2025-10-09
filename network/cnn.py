import torch 
import torch.nn as nn

class CNNDQN(nn.Module):
    def __init__(self, grid_size, num_chanels=4, action_size=4):
        super().__init__()
        self.grid_size = grid_size
        self.conv = nn.Sequential(
            nn.Conv2d(num_chanels, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
                                   
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU()
        )
        conv_out = 64 * self.grid_size * self.grid_size
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        x = self.conv(x)              
        return self.fc(x)