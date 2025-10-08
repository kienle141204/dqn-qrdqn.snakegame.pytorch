import torch 
import torch.nn as nn

class CNNDQN(nn.Module):
    def __init__(self, grid_size, input_shape=(4, 20, 20), action_size=4):
        super().__init__()
        self.grid_size = grid_size
        H, W = grid_size, grid_size
        C, _, _ = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),                                      
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU()
        )
        conv_out = 64 * H * W // 4
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, action_size)
        )

    def forward(self, x):
        x = self.conv(x)              
        return self.fc(x)