import torch
import torch.nn as nn

class CSIClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            # Input: 2 channels (Amplitude, Phase)
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Compress any matrix grid (Packets x Subcarriers) to 4x8
            nn.AdaptiveAvgPool2d((4, 8)),
            nn.Flatten(),
            
            nn.Linear(32 * 4 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)