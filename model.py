import torch
import torch.nn as nn

class CSIClassifier(nn.Module):
    def __init__(self, num_classes=3, hidden_size=64, num_layers=1):
        super().__init__()
        
        # 1. 2D CNN Frontend to extract local temporal-frequency patterns
        self.cnn = nn.Sequential(
            # Input: 2 channels (Amplitude, Phase)
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # We pool the subcarrier dimension to a fixed size of 8
        self.subcarrier_pool_size = 8
        self.lstm_input_size = 32 * self.subcarrier_pool_size  # 256 features per packet
        
        # 2. LSTM Backend for sequence learning across packets
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 3. Dense Classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # Input shape: [batch_size, 2, packets, subcarriers]
        batch_size, channels, packets, subcarriers = x.shape
        
        # Extract features: [batch_size, 32, packets, subcarriers]
        features = self.cnn(x)
        
        # Reshape to pool along subcarriers per packet:
        # [batch_size * packets, 32, subcarriers]
        features = features.permute(0, 2, 1, 3).contiguous()
        features = features.view(batch_size * packets, 32, subcarriers)
        
        # Pool subcarriers to fixed dimension
        pooled = nn.functional.adaptive_avg_pool1d(features, self.subcarrier_pool_size)
        
        # Format sequence for LSTM: [batch_size, packets, 256]
        lstm_in = pooled.view(batch_size, packets, self.lstm_input_size)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(lstm_in)
        
        # Use final hidden state of the last layer: [batch_size, hidden_size]
        out = self.fc(h_n[-1])
        return out