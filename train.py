import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from dataset import CSIDataset
from model import CSIClassifier

# 1. PATHS
AMP_FILE = "amplitude_windows.npy"
PHASE_FILE = "phase_windows.npy"
LABELS_FILE = "labels.npy"
SAVE_PATH = "csi_model.pth"

# 2. LOAD DATA
dataset = CSIDataset(AMP_FILE, PHASE_FILE, LABELS_FILE)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

num_classes = len(torch.unique(dataset.y))
model = CSIClassifier(num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. TRAIN LOOP (20 Epochs)
print("Training started...")
model.train()
for epoch in range(1, 21):
    running_loss = 0.0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch:02d}/20 | Loss: {running_loss / len(loader):.4f}")

# 4. SAVE WEIGHTS
torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved successfully as '{SAVE_PATH}'!")