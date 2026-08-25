import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import CSICSVDataset
from model import CSIClassifier

# 1. Configuration
CSV_PATH = "data/raw/csi_data.csv"
SAVE_PATH = "csi_model.pth"
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.001

# 2. Load Dataset
dataset = CSICSVDataset(CSV_PATH)
num_sessions = len(dataset)
train_size = int(0.8 * num_sessions)
val_size = num_sessions - train_size

train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size], 
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Total sessions: {num_sessions}")
print(f"Train sessions: {train_size} | Validation sessions: {val_size}")

# 3. Model, Loss, Optimizer
num_classes = len(dataset.label_map)
model = CSIClassifier(num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 4. Training Loop
import time

duration_seconds = 5 * 60  # 5 minutes
start_time = time.time()
epoch = 0

print(f"Training started (will run continuously for 5 minutes / {duration_seconds}s)...")
while (time.time() - start_time) < duration_seconds:
    epoch += 1
    model.train()
    running_loss = 0.0
    for x_batch, y_batch in train_loader:
        # Add random Gaussian noise to training batch
        # std=0.3 simulates a realistic noise level for amplitude and phase
        noise = torch.randn_like(x_batch) * 0.3
        x_batch = x_batch + noise
        
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    # Validation (evaluate on clean validation data)
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            
    train_epoch_loss = running_loss / len(train_loader)
    val_epoch_loss = val_loss / len(val_loader)
    val_accuracy = correct / total * 100
    
    elapsed = time.time() - start_time
    minutes_left = max(0.0, (duration_seconds - elapsed) / 60)
    print(f"Epoch {epoch:03d} | Train Loss (with noise): {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_accuracy:.2f}% | Time Elapsed: {elapsed:.1f}s ({minutes_left:.1f}m remaining)")

# 5. Save Weights
torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved successfully to '{SAVE_PATH}'!")

