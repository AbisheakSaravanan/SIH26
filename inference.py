import torch
import numpy as np
from model import CSIClassifier

# 1. LOAD MODEL
num_classes = 3  # Match target class count
model = CSIClassifier(num_classes=num_classes)
model.load_state_dict(torch.load("csi_model.pth"))
model.eval()

def predict_single_window(amp_window, phase_window):
    """
    amp_window: 2D array (Packets, Subcarriers)
    phase_window: 2D array (Packets, Subcarriers)
    """
    # Combine channels -> Shape: (1, 2, Packets, Subcarriers)
    combined = np.stack([amp_window, phase_window], axis=0)
    tensor_input = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(tensor_input)
        predicted_class = torch.argmax(outputs, dim=1).item()
    
    return predicted_class

# --- QUICK TEST WITH RANDOM SAMPLE ---
sample_amp = np.random.rand(10, 56)    # 10 Packets x 56 Subcarriers
sample_phase = np.random.rand(10, 56)

result = predict_single_window(sample_amp, sample_phase)
print(f"Live SPU Window Prediction: Class ID {result}")