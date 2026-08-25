import numpy as np
import torch
from torch.utils.data import Dataset

class CSIDataset(Dataset):
    def __init__(self, amp_path, phase_path, labels_path=None):
        # Load 3D arrays: (Windows, Packets, Subcarriers)
        amp = np.load(amp_path)
        phase = np.load(phase_path)
        
        # Combine Amp + Phase into shape: (Windows, 2, Packets, Subcarriers)
        combined = np.stack([amp, phase], axis=1)
        self.x = torch.tensor(combined, dtype=torch.float32)
        
        if labels_path:
            self.y = torch.tensor(np.load(labels_path), dtype=torch.long)
        else:
            self.y = None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        if self.y is not None:
            return self.x[index], self.y[index]
        return self.x[index]