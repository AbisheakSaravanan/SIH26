import csv
import torch
from torch.utils.data import Dataset
import numpy as np

class CSICSVDataset(Dataset):
    def __init__(self, csv_path, label_map=None, target_subcarriers=114):
        if label_map is None:
            self.label_map = {
                "Standing": 0,
                "Sitting": 1,
                "Walking": 2,
                "Running": 3,
                "Falling": 4
            }
        else:
            self.label_map = label_map
            
        self.sessions = []
        
        # Load and parse the CSV
        sessions_dict = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sess_id = row['session_id']
                if sess_id not in sessions_dict:
                    sessions_dict[sess_id] = {
                        'amp': [],
                        'phase': [],
                        'label': row['predicted_movement'],
                        'times': []
                    }
                # Parse PostgreSQL style array strings
                amp_wave = [float(x) for x in row['amplitude_waveform'].strip('{}').split(',')]
                phase_wave = [float(x) for x in row['phase_waveform'].strip('{}').split(',')]
                sessions_dict[sess_id]['amp'].append(amp_wave)
                sessions_dict[sess_id]['phase'].append(phase_wave)
                sessions_dict[sess_id]['times'].append(row['capture_time'])
                
        # Format sessions into tensors
        for sess_id, data in sessions_dict.items():
            # Sort by capture time to maintain temporal order
            sort_indices = np.argsort(data['times'])
            amp_sorted = [data['amp'][i] for i in sort_indices]
            phase_sorted = [data['phase'][i] for i in sort_indices]
            
            # Stack into shape: (2, Packets, Subcarriers)
            amp_arr = np.array(amp_sorted, dtype=np.float32)      # (Packets, Subcarriers)
            phase_arr = np.array(phase_sorted, dtype=np.float32)  # (Packets, Subcarriers)
            
            combined = np.stack([amp_arr, phase_arr], axis=0)     # (2, Packets, Subcarriers)
            
            x = torch.tensor(combined, dtype=torch.float32)
            
            # Interpolate subcarrier dimension to target size if necessary
            if x.shape[-1] != target_subcarriers:
                x = torch.nn.functional.interpolate(x, size=target_subcarriers, mode='linear', align_corners=False)
                
            y = torch.tensor(self.label_map[data['label']], dtype=torch.long)
            
            self.sessions.append((x, y))
            
    def __len__(self):
        return len(self.sessions)
        
    def __getitem__(self, index):
        return self.sessions[index]
