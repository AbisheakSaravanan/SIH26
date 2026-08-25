import numpy as np

# Generate dummy CSI data
# Shape: (Windows, Packets, Subcarriers)
windows = 100
packets = 10
subcarriers = 56
num_classes = 3

# Generate random amplitude and phase windows
amp_windows = np.random.rand(windows, packets, subcarriers).astype(np.float32)
phase_windows = np.random.rand(windows, packets, subcarriers).astype(np.float32)

# Generate labels between 0 and num_classes-1
labels = np.random.randint(0, num_classes, size=(windows,)).astype(np.int64)

# Save arrays
np.save("amplitude_windows.npy", amp_windows)
np.save("phase_windows.npy", phase_windows)
np.save("labels.npy", labels)

print("Dummy CSI data generated successfully!")
