import numpy as np
from scipy.signal import butter, filtfilt

IN  = "data/ABCDEF.npz"
OUT = "data/ABCDEF_smoothed.npz"

data = dict(np.load(IN))
# Classes to use
labels = ['A', 'B', 'C', 'D', 'E', 'F']

def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    filtered = filtfilt(b, a, data, axis=1)
    return filtered

print(f"all labels:  {list(data.keys())}")
print(f"used labels: {labels}")

for label in labels:
    # only accelerometer data
    data[label] = data[label][:, :, :3]
    data[label] = lowpass_filter(data[label], cutoff=5, fs=50)

np.savez(OUT, **data)
