import numpy as np
from scipy.signal import butter, filtfilt

data = dict(np.load("ABCDEF.npz"))
labels = ['A', 'B', 'C', 'D']

def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    filtered = filtfilt(b, a, data, axis=1)
    return filtered

print(f"all labels:  {list(data.keys())}")
print(f"used labels: {labels}")

for letter in data:
    # only accelero
    data[letter] = data[letter][:, :, :3]
    data[letter] = lowpass_filter(data[letter], cutoff=5, fs=50)

np.savez("ABCD_smoothed.npz", **data)
