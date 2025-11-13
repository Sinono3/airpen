import torch
from scipy.signal import butter, filtfilt

def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    filtered = filtfilt(b, a, data, axis=1)
    return filtered

def process_raw(x: torch.Tensor):
    """
    Converts a raw data sample, and applies all necessary processing.
    """
    device = x.device
    x = x.cpu()
    x = lowpass_filter(x, cutoff=5, fs=50)
    x = torch.tensor(x.copy(), dtype=torch.float32, device=device)
    return x
