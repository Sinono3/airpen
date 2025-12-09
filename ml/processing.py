import math
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


def random_rotation(sample: torch.Tensor, plane: tuple[int, int] = (0, 1)) -> torch.Tensor:
    """
    Random rotation in a specified plane (default XY).
    sample: (channels, time)
    plane: indices of the two axes to rotate, e.g. (0,2) for XZ.
    """
    if sample.ndim != 2:
        raise ValueError("Expected (channels, time)")

    a, b = plane
    if a >= sample.shape[0] or b >= sample.shape[0]:
        raise ValueError("Rotation plane indices out of range")

    sample = sample.clone()
    theta = torch.rand(1, device=sample.device).item() * 2 * math.pi
    c, s = math.cos(theta), math.sin(theta)

    R = sample.new_tensor([[c, -s], [s, c]])
    sample[[a, b], :] = R @ sample[[a, b], :]

    return sample
