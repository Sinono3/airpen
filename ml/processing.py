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


def random_xz_rotation(sample: torch.Tensor) -> torch.Tensor:
    """
    Applies a random rotation in the XZ plane (around the Y axis) to
    accelerometer channels of a sample shaped as (channels, time).
    """
    if sample.ndim != 2:
        raise ValueError("Expected sample tensor with shape (channels, time)")

    # Avoid mutating the underlying dataset tensor
    sample = sample.clone()
    num_channels = sample.shape[0]

    theta = torch.rand(1, device=sample.device).item() * 2 * math.pi
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    if num_channels >= 3:
        rot_matrix = sample.new_tensor(
            [
                [cos_t, 0.0, sin_t],
                [0.0, 1.0, 0.0],
                [-sin_t, 0.0, cos_t],
            ]
        )
        sample[:3, :] = rot_matrix @ sample[:3, :]
    elif num_channels == 2:
        # Assume channels correspond to X and Z in that order.
        rot_matrix = sample.new_tensor(
            [
                [cos_t, sin_t],
                [-sin_t, cos_t],
            ]
        )
        sample[:2, :] = rot_matrix @ sample[:2, :]
    else:
        raise ValueError("Random XZ rotation requires at least X and Z channels")

    return sample
