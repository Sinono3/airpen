import math
import torch
import torch.nn.functional as F
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


def random_rotation(sample: torch.Tensor, normal: torch.Tensor, normal_std: float = 0.0) -> torch.Tensor:
    if sample.ndim != 2:
        raise ValueError("Expected (channels, time)")
    if normal.shape != (sample.shape[0],):
        raise ValueError("Normal must match number of channels")
    
    sample = sample.clone()
    normal = normal / normal.norm()
    
    # Perturb normal direction if normal_std > 0
    if normal_std > 0:
        angle = torch.randn(1, device=sample.device).item() * normal_std
        axis = torch.randn_like(normal)
        axis = axis - (axis @ normal) * normal  # Orthogonal to normal
        if axis.norm() > 1e-6:
            axis = axis / axis.norm()
            K = axis.unsqueeze(1) @ axis.unsqueeze(0)
            I = torch.eye(sample.shape[0], device=sample.device)
            R_perturb = I + math.sin(angle) * (K - I) + (1 - math.cos(angle)) * (K - I) @ (K - I)
            normal = R_perturb @ normal
            normal = normal / normal.norm()
    
    theta = torch.rand(1, device=sample.device).item() * 2 * math.pi
    K = normal.unsqueeze(1) @ normal.unsqueeze(0)
    I = torch.eye(sample.shape[0], device=sample.device)
    R = I + math.sin(theta) * (K - I) + (1 - math.cos(theta)) * (K - I) @ (K - I)
    
    return R @ sample

def augment_accelerometer_temporal(accel_data, 
                                   scale_range=(0.8, 1.2),
                                   translate_range=(-0.2, 0.2)):
    """
    Apply random temporal translation and scaling to accelerometer data.
    
    Args:
        accel_data: torch.Tensor of shape (3, T) where T is time steps
        scale_range: tuple of (min_scale, max_scale) for time axis
        translate_range: tuple of (min_translate, max_translate) as fraction of sequence length
    
    Returns:
        torch.Tensor of shape (3, T) with augmented accelerometer data
    
    Note: When time is scaled by factor s, acceleration magnitudes scale by 1/s²
    because a = d²x/dt². If we compress time (s < 1), accelerations increase.
    """
    device = accel_data.device
    dtype = accel_data.dtype
    _, T = accel_data.shape
    
    # Sample random scale and translation
    scale = torch.empty(1).uniform_(*scale_range).item()
    translate = torch.empty(1).uniform_(*translate_range).item() * T
    
    # Create sampling grid for interpolation
    # Original time indices: [0, 1, 2, ..., T-1]
    original_indices = torch.arange(T, dtype=dtype, device=device)
    
    # Apply inverse transform to find where to sample from
    # If we want output[i] at time t, we sample from input at time (t - translate) / scale
    sampling_indices = (original_indices - translate) / scale
    
    # Normalize to [-1, 1] for grid_sample
    normalized_indices = 2 * sampling_indices / (T - 1) - 1
    
    # Reshape for grid_sample: (N, C, W) requires grid of shape (N, H_out, W_out, 2)
    # For 1D, H_out=1, and we only use the x-coordinate
    grid = torch.stack([
        normalized_indices,
        torch.zeros_like(normalized_indices)
    ], dim=-1).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, T, 2)
    
    # Reshape input for grid_sample: (N, C, H, W)
    accel_reshaped = accel_data.unsqueeze(0).unsqueeze(2)  # Shape: (1, 3, 1, T)
    
    # Interpolate
    augmented = F.grid_sample(
        accel_reshaped, 
        grid, 
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    
    # Reshape back to (3, T)
    augmented = augmented.squeeze(0).squeeze(1)
    
    # Apply acceleration scaling: a_new = a_old / scale²
    # When time is compressed (scale < 1), accelerations increase
    augmented = augmented / (scale ** 2)
    
    return augmented

