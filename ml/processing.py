import math

import einops
import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt


def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    filtered = filtfilt(b, a, data, axis=1)
    return filtered

def smooth(x: torch.Tensor):
    device = x.device
    x = x.cpu()
    x = lowpass_filter(x, cutoff=10, fs=250)
    x = torch.tensor(x.copy(), dtype=torch.float32, device=device)
    return x

GRAVITY = [1.2601, 0.2099, -10.0409]
def remove_gravity_constant(x: torch.Tensor):
    gravity = torch.tensor(GRAVITY, device=x.device)
    x[:3] = x[:3] - gravity.reshape(3, 1)
    return x

def perturb_vector(vector: torch.Tensor, angle_rad_std: float) -> torch.Tensor:
    """
    Rotates the vector by a random angle in a random axis.
    The angle is sampled from a Gaussian distribution with std `angle_rad_std`
    """
    if angle_rad_std <= 0:
        return vector

    angle = torch.randn(1).item() * angle_rad_std
    axis = torch.randn_like(vector)
    axis = axis - (axis @ vector) * vector

    eps = 1e-6
    if axis.norm() < eps:
        return vector
    axis = axis / axis.norm()
    
    # Rodrigues' rotation formula
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])   
    I = torch.eye(3)
    R_perturb = I + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    perturbed = R_perturb @ vector
    return perturbed

def random_rotation(sample: torch.Tensor, axis: torch.Tensor, angle_rad_std: float = 0.0) -> torch.Tensor:
    if sample.ndim != 2:
        raise ValueError("Expected (channels, time)")
    if axis.shape[0] != 3 or sample.shape[0] != 3:
        raise ValueError("Only 3D supported")
    if axis.norm() <= 1e-6:
        raise ValueError("Axis must not be near zero")
    
    axis = axis / axis.norm()
    axis = perturb_vector(axis, angle_rad_std)
    theta = torch.rand(1, device=sample.device).item() * 2 * math.pi
    
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], device=sample.device)
    I = torch.eye(3, device=sample.device)
    R = I + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    return R @ sample


def simple_physical_augmentation(sample: torch.Tensor) -> torch.Tensor:
    """Simpler version that estimates tilt from data"""
    from scipy.signal import butter, filtfilt
    
    accel = sample[:3].cpu().numpy()  # (3, time)
    gyro = sample[3:].cpu().numpy()
    
    # Low-pass filter to extract gravity
    b, a = butter(3, 0.5, btype='low', fs=250)
    gravity = filtfilt(b, a, accel, axis=1)  # (3, time)
    linear_accel = accel - gravity
    
    # Random yaw rotation
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R_yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    # Rotate everything
    linear_accel_rot = R_yaw @ linear_accel
    gyro_rot = R_yaw @ gyro
    gravity_rot = R_yaw @ gravity
    
    accel_rot = linear_accel_rot + gravity_rot
    
    result = np.vstack([accel_rot, gyro_rot])
    return torch.from_numpy(result).float()


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
 
def gravity_invariant_augmentation(
    accel: torch.Tensor,
    gyro: torch.Tensor | None = None,
    gravity_cutoff: float = 0.5,
    fs: float = 250.0,
    filter_order: int = 3,
):
    """
    Apply a yaw rotation that is (approximately) invariant to gravity.

    The idea is:
    1. Estimate gravity with a strong low‑pass filter on the accelerometer.
    2. Subtract gravity to get linear acceleration.
    3. Apply a random yaw rotation (around the Z axis) to linear acceleration
       and gyro.
    4. Rotate the estimated gravity by the same yaw and add it back.

    Args:
        accel: Tensor of shape (3, T) with accelerometer data.
        gyro: Optional tensor of shape (3, T) with gyroscope data.
        gravity_cutoff: Low‑pass cutoff frequency (Hz) for gravity estimation.
        fs: Sampling frequency (Hz).
        filter_order: Order of the Butterworth filter.

    Returns:
        Tuple (accel_aug, gyro_aug) where:
            accel_aug: Augmented accelerometer data, shape (3, T).
            gyro_aug: Augmented gyroscope data, same shape as `gyro`
                      (or None if `gyro` is None).
    """
    if accel.ndim != 2 or accel.shape[0] != 3:
        raise ValueError("accel must have shape (3, T)")
    if gyro is not None and (gyro.ndim != 2 or gyro.shape[0] != 3 or gyro.shape[1] != accel.shape[1]):
        raise ValueError("gyro must have shape (3, T) and match accel in time dimension")

    device = accel.device
    dtype = accel.dtype

    # 1. Estimate gravity using a strong low‑pass filter on the accelerometer
    accel_np = accel.detach().cpu().numpy()
    gravity_np = lowpass_filter(accel_np, cutoff=gravity_cutoff, fs=fs, order=filter_order)
    gravity = torch.tensor(gravity_np, device=device, dtype=dtype)

    # 2. Remove gravity to get linear acceleration
    linear_accel = accel - gravity

    # 3. Random yaw rotation in the XY plane (around Z axis)
    yaw = torch.empty(1, device=device).uniform_(0.0, 2.0 * math.pi).item()
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    R_yaw = torch.tensor(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw,  cos_yaw, 0.0],
            [0.0,      0.0,     1.0],
        ],
        device=device,
        dtype=dtype,
    )

    # 4. Rotate linear acceleration, gyro, and gravity, then add gravity back
    linear_accel_aug = R_yaw @ linear_accel
    gravity_aug = R_yaw @ gravity
    accel_aug = linear_accel_aug + gravity_aug

    gyro_aug = None
    if gyro is not None:
        gyro_aug = R_yaw @ gyro

    return accel_aug, gyro_aug


def random_time_crop_physical(
    sample: torch.Tensor,  # (6, time) -> [ax, ay, az, gx, gy, gz]
    dt: float = 1/250,  # IMU sampling period
    crop_ratio_range: tuple = (0.8, 1.2),  # Speed variation range
    max_duration: float = 2.5,  # Maximum duration in seconds
) -> torch.Tensor:
    """
    Apply random time cropping that respects physics of acceleration.
    
    Args:
        sample: (channels, time) IMU data
        dt: Time step between samples
        crop_ratio_range: (min, max) ratio for time stretching
            - ratio < 1.0: faster motion (compressed time)
            - ratio > 1.0: slower motion (stretched time)
        max_duration: Maximum allowed duration
    
    Returns:
        Augmented sample with same shape as input
    """
    accel = sample[:3].cpu().numpy()  # (3, time)
    gyro = sample[3:].cpu().numpy()   # (3, time)
    n_samples = accel.shape[1]
    
    # Random speed factor
    speed_factor = np.random.uniform(*crop_ratio_range)
    
    # 1. Separate gravity from linear acceleration
    gravity = estimate_gravity(accel)  # (3, time)
    linear_accel = accel - gravity
    
    # 2. Integrate to get velocity and position
    velocity = cumulative_integrate(linear_accel, dt)  # (3, time)
    position = cumulative_integrate(velocity, dt)      # (3, time)
    
    # 3. Create original and new time axes
    t_original = np.arange(n_samples) * dt  # [0, dt, 2*dt, ..., (n-1)*dt]
    
    # New time axis with speed factor applied
    # speed_factor < 1: faster motion, shorter duration
    # speed_factor > 1: slower motion, longer duration
    t_new = t_original * speed_factor
    
    # Ensure we don't exceed max duration
    if t_new[-1] > max_duration:
        scale = max_duration / t_new[-1]
        t_new = t_new * scale
        speed_factor = speed_factor * scale
    
    # 4. Resample position using cubic spline interpolation
    # Position is smooth and can be interpolated
    position_interp = CubicSpline(t_original, position.T, axis=0)
    position_new = position_interp(t_new).T  # (3, n_samples)
    
    # 5. Differentiate to get velocity
    # Use central differences for interior points
    velocity_new = np.zeros_like(position_new)
    velocity_new[:, 0] = (position_new[:, 1] - position_new[:, 0]) / (t_new[1] - t_new[0])
    velocity_new[:, -1] = (position_new[:, -1] - position_new[:, -2]) / (t_new[-1] - t_new[-2])
    
    for i in range(1, n_samples - 1):
        dt_forward = t_new[i+1] - t_new[i]
        dt_backward = t_new[i] - t_new[i-1]
        # Central difference with non-uniform spacing
        velocity_new[:, i] = (
            (position_new[:, i+1] - position_new[:, i-1]) / 
            (dt_forward + dt_backward)
        )
    
    # 6. Differentiate velocity to get acceleration
    linear_accel_new = np.zeros_like(velocity_new)
    linear_accel_new[:, 0] = (velocity_new[:, 1] - velocity_new[:, 0]) / (t_new[1] - t_new[0])
    linear_accel_new[:, -1] = (velocity_new[:, -1] - velocity_new[:, -2]) / (t_new[-1] - t_new[-2])
    
    for i in range(1, n_samples - 1):
        dt_forward = t_new[i+1] - t_new[i]
        dt_backward = t_new[i] - t_new[i-1]
        linear_accel_new[:, i] = (
            (velocity_new[:, i+1] - velocity_new[:, i-1]) / 
            (dt_forward + dt_backward)
        )
    
    # 7. Resample gravity (gravity doesn't change with speed, just time)
    gravity_interp = CubicSpline(t_original, gravity.T, axis=0)
    gravity_new = gravity_interp(t_new).T
    
    # 8. Combine linear acceleration and gravity
    accel_new = linear_accel_new + gravity_new
    
    # 9. Handle gyroscope data
    # Angular velocity can be interpolated directly
    gyro_interp = CubicSpline(t_original, gyro.T, axis=0)
    gyro_new = gyro_interp(t_new).T
    
    # 10. Return to original time grid (resample to fixed n_samples)
    t_fixed = np.linspace(0, t_new[-1], n_samples)
    
    accel_final = CubicSpline(t_new, accel_new.T, axis=0)(t_fixed).T
    gyro_final = CubicSpline(t_new, gyro_new.T, axis=0)(t_fixed).T
    
    result = np.vstack([accel_final, gyro_final])
    return torch.from_numpy(result).float().to(sample.device)


def estimate_gravity(accel: np.ndarray, cutoff_hz: float = 0.5, fs: float = 250) -> np.ndarray:
    """
    Estimate gravity component using low-pass filter.
    
    Args:
        accel: (3, time) acceleration data
        cutoff_hz: Cutoff frequency for low-pass filter
        fs: Sampling frequency
    
    Returns:
        gravity: (3, time) estimated gravity component
    """
    from scipy.signal import butter, filtfilt
    
    b, a = butter(3, cutoff_hz, btype='low', fs=fs)
    gravity = filtfilt(b, a, accel, axis=1)
    
    return gravity


def cumulative_integrate(data: np.ndarray, dt: float) -> np.ndarray:
    """
    Integrate data using cumulative trapezoidal rule.
    
    Args:
        data: (3, time) array to integrate
        dt: Time step
    
    Returns:
        integrated: (3, time) integrated data
    """
    from scipy.integrate import cumulative_trapezoid
    
    # cumulative_trapezoid returns n-1 points, prepend zero for initial condition
    integrated = cumulative_trapezoid(data, dx=dt, axis=1, initial=0)
    
    return integrated

def align_to_plane(vector: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector so its Z-axis aligns with the given normal.
    
    Args:
        vector: (3, ...) tensor to rotate
        normal: (3,) plane normal vector
    
    Returns:
        Rotated vector with Z aligned to normal
    """
    normal = normal / normal.norm()
    z_axis = torch.tensor([0., 0., 1.], device=vector.device, dtype=vector.dtype)
    
    # Rotation axis and angle
    axis = torch.cross(z_axis, normal)
    cos_angle = torch.dot(z_axis, normal)
    
    # Handle parallel/antiparallel cases
    if axis.norm() < 1e-6:
        return vector if cos_angle > 0 else -vector
    
    axis = axis / axis.norm()
    
    # Rodrigues' formula
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], device=vector.device, dtype=vector.dtype)
    
    I = torch.eye(3, device=vector.device, dtype=vector.dtype)
    R = I + torch.sin(torch.acos(cos_angle)) * K + (1 - cos_angle) * (K @ K)
    
    return R @ vector

# def pca_transform_3(x):
#     """Converts to PCA coordinates"""
#     x = x - x.mean(dim=1, keepdim=True)
#     U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    
#     # We have the top-3 components
#     components = U[:3]
#     pca_1 = einops.einsum(components[0], x, "coord, coord time -> time")
#     pca_2 = einops.einsum(components[1], x, "coord, coord time -> time")
#     pca_3 = einops.einsum(components[2], x, "coord, coord time -> time")
#     pca, _ = einops.pack([pca_1, pca_2, pca_3], "* time")
#     return pca

def pca_transform_3(x):
    """
    x: (c, t) tensor (features, samples)
    returns: (3, t) PCA coordinates
    """
    x = x - x.mean(dim=1, keepdim=True)

    U, S, Vh = torch.linalg.svd(x, full_matrices=False)

    components = U[:, :3]      # (c, 3)
    pca = components.T @ x     # (3, t)
    return pca

def pca_transform_3_handedness(x):
    """
    x: (c, t) tensor (features, samples)
    returns: (3, t) PCA coordinates with fixed handedness
    """
    x = x - x.mean(dim=1, keepdim=True)

    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    components = U[:, :3]  # (c, 3)

    # ---- handedness fix ----
    z_up = torch.zeros_like(components[:, 0])
    z_up[2] = 1.0  # global Z+

    e1 = components[:, 0]
    e2 = components[:, 1]

    handedness = torch.dot(torch.cross(e1, e2, dim=0), z_up)

    if handedness < 0:
        # flip second axis (or swap e1/e2 — both are valid)
        components[:, 1] *= -1

    pca = components.T @ x
    return pca

def pca_transform_2(x):
    """Converts to PCA coordinates"""
    x = x - x.mean(dim=1, keepdim=True)
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    
    # We have the top-3 components
    components = U[:3]
    pca_1 = einops.einsum(components[0], x, "coord, coord time -> time")
    pca_2 = einops.einsum(components[1], x, "coord, coord time -> time")
    pca_3 = einops.einsum(components[2], x, "coord, coord time -> time")
    pca, _ = einops.pack([pca_1, pca_2, pca_3], "* time")
    return pca

def canonical_transform_robust(x, samples_for_velocity=10):
    """
    x: (3, t) tensor of accelerometer readings.
    returns: (3, t) tensor projected into the Canonical Kinematic Frame.
             - Row 0: Aligned with initial movement direction (Forward)
             - Row 1: Orthogonal planar direction (Left)
             - Row 2: Vertical/Normal direction (Up)
    """
    # 1. Center the data
    # We use the mean as a proxy for the static gravity component + bias
    mean_vec = x.mean(dim=1, keepdim=True)
    x_centered = x - mean_vec
    
    # 2. Find the Plane of Motion (Z-axis) using SVD
    # The component with the SMALLEST variance (singular value) is the normal to the table.
    U, S, Vh = torch.linalg.svd(x_centered, full_matrices=False)
    
    # U[:, 0] is Max Variance Direction (Unstable X)
    # U[:, 1] is 2nd Max Variance (Unstable Y)
    # U[:, 2] is Min Variance (Stable Z / Normal)
    normal = U[:, 2] 

    # 3. Enforce Z-Direction (Upward)
    # The mean_vec is roughly [0, 0, -9.8] (Gravity points down). 
    # We want our Normal to point UP (opposite to gravity).
    # If dot(normal, gravity) > 0, normal is pointing down. Flip it.
    if torch.dot(normal, mean_vec.squeeze()) > 0:
        normal = -normal
    
    # 4. Enforce X-Direction (Initial Velocity)
    # Project the "Start Vector" onto the plane defined by 'normal' to ensure orthogonality.
    # We take the vector from t=0 to t=10 to get a robust direction of initial movement.
    # Note: If there is a delay in movement, you might need to find the first index 
    # where velocity > threshold.
    x_centered_norm = torch.norm(x_centered, dim=0)
    start = 0
    for i in range(x_centered.shape[1]):
        if x_centered_norm[i] > 10:
            start = i
            break

    R = 10
    start_vec = x_centered[:, start+R] - x_centered[:, start]
    
    # Remove any component of start_vec that points in the Z direction (flatten it)
    start_vec_proj = start_vec - torch.dot(start_vec, normal) * normal
    
    # Normalize to get the X-axis unit vector
    # Handle edge case where start_vec is zero (no movement at start)
    norm = torch.norm(start_vec_proj)
    if norm < 1e-6:
        # Fallback: Use the Major PCA Axis if no initial movement detected
        # (This brings back instability, but prevents crash)
        x_axis = U[:, 0] 
        x_axis = x_axis - torch.dot(x_axis, normal) * normal
        x_axis = x_axis / torch.norm(x_axis)
    else:
        x_axis = start_vec_proj / norm

    # 5. Enforce Y-Direction (Chirality)
    # Use Right-Hand Rule: Z cross X = Y (or X cross Y = Z -> Y = Z cross X)
    # Actually, standard is X cross Y = Z. So Y = Z cross X.
    y_axis = torch.linalg.cross(normal, x_axis)

    # 6. Construct the Projection Matrix
    # Stack rows: [X, Y, Z]
    basis = torch.stack([x_axis, y_axis, normal]) # Shape (3, 3)
    
    # 7. Project
    # x_centered is (3, T), basis is (3, 3). 
    # We want basis @ x_centered
    projected = basis @ x_centered
    
    return projected

def align_to_first_movement(pca_coords, accel_threshold=0.2, window=5):
    """
    Rotate the first two PCA components so that the first component points
    in the direction of initial movement.
    
    Args:
        pca_coords: (3, t) PCA-transformed coordinates
        accel_threshold: Threshold for detecting movement start
        window: Number of samples to average for initial direction
    
    Returns:
        (3, t) rotated PCA coordinates where first axis aligns with initial movement
    """
    # Find where movement starts
    # Compute velocity (derivative of position)
    velocity = torch.diff(pca_coords[:2], dim=1)  # (2, t-1)
    speed = torch.norm(velocity, dim=0)  # (t-1,)
    
    # Find first point where speed exceeds threshold
    movement_start = torch.where(speed > accel_threshold)[0]
    
    if len(movement_start) == 0:
        # No significant movement detected, return as-is
        return pca_coords
    
    start_idx = movement_start[0].item()
    end_idx = min(start_idx + window, velocity.shape[1])
    
    # Average initial velocity direction
    initial_velocity = velocity[:, start_idx:end_idx].mean(dim=1)  # (2,)
    
    if torch.norm(initial_velocity) < 1e-6:
        # No movement, return as-is
        return pca_coords
    
    initial_velocity = initial_velocity / torch.norm(initial_velocity)
    
    # Compute rotation angle to align initial_velocity with [1, 0]
    # current direction is [initial_velocity[0], initial_velocity[1]]
    # we want to rotate to [1, 0]
    cos_theta = initial_velocity[0]
    sin_theta = initial_velocity[1]
    
    # 2D rotation matrix (in the plane of first two components)
    R = torch.tensor([
        [cos_theta, sin_theta],
        [-sin_theta, cos_theta]
    ], device=pca_coords.device, dtype=pca_coords.dtype)
    
    # Apply rotation to first two components only
    result = pca_coords.clone()
    result[:2] = R @ pca_coords[:2]
    
    return result

def integrate_crop_resample_diff(
    x: torch.Tensor,
    crop_min: float = 0.5,
    crop_max: float = 1.0,
) -> torch.Tensor:
    """
    x: (C, T) tensor
    """
    C, T = x.shape

    # integrate twice along T
    y = torch.cumsum(torch.cumsum(x, dim=1), dim=1)

    # random crop length
    ratio = torch.empty(1).uniform_(crop_min, crop_max).item()
    L = int(T * ratio)
    start = torch.randint(0, T - L + 1, (1,)).item()
    y = y[:, start:start + L]

    # upsample back to T
    y = F.interpolate(y.unsqueeze(0), size=T, mode="linear", align_corners=False)
    y = y.squeeze(0)

    # differentiate twice
    y = torch.diff(torch.diff(y, dim=1), dim=1)

    return y
