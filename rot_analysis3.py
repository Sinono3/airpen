import einops
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt

dir = Path("outputs/rotations/20251209_111619")
angles = [0, 90, 180, 270]
sample_count = 8
colors = ['red', 'green', 'blue', 'orange']

# Load all data
all_data = {}
for angle in angles:
    x = np.stack([np.loadtxt(dir / f"rot{angle}_recording{i}.csv", delimiter=",") 
                  for i in range(sample_count)])
    x = torch.from_numpy(x).float()
    x = einops.rearrange(x, "file time channel -> file channel time")
    all_data[angle] = x

# ============================================================================
# DIAGNOSTIC 1: Check if gravity vector actually rotates
# ============================================================================
fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
fig1.suptitle("Diagnostic 1: Gravity Vectors at Different Orientations")

for idx, angle in enumerate(angles):
    ax = axes[idx // 2, idx % 2]
    x = all_data[angle]
    
    # Extract gravity from first 25 samples (stationary period)
    gravity_estimates = x[:, 0:3, :25].mean(dim=2)  # (files, 3)
    
    # Plot each file's gravity estimate
    for i in range(gravity_estimates.shape[0]):
        g = gravity_estimates[i].numpy()
        ax.quiver(0, 0, 0, g[0], g[1], g[2], 
                 color=colors[idx], alpha=0.5, arrow_length_ratio=0.1)
    
    # Mean gravity
    g_mean = gravity_estimates.mean(dim=0).numpy()
    ax.quiver(0, 0, 0, g_mean[0], g_mean[1], g_mean[2],
             color=colors[idx], linewidth=3, arrow_length_ratio=0.1,
             label=f"{angle}°: [{g_mean[0]:.2f}, {g_mean[1]:.2f}, {g_mean[2]:.2f}]")
    
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-12, 0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(f"Gravity at {angle}°")

plt.tight_layout()

# Print gravity magnitudes and XY components
print("\n=== GRAVITY ANALYSIS ===")
for angle in angles:
    x = all_data[angle]
    gravity = x[:, 0:3, :25].mean(dim=2).mean(dim=0)  # Average gravity
    g_xy = torch.sqrt(gravity[0]**2 + gravity[1]**2)
    g_mag = torch.sqrt((gravity**2).sum())
    g_angle = torch.atan2(gravity[1], gravity[0]) * 180 / np.pi
    
    print(f"{angle:3d}°: gravity=[{gravity[0]:6.3f}, {gravity[1]:6.3f}, {gravity[2]:6.3f}], "
          f"XY_mag={g_xy:.3f}, total_mag={g_mag:.3f}, XY_angle={g_angle:.1f}°")

# EXPECTED: If the sensor is truly tilted ~10°, you should see:
# - Gravity XY component rotating: at 0° might be [1.7, 0, -9.65], at 90° should be [0, 1.7, -9.65]
# - If XY components don't rotate, your sensor might not be tilted, or your data is wrong

# ============================================================================
# DIAGNOSTIC 2: Gravity-removed linear acceleration
# ============================================================================
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle("Diagnostic 2: Linear Acceleration (Gravity Removed) - Should Look Similar")

for idx, angle in enumerate(angles):
    ax = axes[idx // 2, idx % 2]
    x = all_data[angle][:, 0:3, :]  # Just accelerometer
    
    # Remove gravity using low-pass filter
    b, a = butter(3, 0.5, btype='low', fs=250)
    
    for i in range(min(3, x.shape[0])):  # Plot first 3 files
        accel = x[i].numpy()
        gravity = filtfilt(b, a, accel, axis=1)
        linear = accel - gravity
        
        # Plot XY trajectory of linear acceleration
        ax.plot(linear[0], linear[1], alpha=0.6, label=f"Sample {i}")
    
    ax.set_xlabel('Linear Accel X')
    ax.set_ylabel('Linear Accel Y')
    ax.set_title(f"{angle}° - Linear Motion (World Frame)")
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

plt.tight_layout()

# EXPECTED: If augmentation works correctly, these trajectories should look 
# like rotated versions of each other. If they're already identical, 
# something is wrong with your data collection.

# ============================================================================
# DIAGNOSTIC 3: Test augmentation on a single sample
# ============================================================================
fig3 = plt.figure(figsize=(16, 10))
fig3.suptitle("Diagnostic 3: Does Augmentation Create Realistic Rotations?")

# Take one sample at 0°
reference_sample = all_data[0][0]  # (6, time)

# Apply your augmentation multiple times
from your_augmentation_module import simple_physical_augmentation

n_augmentations = 8
aug_samples = [simple_physical_augmentation(reference_sample) for _ in range(n_augmentations)]

# Compare initial gravity vectors
ax1 = fig3.add_subplot(2, 3, 1, projection='3d')
ax1.set_title("Initial Gravity Vectors (Augmented)")
for i, aug in enumerate(aug_samples):
    g = aug[0:3, :25].mean(dim=1).numpy()
    ax1.quiver(0, 0, 0, g[0], g[1], g[2], alpha=0.6, arrow_length_ratio=0.1)
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')

# Compare with real gravity vectors at different angles
ax2 = fig3.add_subplot(2, 3, 2, projection='3d')
ax2.set_title("Real Gravity Vectors (Different Orientations)")
for angle, color in zip(angles, colors):
    x = all_data[angle]
    g = x[:, 0:3, :25].mean(dim=2).mean(dim=0).numpy()
    ax2.quiver(0, 0, 0, g[0], g[1], g[2], color=color, 
              arrow_length_ratio=0.1, linewidth=2, label=f"{angle}°")
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
ax2.legend()

# Check if gravity magnitudes are preserved
ax3 = fig3.add_subplot(2, 3, 3)
ax3.set_title("Gravity Magnitude Over Time")
for i, aug in enumerate(aug_samples[:3]):
    g_mag = torch.sqrt((aug[0:3]**2).sum(dim=0)).numpy()
    ax3.plot(g_mag, alpha=0.6, label=f"Aug {i}")
ax3.axhline(y=9.8, color='red', linestyle='--', label='Expected (9.8)')
ax3.set_xlabel('Time'); ax3.set_ylabel('|g|')
ax3.legend()

# Plot XY trajectory of linear acceleration (augmented)
ax4 = fig3.add_subplot(2, 3, 4)
ax4.set_title("Linear Accel XY - Augmented")
b, a = butter(3, 0.5, btype='low', fs=250)
for i, aug in enumerate(aug_samples[:4]):
    accel = aug[0:3].numpy()
    gravity = filtfilt(b, a, accel, axis=1)
    linear = accel - gravity
    ax4.plot(linear[0], linear[1], alpha=0.6, label=f"Aug {i}")
ax4.set_xlabel('X'); ax4.set_ylabel('Y')
ax4.axis('equal'); ax4.grid(True); ax4.legend()

# Plot XY trajectory of linear acceleration (real data)
ax5 = fig3.add_subplot(2, 3, 5)
ax5.set_title("Linear Accel XY - Real Data")
for angle, color in zip(angles, colors):
    x = all_data[angle][0, 0:3, :].numpy()  # First sample
    gravity = filtfilt(b, a, x, axis=1)
    linear = x - gravity
    ax5.plot(linear[0], linear[1], alpha=0.8, color=color, label=f"{angle}°")
ax5.set_xlabel('X'); ax5.set_ylabel('Y')
ax5.axis('equal'); ax5.grid(True); ax5.legend()

# Compare raw acceleration magnitude
ax6 = fig3.add_subplot(2, 3, 6)
ax6.set_title("Total Acceleration Magnitude")
# Augmented
aug_mag = torch.sqrt((aug_samples[0][0:3]**2).sum(dim=0)).numpy()
ax6.plot(aug_mag, label='Augmented', alpha=0.7)
# Real at different angles
for angle, color in zip([0, 90], ['red', 'green']):
    x = all_data[angle][0, 0:3, :]
    mag = torch.sqrt((x**2).sum(dim=0)).numpy()
    ax6.plot(mag, color=color, alpha=0.7, label=f"Real {angle}°")
ax6.set_xlabel('Time'); ax6.set_ylabel('|a|')
ax6.legend()

plt.tight_layout()

# ============================================================================
# DIAGNOSTIC 4: Statistical comparison
# ============================================================================
print("\n=== STATISTICAL COMPARISON ===")

# Extract features from each orientation
def extract_features(x):
    """Extract features that should be rotation-invariant"""
    accel = x[:, 0:3, :]
    
    # Remove gravity
    b, a = butter(3, 0.5, btype='low', fs=250)
    features = []
    
    for i in range(accel.shape[0]):
        a_np = accel[i].numpy()
        gravity = filtfilt(b, a, a_np, axis=1)
        linear = a_np - gravity
        
        # Compute features
        linear_mag = np.sqrt((linear**2).sum(axis=0))
        features.append({
            'linear_mag_mean': linear_mag.mean(),
            'linear_mag_std': linear_mag.std(),
            'linear_mag_max': linear_mag.max(),
            'linear_x_std': linear[0].std(),
            'linear_y_std': linear[1].std(),
            'linear_z_std': linear[2].std(),
        })
    
    return features

for angle in angles:
    features = extract_features(all_data[angle])
    
    # Average over all samples
    avg_features = {k: np.mean([f[k] for f in features]) for k in features[0].keys()}
    
    print(f"\n{angle:3d}°:")
    print(f"  Linear mag: mean={avg_features['linear_mag_mean']:.3f}, "
          f"std={avg_features['linear_mag_std']:.3f}, max={avg_features['linear_mag_max']:.3f}")
    print(f"  Linear XYZ std: X={avg_features['linear_x_std']:.3f}, "
          f"Y={avg_features['linear_y_std']:.3f}, Z={avg_features['linear_z_std']:.3f}")

# EXPECTED: If your sensor is tilted and you're not compensating:
# - linear_x_std and linear_y_std should vary across orientations
# - If they're identical, either: (1) sensor isn't tilted, (2) motion is purely vertical

plt.show()
