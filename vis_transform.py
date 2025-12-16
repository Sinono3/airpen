import torch
import matplotlib.pyplot as plt
import numpy as np
import ml.processing as processing

def visualize_augmentation_pipeline(
    sample: torch.Tensor,  # (6, 500)
    augment: bool = True,
    smooth: bool = True,
    normalize: str | None = 'sample',  # None, 'sample'
    ignore_gyro: bool = True,
):
    """Visualize each step of the augmentation pipeline."""
    
    gravity = torch.tensor(processing.GRAVITY)
    eps = 1e-10
    dt = 1/250  # Sampling period
    
    # Store each step for plotting
    steps = {}
    
    # Step 0: Original
    steps['0. Original'] = sample[:3].clone()
    
    # Step 1: Remove gravity constant
    sample = processing.remove_gravity_constant(sample)
    steps['1. Gravity removed'] = sample[:3].clone()
    
    # Step 2: Random rotation (optional)
    if augment:
        sample[:3] = processing.random_rotation(sample[:3], axis=gravity, angle_rad_std=0.02)
        steps['2. Rotated'] = sample[:3].clone()
    
    # Step 3: Align to plane
    sample[:3] = processing.align_to_plane(sample[:3], gravity)
    steps['3. Aligned to plane'] = sample[:3].clone()
    
    # Step 4: Smooth (optional)
    if smooth:
        sample = processing.smooth(sample)
        steps['4. Smoothed'] = sample[:3].clone()
    
    # Step 5: Normalize (optional)
    if normalize == 'sample':
        sample = (sample - sample.mean(dim=1, keepdim=True)) / (sample.std(dim=1, keepdim=True) + eps)
        steps['5. Normalized'] = sample[:3].clone()
    
    # Create figure with two rows
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1])
    
    # Left column: Time series plots
    axes_time = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    
    # Right column: 3D trajectory
    ax_3d = fig.add_subplot(gs[:, 1], projection='3d')
    
    colors = plt.cm.tab10(range(len(steps)))
    
    # Plot time series
    for ax, axis_name in zip(axes_time, ['X', 'Y', 'Z']):
        for (step_name, data), color in zip(steps.items(), colors):
            axis_idx = ['X', 'Y', 'Z'].index(axis_name)
            ax.plot(data[axis_idx].numpy(), label=step_name, color=color, alpha=0.7, linewidth=1.5)
        
        ax.set_ylabel(f'Accel {axis_name}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    axes_time[-1].set_xlabel('Time (samples)', fontsize=12)
    
    # Plot 3D trajectories
    for (step_name, accel_data), color in zip(steps.items(), colors):
        # Double integrate to get position
        accel = accel_data.numpy()  # (3, T)
        
        # # Integrate acceleration -> velocity
        # velocity = np.cumsum(accel, axis=1) * dt
        
        # # Integrate velocity -> position
        # position = np.cumsum(velocity, axis=1) * dt
        
        # Plot trajectory
        ax_3d.plot(accel[0], accel[1], accel[2], 
                   label=step_name, color=color, alpha=0.7, linewidth=2)
        
        # Mark start and end points
        ax_3d.scatter(accel[0, 0], accel[1, 0], accel[2, 0], 
                     color=color, s=100, marker='o', alpha=0.8)
        ax_3d.scatter(accel[0, -1], accel[1, -1], accel[2, -1], 
                     color=color, s=100, marker='s', alpha=0.8)
    
    ax_3d.set_xlabel('X (m)', fontsize=11)
    ax_3d.set_ylabel('Y (m)', fontsize=11)
    ax_3d.set_zlabel('Z (m)', fontsize=11)
    ax_3d.set_title('Acceleration (○=start, □=end)', fontsize=12)
    ax_3d.legend(fontsize=9, loc='upper left')
    ax_3d.grid(True, alpha=0.3)
    
    # Make aspect ratio equal
    max_range = np.array([
        accel[0].max() - accel[0].min(),
        accel[1].max() - accel[1].min(),
        accel[2].max() - accel[2].min()
    ]).max() / 2.0
    
    mid_x = (accel[0].max() + accel[0].min()) * 0.5
    mid_y = (accel[1].max() + accel[1].min()) * 0.5
    mid_z = (accel[2].max() + accel[2].min()) * 0.5
    
    ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.suptitle(
        f'Augmentation Pipeline (augment={augment}, smooth={smooth}, normalize={normalize})',
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Load a sample
    import numpy as np
    data = np.load('data/samples_processed_split.npz')
    
    # Visualize with different settings
    # visualize_augmentation_pipeline(sample, augment=True, smooth=True, normalize='sample')
    # visualize_augmentation_pipeline(sample, augment=False, smooth=True, normalize='sample')
    for i in range(10):
        sample = torch.from_numpy(data['train_x'][i]).float()  # (6, 500)
        visualize_augmentation_pipeline(sample, augment=False, smooth=False, normalize=None)
