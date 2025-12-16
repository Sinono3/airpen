import torch
import einops
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.animation import FuncAnimation

dir = Path("outputs/rotations/20251209_111619")
angles = [0, 90, 180, 270]
sample_count = 8
colors = ['red', 'green', 'blue', 'orange']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

all_data = []
for angle, color in zip(angles, colors):
    x = np.stack([np.loadtxt(dir / f"rot{angle}_recording{i}.csv", delimiter=",") 
                  for i in range(sample_count)])
    x = torch.from_numpy(x)
    x = einops.rearrange(x, "file time channel -> file channel time")
    x = x[:, 0:3]  # Remove gyro
    all_data.append((x, color, angle))
    
    # Mean vector
    x_mean = x.mean(dim=(0, 2))
    ax.quiver(0, 0, 0, x_mean[0], x_mean[1], x_mean[2],
              color=color, label=f"{angle}°", arrow_length_ratio=0.15, linewidth=2)
    
    # Compute plane using PCA
    points = einops.rearrange(x, "file channel time -> (file time) channel").numpy()
    centroid = points.mean(axis=0)
    points_centered = points - centroid
    _, S, Vt = np.linalg.svd(points_centered)
    
    # Plot first two PCA directions
    scale = 3
    for i in range(2):
        direction = Vt[i] * S[i] * scale / S[0]
        ax.quiver(centroid[0], centroid[1], centroid[2],
                  direction[0], direction[1], direction[2],
                  color=color, alpha=0.6, arrow_length_ratio=0.2, linewidth=1.5)
    
    # Plane
    normal = Vt[2]
    d = -np.dot(normal, centroid)
    xx, yy = np.meshgrid(np.linspace(-8, 8, 10), np.linspace(-8, 8, 10))
    zz = (-normal[0] * xx - normal[1] * yy - d) / (normal[2] + 1e-10)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color=color)

# Animation setup
scatters = [ax.scatter([], [], [], color=color, s=20, alpha=0.7) 
            for _, color, _ in all_data]

def animate(frame):
    for scatter, (x, color, angle) in zip(scatters, all_data):
        # Get all files at this time point
        file_count = x.shape[0]
        time_len = x.shape[2]
        t = frame % time_len
        
        points = x[:, :, t].numpy()  # shape: (files, 3)
        scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
    return scatters

ax.set_xlim(-10.0, 10.0)
ax.set_ylim(-10.0, 10.0)
ax.set_zlim(-10.0, 10.0)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

def fix_roll(event):
    ax.view_init(elev=ax.elev, azim=ax.azim)

fig.canvas.mpl_connect("motion_notify_event", fix_roll)

# Animate (assumes all recordings have same length)
max_frames = all_data[0][0].shape[2]
anim = FuncAnimation(fig, animate, frames=max_frames, interval=50, blit=False)

plt.show()

