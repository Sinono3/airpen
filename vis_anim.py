import einops
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Path3DCollection
from scipy.signal import butter, filtfilt

labels = ['A', 'B', 'C', 'D', 'E', 'F']

# Load data
data = np.load("ABCDEF.npz")
full = []
classes = []

for cls_idx, letter in enumerate(labels):
    for sample in data[letter]:
        full.append(sample)
        classes.append(cls_idx)

full = np.stack(full)
classes = np.stack(classes)

print(full.shape)
print(classes.shape)

def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    filtered = filtfilt(b, a, data, axis=1)
    return filtered

full = lowpass_filter(full, cutoff=5, fs=50)

# mean_A = np.mean(data['A'], axis=(0, 1))
# mean_B = np.mean(data['B'], axis=(0, 1))
# mean_C = np.mean(data['C'], axis=(0, 1))
# mean_D = np.mean(data['D'], axis=(0, 1))
# mean_E = np.mean(data['E'], axis=(0, 1))
# mean_F = np.mean(data['F'], axis=(0, 1))
# print(f"{mean_A}")
# print(f"{mean_B}")
# print(f"{mean_C}")
# print(f"{mean_D}")
# print(f"{mean_E}")
# print(f"{mean_F}")

def generate_gradient(N=256):
    cmap = plt.get_cmap('viridis')
    gradient = cmap(np.linspace(0, 1, N))  # shape: (N, 4)
    return gradient

def lines_to_segments(points):
    return einops.rearrange([points[:-1], points[1:]], "startend time coord -> time startend coord")
    # return np.concatenate([points[:-1], points[1:]], axis=1)

fig = plt.figure()
ax: Axes3D = fig.add_subplot(1, 1, 1, projection='3d')
# points: Path3DCollection = ax.scatter(lin)  # Points
print(full[0, :, :3].shape)
print(lines_to_segments(full[0, :, :3]).shape)

lines = Line3DCollection(lines_to_segments(full[0, :, :3]), cmap="viridis")
fig.colorbar(lines,ax=ax)

Axes3D.add_collection3d(ax, lines)
ax.set_xlim(np.min(full[:, :, 0]), np.max(full[:, :, 0]))
ax.set_ylim(np.min(full[:, :, 1]), np.max(full[:, :, 1]))
ax.set_zlim(np.min(full[:, :, 2]), np.max(full[:, :, 2]))
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

access = np.arange(full.shape[0])
np.random.shuffle(access)

def update(i):
    sample_idx = access[i]
    sample = full[sample_idx, :, :]
    sample_lines = lines_to_segments(sample[:, :3])
    gradient = generate_gradient(sample_lines.shape[0])
    lines.set_segments(sample_lines)
    lines.set_colors(gradient)
    # ax.set_xlim(np.min(full[:, :, 0]), np.max(full[:, :, 0]))
    # ax.set_ylim(np.min(full[:, :, 1]), np.max(full[:, :, 1]))
    # ax.set_zlim(np.min(full[:, :, 2]), np.max(full[:, :, 2]))
    ax.set_xlim(np.min(sample[:, 0]), np.max(sample[:, 0]))
    ax.set_ylim(np.min(sample[:, 1]), np.max(sample[:, 1]))
    ax.set_zlim(np.min(sample[:, 2]), np.max(sample[:, 2]))
    title = f"class: {labels[classes[sample_idx]]}"
    ax.set_title(title, fontsize=16)
    print(title)

# Create animation
ani = FuncAnimation(
    fig,
    update,
    frames=range(full.shape[0]),
    interval=4000  # milliseconds per frame (0.5 sec)
)

plt.show()
