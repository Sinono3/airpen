import torch
import einops
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.signal import butter, filtfilt

def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    filtered = filtfilt(b, a, data, axis=1)
    return filtered

def generate_gradient(N=256, cmap='viridis'):
    cmap = plt.get_cmap(cmap)
    gradient = cmap(np.linspace(0, 1, N))  # shape: (N, 4)
    return gradient

def lines_to_segments(points):
    return einops.rearrange(
        [points[:-1], points[1:]], "startend time coord -> time startend coord"
    )

labels = ['A', 'B', 'C', 'D']

chosen = 'A'
data = dict(np.load("data/ABCDEF.npz"))
data = data[chosen][:,:,:3]
data2 = torch.load(f"./ml/recordings/{chosen}.pt").cpu().numpy()
data2 = einops.rearrange(data2, 'c t -> 1 t c')

data = lowpass_filter(data, cutoff=5, fs=50)
data2 = lowpass_filter(data2, cutoff=5, fs=50)

fig = plt.figure()
ax: Axes3D = fig.add_subplot(1, 1, 1, projection='3d')

lines1 = lines_to_segments(data[0, :, :3])
grad1 = generate_gradient(lines1.shape[0], 'spring')
lines2 = lines_to_segments(data2[0, :, :3])
grad2 = generate_gradient(lines2.shape[0], 'winter')
lines1 = Line3DCollection(lines1)
lines1.set_colors(grad1)
lines2 = Line3DCollection(lines2)
lines2.set_colors(grad2)
sm1 = plt.cm.ScalarMappable(cmap='spring')
sm2 = plt.cm.ScalarMappable(cmap='winter')
fig.colorbar(sm1, ax=ax)
fig.colorbar(sm2, ax=ax)
Axes3D.add_collection3d(ax, lines1)
Axes3D.add_collection3d(ax, lines2)

ax.set_xlim(  np.min(data[:, :, 0]), np.max(data[:, :, 0]))
ax.set_ylim(np.min(data[:, :, 1]), np.max(data[:, :, 1]))
ax.set_zlim(np.min(data[:, :, 2]), np.max(data[:, :, 2]))
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

fig.show()
plt.show()
