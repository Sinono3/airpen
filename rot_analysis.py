import math
import torch
import ml.record
import einops
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

dir = Path("outputs/rotations/20251209_111619")
# files = [x.parts[-1] for x in dir.iterdir()]
angle = 0
sample_count = 8

X = np.stack(
    [np.loadtxt(dir / f"rot{angle}_recording{i}.csv", delimiter=",") for i in range(sample_count)]
)

X = torch.from_numpy(X)
X = einops.rearrange(X, "file time channel -> file channel time")
X = X[:, 1:4]
X = X.mean(dim=(1, 2), keepdim=True)

length = torch.linalg.vector_norm(X.squeeze(dim=1)).item()
print(f"mean: {X}")
print(f"length of mean: {length}")
angle = torch.arccos(-X[2] / length)
print(f"angle with Z- in radians: {angle}")
print(f"angle with Z- in radians: {(angle * 180) / math.pi}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.quiver(0, 0, 0, X[0,0], X[1,0], X[2,0])
ax.set_xlim(-10.0, 10.0)
ax.set_ylim(-10.0, 10.0)
ax.set_zlim(-10.0, 10.0)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

def fix_roll(event):
    elev = ax.elev
    azim = ax.azim
    ax.view_init(elev=elev, azim=azim)
fig.canvas.mpl_connect("motion_notify_event", fix_roll)
plt.show()

