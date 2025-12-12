import math
import torch
import ml.record
import matplotlib.pyplot as plt

X = ml.record.record(1000, torch.device('cpu'))
X = X[1:4]
X = X.mean(dim=1, keepdim=True)
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

