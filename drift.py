import torch
import ml.record
import matplotlib.pyplot as plt

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

X = ml.record.record(1000, torch.device('cpu'))
X = X[1:4]
X = X.mean(dim=1, keepdim=True)
length = torch.linalg.vector_norm(X.squeeze(dim=1)).item()
print(f"mean: {X}")
print(f"length of mean: {length}")

rotation = rotation_matrix_from_vectors

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.quiver(0, 0, 0, X[0,0], X[1,0], X[2,0])
ax.set_xlim(-10.0, 10.0)
ax.set_ylim(-10.0, 10.0)
ax.set_zlim(-10.0, 10.0)

def fix_roll(event):
    elev = ax.elev
    azim = ax.azim
    ax.view_init(elev=elev, azim=azim)
fig.canvas.mpl_connect("motion_notify_event", fix_roll)
plt.show()

