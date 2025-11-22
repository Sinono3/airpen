import numpy as np
import matplotlib.pyplot as plt
import ml.processing as P
import einops

IN = "data/ABCD_smoothed.npz"
OUT = "./data/ABCD_smoothed_no_outliers.npz"

def plot_mean_y(data, title):
    fig, ax = plt.subplots(1, 1)

    acc = 0
    for class_k in data:
        data_k = data[class_k]
        # Place each class side by side
        x = acc + np.arange(data_k.shape[0])
        acc += data_k.shape[0]

        y = data_k[:,:,1].mean(axis=1)
        ax.scatter(x, y, label=class_k)

    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    plt.show()
    plt.close(fig)

data = dict(np.load(IN))

# Delete E and F which are unbalanced classes
del data['E']
del data['F']

# plot_mean_y(data, "Before fixing outliers")
# Row major
ROTATION_MATRIX = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1],
], dtype=np.float32)

# FIX OUTLIERS
for class_k in data:
    data_k = data[class_k]
    mean_y = data_k[:,:,1].mean(axis=1)
    # mean_y 
    outlier_idxs = np.where(mean_y > 0)[0]
    print(f"{class_k}:")
    print(f"Samples: {len(data_k)}")
    print(f"Outliers: {len(outlier_idxs)} ({len(outlier_idxs)/len(data_k)*100:.2f}%)")

    for outlier_idx in outlier_idxs:
        # Multiply by 180-degree rotation matrix
        data_k[outlier_idx] = einops.einsum(ROTATION_MATRIX, data_k[outlier_idx], "rows cols, time rows -> time cols")

    data[class_k] = data_k


plot_mean_y(data, "After fixing outliers")
np.savez(OUT, **data)
print(f"File saved to {OUT}")

# DEBUG: Merge all data
# data = np.concat([data[k] for k in data], axis=0)
# a = len(y)
# b = (y > 0).sum()
# for i in range(100):
#     plt.plot(np.arange(132), data[y > 0][i, :, :])
#     plt.show()
# print(f"{a}")
# print(f"{b}")
# print(f"{b/a}")
# a = data['C'][0, :, :]
# fig, ax = plt.subplots()
# ax.plot(a[:, 0], a[:, 2], color='red')
# plt.show()
