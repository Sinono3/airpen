import numpy as np
import matplotlib.pyplot as plt
import einops
from scipy.signal import butter, filtfilt

IN = "./samples.npz"
OUT = "./samples_processed.npz"

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

def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    filtered = filtfilt(b, a, data, axis=1)
    return filtered

data = dict(np.load(IN))

# ---------------------------------------------
print("STEP1: Delete unused classes")
unused_classes = []
# unused_classes = ['E', 'F']
print(f"Deleting unused classes: {unused_classes}")
for class_idx in unused_classes:
    del data[class_idx]

# ---------------------------------------------
print("STEP2: Apply smoothing with lowpass filter")
for class_k in data:
    data[class_k] = lowpass_filter(data[class_k], cutoff=5, fs=50)

# # ---------------------------------------------
# print("STEP3: Fix outliers")
# # (rows, cols)
# ROTATION_MATRIX = np.array([
#     [1, 0, 0],
#     [0, -1, 0],
#     [0, 0, -1],
# ], dtype=np.float32)

# # DEBUG: plot before
# plot_mean_y(data, "Before fixing outliers")

# for class_k in data:
#     data_k = data[class_k]
#     mean_y = data_k[:,:,1].mean(axis=1)
#     # mean_y > 0.0 --> outlier
#     outlier_idxs = np.where(mean_y > 0.0)[0]

#     print(f"{class_k}:")
#     print(f"- Samples: {len(data_k)}")
#     print(f"- Outliers: {len(outlier_idxs)} ({len(outlier_idxs)/len(data_k)*100:.2f}%)")

#     for outlier_idx in outlier_idxs:
#         # Multiply by 180-degree rotation matrix to the accelerometer cols
#         data_k[outlier_idx, :, :3] = einops.einsum(ROTATION_MATRIX, data_k[outlier_idx, :, :3], "rows cols, time rows -> time cols")

#     data[class_k] = data_k

# # DEBUG: plot after
# plot_mean_y(data, "After fixing outliers")
# exit()

# # ---------------------------------------------
# ROTATIONS = 24
# DEGREES = 360 / ROTATIONS
# print(f"STEP4: Augment with {ROTATIONS} rotation ({DEGREES} degrees per rotation)")

# for class_k in data:
#     new_data = []
#     data_k = data[class_k]

#     for rot_i in range(ROTATIONS):
#         rot_deg = ROTATIONS * DEGREES
#         rot_rad = (rot_deg / 180) * np.pi
#         rot_matrix = np.array([
#             [np.cos(rot_rad), 0, np.sin(rot_rad)],
#             [0, 1, 0],
#             [-np.sin(rot_rad), 0, np.cos(rot_rad)],
#         ])

#         new_data_i = np.copy(data_k)
#         # Only for accelerometer, thus the :3
#         new_data_i[:, :, :3] = einops.einsum(rot_matrix, data_k[:, :, :3], "rows cols, batch time rows -> batch time cols")
#         new_data.append(new_data_i)

#     new_data = np.concatenate(new_data, axis=0)
#     data[class_k] = new_data

# ---------------------------------------------
print("STEP5: (time, channel) -> (channel, time)")
for class_k in data:
    data[class_k]  = einops.rearrange(data[class_k], "sample time channel -> sample channel time")

# ---------------------------------------------
print(f"Saving output to {OUT}...")
np.savez(OUT, **data)
print(f"Output saved to {OUT}")
