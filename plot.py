import numpy as np
import matplotlib.pyplot as plt

data = dict(np.load("data/samples_processed.npz"))
channel_labels = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
arr = data['A'][0, :3, :]

plt.figure(figsize=(8, 4))
for i in range(arr.shape[0]):
    plt.plot(arr[i], label=channel_labels[i])

plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()
