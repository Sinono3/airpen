import torch
import numpy as np
import matplotlib.pyplot as plt

data = dict(np.load("data/ABCDEF_noaug.npz"))
classes = ['A', 'B', 'C', 'D']
data = {c: torch.from_numpy(data[c]) for c in classes}

for c in classes:
    # Normalize samples
    data[c] = (data[c] - data[c].mean(dim=2, keepdim=True)) / data[c].std(dim=2, keepdim=True)
    # Only XZ
    data[c] = data[c][:, [0, 2], :]

def show_static(sample):
    fig, ax = plt.subplots()
    ax.set_xlim(sample[0].min(), sample[0].max())
    ax.set_ylim(sample[1].min(), sample[1].max())
    ax.plot(sample[0], sample[1], color='red')
    plt.show()

def show_animated(sample, title):
    T = sample.shape[1]
    fig, ax = plt.subplots()
    ax.set_xlim(sample[0].min(), sample[0].max())
    ax.set_ylim(sample[1].min(), sample[1].max())

    for t in range(T):
        ax.clear()
        ax.plot(sample[0, :t+1], sample[1, :t+1], color='red')
        ax.set_title(f"{title}: {t}/{T}  {t/66}/{T/66}")
        ax.set_xlim(sample[0].min(), sample[0].max())
        ax.set_ylim(sample[1].min(), sample[1].max())
        plt.pause(0.000001)

    plt.show()

# Calculate integrated
dt = 1.0 / 50.0
data_vel, data_pos = {}, {}

for c in classes:
    acc = data[c]
    v = torch.zeros_like(acc)
    v[..., 1:] = torch.cumsum((acc[..., 1:] + acc[..., :-1]) * 0.5 * dt, dim=-1)

    pos = torch.zeros_like(acc)
    pos[..., 1:] = torch.cumsum((v[..., 1:] + v[..., :-1]) * 0.5 * dt, dim=-1)

    data_vel[c] = v                      # (N, 2, T)
    data_pos[c] = pos   # final positions (N, 2, T)

for i in range(20):
    show_animated(data_pos['D'][i], f"Sample {i}")
