import ml.record as record
import ml.processing as processing
import torch
import numpy as np
import matplotlib.pyplot as plt

for i in range(1):
    record.countdown("recording", 1)
    x = torch.arange(500)
    y = record.record(500, device=torch.device('cpu'))
    y =  y - y.mean(dim=1, keepdim=True)
    y = processing.smooth(y)
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(x, y[0, :].numpy(), color='red', label='Smoothed (x)')
    # ax.plot(x, y[1, :].numpy(), color='green', label='Smoothed (y)')
    # ax.plot(x, y[2, :].numpy(), color='blue', label='Smoothed (z)')
    # ax.set_yticks(np.linspace(-20, 20, 20))
    # ax.legend()
    # plt.show()

    y_pca = processing.pca_transform_3_handedness(y[:3])
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(x, y_pca[0, :].numpy(), color='red', label='PCA (x)')
    # ax.plot(x, y_pca[1, :].numpy(), color='green', label='PCA (y)')
    # ax.plot(x, y_pca[2, :].numpy(), color='blue', label='PCA (z)')
    # ax.set_yticks(np.linspace(-20, 20, 20))
    # ax.legend()
    # plt.show()   

    # pca_vel = torch.diff(y_pca[:2], dim=1)  # (2, t-1)
    # pca_speed = torch.norm(pca_vel, dim=0)  # (t-1,)
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(x[:-1], pca_speed.numpy(), color='red', label='PCA speed')
    # ax.set_yticks(np.linspace(-3, 3, 20))
    # ax.legend()
    # plt.show()

    y_pca_aligned = processing.align_to_first_movement(y[:3])
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y_pca_aligned[0, :].numpy(), color='red', label='PCA+align (x)')
    ax.plot(x, y_pca_aligned[1, :].numpy(), color='green', label='PCA+align (y)')
    ax.plot(x, y_pca_aligned[2, :].numpy(), color='blue', label='PCA+align (z)')
    ax.set_yticks(np.linspace(-20, 20, 20))
    ax.legend()
    plt.show()   

    
    # y_norm = torch.norm(y[:], dim=0)
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(x, y_norm.numpy(), color='red', label='norm')
    # ax.set_yticks(np.linspace(-20, 20, 20))
    # ax.legend()
    # plt.show()
