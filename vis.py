import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = np.load("ABCDEF.npz")

a = data['C'][0, :, :]
fig, ax = plt.subplots()
ax.plot(a[:, 0], a[:, 2], color='red')
plt.show()
