import numpy as np
import matplotlib.pyplot as plt
import glob, os

markers = ["o", "s", "^", "D"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

npy_dir = os.path.dirname(__file__)
files = sorted(glob.glob(os.path.join(npy_dir, "acc_*.npy")))

for i, f in enumerate(files):
    data = np.load(f)
    label = os.path.basename(f).replace("acc_", "").replace(".npy", "")
    plt.plot(data[:, 0], data[:, 1],
             marker=markers[i % len(markers)],
             color=colors[i % len(colors)],
             label=label)

plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.xlim(0, max(data[:,0]))
plt.ylim(0, 1)
plt.legend()
plt.show()
