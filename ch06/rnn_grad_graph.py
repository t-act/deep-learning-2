import numpy as np
import matplotlib.pyplot as plt

# N: ミニバッチ数, H: 隠れ状態ベクトルの次元数, T: 時系列データの長さ
N, H, T = 2, 3, 20

dh = np.ones((N, H))
np.random.seed(42)
Wh = np.random.randn(H, H)
# Wh = np.random.randn(H, H) / np.sqrt(H)

norm_list = []
for t in range(T):
    dh = np.dot(dh, Wh.T)
    norm = np.sqrt(np.sum(dh**2)) / N
    norm_list.append([t, norm])

norm_list = np.array(norm_list)

plt.plot(norm_list[:,0], norm_list[:,1])
plt.xlabel("time step")
plt.ylabel("norm")
plt.show()