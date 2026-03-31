import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset import spiral
import matplotlib.pyplot as plt

x, t = spiral.load_data()
t_label = t.argmax(axis=1)

for cls in range(3):
    mask = t_label == cls
    plt.scatter(x[mask, 0], x[mask, 1], label=f'class {cls}')

plt.show()