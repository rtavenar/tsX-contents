import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tslearn.metrics import dtw_path

length = 20

fig = plt.figure(figsize=(12, 4))
ax2 = plt.subplot(1, 2, 1)
ax = plt.subplot(1, 2, 2)
for cur_ax in fig.axes:
    cur_ax.set_facecolor(fig.patch.get_facecolor())
colors = sns.color_palette("Paired")

shift = 4

x_ref = np.zeros((length+shift, ))
x_ref[:length] = np.sin(np.linspace(0, 2 * np.pi, num=length))
x_ref[:3*length//4] = 0.

x = np.zeros((length+shift, ))
x[shift:shift+length] = np.sin(np.linspace(0, 2 * np.pi, num=length))
x[:3*length//4+shift] = 0.

x, x_ref = x_ref, x

path, _ = dtw_path(x_ref[3*length//4-1:], x[3*length//4-1:])

x -= 3

ax.plot(x_ref, color=(106/255, 177/255, 208/255), linestyle='-', marker='o', zorder=1)
ax.plot(x, color=(106/255, 177/255, 208/255), linestyle='-', marker='o', zorder=1)
ax.plot(x_ref[:3*length//4-1], color=(135/255, 108/255, 173/255), linestyle='-', marker='o', zorder=2)
ax.plot(x[:3*length//4-1], color=(135/255, 108/255, 173/255), linestyle='-', marker='o', zorder=2)
for idx, (i, j) in enumerate(path):
    ax.plot([3*length//4-1+i, 3*length//4-1+j], [x_ref[3*length//4-1+i], x[3*length//4-1+j]], color='k', alpha=.2, zorder=0)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Dynamic Time Warping")

ax2.plot(x_ref, color=(106/255, 177/255, 208/255), linestyle='-', marker='o', zorder=1)
ax2.plot(x, color=(106/255, 177/255, 208/255), linestyle='-', marker='o', zorder=1)
ax2.plot(x_ref[:3*length//4-1], color=(135/255, 108/255, 173/255), linestyle='-', marker='o', zorder=2)
ax2.plot(x[:3*length//4-1], color=(135/255, 108/255, 173/255), linestyle='-', marker='o', zorder=2)
for idx in range(3*length//4-1, len(x_ref)):
    ax2.plot([idx, idx], [x_ref[idx], x[idx]], color='k', alpha=.2, zorder=0)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title("Mean Squared Error")

plt.tight_layout()
plt.savefig('slides/fig/dtw_mse.svg')
