"""
Checkout also these:
- https://distill.pub/2016/misread-tsne/
"""

from sklearn import datasets
from sjf_data_viz.utils.representations import JumpingTSNE

# Create representation with our modified TSNE
n_components=2

ds = datasets.load_digits(n_class=10)
X = ds.data#[:100]
y = ds.target#[:100]
n_neighbors = 30

jtsne = JumpingTSNE(jump_size=6, n_components=n_components, init='random', random_state=0)

print("Learning the representations")
z = jtsne.fit_transform(X)
z_list = jtsne.X_embedded_jumps
n_exploration, n_finetuning = jtsne.num_jumps_per_type()


# Create animation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

fig, ax = plt.subplots()
print("Preparing animation...")

#base scatter plot
scatter = ax.scatter(z[:, 0], z[:, 1], c=y)
legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Digit")
ax.add_artist(legend1)

#function to edit scatter plot
def animate(i):
    scatter.set_offsets(z_list[i])
    ax.set_title("Exploration {iexp}/{exptot} | Fine tuning {ift}/{fttot}".format(
        iexp=min([i, n_exploration]), exptot=n_exploration, 
        ift=max([i-n_exploration, 0]), fttot=n_finetuning))
    return scatter

#animation
ani = animation.FuncAnimation(
    fig, animate, frames=len(z_list), interval=10, blit=False, save_count=50) #, init_func=init)
writer = animation.FFMpegWriter(fps=5, bitrate=1800)
ani.save("results/viz_learning.mp4", writer=writer)

plt.show()