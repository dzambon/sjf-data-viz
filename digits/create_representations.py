import matplotlib.pyplot as plt
from sjf_data_viz.utils.visualization import *
from sklearn import datasets
import numpy as np

n_components=2
only_method="t-SNE"

ds = datasets.load_digits(n_class=10)
X = ds.data
y = ds.target
n_neighbors = 30

z_list = draw_all(X=X, color=y, n_neighbors=n_neighbors, n_components=n_components,
                  only_method=only_method, base_folder="./results")

if only_method is not None:
    pp = np.random.permutation(X.shape[0])[::10]
    imgs = X[pp].reshape(-1, 8, 8)/16.
    z = z_list[0][1][pp]

    if n_components == 3:
        _ = ImageAnnotations3D(z, imgs)
    else:
        ImageAnnotations2D(z, imgs)

plt.savefig("./results/digits.pdf")
plt.show()
