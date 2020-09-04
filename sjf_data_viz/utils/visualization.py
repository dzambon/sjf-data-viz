from sklearn import manifold
import umap

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D

import os
from functools import partial
from collections import OrderedDict
import pickle
from time import time
import numpy as np

def draw_all(X, color, n_neighbors=10, only_method=None, base_folder=".", n_components=2):
    
    LLE = partial(manifold.LocallyLinearEmbedding,
                  n_neighbors, n_components, eigen_solver='auto')

    methods = OrderedDict()
    methods['LLE'] = LLE(method='standard')
    methods['LTSA'] = LLE(method='ltsa')
    methods['Hessian LLE'] = LLE(method='hessian')
    methods['Modified LLE'] = LLE(method='modified')
    methods['Isomap'] = manifold.Isomap(n_neighbors, n_components)
    methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)
    methods['SE'] = manifold.SpectralEmbedding(n_components=n_components,
                                               n_neighbors=n_neighbors)
    methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca',
                                     random_state=0)
    methods['UMAP'] = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors)

    if only_method is not None:
        m = OrderedDict()
        m[only_method] = methods[only_method]
        methods = m

    # Create figure
    fig = plt.figure(figsize=(12, 12))

    # Plot results
    z_list = []
    for i, (method_name, method) in enumerate(methods.items()):

        # Init the correct subplot grid and axes
        if only_method is None:
            if n_components == 3:
                ax = fig.add_subplot(3, 3, i+1, projection=Axes3D.name)
            else:                
                ax = fig.add_subplot(3, 3, i+1)
        else:
            if n_components == 3:
                ax = fig.add_subplot(111, projection=Axes3D.name)
            else:                
                ax = fig.add_subplot(111)
        
        # Compute or load data
        fname = base_folder + "/" + method_name + ".pkl"
        if os.path.isfile(fname):
            #load
            with open(fname, 'rb') as f:
                [z, dt] = pickle.load(f)
        else:
            #compute
            try:
                t0 = time()
                z = method.fit_transform(X)
                t1 = time()
                dt = t1 - t0
            except Exception as e:
                import traceback
                traceback.print_exc()
                dt, z = -1, None
            if dt > 0:
                #save
                with open(fname, 'wb') as f:  pickle.dump([z, dt], f)

        # Draw representations
        print("{}: {:.2g} sec".format(method_name, dt))
        scatter = ax.scatter(z[:, 0], z[:, 1], c=color, marker=".")
        legend1 = ax.legend(*scatter.legend_elements(), title="Digit")
        ax.add_artist(legend1)
        ax.set_title("{} ({:.2g} sec)".format(method_name, dt))
        ax.axis('tight')
        if only_method is None:
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())

        # Store representations to be returned
        z_list.append((method_name, z))

    return z_list

from mpl_toolkits.mplot3d import proj3d
from matplotlib import offsetbox

class ImageAnnotations2D(object):
    """ Displays images directly in the plot. """

    def __init__(self, xy, imgs, ax2d=None, cmap=plt.get_cmap("Greys")):

        # store attributes
        self.imgs = imgs
        self.ax2d = plt.gca() if ax2d is None else ax2d 
        self.cmap = cmap
        self.xy = xy

        # draw images
        self.annot = []
        for s,im in zip(self.xy, self.imgs):
            x, y = s
            self.annot.append(self.image(im,[x,y]))
        #     ab_list.append(add_image(x=X[p], z=z_list[0][1][p], ax=ax))

    def image(self, img, xy):
        """ Draw image to location xy """
        imagebox = offsetbox.OffsetImage(img, zoom=2, cmap=self.cmap)
        imagebox.image.axes = self.ax2d
        ab = offsetbox.AnnotationBbox(imagebox, xy, xycoords='data', pad=0.0, frameon=True )
        self.ax2d.add_artist(ab)
        return ab
 

class ImageAnnotations3D(object):
    """ Displays images directly in the plot. """

    def __init__(self, xyz, imgs, ax3d=None, ax2d=None, cmap=plt.get_cmap("Greys")):

        # store attributes
        self.xyz = xyz
        self.imgs = imgs
        self.cmap = cmap
        self.ax3d = plt.gca() if ax3d is None else ax3d 
        if ax2d is None:
            ax2d = plt.gcf().add_subplot(111,frame_on=False) 
            ax2d.axis("off")
        self.ax2d = ax2d

        # draw images
        self.annot = []
        for s,im in zip(self.xyz, self.imgs):
            x,y = self.proj(s)
            self.annot.append(self.image(im,[x,y]))
        self.lim = self.ax3d.get_w_lims()
        self.rot = self.ax3d.get_proj()
        self.cid = self.ax3d.figure.canvas.mpl_connect("draw_event",self.update)

        # register events
        self.funcmap = {"button_press_event" : self.ax3d._button_press,
                        "motion_notify_event" : self.ax3d._on_move,
                        "button_release_event" : self.ax3d._button_release}
        self.cfs = [self.ax3d.figure.canvas.mpl_connect(kind, self.cb) \
                        for kind in self.funcmap.keys()]

    def cb(self, event):
        event.inaxes = self.ax3d
        self.funcmap[event.name](event)

    def proj(self, X):
        """ From a 3D point in axes ax1, calculate position in 2D in ax2 """
        x, y, z = X
        x2, y2, _ = proj3d.proj_transform(x,y,z, self.ax3d.get_proj())
        tr = self.ax3d.transData.transform((x2, y2))
        return self.ax2d.transData.inverted().transform(tr)

    def image(self,arr,xy):
        """ Place an image (arr) as annotation at position xy """
        im = offsetbox.OffsetImage(arr, zoom=2, cmap=self.cmap)
        im.image.axes = self.ax3d
        ab = offsetbox.AnnotationBbox(im, xy, #xybox=(-30., 30.),
                            xycoords='data', boxcoords="offset points",
                            pad=0.0,) # arrowprops=dict(arrowstyle="->"))
        self.ax2d.add_artist(ab)

        return ab

    def update(self,event):
        """ Update drawing """
        if np.any(self.ax3d.get_w_lims() != self.lim) or \
                        np.any(self.ax3d.get_proj() != self.rot):
            self.lim = self.ax3d.get_w_lims()
            self.rot = self.ax3d.get_proj()
            for s,ab in zip(self.xyz, self.annot):
                ab.xy = self.proj(s)
