"""
=========================================
Segmenting the picture of Lena in regions
=========================================

This example uses :ref:`spectral_clustering` on a graph created from
voxel-to-voxel difference on an image to break this image into multiple
partly-homogeneous regions.

This procedure (spectral clustering on an image) is an efficient
approximate solution for finding normalized graph cuts.

There are two options to assign labels:

* with 'kmeans' spectral clustering will cluster samples in the embedding space
  using a kmeans algorithm
* whereas 'discrete' will iteratively search for the closest partition
  space to the embedding space.
"""
print(__doc__)

# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>, Brian Cheung
# License: BSD 3 clause

import time

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

import skimage
from skimage import data
from skimage import transform
from skimage import io
from scipy import misc

lena = io.imread('example.png',as_grey=True)[::2, ::2]
lena = skimage.transform.rescale(lena, 0.5, order=1)


from skimage import data
from skimage import filter
from skimage import img_as_float
import matplotlib.pyplot as plt


unsharp_strength = 0.8
blur_size = 8  # Standard deviation in pixels.

# Convert to float so that negatives don't cause problems
image = img_as_float(lena)
blurred = filter.gaussian_filter(image, blur_size)
highpass = image - unsharp_strength * blurred
sharp = image + highpass


fig, axes = plt.subplots(ncols=2)
axes[0].imshow(image, vmin=0, vmax=1)
axes[1].imshow(sharp, vmin=0, vmax=1)
plt.show()


#lena = sp.misc.lena()
# Downsample the image by a factor of 4
#lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
#lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]

# Convert the image into a graph with the value of the gradient on the
# edges.
graph = image.img_to_graph(image)

# Take a decreasing function of the gradient: an exponential
# The smaller beta is, the more independent the segmentation is of the
# actual image. For beta=1, the segmentation is close to a voronoi
beta = 5
eps = 1e-6
graph.data = np.exp(-beta * graph.data / lena.std()) + eps

# Apply spectral clustering (this step goes much faster if you have pyamg
# installed)
N_REGIONS = 7

###############################################################################
# Visualize the resulting regions

for assign_labels in ('kmeans', 'discretize'):
    t0 = time.time()
    labels = spectral_clustering(graph, n_clusters=N_REGIONS,
                                 assign_labels=assign_labels,
                                 random_state=1)
    t1 = time.time()
    labels = labels.reshape(lena.shape)

    plt.figure(figsize=(5, 5))
    plt.imshow(lena,   cmap=plt.cm.gray)
    for l in range(N_REGIONS):
        plt.contour(labels == l, contours=1,
                    colors=[plt.cm.spectral(l / float(N_REGIONS)), ])
    plt.xticks(())
    plt.yticks(())
    plt.title('Spectral clustering: %s, %.2fs' % (assign_labels, (t1 - t0)))

plt.show()
