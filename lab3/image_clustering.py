from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt
import numpy as np
import imageio
from sklearn.cluster import KMeans

sample_img = np.array(imageio.imread('./datasets/task3sample.jpg'))
print(sample_img.shape)

data = sample_img / 255.0
data = data.reshape(sample_img.shape[0] * sample_img.shape[1], 3)


def plot_pixels(data, title, colors=None, N=10000):
    if colors is None:
        colors = data
    
    # choose a random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle(title, size=20);
    plt.show()

plot_pixels(data, title='Input color space: 8 million possible colors')    

import warnings; warnings.simplefilter('ignore')  # Fix NumPy issues.

kmeans = KMeans(n_clusters=8)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

plot_pixels(data, colors=new_colors,title="Reduced color space: 8 colors")

sample_img_recolored = new_colors.reshape(sample_img.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(sample_img)
ax[0].set_title('Original Image', size=16)
ax[1].imshow(sample_img_recolored)
ax[1].set_title('8-color Image', size=16);
plt.show()