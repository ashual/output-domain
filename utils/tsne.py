from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import (manifold)


# Scale and visualize the embedding vectors
def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    fig = plt.figure()
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]), color=plt.cm.Set1(y[i] / 10.), fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    return fig


def run(x, y):
    # t-SNE embedding of the digits dataset
    print('Computing t-SNE embedding')
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    x_tsne = tsne.fit_transform(x)

    fig = plot_embedding(x_tsne, y, "t-SNE embedding of the digits (time %.2fs)" % (time() - t0))
    return fig
