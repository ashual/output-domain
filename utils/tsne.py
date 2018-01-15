from sklearn import (manifold)


def run(x):
    # t-SNE embedding of the digits dataset
    print('Computing t-SNE embedding')
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_tsne = tsne.fit_transform(x)
    return x_tsne
