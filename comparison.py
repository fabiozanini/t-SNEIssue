# vim: fdm=indent
'''
author:     Fabio Zanini
date:       17/02/17
content:    Test bhtsne versus sklearn
'''
# Modules
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from bhtsne import tsne
import matplotlib.pyplot as plt


# Script
if __name__ == '__main__':

    iris = load_iris()
    data = iris.data

    perplexity = 30
    random_seed = 0

    fig, axs = plt.subplots(3, 1, figsize=(9, 22))

    # bhtsne
    Ybh = tsne(iris.data, dimensions=2, perplexity=perplexity, rand_seed=random_seed)

    # sklearn eta=1000
    model = TSNE(n_components=2, random_state=random_seed, perplexity=perplexity,
                 learning_rate=1000,
                 metric='euclidean')
    Ysk = model.fit_transform(iris.data)

    # sklearn eta=200
    model = TSNE(n_components=2, random_state=random_seed, perplexity=perplexity,
                 learning_rate=200,
                 metric='euclidean')
    Ysk_200 = model.fit_transform(iris.data)

    axs[0].scatter(Ybh[:, 0], Ybh[:, 1], c=iris.target)
    axs[0].set_title('bhtsne')
    axs[1].scatter(Ysk[:, 0], Ysk[:, 1], c=iris.target)
    axs[1].set_title('sklearn.manifold.TSNE (eta = 1000)')
    axs[2].scatter(Ysk_200[:, 0], Ysk_200[:, 1], c=iris.target)
    axs[2].set_title('sklearn.manifold.TSNE (eta = 200)')

    plt.tight_layout()
    plt.savefig('comparison_bhtsne_iris.png')

    plt.ion()
    plt.show()
