import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import train_test_split
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances

sns.set_style('ticks')


def plot_lsi(svd_trans, genres):
    genres = np.array(genres)
    genre_sel = np.not_equal(genres, None)
    X, y = svd_trans[genre_sel], genres[genre_sel]
    plot_embedding(X, y)


def plot_mds(points, genres, n_points=500):
    genres = np.array(genres)
    genre_sel = np.not_equal(genres, None)
    X, y = points[genre_sel], genres[genre_sel]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, train_size=n_points)

    distances = cosine_distances(X_train, X_train)
    mds = MDS(n_components=2, dissimilarity='precomputed')
    mds.fit(distances)

    plot_embedding(mds.embedding_, y_train)


def plot_embedding(embedding, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = sns.color_palette('Set1', len(np.unique(labels)))
    for i, label in enumerate(np.sort(np.unique(labels))):
        cur_X = embedding[labels == label]
        ax.add_line(plt.Line2D(cur_X[:, 0], cur_X[:, 1], color=colors[i],
                               linestyle='none', marker='o', alpha=0.75))
    ax.set_xlim(1.1*np.min(embedding[:, 0], 1.1*np.max(embedding[:, 0])))
    ax.set_ylim(1.1*np.min(embedding[:, 1], 1.1*np.max(embedding[:, 1])))
    sns.despine(offset=5, trim=True)
    ax.set_xlabel('MDS dim 1')
    ax.set_ylabel('MDS dim 2')
    ax.legend(np.sort(np.unique(labels)))
