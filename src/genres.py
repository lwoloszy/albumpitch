import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import train_test_split
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

sns.set_style('ticks')


def plot_lsi(svd_trans, genres):
    '''
    Plots a set of documents in the LSI space

    Args:
        svd_trans: dense array with svd transformed data
        genres: list of genres for each entry in svd_trans
    Returns:
        None
    '''

    genres = np.array(genres)
    genre_sel = np.not_equal(genres, None)
    X, y = svd_trans[genre_sel], genres[genre_sel]
    plot_embedding(X, y)


def plot_mds(points, genres, n_points=500):
    '''
    Plots a set of documents in MDS space

    Args:
        points: dense array with coordinates of each document
        genres: list of genres for each entry in points
    Returns:
        None
    '''

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
    '''
    Utility function that actually does the plotting
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = sns.color_palette('Set1', len(np.unique(labels)))
    for i, label in enumerate(np.sort(np.unique(labels))[::-1]):
        cur_X = embedding[labels == label]
        ax.add_line(plt.Line2D(cur_X[:, 0], cur_X[:, 1], color=colors[i],
                               linestyle='none', marker='o', alpha=0.75))
    #ax.set_xlim(1.1*np.min(embedding[:, 0], 1.1*np.max(embedding[:, 0])))
    #ax.set_ylim(1.1*np.min(embedding[:, 1], 1.1*np.max(embedding[:, 1])))
    ax.set_xlim(-1.51, 1.51)
    ax.set_ylim(-1.51, 1.51)
    sns.despine(offset=5, trim=True)
    ax.set_xlabel('MDS dim 1')
    ax.set_ylabel('MDS dim 2')
    ax.legend(np.sort(np.unique(labels)))


def show_lsi(tfidf, svd, svd_trans,
             n_comp=10, n_words=10):
    '''
    Shows the individual words that make up each svd component

    Args:
        tfidf: sklearn fitted TfidfVectorizer
        svd: sklearn fitted TruncatedSVD
        svd_trans: dense array with lsi transformed data
        n_comp: number of components to show
        n_words: number of words for each component to show
    Returns:
        None
    '''

    components = svd.components_
    words = tfidf.get_feature_names()
    words = prettify(words)
    words = np.array(words)

    fig = plt.figure(figsize=(10, 8))

    for i, component in enumerate(components[0:n_comp], 1):
        sorted_idx = np.argsort(component)
        print('Component #{:d}'.format(i))
        print('-'*20)
        print('\nMost negative words:')
        print('\n\t'+'\n\t'.join(words[sorted_idx[:n_words]]))
        print('\nMost positive words:')
        print('\n\t'+'\n\t'.join(words[sorted_idx[-n_words:]]))

        # Make a figure and axes with dimensions as desired.
        ax = fig.add_subplot(2, 5, i)
        ax.set_title('Component {:d}'.format(i))

        # Set the colormap and norm to correspond to the data for which
        # the colorbar will be used.
        cmap = plt.cm.Spectral_r
        mn = np.min(component)
        mx = np.max(component)
        norm = mpl.colors.Normalize(mn, mx)

        cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                       norm=norm,
                                       orientation='vertical')
        colors = sns.color_palette('Spectral_r', 9).as_hex()
        colors = np.r_[np.repeat(colors[0], n_words),
                       np.repeat(colors[-1], n_words)]

        cb.set_ticks(np.linspace(mn, mx, n_words*2+2)[1:-1])
        cb.ax.tick_params(labelsize=10)
        for color, tick in zip(colors, cb.ax.get_yticklabels()):
            tick.set_color(color)
            tick.set_fontsize(14)
        cb.set_ticklabels(np.r_[
            words[sorted_idx[:n_words]],
            words[sorted_idx[-n_words:]]].flatten())

    plt.tight_layout()


def explore_k(svd_trans, k_range):
    '''
    Explores various values of k in KMeans

    Args:
        svd_trans: dense array with lsi transformed data
        k_range: the range of k-values to explore
    Returns:
        scores: list of intertia scores for each k value
    '''

    scores = []
    # spherical kmeans, so normalize
    normalizer = Normalizer()
    norm_data = normalizer.fit_transform(svd_trans)
    for k in np.arange:
        km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1,
                    verbose=2)
        km.fit(norm_data)
        scores.append(-1*km.score(norm_data))
    plt.plot(k_range, scores)
    plt.xlabel('# of clusters')
    plt.ylabel('Inertia')
    sns.despine(offset=5, trim=True)
    return scores


def kmeans(tfidf, svd, svd_trans, k=200, n_words=10):
    '''
    Performs k-means clustering on svd transformed data and plots it

    Args:
        tfidf: sklearn fitted TfidfVectorizer
        svd: sklearn fitted TruncatedSVD
        svd_trans: dense array with lsi transformed data
        k: the k in k-means
    Returns:
        km: the fitted KMean object
    '''

    # spherical kmeans, so normalize
    normalizer = Normalizer()
    norm_data = normalizer.fit_transform(svd_trans)
    km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=5,
                verbose=2)
    km.fit(norm_data)

    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]

    terms = tfidf.get_feature_names()
    terms = prettify(terms)
    terms = np.array(terms)
    fig = plt.figure(figsize=(10, 8))
    for i in range(10):
        print("Cluster {:d}:".format(i))
        for ind in order_centroids[i, :n_words]:
            print(' {:s}'.format(terms[ind]))
        print('\n')

        # Make a figure and axes with dimensions as desired.
        ax = fig.add_subplot(2, 5, i+1)
        ax.set_title('Cluster {:d}'.format(i+1))

        component = order_centroids[i]
        cmap = plt.cm.Purples
        mn = np.min(component[:n_words])
        mx = np.max(component[:n_words])
        norm = mpl.colors.Normalize(mn, mx)

        cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm,
                                       orientation='vertical')
        # sorted_component = np.sort(component)
        colors = sns.color_palette('Purples', 9).as_hex()
        colors = np.repeat(colors[-1], n_words)

        cb.set_ticks(np.linspace(mn, mx, n_words))
        cb.ax.tick_params(labelsize=10)
        for color, tick in zip(colors, cb.ax.get_yticklabels()):
            tick.set_color(color)
            tick.set_fontsize(14)
        cb.set_ticklabels(np.array(terms)[order_centroids[i, :n_words][::-1]])
    plt.tight_layout()
    return km


def prettify(words):
    '''
    Re-format my mangling of words back to nice looking ones

    Args:
        words: list of words
    Returns:
        out: reformatted list of words
    '''

    words = [word.replace('__ampersand__', '&') for word in words]
    words = [word.replace('__dollar_sign__', '$') for word in words]
    words = [word.replace('__exclamation__', '!') for word in words]
    words = [word.replace('__s', 's') for word in words]
    words = [word.replace('_', ' ') for word in words]
    return words
