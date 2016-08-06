import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
import scipy.stats as scp

sns.set_style('ticks')
sns.set_context('talk')

FEATURE_NAMES = ['acoustic', 'dance', 'energy', 'instrument',
                 'live', 'loud', 'speech', 'tempo', 'valence']
LABEL_DICT = {'acoustic': 'Acousticness',
              'dance': 'Danceability',
              'energy': 'Energy',
              'instrument': 'Instrumentalness',
              'live': 'Liveness',
              'loud': 'Loudness',
              'speech': 'Speechiness',
              'tempo': 'Tempo',
              'valence': 'Valence'}


def get_af_diffs(df, svd_trans, func='raw', normalize=False,
                     restrict=False, n=250):
    """
    Computes distances between the mean track audio features
    of a seed album and those albums recommended by a recommendation
    algorithm. Performs this for all albums in collection that have
    audio features.

    Args:
        df: pandas dataframe that has all the audio features
            as well as auxiliary information identifying the album
            such as its artists, genres, etc.
        svd_trans: the representation of each and every album in df
                   in the reduced dimensionality space
        func: compute differences between track features
              on a per feature basis ('raw') or collapse
              those difference to a single number with a function
              such as cosine_distance (default='raw')
        normalize: normalize the audiofeatures across
                   the albums before computing differences (default=False)
        restrict: restrict the set of distance calculations to
                  albums in the same genre and excluding the same artist
                  (control manipulation)
        n: how far down the rankings to go (default=250)
    Returns:
        all_lsi_sims: list of dataframes, where each dataframe has
                      the cosine similarities between a given seed
                      album and all the other albums
        all_audio_diffs: list of dataframes, where each dataframe
                         has the audio differences between a given
                         seed album and all the other albums
    """

    # for validation, use only tracks that have audio information
    # as scraped from spotify (and matched in pitchfork)
    df['has_audio_features'] = ~df['acoustic'].isnull()

    features_and_sims = FEATURE_NAMES + ['lsi_sims']

    valid_idx = np.where(df['has_audio_features'])[0]
    df = df.iloc[valid_idx]
    svd_trans = svd_trans[valid_idx]
    sims = cosine_similarity(svd_trans, svd_trans)

    if normalize:
        ss = StandardScaler()
        ss.fit(np.float64(df[FEATURE_NAMES].values))
        df[FEATURE_NAMES] = ss.transform(df[FEATURE_NAMES])

    features = np.float64(df[FEATURE_NAMES].values)
    all_audio_diffs = []
    all_lsi_sims = []

    df['artists_str'] = df['artists'].apply(lambda x: ' '.join(x))
    df['genres_str'] = df['genres'].apply(lambda x: ' '.join(x))
    artists_list = df['artists'].tolist()
    genres_list = df['genres'].tolist()

    for i, (sim_vector, seed_features) in enumerate(zip(sims, features), 1):
        if i % 100 == 0:
            print('Gone through {:d} albums'.format(i))

        df.loc[:, 'lsi_sims'] = sim_vector

        if restrict:
            cur_artists = artists_list[i-1]
            sel_artist = np.zeros(df.shape[0])
            for artist in cur_artists:
                sel_artist += ~df['artists_str'].str.contains(artist).values

            cur_genres = genres_list[i-1]
            if not cur_genres:  # all genres good if album doesn't have genre
                sel_genre = np.ones(df.shape[0])
            else:
                sel_genre = np.zeros(df.shape[0])
            for genre in cur_genres:
                sel_genre += df['genres_str'].str.contains(genre).values

            sel_genre[df['genres'].apply(lambda x: len(x) == 0).values] = 1

            sel = np.logical_and(sel_artist > 0, sel_genre > 0)

            # blank out all albums that are not valid
            if np.sum(sel) < n:
                print('Not enough data to make {:d} reccs for {:s}, {:s}, {:d}'
                      .format(cur_artists, cur_genres, np.sum(sel)))
                # n = np.sum(sel)
                continue
            df.loc[~sel, 'lsi_sims'] = -1

        # print(i)
        df_audiofeats = df[features_and_sims].sort_values('lsi_sims').iloc[-n:]
        other_features = np.float64(df_audiofeats[FEATURE_NAMES].values)

        if not hasattr(func, '__call__'):
            feature_diffs = np.sqrt((other_features - seed_features)**2)
            all_audio_diffs.append(
                pd.DataFrame(feature_diffs[::-1],
                             columns=FEATURE_NAMES).reset_index(drop=True))
        else:
            all_audio_diffs.append(
                pd.DataFrame(
                    func(seed_features.reshape(1, -1), other_features).T[::-1],
                    columns=['sim_func']).
                reset_index(drop=True))

        all_lsi_sims.append(df_audiofeats[['lsi_sims']][::-1]
                            .reset_index(drop=True))

    return all_lsi_sims, all_audio_diffs


def plot_af_diffs(all_audio_diffs):
    """
    Plots the mean album audio features differences as a function
    of recommendation rank. Does this for all individual audio features.

    Args:
        all_audio_diffs: list of dataframes as returned from get_af_diffs
    Returns:
        None
    """

    plt.close('all')
    fig = plt.figure()
    colors = sns.color_palette('Set1', 10)
    for i, feature_name in enumerate(FEATURE_NAMES):
        temp = [seed[feature_name] for seed in all_audio_diffs]
        df = pd.concat(temp)
        df.index.name = 'recc_rank'
        mean_diff = df.groupby(df.index).mean()[:].values.flatten()
        sem_diff = df.groupby(df.index).sem()[:].values.flatten()
        ax = fig.add_subplot(3, 3, i+1)
        plot_indiv_diff(ax, mean_diff, sem_diff, LABEL_DICT[feature_name],
                        colors[i])
        if i != 6:
            # ax.xaxis.set_major_formatter(NullFormatter())
            # ax.yaxis.set_major_formatter(NullFormatter())
            pass
        else:
            ax.set_xlabel('Recommendation rank')
            # ax.set_ylabel('# of instances')

    sns.despine(offset=5, trim=True)
    plt.tight_layout()


def plot_af_beta_dists(all_audio_diffs):
    """
    Plots distribution of individual beta coefficients, where the betas
    try to capture linear relationship between recommendation rank and
    audio features diff (compute one beta coefficient for each and every
    seed album and each and every audio feature)

    Args:
        all_audio_diffs: list of dataframes as returned from get_af_diffs
    Returns:
        None
    """

    plt.close('all')
    fig = plt.figure()
    colors = sns.color_palette('Set1', 10)
    for i, feature_name in enumerate(FEATURE_NAMES):
        cur_feature = [seed[feature_name] for seed in all_audio_diffs]
        betas = np.zeros(len(cur_feature))
        pvals = np.zeros(len(cur_feature))
        for j, seed_album in enumerate(cur_feature):
            beta, pval = compute_regression(seed_album.index.values[1:],
                                            seed_album.values[1:])
            betas[j], pvals[j] = beta, pval
        ax = fig.add_subplot(3, 3, i+1)
        plot_indiv_hist(ax, betas, pvals, LABEL_DICT[feature_name], colors[i],
                        mn=-.004, mx=.004, n_bins=40)
        ax.set_xlim(-.004, .004)
        ax.set_xticks(np.arange(-.004, .0041, .002))
        ax.set_ylim(0, 2500)
        ax.set_yticks(np.arange(0, 2501, 500))
        if i != 6:
            # ax.xaxis.set_major_formatter(NullFormatter())
            # ax.yaxis.set_major_formatter(NullFormatter())
            pass
        else:
            ax.set_xlabel(r"$\beta_1$")
            ax.set_ylabel('# of instances')

    sns.despine(offset=5, trim=True)


def plot_cos_diffs(all_cos_diffs):
    """
    Plots mean cosine difference between seed album and recommended
    albums sorted by recommendation rank

    Args:
        all_cos_diffs: list of dataframes as returned from get_af_diffs
    Returns:
        None
    """

    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    feature_name = ['sim_func']
    df = pd.concat([seed[feature_name] for seed in all_cos_diffs])
    df.index.name = 'recc_rank'

    mean_diff = df.groupby(df.index).mean().values.flatten()
    sem_diff = df.groupby(df.index).sem().values.flatten()
    # color = sns.color_palette('Set1', 2)[1]
    plot_indiv_diff(ax, mean_diff, sem_diff, 'Cosine Distance', 'k')

    # n = len(mean_diff)
    # inset_ax = fig.add_axes([.25, .6, .1, .3])
    # lh, rh = mean_diff[:n/2], mean_diff[n/2:n]
    # m_lh, m_rh = np.mean(lh), np.mean(rh)
    # sem_lh, sem_rh = np.std(lh)/np.sqrt(n/2), np.std(rh)/np.sqrt(n/2)

    # model = sm.OLS(df['sim_func'].values, sm.add_constant(df.index.values))
    # params = model.fit().params
    # ax.add_line(plt.Line2D(np.arange(n), np.arange(n)*params[1]+params[0],
    #                       color=color, linestyle='--'))
    # ax.set_xlabel('Recommendation rank')

    # N = np.arange(2)
    # width = 0.75
    # inset_ax.bar(N, [m_lh, m_rh], width=width, yerr=[sem_lh, sem_rh])
    # inset_ax.set_xticks(N + width/2.)
    # inset_ax.set_ylabel('Cosine Distance')

    sns.despine(offset=5, trim=True)

    # gotta do this after despine
    # inset_ax.set_xticklabels(('Ranks 1-{:d}'.format(n/2),
    #                          'Ranks {:d}-{:d}'.format(n/2, n)),
    #                         horizontalalignment='right', rotation=45)


def plot_indiv_diff(ax, y, y_e, y_label, color):
    """
    Plots y as a function of its index

    Args:
        ax: axis object on which to plot
        y: the set of y values
        y: the set of y errors
        y_label: axis y label
        color: line color
    Returns:
        None
    """

    x = np.arange(len(y))
    ax.add_line(plt.Line2D(x, y, color=color))
    ax.fill_between(x, y-y_e, y+y_e, color=color, alpha=0.5)
    ax.set_ylabel(y_label)
    ax.set_xlim(0, len(y)+1)
    ax.set_ylim(np.min(y), np.max(y)*1.1)


def plot_indiv_hist(ax, vals, pvals, label, color='k', alpha=0.05,
                    mn=None, mx=None, n_bins=20):
    """
    Plots distribution of values, where some values are significant
    and indicated by filled-in bars and other values are not
    significant as indicated by open bars

    Args:
        ax: axis object on which to plot
        vals: the set of values over which to plot hist
        pvals: array of same length as vals indicating
               the pvalue
        label: the label for the plot
        color: the color for the bars
        alpha: the alpha level to use to determining pval sig
        mn: the min value over which to compute histogram;
            if None (default) will be compute from data
        mx: the max value over which to compute histogram;
            if None (default) will be compute from data
        n_bins: number of bins to use for histogram (default=20)
    Returns:
        None
    """

    if not mn:
        mn = np.min(vals)
    if not mx:
        mx = np.max(vals)
    bin_edges = np.linspace(mn, mx, n_bins)
    bin_width = np.diff(bin_edges)[0]
    counts_sig, edges = np.histogram(vals[pvals < alpha], bin_edges)
    counts_nonsig, edges = np.histogram(vals[pvals >= alpha], bin_edges)

    ax.bar(edges[0:-1], height=counts_sig+counts_nonsig, width=bin_width,
           color='none', edgecolor=color)
    ax.bar(edges[0:-1], height=counts_sig, width=bin_width,
           color=color, edgecolor=color)
    ax.axvline(0, linestyle='--', color='k', linewidth=0.75)

    trans = transforms.blended_transform_factory(
        ax.transData, ax.transAxes)
    ax.annotate("", xy=(np.mean(vals), 0.9), xycoords=trans,
                xytext=(np.mean(vals), 1.0),
                ha='center', va='center', color='k',
                arrowprops=dict(arrowstyle="-|>",
                                facecolor='k', edgecolor='k',
                                linewidth=1))
    ax.text(0, 0.8, label, transform=ax.transAxes, fontsize=12,
            horizontalalignment='left')

    print(scp.ttest_1samp(vals, 0))


def plot_cosafs_vs_coslsi(all_lsi_sims, all_cos_diffs, win_size=100):
    """
    Plots mean audio features cos difference as a function of mean
    lsi difference

    Args:
        all_lsi_sims: list of dataframes as returned from get_af_diffs
        all_cos_diffs: list of dataframes as returned from get_af_diffs
    Returns:
        None

    """
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    feature_name = ['sim_func']
    df_cos = pd.concat([seed[feature_name][1:] for seed in all_cos_diffs])
    df_lsi = pd.concat([seed[:][1:] for seed in all_lsi_sims])

    x = df_lsi.values.flatten()
    y = df_cos.values.flatten()

    argsorted = np.argsort(x)[::-1]
    x = 1-x[argsorted]
    y = y[argsorted]
    x = pd.rolling_mean(x, win_size)
    y = pd.rolling_mean(y, win_size)
    ax.add_line(plt.Line2D(x[~np.isnan(x)], y[~np.isnan(y)], color='k'))
    ax.set_xlim(np.nanmin(x), np.nanmax(x))
    ax.set_ylim(np.nanmin(y), np.nanmax(y))

    xmin, xmax = np.min(df_cos.values), np.max(df_cos.values)
    model = sm.OLS(df_cos.values.flatten(),
                   sm.add_constant(1-df_lsi.values.flatten()))
    params = model.fit().params
    ax.add_line(plt.Line2D(np.linspace(xmin, xmax, 100),
                           np.linspace(xmin, xmax, 100)*params[1]+params[0],
                           color='k', linestyle='--'))

    sns.despine(offset=5, trim=True)
    ax.set_xlabel('LSI cosine distance')
    ax.set_ylabel('Spotify AudioFeatures cosine distance')


def compute_regression(X, y):
    """
    Compute simple regression coefficient between X and y
    """
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.params[1], model.pvalues[1]
