import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm

sns.set_style('ticks')


def get_similarities(df, svd_trans, func='raw', normalize=False,
                     restrict=False, n=200):
    # for validation, use only tracks that have audio information
    # as scraped from spotify (and matched in pitchfork)
    df['has_audio_features'] = ~df['acoustic'].isnull()

    feature_names = ['acoustic', 'dance', 'energy', 'instrument',
                     'live', 'loud', 'speech', 'tempo', 'valence']
    features_and_sims = feature_names + ['lsi_sims']

    valid_idx = np.where(df['has_audio_features'])[0][0:5000]
    df = df.iloc[valid_idx]
    svd_trans = svd_trans[valid_idx]
    sims = cosine_similarity(svd_trans, svd_trans)

    if normalize:
        ss = StandardScaler()
        ss.fit(np.float64(df[feature_names].values))
        df[feature_names] = ss.transform(df[feature_names])

    features = np.float64(df[feature_names].values)
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
        other_features = np.float64(df_audiofeats[feature_names].values)

        if not hasattr(func, '__call__'):
            feature_diffs = np.sqrt((other_features - seed_features)**2)
            all_audio_diffs.append(
                pd.DataFrame(feature_diffs[::-1],
                             columns=feature_names).reset_index(drop=True))
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
    plt.close('all')
    fig = plt.figure()
    feature_names = ['acoustic', 'dance', 'energy', 'instrument',
                     'live', 'loud', 'speech', 'tempo', 'valence']
    colors = sns.color_palette('Set1', 10)
    for i, feature_name in enumerate(feature_names):
        temp = [seed[feature_name] for seed in all_audio_diffs]
        df = pd.concat(temp)
        df.index.name = 'recc_rank'
        mean_diff = df.groupby(df.index).mean()[1:].values.flatten()
        sem_diff = df.groupby(df.index).sem()[1:].values.flatten()
        ax = fig.add_subplot(3, 3, i+1)
        plot_indiv_diff(ax, mean_diff, sem_diff, feature_name, colors[i])

    sns.despine(offset=5, trim=True)


def plot_af_beta_dists(all_audio_diffs):
    plt.close('all')
    fig = plt.figure()
    feature_names = ['acoustic', 'dance', 'energy', 'instrument',
                     'live', 'loud', 'speech', 'tempo', 'valence']
    colors = sns.color_palette('Set1', 10)
    for i, feature_name in enumerate(feature_names):
        cur_feature = [seed[feature_name] for seed in all_audio_diffs]
        betas = np.zeros(len(cur_feature))
        pvals = np.zeros(len(cur_feature))
        for j, seed_album in enumerate(cur_feature):
            beta, pval = compute_regression(seed_album.index.values,
                                            seed_album.values)
            betas[j], pvals[j] = beta, pval
        ax = fig.add_subplot(3, 3, i+1)
        plot_indiv_hist(ax, betas, pvals, feature_name, colors[i])

    sns.despine(offset=5, trim=True)


def plot_cos_sims(all_cos_sims):
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    feature_name = ['sim_func']
    df = pd.concat([seed[feature_name] for seed in all_cos_sims])
    df.index.name = 'recc_rank'

    mean_diff = df.groupby(df.index).mean()[1:].values.flatten()
    sem_diff = df.groupby(df.index).sem()[1:].values.flatten()
    # return df.groupby(df.index).std()

    plot_indiv_diff(ax, mean_diff, sem_diff, 'cosine similarity', 'k')

    inset_ax = fig.add_axes([.75, .5, .1, .3])
    n = 500
    lh, rh = mean_diff[:n/2], mean_diff[n/2:n]
    m_lh, m_rh = np.mean(lh), np.mean(rh)
    sem_lh, sem_rh = np.std(lh)/np.sqrt(n/2), np.std(rh)/np.sqrt(n/2)

    N = np.arange(2)
    width = 0.75
    inset_ax.bar(N, [m_lh, m_rh], width=width, yerr=[sem_lh, sem_rh])
    inset_ax.set_xticks(N + width/2.)
    inset_ax.set_ylabel('cosine similarity')

    sns.despine(offset=5, trim=True)

    # gotta do this after despine
    inset_ax.set_xticklabels(('Left half', 'Right half'),
                             horizontalalignment='right', rotation=45)


def plot_indiv_diff(ax, y, y_e, y_label, color):
    x = np.arange(len(y))
    ax.add_line(plt.Line2D(x, y, color=color))
    ax.fill_between(x, y-y_e, y+y_e, color=color, alpha=0.5)
    ax.set_xlabel('Recommendation rank')
    ax.set_ylabel(y_label)
    ax.set_xlim(0, 500)
    ax.set_ylim(np.min(y), np.max(y)*1.1)


def plot_indiv_hist(ax, vals, pvals, label, color='k', alpha=0.05, n_bins=20):
    mn = np.min(vals)
    mx = np.max(vals)
    bin_edges = np.linspace(mn, mx, n_bins)
    bin_width = np.diff(bin_edges)[0]
    counts_sig, edges = np.histogram(vals[pvals < alpha], bin_edges)
    counts_nonsig, edges = np.histogram(vals[pvals >= alpha], bin_edges)

    ax.bar(edges[0:-1], height=counts_sig+counts_nonsig, width=bin_width,
           color='none', edgecolor=color)
    ax.bar(edges[0:-1], height=counts_sig, width=bin_width,
           color=color, edgecolor=color)
    ax.axvline(np.mean(vals), linestyle='--', color='k')


def plot_cosafs_vs_coslsi(all_lsi_sims, all_cos_sims, win_size=100):
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    feature_name = ['sim_func']
    df_cos = pd.concat([seed[feature_name][1:] for seed in all_cos_sims])
    df_lsi = pd.concat([seed[:][1:] for seed in all_lsi_sims])

    x = df_lsi.values
    y = df_cos.values.flatten()

    argsorted = np.argsort(x)[::-1]
    x = x[argsorted]
    y = y[argsorted]
    x = pd.rolling_mean(x, win_size)
    y = pd.rolling_mean(y, win_size)
    ax.add_line(plt.Line2D(x[~np.isnan(x)], y[~np.isnan(y)], color='k'))
    ax.set_xlim(np.nanmin(x), np.nanmax(x))
    ax.set_ylim(np.nanmin(y), np.nanmax(y))

    sns.despine(offset=5, trim=True)
    ax.set_xlabel('LSI cosine similarity')
    ax.set_ylabel('Spotify AF cosine similarity')


def permutation_cosafs_vs_coslsi(all_lsi_sims, all_cos_sims, n_perm=1000):
    plt.close('all')
    feature_name = ['sim_func']
    df_cos = pd.concat([seed[feature_name][1:] for seed in all_cos_sims])
    df_lsi = pd.concat([seed['sims'][1:] for seed in all_lsi_sims])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = df_lsi.values
    y = df_cos.values.flatten()

    real_corr = np.corrcoef(x, y)[0, 1]

    permuted_corrs = []
    for i in xrange(n_perm):
        print(i)
        permuted_x = x[np.random.permutation(len(x))]
        permuted_corrs.append(np.corrcoef(permuted_x, y)[0, 1])

    real_corr = np.corrcoef(x, y)[0, 1]

    return permuted_corrs, real_corr


def compute_regression(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.params[1], model.pvalues[1]
