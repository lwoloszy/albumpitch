from __future__ import division
import re

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pymongo import MongoClient

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

from stop_words import get_stop_words
# import requests
import nltk.corpus
from nltk.stem.porter import PorterStemmer

import text_preprocess as textpre
import utility_funcs as uf
reload(textpre)

sns.set_style('ticks')


def get_documents():
    '''
    Get Pitchfork album reviews from MongoDB

    Args:
        None
    Returns:
        Pandas DataFrame with album information, including
        artists, album name, abstract, review, genres, labels,
        and pitchfork url
    '''

    client = MongoClient()
    db = client['albumpitch']
    coll = db['pitchfork']

    agg = coll.aggregate(
        [{'$project':
          {'_id': 0, 'abstract': 1, 'album': 1, 'artists': 1,
           'review': 1, 'genres': 1, 'labels': 1, 'url': 1}}])
    client.close()

    agg = list(agg)
    df = pd.DataFrame(agg)
    return df


def append_genres(df):
    '''
    Append to a dataframe a genre column, where we
    take the first genre from the genres list, if any

    Args:
        df: dataframe with Pitchfork reviews
    Returns:
        None

    '''

    df['genre'] = uf.get_genres(df)


def plot_albums_genre(df):
    '''
    Plot number of reviews for each music genre

    Args:
        df: dataframe with Pitchfork reviews
    Returns:
        None

    '''

    genres = df['genres'].tolist()
    u_genres = np.unique([item for sublist in genres for item in sublist])
    tuples_list = []
    for u_genre in u_genres:
        n_genre = np.sum(df['genres'].apply(lambda x: u_genre in x))
        tuples_list.append((u_genre, n_genre))
    n_unknown = np.sum(df['genres'].apply(lambda x: len(x) == 0))
    tuples_list.append(('Uncategorized', n_unknown))
    df = pd.DataFrame(tuples_list, columns=['genre', 'count'])

    plt.close('all')
    color = sns.color_palette('Set1', 2)[1]
    df_sorted = df.sort_values('count', ascending=False)
    ax = sns.barplot(x='count', y='genre', data=df_sorted,
                     color=color)
    # ax.grid('on', alpha=0.5, linestyle='--')
    ax.set_xlabel('# of reviews')
    ax.set_ylabel('Genre')
    sns.despine(offset=5, trim=True)
    plt.tight_layout()

    return df


def naive_bayes_genre_cv(df):
    '''
    Cross-validate a simple Multinomial Naive Bayes
    classifier to see how well we can predict genre
    tags from Pitchfork text reviews (sanity check)

    Args:
        df: dataframe with Pitchfork reviews
    Returns:
        cv_scores: cross-validation scores (accuracy)

    '''

    X = df['review']
    y = df['genre']

    clf = Pipeline([
        ('counts', CountVectorizer(stop_words='english')),
        ('mnb', MultinomialNB())
    ])

    cv_scores = cross_val_score(clf, X, y, scoring='accuracy', cv=4)
    return cv_scores


def naive_bayes_genre_grid(df):
    '''
    Grid search a simple Multinomial Naive Bayes
    classifier to see roughly which set of parameters
    work well at classifying genres (playing around)

    Args:
        df: dataframe with Pitchfork reviews
    Returns:
        grid_search: fitted GridSearchCV object

    '''

    X = df['review']
    y = df['genre']

    clf = Pipeline([
        ('counts', CountVectorizer()),
        ('mnb', MultinomialNB())
    ])

    parameters = {'counts__stop_words': [None, 'english'],
                  'counts__ngram_range': [(1, 1), (1, 2)],
                  'counts__lowercase': [True, False],
                  'counts__max_df': [0.9, 1.0],
                  'counts__min_df': [0, 0.01, 0.05]
                  }

    grid_search = GridSearchCV(clf, parameters, cv=3, scoring='accuracy',
                               verbose=2)
    grid_search.fit(X, y)
    return grid_search


def get_important_words(df, n_words):
    '''
    Fit a simple Multinomial Naive Bayes classifier to
    Pitchfork reviews and print out words which reduce
    most the entropy of genre distributions

    Args:
        df: dataframe with Pitchfork reviews
        n_words: number of informative words to return
    Returns:
        words: list of informative words

    '''

    X = df['review']
    y = df['genre']

    clf = Pipeline([
        ('counts', CountVectorizer(stop_words='english')),
        ('mnb', MultinomialNB())
    ])

    clf.fit(X, y)
    return h_y_giv_x(clf.steps[0][1], clf.steps[1][1], n_words=n_words)


def h_y_giv_x(cv, mnb, n_words=100):
    '''
    Compute the entropy of genre distributions given the
    presence of a word in a document

    Args:
        cv: sklearn fitted CountVectorizer
        mnb: sklearn fitted MulinomialNB
        n_words: number of most informative words to return
    Returns:
        words: list of informative words

    '''
    p_x_giv_y = np.exp(mnb.coef_)
    p_y = np.exp(mnb.intercept_)

    p_x_and_y = p_x_giv_y * p_y[:, np.newaxis]
    p_y_giv_x = p_x_and_y / np.sum(p_x_and_y, axis=0)
    entropy = -np.sum(p_y_giv_x * np.log2(p_y_giv_x), 0)

    # p_x = np.sum(p_x_and_y, axis=0)
    # entropy_scaled = entropy * p_x

    words = np.array(cv.get_feature_names())
    return words[np.argsort(entropy)[:n_words]]


def basic_lsi(df, n_components=200, max_df=0.5, min_df=5):
    '''
    Basic LSI model for album recommendations

    Args:
        df: dataframe with Pitchfork reviews
        n_components: number of lsi dimensions
        max_df: max_df in TfidfVectorizer
        min_df: min_df in TfidfVectorizer
    Returns:
        tfidf: sklearn fitted TfidfVectorizer
        tfidf_trans: sparse matrix with tfidf transformed data
        svd: sklearn fitted TruncatedSVD
        svd_trans: dense array with lsi transformed data

    '''

    X = df['review']
    stopwords = nltk.corpus.stopwords.words('english')

    tfidf = TfidfVectorizer(stop_words=stopwords,
                            max_df=max_df, min_df=min_df)
    tfidf_trans = tfidf.fit_transform(X)

    svd = TruncatedSVD(n_components=n_components)
    svd_trans = svd.fit_transform(tfidf_trans)

    return tfidf, tfidf_trans, svd, svd_trans


def extended_tfidf(df):
    abstracts = df['abstract'].tolist()
    reviews = df['review'].tolist()
    genres = df['genres'].map(lambda x: ', '.join(x)).tolist()
    artists = df['artists'].map(lambda x: ', '.join(x)).tolist()
    album = df['album'].tolist()

    new_reviews = []
    for i, (artist, review) in enumerate(zip(artists, reviews)):
        new_reviews.append(textpre.prepend_first_name(artist, review))

    #reviews = new_reviews
    together = [abstracts, reviews, genres, artists, album]
    entries = [', '.join(entry) for entry in zip(*together)]

    # r = requests.get('http://fs1.position2.com/bm/txt/stopwords.txt')
    # stopwords = r.content.split('\n')
    # stopwords = nltk.corpus.stopwords.words('english')

    # with open('data/stopwords.txt') as f:
    #    stopwords = f.readlines()
    #    stopwords = [stopword.strip() for stopword in stopwords]

    stopset = textpre.get_stopset()
    stemmer = textpre.get_stemmer('snowball')

    ctpre = textpre.CustomTextPreprocessor(merge_capitalized=True)
    ctok = textpre.CustomTokenizer(stopset, stemmer)

    tfidf = TfidfVectorizer(stop_words=stopset,
                            preprocessor=ctpre, tokenizer=ctok,
                            max_df=0.5, min_df=5)
    tfidf_trans = tfidf.fit_transform(entries)
    return tfidf, tfidf_trans


def extended_lsi(df, n_components=200):
    print('Starting TfIdf')
    tfidf, tfidf_trans = extended_tfidf(df)

    print('Starting SVD')
    svd = TruncatedSVD(n_components=n_components)
    svd_trans = svd.fit_transform(tfidf_trans)

    return tfidf, tfidf_trans, svd, svd_trans


def basic_lda(df, max_df=0.5, min_df=5):
    X = df['review']
    cv = CountVectorizer(stop_words='english',
                         min_df=5,
                         max_df=0.5)
    cv_trans = cv.fit_transform(X)

    lda = LatentDirichletAllocation(n_topics=200, max_iter=7)
    lda_trans = lda.fit_transform(cv_trans)

    return cv, cv_trans, lda, lda_trans


def extended_lda(df):
    tfidf, tfidf_trans = extended_tfidf(df)
    lda = LatentDirichletAllocation(n_topics=100, max_iter=5)
    lda_trans = lda.fit_transform(tfidf_trans)

    return tfidf, tfidf_trans, lda, lda_trans


def print_top_words(vectorizer, clf, class_labels, n=10):
    '''
    Prints features with the highest coefficient values, per class

    Args:
        vectorizer: sklearn fitted CountVectorizer
        clf: sklearn fitted MultinomialNB
        class_labels: ordered list of class labels
    Returns:
        None

    '''
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top_n = np.argsort(clf.coef_[i])[-n:]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top_n)))


def print_recommendations(df, svd_trans, album_idx, n=25):
    sims = cosine_similarity(svd_trans[album_idx, :].reshape(1, -1), svd_trans)
    df_temp = df.iloc[np.argsort(sims).flatten()[-n:]]
    df_temp['sim_scores'] = np.sort(sims.flatten())[-n:]
    print df_temp[['url', 'genres', 'sim_scores']][::-1]


def print_components(tfidf, svd, svd_trans, df,
                     n_comp=10, n_words=10, n_docs=10):
    # transformed = svd.transform(tfidf.transform(df['review']))
    # top_docs = np.argsort(transformed, axis=0)

    components = svd.components_
    words = np.array(tfidf.get_feature_names())
    for i, component in enumerate(components[0:n_comp], 1):
        sorted_idx = np.argsort(component)
        print('Component #{:d}'.format(i))
        print('-'*20)
        print('\nMost negative words:')
        print('\n\t'+'\n\t'.join(words[sorted_idx[:n_words]]))
        print('\nMost positive words:')
        print('\n\t'+'\n\t'.join(words[sorted_idx[-n_words:]]))

        top_docs = np.argsort(svd_trans[:, i-1])[-n_docs:]
        artists = df.iloc[top_docs]['artists']
        artists = [' / '.join(artist) for artist in artists]
        album = df.iloc[top_docs]['album'].tolist()

        artist_album = [' : '.join([ar, al]) for ar, al in zip(artists, album)]

        print('\nTop albums:')
        print('\n\t'+'\n\t'.join(artist_album))
        print('\n')


def print_top(tfidf, tfidf_trans, n_words=10):
    top_idx = np.argsort(np.mean(tfidf_trans.toarray(), axis=0))[-n_words:]
    words = np.array(tfidf.get_feature_names())[top_idx]
    print(words)
