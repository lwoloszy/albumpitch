from __future__ import division
import re

import pandas as pd
import numpy as np
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

import text_preprocess as textpre
reload(textpre)


def get_documents(collection='pitchfork_full'):
    client = MongoClient()
    db = client['album_reviews']
    coll = db[collection]

    agg = coll.aggregate(
        [{'$project':
          {'_id': 0, 'abstract': 1, 'album': 1, 'artists': 1,
           'review': 1, 'genres': 1, 'url': 1}}])
    client.close()

    agg = list(agg)
    df = pd.DataFrame(agg)
    return df


def compute_genres(df):
    sel = df['genres'].apply(lambda x: len(x) > 0)
    df = df[sel]
    df['genre'] = df['genres'].map(lambda x: x[0])
    return df


def naive_bayes_genre_cv(df):
    X = df['review']
    y = df['genre']

    clf = Pipeline([
        ('counts', CountVectorizer(stop_words='english')),
        ('mnb', MultinomialNB())
    ])

    cv_scores = cross_val_score(clf, X, y, scoring='accuracy', cv=4)
    return cv_scores


def naive_bayes_genre_grid(df):
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


def get_important_words(df):
    X = df['review']
    y = df['genre']

    clf = Pipeline([
        ('counts', CountVectorizer(stop_words='english')),
        ('mnb', MultinomialNB())
    ])

    clf.fit(X, y)
    return clf


def h_y_giv_x(cv, mnb, n_words=100):
    '''
    Get most informative words, as quantified by entropy
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


def information_gain(cv, mnb, n_words=100):
    p_x_giv_y = np.exp(mnb.coef_)
    p_y = np.exp(mnb.intercept_)[:, np.newaxis]

    p_x_and_y = p_x_giv_y * p_y[:, np.newaxis]
    p_x = np.sum(p_x_and_y, axis=0)

    cond_ent_tmp = -p_x_giv_y*np.log2(p_x_giv_y) - \
                   (1-p_x_giv_y)*np.log2(1-p_x_giv_y)
    cond_ent = np.sum(cond_ent_tmp * p_y, axis=0)
    marg_ent = -p_x*np.log2(p_x)-(1-p_x)*np.log2(1-p_x)
    entropy = marg_ent - cond_ent

    words = np.array(cv.get_feature_names())
    return words[np.argsort(entropy)[-n_words:]]


def basic_lsi(df):
    X = df['review']
    stopwords = nltk.corpus.stopwords.words('english')

    cv = CountVectorizer(stop_words='english', binary=True)
    counts = cv.fit_transform(X)
    probs = np.array(counts.mean(axis=0)).flatten()
    add_stopwords = np.array(cv.get_feature_names())[np.where(probs > 0.5)[0]]

    stopwords.extend(add_stopwords)

    tfidf = TfidfVectorizer(stop_words=stopwords, max_df=0.7, min_df=0.01)
    tfidf_trans = tfidf.fit_transform(X)

    svd = TruncatedSVD(n_components=100)
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
        artist_parts = artist.split()
        if len(artist_parts) == 2:
            if re.match(r'\b(the|a|an)\b', artist_parts[0], re.IGNORECASE):
                new_reviews.append(review)
                continue
            artist = artist.encode('utf-8')
            review = review.encode('utf-8')
            new_reviews.append(
                textpre.prepend_first_name(artist, review).decode('utf-8'))
        else:
            new_reviews.append(review)

    reviews = new_reviews
    together = [abstracts, reviews, genres, artists, album]
    entries = [', '.join(entry) for entry in zip(*together)]

    # r = requests.get('http://fs1.position2.com/bm/txt/stopwords.txt')
    # stopwords = r.content.split('\n')
    # stopwords = nltk.corpus.stopwords.words('english')
    stopwords = get_stop_words('en')
    # with open('data/stopwords.txt') as f:
    #    stopwords = f.readlines()
    #    stopwords = [stopword.strip() for stopword in stopwords]

    # get those contractions
    stopwords.extend(nltk.word_tokenize(' '.join(stopwords)))

    # custom stop words
    stopwords.extend(['lp', 'ep',
                      'record', 'records',
                      'label', 'labels',
                      'release', 'releases', 'released',
                      'listen', 'listens', 'listened', 'listener',
                      'version', 'versions',
                      'album', 'albums',
                      'song', 'songs',
                      'track', 'tracks',
                      'sound', 'sounds',
                      # 'feel', 'feels',
                      # 'think', 'thinks',
                      'thing', 'things', 'something'
                      'music'])
    stopset = set(stopwords)

    tfidf = TfidfVectorizer(stop_words=stopset,
                            preprocessor=textpre.CustomTextPreprocessor(),
                            tokenizer=textpre.CustomTokenizer(stopset),
                            max_df=0.5, min_df=5)
    tfidf_trans = tfidf.fit_transform(entries)
    return tfidf, tfidf_trans


def extended_lsi(df):
    tfidf, tfidf_trans = extended_tfidf(df)
    svd = TruncatedSVD(n_components=250)
    svd_trans = svd.fit_transform(tfidf_trans)

    return tfidf, tfidf_trans, svd, svd_trans


def basic_lda(df):
    X = df['review']
    tfidf = TfidfVectorizer(stop_words='english',
                            min_df=5,
                            max_df=0.5)
    tfidf_trans = tfidf.fit_transform(X)

    lda = LatentDirichletAllocation(n_topics=100, max_iter=5)
    lda_trans = lda.fit_transform(tfidf_trans)

    return tfidf, tfidf_trans, lda, lda_trans


def extended_lda(df):
    tfidf, tfidf_trans = extended_tfidf(df)
    lda = LatentDirichletAllocation(n_topics=100, max_iter=5)
    lda_trans = lda.fit_transform(tfidf_trans)

    return tfidf, tfidf_trans, lda, lda_trans


def print_top_words(vectorizer, clf, class_labels, n=10):
    """
    Prints features with the highest coefficient values, per class
    """
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top_n = np.argsort(clf.coef_[i])[-n:]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top_n)))


def print_recommendations(df, svd_trans, album_idx):
    sims = cosine_similarity(svd_trans[album_idx, :].reshape(1, -1), svd_trans)
    df_temp = df.iloc[np.argsort(sims).flatten()[-25:]]
    df_temp['sim_scores'] = np.sort(sims.flatten())[-25:]
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
