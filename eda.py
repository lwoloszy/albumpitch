from __future__ import division

import pandas as pd
import numpy as np
from pymongo import MongoClient

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation

import nltk.corpus
import re


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

    return tfidf, svd


def print_components_and_albums(tfidf, svd, df, n_words=10, n_docs=10):
    transformed = svd.transform(tfidf.transform(df['review']))
    top_docs = np.argsort(transformed, axis=0)

    components = svd.components_
    words = np.array(tfidf.get_feature_names())
    for i, component in enumerate(components[0:10], 1):
        sorted_idx = np.argsort(component)
        print('Component #{:d}'.format(i))
        print('Most negative words:')
        print(' '.join(words[sorted_idx[:n_words]]))
        print('Most positive words:')
        print(' '.join(words[sorted_idx[-n_words:]]))
        print('Top albums:')
        print(df['url'].iloc[top_docs[-n_docs:, i-1]])
        print('\n')



def basic_lda(df):
    X = df['review']
    tfidf = CountVectorizer(stop_words='english',
                            min_df=2,
                            max_df=0.95)
    tfidf_trans = tfidf.fit_transform(X)

    lda = LatentDirichletAllocation(n_topics=100, max_iter=5)
    lda_trans = lda.fit_transform(tfidf_trans)

    return lda_trans




#  pY=repmat(pY,1,size(p_X_giv_Y,2));

#  %marginal probability of x
#  p_X=sum(p_X_and_Y,1);

#  %inner sum of conditional entropy
#  cond_ent_tmp=-p_X_giv_Y.*log2(p_X_giv_Y)-(1-p_X_giv_Y).*log2(1-p_X_giv_Y);
#  %outer sum of conditional entropy
#  cond_ent=sum(cond_ent_tmp.*pY,1);
#  marg_ent=-p_X.*log2(p_X)-(1-p_X).*log2(1-p_X);
#  ent=marg_ent-cond_ent;

#  [vals idx]=sort(ent,'descend');
#  idx=idx(1:numfeats);

class PorterTokenizer(object):

    def __init__(self):
        self.stemmer = PorterStemmer()

    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in word_tokenize(doc)]


def print_top_words(vectorizer, clf, class_labels, n=10):
    """
    Prints features with the highest coefficient values, per class
    """
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top_n = np.argsort(clf.coef_[i])[-n:]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top_n)))


#def print_top_words(nmf, vectorizer, n=10):
#    words = vectorizer.get_feature_names()
#    for i, component in enumerate(nmf.components_):
#        top10 = np.argsort(nmf.components_[i])[-n:]
#        print('Component {:d}: {:s}'.format(i,
#            ' '.join([words[j] for j in top10])))


def unicode_preprocessing(text):
    single_quotes = ur"""['\u2018\u2019\u0060\u00b4]"""
    text = re.sub(single_quotes, "'", text)

    double_quotes = ur"""["\u201c\u201d]"""
    text = re.sub(double_quotes, '"', text)

    # pound_signs = ur"""#"""
    # text = re.sub(pound_signs, '-', text)

    space = ur"""\xa0"""
    text = re.sub(space, ' ', text)

    em_dashes = ur"""\u2014"""
    text = re.sub(em_dashes, '--', text)

    text = re.sub(ur'/', ' / ', text)
    return text