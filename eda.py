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

from stop_words import get_stop_words
# import requests
import nltk.corpus
import re
import string


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


def extended_lsi(df):
    abstracts = df['abstract']
    reviews = df['review']
    genres = df['genres'].map(lambda x: ' '.join(x))
    artists = df['artists'].map(lambda x: ' '.join(x))
    album = df['album']

    together = [abstracts, reviews, genres, artists, album]

    entries = [' '.join(entry) for entry in zip(*together)]

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
                      'feel', 'feels',
                      'sound', 'sounds',
                      'album', 'albums',
                      'song', 'songs',
                      'music'])
    stopset = set(stopwords)

    tfidf = TfidfVectorizer(stop_words=stopset,
                            preprocessor=CustomTextPreprocessor(),
                            tokenizer=CustomTokenizer(stopset),
                            max_df=0.5, min_df=5)
    tfidf_trans = tfidf.fit_transform(entries)

    svd = TruncatedSVD(n_components=100)
    svd_trans = svd.fit_transform(tfidf_trans)

    return tfidf, tfidf_trans, svd, svd_trans


def print_components(tfidf, svd, n_words=10, n_docs=10):
    # transformed = svd.transform(tfidf.transform(df['review']))
    # top_docs = np.argsort(transformed, axis=0)

    components = svd.components_
    words = np.array(tfidf.get_feature_names())
    for i, component in enumerate(components[0:10], 1):
        sorted_idx = np.argsort(component)
        print('Component #{:d}'.format(i))
        print('-'*20)
        print('\nMost negative words:')
        print('\n\t'+'\n\t'.join(words[sorted_idx[:n_words]]))
        print('\nMost positive words:')
        print('\n\t'+'\n\t'.join(words[sorted_idx[-n_words:]]))
        # print('Top albums:')
        # print('\t\n'.join(df['url'].iloc[top_docs[-n_docs:, i-1]]))
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

class CustomTokenizer(object):
    def __init__(self, stopset=None, stemmer=PorterStemmer()):
        self.tokenizer = nltk.tokenize.word_tokenize
        self.punctset = set(string.punctuation)
        if stopset:
            self.stopset = set(stopset)
        if stemmer:
            self.stemmer = PorterStemmer()

    def __call__(self, doc):
        # use nltk's intelligent tokenizer, but we'll do some postprocessing
        words = self.tokenizer(doc)

        # now split on hyphens manually
        words = [subword for word in words for subword in word.split('-')]

        # remove stopwords/punctuation
        words = filter(lambda x: (x not in self.punctset and
                                  x not in self.stopset and
                                  x.strip(string.punctuation)),
                       words)

        # strip punctuation and stem
        words = [self.stemmer.stem(word.strip(string.punctuation))
                 for word in words]

        return words


class CustomTextPreprocessor(object):
    def __init__(self):
        self.u_single_quotes = ur"['\u2018\u2019\u0060\u00b4]"
        self.u_double_quotes = ur'["\u201c\u201d]'
        self.u_spaces = ur'\xa0'
        self.u_en_dashes = ur'\u2013'
        self.u_em_dashes = ur'\u2014'
        self.u_infinity_signs = ur'\u221e'

    def _lowercase(self, text):
        return text.lower()

    def _preprocess_unicode(self, text):
        text = re.sub(self.u_single_quotes, "'", text)
        text = re.sub(self.u_double_quotes, '"', text)
        text = re.sub(self.u_spaces, ' ', text)
        text = re.sub(self.u_en_dashes, '-', text)
        text = re.sub(self.u_em_dashes, '--', text)
        text = re.sub(self.u_infinity_signs, '__INF__', text)
        return text

    def _preprocess_custom_general(self, text):
        # doing this so nltk tokenizer works better
        text = re.sub(r'\$', '__DOLLAR_SIGN__', text)
        text = re.sub(r'%', '__PERCENT_SIGN__', text)
        text = re.sub(r'\^', '__CARET__', text)
        text = re.sub(r'&', '__AMPERSAND__', text)
        text = re.sub(r'\*', '__ASTERISK__', text)

        # deal with music "decades": change e.g. 1980 to '80s and '80s to 80s
        text = re.sub(r"(19|20)(\d0)s", r"'\2s", text)
        text = re.sub(r"'(\d0s)", r"\1", text)

        # get rid of pesky newlines, carriages, tabs which screw up tokenizer
        text = re.sub(r'(\n|\r|\t)', ' ', text)

        return text

    def _preprocess_custom_specific(self, text):
        # band chk chk chk
        text = re.sub(r"""\B!!!([\s,.'"])""", r'chk chk chk\1', text)

        # bands like Matt & Kim
        text = re.sub(r'\b([A-Z][a-z]*)(\s+)&(\s+)+([A-Z][a-z]*)\b',
                      r'\1__AMPERSAND__\4', text)

        ################################
        #  custom genre manipulation   #
        ################################

        # genres that have a -rock, -pop, etc.
        text = re.sub(r'(\w+)-rock\b', r'\1_rock', text,
                      flags=re.IGNORECASE)
        text = re.sub(r'(\w+)-pop\b', r'\1_pop', text,
                      flags=re.IGNORECASE)
        text = re.sub(r'(\w+)-punk\b', r'\1_punk', text,
                      flags=re.IGNORECASE)
        text = re.sub(r'(\w+)-metal\b', r'\1_metal', text,
                      flags=re.IGNORECASE)
        text = re.sub(r'(\w+)-country\b', r'\1_country', text,
                      flags=re.IGNORECASE)
        text = re.sub(r'(\w+)-blues\b', r'\1_blues', text,
                      flags=re.IGNORECASE)
        text = re.sub(r'(\w+)-hop', r'\1_hop', text,
                      flags=re.IGNORECASE)
        text = re.sub(r'(\w+)-rap\b', r'\1_rap', text,
                      flags=re.IGNORECASE)
        text = re.sub(r'(\w+)-R&B\b', r'\1_R&B', text,
                      flags=re.IGNORECASE)
        text = re.sub(r'(\w+)-jazz\b', r'\1_jazz', text,
                      flags=re.IGNORECASE)
        text = re.sub(r'(\w+)-soul\b', r'\1_soul', text,
                      flags=re.IGNORECASE)
        text = re.sub(r'(\w+)-classical\b', r'\1_classical', text,
                      flags=re.IGNORECASE)
        text = re.sub(r'(\w+)-contemporary\b', r'\1_contemporary', text,
                      flags=re.IGNORECASE)
        text = re.sub(r'(\w+)-techno\b', r'\1_techno', text,
                      re.IGNORECASE)
        text = re.sub(r'(\w+)-electronica\b', r'\1_electronica', text,
                      flags=re.IGNORECASE)

        # split quoted verses (usually rap lyrics)
        text = re.sub(r'/', ' / ', text)

        return text

    def __call__(self, text):
        text = self._preprocess_unicode(text)
        text = self._preprocess_custom_specific(text)
        return text
        text = self._preprocess_custom_general(text)

        # lowercase last!
        text = self._lowercase(text)
        return text
