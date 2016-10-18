import os
import dill
import numpy as np
import eda
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation

import text_preprocess as textpre

basedir = os.path.dirname(os.path.abspath(__file__))


def extended_tfidf(df, norm='l2', use_idf=True, sublinear_tf=False):
    '''
    Trains an extended TFIDF model with custom text preprocessor
    and custom tokenizer

    Args:
        df: dataframe with Pitchfork reviews
        norm: the norm to use in tfidf (default=l2)
        use_idf: whether to use idf in tfidf (default=True)
        sublinear_tf: log transform tf counts (default=False)

    Returns:
        tfidf: sklearn fitted TfidfVectorizer
        tfidf_trans: sparse matrix with tfidf transformed data
    '''

    abstracts = df['abstract'].tolist()
    reviews = df['review'].tolist()
    genres = df['genres'].map(lambda x: ' '.join(x)).tolist()
    artists = df['artists'].map(lambda x: ' '.join(x)).tolist()
    albums = df['album'].tolist()
    labels = df['labels'].map(lambda x: ' '.join(x)).tolist()

    new_reviews = []
    for i, (artist, review) in enumerate(zip(artists, reviews)):
        new_reviews.append(textpre.prepend_first_name(artist, review))

    reviews = new_reviews
    together = [abstracts, reviews, genres, artists, albums, labels]
    entries = [' '.join(entry) for entry in zip(*together)]

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
                            preprocessor=ctpre,
                            tokenizer=ctok,
                            max_df=0.75, min_df=5,
                            sublinear_tf=sublinear_tf,
                            use_idf=use_idf, norm=norm)
    tfidf_trans = tfidf.fit_transform(entries)
    return tfidf, tfidf_trans


def extended_lsi(df, n_components=200):
    '''
    Trains an extended LSI model with custom text preprocessor
    and custom tokenizer

    Args:
        df: dataframe with Pitchfork reviews
        n_components: number of components in LSI model
    Returns:
        tfidf: sklearn fitted TfidfVectorizer
        tfidf_trans: sparse matrix with tfidf transformed data
        svd: sklearn fitted TruncatedSVD
        svd_trans: dense array with lsi transformed data
    '''

    print('Starting TfIdf')
    tfidf, tfidf_trans = extended_tfidf(df, sublinear_tf=True)

    print('Starting SVD')
    svd = TruncatedSVD(n_components=n_components)
    svd_trans = svd.fit_transform(tfidf_trans)

    return tfidf, tfidf_trans, svd, svd_trans


def extended_lda(df, n_topics=200):
    '''
    Trains an extended LDA model with custom text preprocessor
    and custom tokenizer

    Args:
        df: dataframe with Pitchfork reviews
        n_topics: number of topis in LDA model
    Returns:
        tfidf: sklearn fitted TfidfVectorizer
        tfidf_trans: sparse matrix with tf! transformed data
        lda: sklearn fitted LatentDirichletAllocation
        lda_trans: dense array with lda transformed data
    '''

    print('Starting TfIdf')
    # for LDA, use raw counts; that is, tfidf with appropriate parameters
    tfidf, tfidf_trans = extended_tfidf(df, use_idf=False, norm=None)

    print('Starting LDA')
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5)
    lda_trans = lda.fit_transform(tfidf_trans)

    return tfidf, tfidf_trans, lda, lda_trans


def train_and_save_models(n_components=200):
    df = eda.get_documents()
    tfidf, tfidf_trans, svd, svd_trans = extended_lsi(df, n_components)
    save_models(df['url'].values, tfidf, svd, svd_trans)


def save_models(urls, tfidf, svd, svd_trans):
    path = basedir + '/../flask_app/app/main/models/'
    np.save(path + 'urls.npy', urls)
    np.save(path + 'svd_trans.npy', svd_trans)
    save_dill(path + 'tfidf.dill', tfidf)
    save_dill(path + 'svd.dill', svd)


def save_dill(filename, obj):
    with open(filename, 'w') as f:
        f.write(dill.dumps(obj))
