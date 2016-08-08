# playground for gensim stuff

import logging
import numpy as np
from gensim import corpora, models, similarities, matutils
import text_preprocess as textpre
from sklearn.neighbors import DistanceMetric

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class MyCorpusForDict(object):

    def __init__(self, df):
        self.df = df
        if 'tokenized_text' not in self.df:
            stopset = textpre.get_stopset()
            stemmer = textpre.get_stemmer('snowball')
            self.preprocessor = textpre.CustomTextPreprocessor(
                merge_capitalized=True)
            self.tokenizer = textpre.CustomTokenizer(
                stopset=stopset, stemmer=stemmer)
            self.pretokenized = False
        else:
            self.pretokenized = True

    def __iter__(self):
        for i, doc in enumerate(self.df.itertuples()):
            doc = doc._asdict()
            if self.pretokenized:
                yield doc['tokenized_text']
            else:
                text = textpre.concat_review_elements(doc)
                yield(self.tokenizer(self.preprocessor(text)))


def create_dictionary(df):
    dictionary = corpora.Dictionary(MyCorpusForDict(df))
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    dictionary.compactify()
    return dictionary


def create_corpus(dictionary, df):
    corpus = [dictionary.doc2bow(text) for text in df['tokenized_text']]
    return corpus


def create_tfidf(dictionary, corpus):
    tfidf = models.TfidfModel(corpus, id2word=dictionary)
    corpus_tfidf = tfidf[corpus]
    return tfidf, corpus_tfidf


def create_lsi(dictionary, corpus_tfidf, num_topics=200):
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary,
                          num_topics=num_topics)
    corpus_lsi = lsi[corpus_tfidf]
    return lsi, corpus_lsi


def create_lda(dictionary, corpus, num_topics=100, passes=1):
    lda = models.LdaModel(corpus, id2word=dictionary,
                          num_topics=100, passes=passes)
    corpus_lda = lda[corpus]
    return lda, corpus_lda


def get_doc_topics(lda, bow):
    gamma, _ = lda.inference([bow])
    topic_dist = gamma[0] / sum(gamma[0])  # normalize distribution
    return [(topicid, topicvalue) for topicid, topicvalue in enumerate(topic_dist)]


def get_indexed_sims(model, corpus):
    index = similarities.MatrixSimilarity(model[corpus])
    return index


def print_most_similar_lsi(df, seed_num, dictionary, tfidf, lsi, index, n=10):
    vec_bow = dictionary.doc2bow(df['tokenized_text'].iloc[seed_num])
    vec_tfidf = tfidf[vec_bow]
    vec_lsi = lsi[vec_tfidf]
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    idxs = [tup[0] for tup in sims[0:n]]
    scores = [tup[1] for tup in sims[0:n]]
    df_temp = df.iloc[idxs]
    df_temp['sim_scores'] = scores
    print df_temp[['url', 'genres', 'sim_scores']]


def print_most_similar_lda(df, seed_num, dictionary, lda, index, n=10):
    vec_bow = dictionary.doc2bow(df['tokenized_text'].iloc[seed_num])
    vec_lda = lda[vec_bow]
    sims = index[vec_lda]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    idxs = [tup[0] for tup in sims[0:n]]
    scores = [tup[1] for tup in sims[0:n]]
    df_temp = df.iloc[idxs]
    df_temp['sim_scores'] = scores
    print df_temp[['url', 'genres', 'sim_scores']]


def print_most_similar_lda_hellinger(df, seed_num, X_dense, n=10):

    def hellinger(v1, v2):
        sim = np.sqrt(0.5 * ((np.sqrt(v1) - np.sqrt(v2))**2).sum())
        return sim

    dist = DistanceMetric.get_metric(hellinger)
    dists = dist.pairwise(X_dense[seed_num, :].reshape(1, -1), X_dense).flatten()
    idx = np.argsort(dists)
    sims = 1 - dists
    sims = sims[idx]

    df_temp = df.iloc[idx[0:n]]
    df_temp['sim_scores'] = sims[0:n]
    print df_temp[['url', 'genres', 'sim_scores']]


def lda_to_dense(lda, corpus):
    X_lda = np.zeros((len(corpus), lda.num_topics))
    for i, doc in enumerate(corpus):
        X_lda[i, :] = matutils.sparse2full(get_doc_topics(lda, doc),
                                           lda.num_topics)
    return X_lda


def hellinger(v1, v2):
    sim = np.sqrt(0.5 * ((np.sqrt(v1) - np.sqrt(v2))**2).sum())
    return sim
