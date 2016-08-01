import logging
from gensim import corpora, models, similarities
import text_preprocess as textpre

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
            print(i)
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


def create_lsi(dictionary, corpus_tfidf):
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=200)
    corpus_lsi = lsi[corpus_tfidf]
    return lsi, corpus_lsi


def create_lda(dictionary, corpus):
    lda = models.LdaModel(corpus, id2word=dictionary, num_topics=100)
    corpus_lda = lda[corpus]
    return lda, corpus_lda


def get_indexed_sims(model, corpus):
    index = similarities.MatrixSimilarity(model[corpus])
    return index


def get_most_similar(df, seed, dictionary, tfidf, lsi, index, n=10):
    vec_bow = dictionary.doc2bow(df['tokenized_text'].iloc[seed])
    vec_tfidf = tfidf[vec_bow]
    vec_lsi = lsi[vec_tfidf]
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    idxs = [tup[0] for tup in sims[0:n]]
    scores = [tup[1] for tup in sims[0:n]]
    df_temp = df.iloc[idxs]
    df_temp['sim_scores'] = scores
    print df_temp[['url', 'genres', 'sim_scores']]
