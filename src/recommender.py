from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation

import text_preprocess as textpre


def extended_tfidf(df):
    '''
    Trains an extended TFIDF model with custom text preprocessor
    and custom tokenizer

    Args:
        df: dataframe with Pitchfork reviews
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
                            preprocessor=ctpre, tokenizer=ctok,
                            max_df=0.75, min_df=5)
    tfidf_trans = tfidf.fit_transform(entries)
    return tfidf, tfidf_trans


def extended_lsi(df, n_components=200):
    '''
    Trains an extended LSI model with custom text preprocessor
    and custom tokenizer

    Args:
        df: dataframe with Pitchfork reviews
    Returns:
        tfidf: sklearn fitted TfidfVectorizer
        tfidf_trans: sparse matrix with tfidf transformed data
        svd: sklearn fitted TruncatedSVD
        svd_trans: dense array with lsi transformed data
    '''

    print('Starting TfIdf')
    tfidf, tfidf_trans = extended_tfidf(df)

    print('Starting SVD')
    svd = TruncatedSVD(n_components=n_components)
    svd_trans = svd.fit_transform(tfidf_trans)

    return tfidf, tfidf_trans, svd, svd_trans


def extended_lda(df):
    '''
    Trains an extended LDA model with custom text preprocessor
    and custom tokenizer

    Args:
        df: dataframe with Pitchfork reviews
    Returns:
        tfidf: sklearn fitted TfidfVectorizer
        tfidf_trans: sparse matrix with tfidf transformed data
        lda: sklearn fitted LatentDirichletAllocation
        lda_trans: dense array with lda transformed data
    '''

    tfidf, tfidf_trans = extended_tfidf(df)
    lda = LatentDirichletAllocation(n_topics=100, max_iter=5)
    lda_trans = lda.fit_transform(tfidf_trans)

    return tfidf, tfidf_trans, lda, lda_trans
