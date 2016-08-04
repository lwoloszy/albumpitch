import re
import string
from itertools import groupby
from unidecode import unidecode

import nltk
from nltk.stem import PorterStemmer, SnowballStemmer

from stop_words import get_stop_words

from pymongo import MongoClient


class CustomTokenizer(object):

    def __init__(self, stopset=None, stemmer=None):
        if stopset:
            self.stopset = set(stopset)
        else:
            self.stopset = set()

        if stemmer:
            self.stemmer = stemmer
        else:
            self.stemmer = lambda x: x

    def tokenize(self, text):
        words = nltk.word_tokenize(text)

        # split on hyphens manually and lowercase
        words = [subword.lower() for word in words for subword in word.split('-')]

        # strip punctuation, remove stopwords and stem
        words = [self.stemmer.stem(word.strip(string.punctuation))
                 for word in words
                 if word not in self.stopset and word.strip(string.punctuation)]

        return words

    __call__ = tokenize


class CustomTokenizerWithPOS(CustomTokenizer):

    def __init__(self, stopset=None, stemmer=None):
        super(CustomTokenizerWithPOS, self).__init__(stopset=stopset,
                                                     stemmer=stemmer)

    def tokenize(self, text):
        # tokenize into sentences
        sents = nltk.tokenize.sent_tokenize(text)

        # tag with putative parts of speech
        word_pos_tups_lst = [nltk.pos_tag(nltk.word_tokenize(sent))
                             for sent in sents]

        # merge consecutive proper nouns in hopes of identifying band names
        words = [merge_proper_nouns(word_pos_tups)
                 for word_pos_tups in word_pos_tups_lst]

        # unpack, lowercase and strip punctuation
        words = [word.lower().strip(string.punctuation)
                 for sublist in words for word in sublist]

        # split on hyphens and remove zero length strings and stopwords
        words = [subword for word in words for subword in word.split('-')
                 if subword and subword not in self.stopset]

        # stem
        words = [self.stemmer(word) for word in words]

        return words

    __call__ = tokenize


class CustomTextPreprocessor(object):

    def __init__(self, merge_capitalized=True):
        # are we going to merge consecutive capitalized words?
        self.merge_capitalized = merge_capitalized

    def normalize(self, text):
        # keep hip-hop as one word
        text = re.sub(r'(hip)[-/\s](hop)', r'\1_\2', text)

        # keep singer/songwriter as one word
        text = re.sub(r'(singer)[-/\s](songwriter)', r'\1_\2', text)

        # subgenre discovery... construct "bigrams" but only for musical
        # terms, separator can be hyphen, forward slash or space character
        genres = ['(rock', 'pop', 'punk', 'metal', 'country', 'blues',
                  'rap', 'R&B', 'jazz', 'soul', 'classical', 'hip_hop',
                  'contemporary', 'funk', 'techno', 'wave', 'electro',
                  'electronica?', 'experimental', 'singer_songwriter)']
        regex = re.compile(r'\b(\w{3,})[-/\s]' + '|'.join(genres))
        text = regex.sub(r'\1 \2 \1_\2', text)

        # merge consecutive capitalized words, but not if at the start
        # of sentence (or something resembling that)
        if self.merge_capitalized:
            text = re.sub(
                r'(?<![.!?])([-\s\"\']+)([A-Z]+[\w$&%]*)\s+([A-Z]+[\w$&%]*)',
                r'\1\2_\3', text)

        # band chk chk chk, doing the best I can here, likely not perfect
        text = re.sub(r"""\s!!!([\s,.'"])""", r'chk_chk_chk\1', text)

        # DJ/rupture correction
        text = re.sub(r'DJ\s?/rupture', 'DJ_Rupture', text)

        # split quoted verses (usually lyrics, but not always)
        text = re.sub(r'/', ' ', text)

        # remove parentheses as these screw up part of speech tagger
        text = re.sub(r'[\(\)]', '', text)

        # keep embedded ampersands (mostly for R&B)
        text = re.sub(r'(\w+)&(\w+)', r'\1__AMPERSAND__\2', text)

        # keep dollar signs if they don't precede digit (rapper names)
        text = re.sub(r'\$(?!\d)', '__DOLLAR_SIGN__', text)

        # keep exclamation points if they follow capitalized word (band names)
        text = re.sub(r'([A-Z]+\w*)!', r'\1__EXCLAMATION__', text)

        # convert symbol % to word if it follows digit
        text = re.sub(r'(?<=\d)%', ' percent', text)

        # deal with music "decades": change e.g. 1980s to '80s and 80s to '80s
        text = re.sub(r"\b(19|20| )(\d0)'?s\b", r"'\2__s", text)

        # add decades information to articles that only have specific year
        text = re.sub(r"\b(19|20)(\d)(\d)\b", r"\1\2\3 '\g<2>0__s", text)

        # Get rid of pesky newlines, carriages, tabs which screw up tokenizer
        text = re.sub(r'(\n|\r|\t)', ' ', text)

        return text

    def preprocess(self, text):
        text = unidecode(text)
        text = self.normalize(text)
        return text

    __call__ = preprocess


# utility functions for pre-processing text

def prepend_first_name(artist, text):
    artist_parts = artist.split()

    if len(artist_parts) != 2:
        return text

    if re.match(r'\b(the|a|an)\b', artist_parts[0], re.IGNORECASE):
        return text

    artist = artist.encode('utf-8')
    text = text.encode('utf-8')

    fn, ln = artist.split()
    fn = fn.replace(')', '\)')
    ln = ln.replace(')', '\)')
    fn = fn.replace('(', '\(')
    ln = ln.replace('(', '\(')
    regex = re.compile(r'(?<!{:s})\s+({:s})'.format(fn, ln))
    replacement = ' {:s}'.format(fn) + r' \1'
    out = regex.sub(replacement, text).decode('utf-8')

    return out


def merge_proper_nouns(word_pos_tups):
    out = []
    iterator = groupby(word_pos_tups,
                       lambda x: 'NNP' in x or 'NNPS' in x)
    for key, group in iterator:
        if key:
            out.append('_'.join([elem[0] for elem in group]))
        else:
            out.extend([elem[0] for elem in group])
    return out


def discover_subgenres(all_text):
    genres = ['(rock', 'pop', 'punk', 'metal', 'country', 'blues', 'hip-hop',
              'rap', 'R&B', 'jazz', 'soul', 'classical', 'songwriter',
              'contemporary', 'funk', 'folk', 'techno', 'wave', 'electronic)']
    regex_part = '|'.join(genres)
    regex_full = re.compile(r'\s(\w{3,})[-/]\b' + regex_part)
    subgenres = set([(item[0].lower(), item[1].lower())
                     for item in regex_full.findall(all_text)])
    with open('../data/subgenres.txt', 'w') as f:
        for subgenre in subgenres:
            f.write('{:s} {:s}\n'.format(*subgenre))
        f.write('hip hop\n')


def tokenize_and_save():
    client = MongoClient()
    db = client['albumpitch']
    coll = db['pitchfork']

    stemmer = get_stemmer()
    stopset = get_stopset()
    text_preprocessor = CustomTextPreprocessor(
        subgenres_file='../data/subgenres.txt', merge_capitals=False)
    text_tokenizer = CustomTokenizerWithPOS(stopset=stopset, stemmer=stemmer)

    for i, doc in enumerate(coll.find()):
        print(i)

        text = concat_review_elements(doc)

        processed_text = text_preprocessor.preprocess(text)
        tokenized_text = text_tokenizer.tokenize(processed_text)

        coll.update_one(
            {'_id': doc['_id']},
            {
                '$set': {'tokenized_text': tokenized_text},
                '$currentDate': {'lastModified': True}
            })

    client.close()


def pretokenize(df):
    stopset = get_stopset()
    stemmer = get_stemmer('snowball')
    preprocessor = CustomTextPreprocessor(merge_capitalized=True)
    tokenizer = CustomTokenizer(stopset=stopset, stemmer=stemmer)

    texts = []
    for i, doc in enumerate(df.itertuples(), 1):
        if i % 100 == 0:
            print('Tokenized {:d} documents'.format(i))
        doc = doc._asdict()
        text = concat_review_elements(doc)
        texts.append(tokenizer(preprocessor(text)))
    df.loc[:, 'tokenized_text'] = texts


def concat_review_elements(doc, prepend=True):
    abstract = doc['abstract']
    review = doc['review']
    genre = u' '.join(doc['genres'])
    artist = u' '.join(doc['artists'])
    label = u' '.join(doc['labels'])
    album = doc['album']

    if prepend:
        review = prepend_first_name(artist, review)

    entry = [abstract, review, genre, artist, album, label]
    return ' '.join(entry)


def get_stopset():
    stopset = set(get_stop_words('en'))

    # get those contractions
    add_stops = nltk.word_tokenize(' '.join(stopset))
    stopset.update(add_stops)

    # make sure to get contractions without punctuation, so that
    # order of operations doesn't matter later
    add_stops = [stopword.strip(string.punctuation)
                 for stopword in stopset]
    stopset.update(add_stops)

    # custom stop words
    add_stops = [u'lp', u'ep',
                 u'record', u'records', u'recorded'
                 u'label', u'labels',
                 u'release', u'releases', u'released',
                 u'listen', u'listens', u'listened', u'listener',
                 u'version', u'versions',
                 u'album', u'albums',
                 u'song', u'songs',
                 u'track', u'tracks',
                 u'sound', u'sounds',
                 u'thing', u'things', u'something',
                 u'music']
    stopset.update(add_stops)
    return stopset


def get_stemmer(name):
    if name == 'porter':
        return PorterStemmer()
    elif name == 'snowball':
        return SnowballStemmer('english').stemmer
