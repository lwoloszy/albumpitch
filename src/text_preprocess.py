import re
import string
from itertools import groupby

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
            self.stemmer = stemmer.stem
        else:
            self.stemmer = lambda x: x

    def tokenize(self, text):
        words = nltk.word_tokenize(text)

        # split on hyphens manually
        words = [subword for word in words for subword in word.split('-')]

        # remove stopwords/punctuation and stem
        words = [self.stemmer(word.strip(string.punctuation)) for word in words
                 if word.strip(string.punctuation) and word not in self.stopset]

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

        # strip punctuation and stem
        words = [self.stemmer(word.strip(string.punctuation))
                 for word in words]

        return words

    __call__ = tokenize


class CustomTextPreprocessor(object):

    def __init__(self, subgenres_file=None, merge_capitalized=False):
        self.u_single_quotes = ur"['\u2018\u2019\u0060\u00b4]"
        self.u_double_quotes = ur'["\u201c\u201d]'
        self.u_spaces = ur'\xa0'
        self.u_en_dashes = ur'\u2013'
        self.u_em_dashes = ur'\u2014'

        # are we going to merge consecutive capitalized words?
        self.merge_capitalized = merge_capitalized

        self.subgenres_regex = None
        if subgenres_file:
            with open(subgenres_file) as f:
                subgenres = f.readlines()

            # create regular expression to normalize subgenres styling
            subgenres = [subgenre.strip() for subgenre in subgenres]
            genre_modifiers = [subgenre.split()[0]
                               for subgenre in subgenres]
            genre_bases = [subgenre.split()[1]
                           for subgenre in subgenres]
            regex1 = r'(' + r'|'.join(genre_modifiers) + r')'
            regex2 = r'(' + r'|'.join(genre_bases) + r')'
            regex = regex1 + r'[-/ ]' + regex2
            self.subgenres_regex = re.compile(regex, flags=re.IGNORECASE)

    def _substitute_unicode(self, text):
        text = re.sub(self.u_single_quotes, "'", text)
        text = re.sub(self.u_double_quotes, '"', text)
        text = re.sub(self.u_spaces, ' ', text)
        text = re.sub(self.u_en_dashes, '-', text)
        text = re.sub(self.u_em_dashes, '--', text)
        return text

    def _substitute_custom(self, text):
        # merge consecutive capitalized words, but not if at the start
        # of sentence
        if self.merge_capitals:
            text = re.sub(r'(?<![.!?]\s)([A-Z][\w$&%]*)\s+([A-Z][\w$&%]*)',
                          r'\1_\2', text)

        # band chk chk chk, doing the best I can here, likely not perfect
        text = re.sub(r"""\s!!!([\s,.'"])""", r'chk_chk_chk\1', text)

        # any DJ followed by word that starts with capital letter,
        # glue together (separator can be either a space or hyphen)
        text = re.sub(r'DJ[- ]([A-Z]+\w*)', r'DJ_\1', text)

        # DJ/rupture correction
        text = re.sub(r'DJ ?/rupture', 'DJ_Rupture', text)

        # glue together things joined with an ampersand
        text = re.sub(r'\s+&\s+', '&', text)

        # construct genre 'bigrams' (based on prior discovery)
        # if self.subgenres_regex:
        #    text = self.subgenres_regex.sub(r'\1 \2 \1_\2', text)

        genres = ['(rock', 'pop', 'punk', 'metal', 'country', 'blues', 'hop',
                  'rap', 'R&B', 'jazz', 'soul', 'classical', 'songwriter',
                  'contemporary', 'funk', 'techno', 'wave', 'electro',
                  'electronica)']
        regex_part = '|'.join(genres)
        regex_full = re.compile(r'\b(\w+)[-/ ]' + regex_part)
        text = regex_full.sub(r'\1 \2 \1_\2', text)

        # split quoted verses (usually lyrics, but now always)
        text = re.sub(r'/', ' ', text)

        # remove parentheses as these screw up part of speech tagger
        text = re.sub(r'[\(\)]', '', text)

        # keep all ampersands
        text = re.sub(r'&', '__AMPERSAND__', text)

        # keep dollar signs if they don't precede digit
        text = re.sub(r'\$(?!\d)', '__DOLLAR_SIGN__', text)

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
        text = self._substitute_unicode(text)
        text = self._substitute_custom(text)
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


def tokenize(text, stopset, stemmer):
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
             if subword and subword not in stopset]

    # strip punctuation and stem
    words = [stemmer.stem(word.strip(string.punctuation))
             for word in words]

    return words


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


def concat_review_elements(doc):
    abstract = doc['abstract']
    review = doc['review']
    genre = u', '.join(doc['genres'])
    artist = u', '.join(doc['artists'])
    album = doc['album']

    review = prepend_first_name(artist, review)

    entry = [abstract, review, genre, artist, album]
    return ', '.join(entry)


def get_stopset():
    stopwords = get_stop_words('en')

    # get those contractions
    stopwords.extend(nltk.word_tokenize(' '.join(stopwords)))

    # make sure to get contractions without punctuation, so that
    # order of operations doesn't matter later
    stopwords.extend([stopword.strip(string.punctuation)
                      for stopword in stopwords])

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
                      'thing', 'things', 'something',
                      'music'])
    stopset = set(stopwords)
    return stopset


def get_stemmer(name):
    if name == 'porter':
        return PorterStemmer()
    elif name == 'snowball':
        return SnowballStemmer('english')
