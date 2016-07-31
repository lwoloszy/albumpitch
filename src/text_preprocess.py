import re
import string
from itertools import groupby

import nltk


class CustomTokenizer(object):
    def __init__(self, stopset=None, stemmer=None):
        self.punctset = set(string.punctuation)

        if stopset:
            self.stopset = set(stopset)
        else:
            self.stopset = set()

        if stemmer:
            self.stemmer = stemmer.stem
        else:
            self.stemmer = lambda x: x

    def __call__(self, doc):
        # tokenize into sentences
        sents = nltk.tokenize.sent_tokenize(doc)

        # tag with putative parts of speech
        word_pos_tups_lst = [nltk.pos_tag(nltk.word_tokenize(sent))
                             for sent in sents]

        # merge consecutive proper nouns in hopes of identifying band names
        words = [merge_proper_nouns(word_pos_tups)
                 for word_pos_tups in word_pos_tups_lst]

        # unpack and lowercase
        words = [word.lower() for sublist in words for word in sublist]

        # split on hyphens and remove stopwords/punctuation
        words = [subword for word in words for subword in word.split('-')
                 if subword.strip(string.punctuation) and subword not in self.stopset]

        # remove stopwords/punctuation
        # words = filter(lambda x: (x.strip(string.punctuation) and
        #                          x not in self.stopset),
        #               words)

        # strip punctuation and stem
        words = [self.stemmer(word.strip(string.punctuation))
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

        # keep all ampersands
        text = re.sub(r'&', '__AMPERSAND__', text)

        # keep dollar signs if they don't precede digit
        text = re.sub(r'\$(?!\d)', '__DOLLAR_SIGN__', text)

        # convert symbol % to word if it follows digit
        text = re.sub(r'(?<=\d)%', ' percent', text)

        # deal with music "decades": change e.g. 1980s to '80s and 80s to '80s
        text = re.sub(r"\b(19|20|'?)(\d0)s", r"'\2__s", text)

        # add decades information to articles that only have specific year
        text = re.sub(r"\b(19|20)(\d)(\d)\b", r"\1\2\3 '\g<2>0__s", text)

        # Get rid of pesky newlines, carriages, tabs which screw up tokenizer
        text = re.sub(r'(\n|\r|\t)', ' ', text)

        return text

    def _preprocess_custom_specific(self, text):
        # merge consecutive capitalized words, but not if they start a sentence
        # meant to extract proper names and some percentage of band names
        #text = re.sub(r'(?<![.!?]\s)([A-Z][\w$&%]*)\s+([A-Z][\w$&%]*)',
        #              r'\1_\2', text)

        # band chk chk chk
        text = re.sub(r"""\s!!!([\s,.'"])""", r'chk_chk_chk\1', text)

        # glue together things joined with an ampersand
        text = re.sub(r'\s+&\s+', '&', text)

        # genres that have a -rock, -pop, R&B, etc. we will create joined
        # words but also include the separate terms
        genres = ['(rock', 'pop', 'punk', 'metal', 'country', 'blues', 'hop',
                  'rap', 'R&B', 'jazz', 'soul', 'classical', 'songwriter',
                  'contemporary', 'funk', 'techno', 'wave', 'electronica)']
        regex_part = '|'.join(genres)
        regex_full = re.compile(r'(\w+)[-/]' + regex_part, flags=re.IGNORECASE)
        text = regex_full.sub(r'\1 \2 \1_\2', text)

        # split quoted verses (usually rap lyrics, but now always)
        text = re.sub(r'/', ' ', text)
        text = re.sub(r'[\(\)]', '', text)

        return text

    def __call__(self, text):
        text = self._preprocess_unicode(text)
        text = self._preprocess_custom_specific(text)
        text = self._preprocess_custom_general(text)

        # lowercase last!
        # text = self._lowercase(text)
        return text


def prepend_first_name(artist, text):
    fn, ln = artist.split()
    fn = fn.replace(')', '\)')
    ln = ln.replace(')', '\)')
    fn = fn.replace('(', '\(')
    ln = ln.replace('(', '\(')
    regex = re.compile(r'(?<!{:s})\s+({:s})'.format(fn, ln))
    replacement = ' {:s}'.format(fn) + r'_\1'
    return regex.sub(replacement, text)


def merge_proper_nouns(words_pos_tups):
    out = []
    iterator = groupby(words_pos_tups,
                       lambda x: 'NNP' in x or 'NNPS' in x)
    for key, group in iterator:
        if key:
            out.append('_'.join([elem[0] for elem in group]))
        else:
            out.extend([elem[0] for elem in group])
    return out
