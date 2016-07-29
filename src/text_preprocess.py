import re
import string

import nltk
from nltk.stem.porter import PorterStemmer


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

        # deal with music "decades": change e.g. 1980s to '80s and 80s to '80s
        text = re.sub(r"(19|20|'?)(\d0)s", r"'\2__s", text)

        # get rid of pesky newlines, carriages, tabs which screw up tokenizer
        text = re.sub(r'(\n|\r|\t)', ' ', text)

        return text

    def _preprocess_custom_specific(self, text):
        # merge consecutive capitalized words, but not if they start a sentence
        # meant to extract proper names and some portion of band names
        text = re.sub(r'(?<![.!?]\s)([A-Z][\w$&%]*)\s+([A-Z][\w$&%]*)',
                      r'\1_\2', text)

        # band chk chk chk
        text = re.sub(r"""\s!!!([\s,.'"])""", r'chk_chk_chk\1', text)

        # glue together things joined with an ampersand
        # text = re.sub(r'\b([A-Z][a-z]*)(\s+)&(\s+)+([A-Z][a-z]*)\b',
        #              r'\1__AMPERSAND__\4', text)
        text = re.sub(r'\s+&\s+', '&', text)

        ################################
        #  custom genre manipulation   #
        ################################

        # genres that have a -rock, -pop, R&B, etc.
        genres = ['(rock', 'pop', 'punk', 'metal', 'country', 'blues', 'hop',
                  'rap', 'R&B', 'jazz', 'soul', 'classical', 'songwriter',
                  'contemporary', 'funk', 'techno', 'electronica)']
        regex_part = '|'.join(genres)
        regex_full = re.compile(r'(\w+)[-/]' + regex_part, flags=re.IGNORECASE)
        text = regex_full.sub(r'\1_\2', text)

        # split quoted verses (usually rap lyrics, but now always)
        text = re.sub(r'/', ' / ', text)

        return text

    def __call__(self, text):
        text = self._preprocess_unicode(text)
        text = self._preprocess_custom_specific(text)
        text = self._preprocess_custom_general(text)

        # lowercase last!
        text = self._lowercase(text)
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
