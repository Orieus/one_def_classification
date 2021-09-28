# -*- coding: utf-8 -*-
'''

@author:  Angel Navia Vázquez
May 2018

import code
code.interact(local=locals())
'''

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class Bow(object):
    '''
    Class to obtain a TFIDF BOW

    ============================================================================
    Methods:
    ============================================================================
    fit(docs):              adjusts the tfidf model using a list of strings,
                            the resulting model is stored in _tfidf
    transform(docs):        obtains tfidf for new data, same format as in fit
    obtain_inv_vocab()  :   obtains the inverse vocabulary
    ============================================================================
    '''

    def __init__(self, min_df=2):
        '''
        Initialization:

        Args:
            min_df: Minimum document frequency for words to be included in
                    the bow
        '''
        self.tfidf = None           # sklearn TfidfVectorizer
        self.inv_vocab = None       # inverse vocabulary index (index -> word)
        self.min_df = min_df

        # # Include any of these parameters in config ???
        self.tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 1),
                                     min_df=self.min_df)

    def fit(self, docs):
        # Fit the TFIDF transformer and obtain the TFIDF representation
        self.tfidf.fit(docs)
        return self.tfidf.transform(docs)

    def transform(self, documents):
        # Obtain the TFIDF representation for a new document (string)
        return self.tfidf.transform(documents)

    def obtain_inv_vocab(self):
        # Obtain inverse vocabulary
        vocab = self.tfidf.vocabulary_
        self.inv_vocab = {}
        for key in list(vocab.keys()):
            self.inv_vocab.update({vocab[key]: key})
        return self.inv_vocab

    def compute_doc_frequencies(self):

        doc_freq = np.sum(self.tfidf > 0, axis=0).tolist()[0]

        return doc_freq
