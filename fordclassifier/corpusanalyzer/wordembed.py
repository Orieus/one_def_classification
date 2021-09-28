# -*- coding: utf-8 -*-
'''

@author:  Angel Navia Vázquez
May 2018

import code
code.interact(local=locals())
'''

from gensim.models import Word2Vec
import numpy as np
import pickle
import os


class WordEmbedding(object):
    '''
    Class to train and predict word embeddings from a collection of texts
    *** Warning ***, training may take VEEEERY LONG, depending on the number of iterations.

    ============================================================================
    Methods:
    ============================================================================
    obtain_sentences:       receives as input a dataframe and produces the list of tokenized, sentences for WE training filtering out those projects without summary in spanish
    train:                  Train the word embedding
    predict:                Obtain the word embedding for a given word
    obtain_wedict:          Obtain the WE as a dict
    save_wedict:            Save the WE dict
    load_wedict:            Load the WE dict
    closest:                Simple (brute force) distance measurement to find
                            the closest element in the vocabulary given a WE
    closest_words:          returns most similar words
    tokens_dist:            Distance between two sets of tokens

    ============================================================================
    '''

    def __init__(self, project_path, dim, nepochs, subfolders, verbose=True):

        self._project_path = None   # working path
        self.sentences = None  # List of tokenized sentences 2 be used in train
        self.model = None      # Gensim Word2Vec model
        self.verbose = None    # messages are printed on screen if verbose=True
        self.vocab = None      # vocabulary in the model
        self.wedict = None     # Dict storing the WE vectors, to save as pickle
        self._subfolders = subfolders   # subfolders structure
        self._min_ocurr = None   # min word occurences 2 be in the model
        self._dim = None         # Dimension of the output vectors
        self._nepochs = None     # Number of epochs
        self.Xwe = None          # Matrix storing the embeddings
        self.xTx = None          # Matrix storing the inner product between embeddings
        self.we_words = None     # words in the embedding
        self.pesos_dict = None   # dictionary with the mean values of the embeddings
        self.inv_words = None    # Inverted word dictionary

        self._project_path = project_path
        # PENDING: read parameters from a configuration file
        self._min_ocurr = 5    # minimum occurrences of a word to be taken into account
        self._dim = dim        # dimension of the embedding space
        self._nepochs = nepochs
        # empty WE model
        self.model = Word2Vec(size=self._dim, window=8,
                              min_count=self._min_ocurr, workers=50)
        self.verbose = verbose

        for subfolder in list(subfolders.keys()):
            folder = os.path.join(self._project_path, subfolders[subfolder])
            if not os.path.exists(folder):
                os.makedirs(folder)

        try:
            # tries to recover the state, loads the data if it was previously
            # trained
            self._recover('Xwe')
            if self.verbose:
                print('Loaded Xwe, xTx, we_words')
            self._recover('wedict')
            if self.verbose:
                print('Loaded we_dict')
            self._recover('model')
            if self.verbose:
                print('Loaded model')

            pesos = np.mean(self.Xwe, axis=1)
            self.pesos_dict = {}
            for k in range(0, len(self.we_words)):
                self.pesos_dict.update({self.we_words[k]: pesos[k]})
            self.inv_words = {}
            for k in range(0, len(self.we_words)):
                word = self.we_words[k]
                self.inv_words.update({word: k})
        except:
            pass

    def _recover(self, field):
        '''
        Loads from disk a previously stored variable, to avoid recomputing it
        '''
        # field:    variable to restore from disk

        if field == 'Xwe':
            input_file = os.path.join(
                self._get_folder('we'), 'WEmatrices_dim_' + str(self._dim) +
                '_nepochs_' + str(self._nepochs) + '.pkl')
            with open(input_file, 'rb') as f:
                [self.Xwe, self.xTx, self.we_words] = pickle.load(f)

        if field == 'wedict':
            filename = ('wedict_dim_' + str(self._dim) + '_nepochs_' +
                        str(self._nepochs) + '.pkl')
            path_filename = os.path.join(self._get_folder('we'), filename)
            with open(path_filename, 'rb') as f:
                self.wedict = pickle.load(f)

        if field == 'model':
            filename = ('WEmodel_dim_' + str(self._dim) + '_nepochs_' +
                        str(self._nepochs) + '.pkl')
            path_filename = os.path.join(self._get_folder('we'), filename)
            with open(path_filename, 'rb') as f:
                self.model = pickle.load(f)
        return

    def _get_folder(self, subfolder):
        '''
        gets full path to a folder
        '''
        # subfolder:    subfolder name
        return os.path.join(self._project_path, self._subfolders[subfolder])

    def obtain_sentences(self, df):
        '''
        Obtains from the DF the sentences to be used for the WE training
        '''
       # df: dataframe with the input data

        df = df.loc[df['Resumen_lang'] == 'es']
        titulo_lang = list(df.loc[:, 'Titulo_lang'])
        resumen_lemas = list(df.loc[:, 'Resumen_lemas'])
        titulo_lemas = list(df.loc[:, 'Titulo_lemas'])

        Ndocs = df.count()[0]
        self.sentences = []
        for kdoc in range(0, Ndocs):
            if titulo_lang[kdoc] == 'es':
                doc = titulo_lemas[kdoc]
                doc = doc.lower()
                frases = doc.split('\n')
                tokens = [fr.split(' ') for fr in frases]
                self.sentences += doc.split('\n')
            doc = resumen_lemas[kdoc]
            doc = doc.lower()
            frases = doc.split('\n')
            tokens = [fr.split(' ') for fr in frases]
            self.sentences += tokens

    def train(self):
        '''
        Trains the model
        '''
        if self.verbose:
            print("-" * 50)
            print('Obtaining vocabulary...')
        self.model.build_vocab(self.sentences)
        vocab = list(self.model.wv.vocab.keys())
        vocab.sort()
        self.vocab = vocab

        if self.verbose:
            print('Done!')
            print("-" * 50)
        if self.verbose:
            print("-" * 50)
            print('Training Word Embedding Model with ' +
                  '{0} sentences and {1} different words'.format(
                    len(self.sentences), len(vocab)))
            print('Warning!, this may take **VERY LONG**...')
        self.model.train(self.sentences, total_examples=len(self.sentences),
                         epochs=self._nepochs)

        self.wedict = {}
        for word in self.vocab:
            try:
                self.wedict.update({word: self.model[word]})
            except:
                pass

        filename = ('wedict_dim_' + str(self._dim) + '_nepochs_' +
                    str(self._nepochs) + '.pkl')
        path_filename = os.path.join(self._get_folder('we'), filename)
        with open(path_filename, 'wb') as f:
            pickle.dump(self.wedict, f)

        if self.verbose:
            print("-" * 50)
            print('WEdict saved to %s' % filename)
            print("-" * 50)

        N = len(self.wedict)
        we_words = list(self.wedict.keys())
        vects = list(self.wedict.values())
        Xwe = vects[0].reshape((1, self._dim))
        for k in range(1, N):
            Xwe = np.vstack((Xwe, vects[k].reshape((1, self._dim))))

        xTx = np.zeros(N)
        for k in range(0, N):
            xTx[k] = np.dot(vects[k], vects[k])
        xTx = xTx.reshape((N, 1))

        filename = os.path.join(self._get_folder('we'), 'WEmatrices_dim_' +
                                str(self._dim) + '_nepochs_' +
                                str(self._nepochs) + '.pkl')
        with open(filename, 'wb') as f:
            pickle.dump([Xwe, xTx, we_words], f)

        if self.verbose:
            print("-" * 50)
            print('WEmatrices saved to %s' % filename)
            print("-" * 50)

        filename = os.path.join(self._get_folder('we'), 'WEmodel_dim_' +
                                str(self._dim) + '_nepochs_' +
                                str(self._nepochs) + '.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

        if self.verbose:
            print("-" * 50)
            print('WE model saved to %s' % filename)
            print("-" * 50)

        if self.verbose:
            print('Done!')
            print("-" * 50)
        return

    def predict(self, word):
        '''
        Obtain the word embedding for a given word
        '''
        # word: word to map to the embedding space
        x = None
        try:
            x = self.model[word]
        except:
            print("Error: word not in vocabulary")
            pass
        return x

    def most_similar(self, pos_words, neg_words, N):
        '''
        # find words mostly similar      
        '''
        return self.model.wv.most_similar(positive=pos_words,
                                          negative=neg_words, topn=N)

    def most_similar_cosmul(self, pos_words, neg_words, N):
        '''
        # find words mostly similar
        '''
        # find words mostly similar cosine distance
        return self.model.wv.most_similar_cosmul(positive=pos_words,
                                                 negative=neg_words, topn=N)

    def closest_words(self, word, M):
        '''
        returns most similar words
        '''
        # word:      reference word
        # M:         number of words to retrieve

        nword = self.we_words.index(word)
        dim = self.Xwe.shape[1]
        y = self.Xwe[nword, :].reshape((dim, 1))
        N = self.Xwe.shape[0]

        # This term does not affect to the distance comparison but we maintain
        # it for completeness
        yTy = np.ones(N) * np.dot(y.T, y)
        dists = self.xTx - 2 * np.dot(self.Xwe, y) + yTy.T
        dists[nword] = 9999999999.0
        index = np.argsort(dists, axis=0)
        result = []
        for k in range(0, M):
            cual = index[k][0]
            result.append((self.we_words[cual], dists[cual][0]))
        return result

    '''
    def closest_words_cosine(self, word, M):
        nword = self.we_words.index(word)
        dim = self.Xwe.shape[1]
        y = self.Xwe[nword, :].reshape((dim, 1))
        N = self.Xwe.shape[0]
        # this term does not affect to the distance comparison but we
        # maintain it for completeness
        yTy = np.ones(N) * np.dot(y.T, y)

        cosine_dists = np.dot(self.Xwe, y)
        cosine_dists = np.divide(cosine_dists, self.xTx)
        index = np.argsort(-1 * cosine_dists, axis=0)

        import code
        code.interact(local=locals())

        dists = self.xTx - 2 * np.dot(self.Xwe, y) + yTy.T
        dists[nword] = 9999999999.0
        index = np.argsort(dists, axis=0)
        result = []
        for k in range(0, M):
            cual = index[k][0]
            result.append((self.we_words[cual], dists[cual][0]))
        return result
    '''

    # Distance between two sets of tokens
    '''
    '''
    def tokens_dist_old(self, tokens1, tokens2):
        N1 = len(tokens1)
        N2 = len(tokens2)
        D = np.zeros((N1, N2))
        for k1 in range(0, N1):
            for k2 in range(0, N2):
                try:
                    v1 = self.wedict[tokens1[k1]]
                    v2 = self.wedict[tokens2[k2]]
                    e = v1 - v2
                    dist2 = np.dot(e, e)
                    '''
                    if len(self.pesos_dict) > 0:
                        try:
                            dist2 = dist2 / self.pesos_dict[tokens2[k2]]
                        except:
                            #print('ERROR WE_dist')
                            pass
                    '''
                    D[k1, k2] = dist2
                except:
                    D[k1, k2] = 9999.0
        mindists = np.zeros((N1, 1))
        for k1 in range(0, N1):
            mindists[k1] = min(D[k1, :])
        dmin = np.mean(mindists)
        return D, dmin

    '''
    # Distance between two sets of tokens
    '''
    def tokens_dist(self, tokens1, tokens2, sigma):
        # tokens1:      set of tokens no. 1
        # tokens2:      set of tokens no. 2
        # sigma:        smoothing parameter
        dmin1 = None
        dmin2 = None

        # Identify the words and the index of every set, the input 
        # words may not be present in the WE vocabulary...
        cuales_1 = []
        words1 = []
        for t in tokens1:
            try:
                cuales_1.append(self.inv_words[t])
                words1.append(t)
            except:
                pass

        cuales_2 = []
        words2 = []
        for t in tokens2:
            try:
                cuales_2.append(self.inv_words[t])
                words2.append(t)
            except:
                pass

        # control
        # if len(words1) == 0 or len(words2) == 0:
        #    import code
        #    code.interact(local=locals())

        if len(cuales_1) > 0 and len(cuales_2) > 0:   # we only compute the distance between nonempty sets
            N1 = len(cuales_1)
            N2 = len(cuales_2)

            Xwe1 = self.Xwe[cuales_1, :]
            Xwe2 = self.Xwe[cuales_2, :]

            x1Tx = self.xTx[cuales_1]
            x2Tx = self.xTx[cuales_2]

            '''
            try:
                np.kron(x2Tx, np.ones((1, N1)))
            except:
                print("Error en tokens_dist")
                import code
                code.interact(local=locals())
                pass
            '''

            # Compute the distance matrix using matrix products
            D = (np.kron(x1Tx, np.ones((1, N2))) - 2.0 * np.dot(Xwe1, Xwe2.T) +
                 np.kron(x2Tx.T, np.ones((N1, 1))))

            # Ensure positive values, roundoff problems and rescale
            S = np.mean(self.xTx)
            D = np.abs(D / S)
            
            # Transform distance to kernel, sigma controls the scale
            K = np.exp(-D / sigma)
        else:
            D = 999999999.0
            K = 0
        return K, D, words1, words2
