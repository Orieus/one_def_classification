# -*- coding: utf-8 -*-
'''

@author:  Angel Navia Vázquez
June 2018

Included Models:

- Minimal distance computed on Word Embeddings between a document a set of
  definitions
'''

import os
import pickle
import numpy as np
import json
import matplotlib.pyplot as plt
import operator
import itertools
from nltk.corpus import stopwords
from math import floor
from collections import OrderedDict
from fordclassifier.corpusanalyzer.bow import Bow
import codecs

from time import time


class UnsupClassifier(object):
    '''
    Class to train and evaluate an unsupervised classifier

    ===============================================================================
    Methods:
    ===============================================================================
    _recover:                           if a variable is not on memory, it tries to recover it from disk
    _get_folder:                        retuns full path to a subfolder
    _exists_file:                       check if the file exists in disk
    get_categories_definitions:         obtains the definition of every category
    get_categories_definitions_tfidf:   obtains the definition of every category with tfidf weighting
    load_predicts:                      loads predictions from file
    obtain_predicts:                    obtains unsupervised predictions based on distances computed in the WE space
    obtain_predicts_wordcount:          obtains unsupervised predictions (wordcounts + tfidf)
    obtain_original_labels:             processes the tags in the DB to obtain a list of strings, removes duplicates and eliminates supercategories
    ===============================================================================
    '''

    def __init__(self, project_path, subfolders, verbose=True):
        '''
        Initialization:
        '''
        self._project_path = project_path      # working directory
        self._verbose = verbose                # messages are printed on screen when True
        self._defs_dict = None                 # definitions of the categories, dict with list of tokens 
        self.weighted_dict = None              # dictionary of terms with weights extracted from tfidf 
        self._subfolders = subfolders          # subfolders structure

    def _get_folder(self, subfolder):
        '''
        gets full path to a folder
        '''
        # subfolder:    subfolder name
        return os.path.join(self._project_path, self._subfolders[subfolder])

    def _exists_file(self, filename):
        '''
        Checks if the file exists
        '''
        # filename:    file name
        try:
            f = open(filename, 'r')
            existe = True
            f.close()
        except:
            existe = False
            pass
        return existe

    def _recover(self, field):
        '''
        Loads from disk a previously stored variable, to avoid recomputing it
        '''
        # field:    variable to restore from disk

        if field == 'defs_dict':
            input_file = os.path.join(self._get_folder('training_data'),
                                      'defs_dict.pkl')
            with open(input_file, 'rb') as f:
                self._defs_dict = pickle.load(f)

        if field == 'weighted_dict':
            input_file = os.path.join(self._get_folder('training_data'),
                                      'weighted_dict.pkl')
            with open(input_file, 'rb') as f:
                self.weighted_dict = pickle.load(f)

        if field == 'preds':
            input_file = os.path.join(self._get_folder('results'),
                                      'unsup_preds.pkl')
            with open(input_file, 'rb') as f:
                self.preds = pickle.load(f)
        return

    def get_category_definitions(self, categories, df_categories,
                                 verbose=True):
        '''
        Extracts from the dataframe the category definitions, list of
        lemmatized terms defining every category
        '''
        # categories:    list of categories
        # df_categories: dataframe with the categories definitions. df_categories must contain lemmatized words
        # returns defs_dict

        defs_dict = {}
        # df_categories must contain lower lemmatized words, no \n
        for cat in categories:
            aux = df_categories.loc[df_categories['Reference'] == cat]
            definition = aux['Def_lemmas'].iloc[0]
            definition_wiki = aux['WikiDefs'].iloc[0]
            definition = definition.replace('\n', ' ')
            definition_wiki = definition_wiki.replace('\n', ' ')
            definition = definition + ' ' + definition_wiki

            # definition = definition.replace('\n', ' ').replace(
            #     '+', ' ').lower()
            tokens = definition.split(' ')
            tokens = list(set(tokens))
            tokens = [t for t in tokens if len(t) > 0]

            defs_dict.update({cat: tokens})
            if verbose:
                print(cat, len(tokens))

        pathfilename = os.path.join(
            self._project_path, self._subfolders['training_data'],
            'defs_dict.pkl')

        with open(pathfilename, 'wb') as f:
            pickle.dump(defs_dict, f)

        return defs_dict

    def get_category_definitions_tfidf(self, categories, df_categories,
                                       verbose=True):
        '''
        Extracts from the dataframe the category definitions, list of
        lemmatized terms defining every category and also computes a relevance
        weight for every word, using a tfidf scheme
        '''

        # categories:    list of categories
        # df_categories: dataframe with the category definitions.
        #                df_categories must contain lemmatized words

        docs_definitions = []
        for cat in categories:
            aux = df_categories.loc[df_categories['Reference'] == cat]
            definition = aux['Def_lemmas'].iloc[0]
            definition_wiki = aux['WikiDefs'].iloc[0]
            definition = definition.replace('\n', ' ')
            definition_wiki = definition_wiki.replace('\n', ' ')
            definition = definition + ' ' + definition_wiki
            docs_definitions.append(definition)

        # Common bow to all definitions
        bow = Bow()
        Xtfidf = bow.fit(docs_definitions)
        vocab = bow.tfidf.vocabulary_
        inv_vocab = bow.obtain_inv_vocab()

        # We write a text report for every category, to easily check the main terms that are being 
        # identified in the category definition
        report = []
        cadena = '=' * 70 + '\r\n'
        report.append(cadena)
        cadena = ('    50 terms with largest weight for every category' +
                  '\r\n')
        report.append(cadena)
        self.weighted_dict = {}
        for k, cat in enumerate(categories):
            cadena = '=' * 70 + '\r\n'
            report.append(cadena)
            cadena = cat + '\r\n'
            report.append(cadena)
            cadena = '=' * 70 + '\r\n'
            report.append(cadena)
            row = Xtfidf[k, :]
            weights = [(i[1], row[i]) for i in zip(*row.nonzero())]
            weights.sort(key=lambda x: -x[1])

            for kk in range(0, min(50, len(weights))):
                cadena = inv_vocab[weights[kk][0]] + '\t' + str(weights[kk][1]) + '\r\n'
                report.append(cadena)

            tmp_dict = {}
            for kk in range(0, len(weights)):
                tmp_dict.update({inv_vocab[weights[kk][0]]: weights[kk][1]})
            self.weighted_dict.update({cat: tmp_dict})

        filename = os.path.join(self._project_path + self._subfolders['results'], 'weighted_dict.txt')
        file = codecs.open(filename, "w", "utf-8")
        for line in report:
            file.write(line)
        file.close()

        pathfilename = os.path.join(self._project_path + self._subfolders['training_data'], 'weighted_dict.pkl')
        with open(pathfilename, 'wb') as f:
            pickle.dump(self.weighted_dict, f)

        return self.weighted_dict

    def load_predicts(self, filename):
        '''
        Loads predictions from file
        '''
        # filename:    file name
        filename_predicts = os.path.join(self._project_path + self._subfolders['results'], filename)
        with open(filename_predicts, 'rb') as f:
            Preds = pickle.load(f)
        return Preds

    def obtain_predicts(self, docs, we, sigma, filename, verbose=True):
        '''
        Computes predictions based on distances computed in the WE space
        '''
        # docs:         list of documents to be classified. docs must be lematized.
        # we:           WE model
        # sigma:        smoothing parameter to control the distance between tokens
        # filename:     file name to store the predictions

        filename_predicts = os.path.join(self._project_path + self._subfolders['results'], filename)

        if self._defs_dict is None:
            self._recover('defs_dict')

        preds = []
        categories = list(self._defs_dict.keys())
        #for k in range(0, 100):
        Ndocs = len(docs)
        for k in range(0, Ndocs):
            tokens = docs[k].split(' ')
            # proyectando sobre cada categoría
            proy = []
            for cat in categories:
                tokens2 = self._defs_dict[cat]
                K, D, words1, words2 = we.tokens_dist(tokens, tokens2, sigma)
                if len(words2) > 0 and len(words1) > 0:
                    try:
                        v1 = np.max(np.round(K), axis=0)
                        v2 = np.max(np.round(K), axis=1)
                        #v = np.sum(v1) + np.sum(v2)    # no da muy buenos resultados
                        v = np.mean(v1) + np.mean(v2)   # no da muy buenos resultados
                    except:
                        print('Error v')
                        import code
                        code.interact(local=locals())
                        pass
                    #v = v / len(words2)
                else:
                    v = 0
                #print(cat + '\t' + str(v))
                proy.append((cat, v))

            proy.sort(key=lambda x: -x[1])
            preds.append(proy)
            if verbose:
                self._muestra_avance(k, Ndocs)
        self.preds = preds

        with open(filename_predicts, 'wb') as f:
            pickle.dump(preds, f)
        return preds

    def obtain_predicts_wordcount(self, docs, we, sigma, filename,
                                  verbose=True):
        '''
        Computes predictions based on coincidences with the weighted definitions
        '''
        # docs:         list of documents to be classified. docs must be leematized.
        # we:           WE model
        # sigma:        parameter to control the distance between tokens
        # filename:     file name to store the predictions
        filename_predicts = os.path.join(self._project_path + self._subfolders['results'], filename)

        if self.weighted_dict is None:
            self._recover('weighted_dict')

        categories = list(self.weighted_dict.keys())
        Ncats = len(categories)

        norm_dict = {cat: sum(self.weighted_dict[cat].values())
                     for cat in categories}

        start = time()
        Ndocs = len(docs)
        # Ndocs = 1000
        Preds = np.zeros((Ndocs, Ncats))
        for kdoc, doc in enumerate(docs[0:Ndocs]):
            tokens = doc.split(' ')

            # proyectando sobre cada categoría
            # acums = np.zeros(Ncats)
            for kcat, cat in enumerate(categories):

                Preds[kdoc, kcat] = (
                    sum(self.weighted_dict[cat][w] for w in tokens
                        if w in self.weighted_dict[cat]) /
                    norm_dict[cat])

            if verbose:
                self._muestra_avance(kdoc, Ndocs)
        print ('---- ---- Predictions computed in {} seconds'.format(
            time() - start))

        with open(filename_predicts, 'wb') as f:
            pickle.dump(Preds, f)
        return Preds

    def _muestra_avance(self, i, N):
        aux = float(i) / (float(floor(N / 100.0)) + 1)
        if (aux % 1.0) == 0 and aux > 0.0:
            print(str(int(aux)) + "%, ", end='', flush=True)
        return

    def obtain_original_labels(self, tags, verbose=True):
        '''
        Obtains labels from the database, removes duplicates and eliminates
        supercategories
        '''
        # tags:         list of tags as retrieved from the database

        new_tags = [t.split(',') for t in tags]

        # removing supercategories, ending in "_"
        tags = []
        for l in new_tags:
            newlist = [tag for tag in l if len(tag) > 0 and tag[-1] != '_']
            tags.append(newlist)

        # remove duplicates in the list
        tags_without_duplicates = []
        for t in tags:
            unique_list = list(OrderedDict((element, None) for element in t))
            tags_without_duplicates.append(unique_list)

        return tags_without_duplicates
