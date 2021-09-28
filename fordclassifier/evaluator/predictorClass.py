# -*- coding: utf-8 -*-
'''

@author:  Angel Navia Vázquez
May 2018

'''

# import code
# code.interact(local=locals())

import os
import pickle
# from fordclassifier.classifier.classifier import Classifier
import numpy as np
# from sklearn.metrics import roc_curve, auc
import json
# import matplotlib.pyplot as plt
# import operator
# from fordclassifier.corpusclassproject.datamanager import DataManager
# import configparser
# from fordclassifier.corpusanalyzer.textproc import BasicNLP
# import itertools
# from fordclassifier.corpusanalyzer.bow import Bow
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt


class Predictor(object):
    '''
    Class to predict labels

    ============================================================================
    Methods:
    ============================================================================
    _recover:                  if a variable is not on memory, it tries to
                               recover it from disk
    _get_folder:               retuns full path to a subfolder
    _exists_file:              check if the file exists in disk
    predict:                   computes predictions for every independent
                               classifier
    obtain_multilabel_preds:   computes predicted output labels
    predict_new_document:      computes predictions for a new document
    ============================================================================
    '''

    def __init__(self, project_path, subfolders, verbose=True):
        '''
        Initialization:
        '''
        self._project_path = project_path   # working directory
        self._verbose = verbose      # messages are printed on screen when True
        self.models2evaluate = None    # models to evaluate (classif, params)
        self._subfolders = subfolders  # subfolders structure
        self.best_auc = None           # Best AUC
        self.best_models = None        # Best models
        self.Xtfidf_tr = None          # Xtfidf for training
        self.vocab = None
        self.inv_vocab = None
        self.tags = None
        self.tags2_tr = None
        self.ths_dict = None
        self.Preds = None
        self.index_tst = None
        self.categories = None
        self.Xtfidf_tst = None
        self.tags_tst = None
        self.CONF = None
        self.partition = None
        self.bow = None

    def _get_folder(self, subfolder):
        '''
        gets full path to a folder
        Inputs:
            - subfolder: target subfolder
        '''
        return os.path.join(self._project_path, self._subfolders[subfolder])

    def _exists_file(self, filename):
        '''
        Checks if the file exists
        Inputs: 
            - filename
        '''
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
        Inputs:
            - field:    variable to restore from disk
        '''
        if field == 'best_models':
            try:
                input_file = os.path.join(
                    self._get_folder('results'), 'best_models.json')
                with open(input_file, 'r') as f:
                    self.best_models = json.load(f)
            except:
                input_file = os.path.join(
                    self._get_folder('export'), 'best_models.json')
                with open(input_file, 'r') as f:
                    self.best_models = json.load(f)
                pass

        if field == 'ths_dict':
            try:
                filename = os.path.join(
                    self._project_path + self._subfolders['results'],
                    'ths_dict.pkl')
                with open(filename, 'rb') as f:
                    self.ths_dict = pickle.load(f)
            except:
                filename = os.path.join(
                    self._project_path + self._subfolders['export'],
                    'ths_dict.pkl')
                with open(filename, 'rb') as f:
                    self.ths_dict = pickle.load(f)
                pass

        if field == 'categories':
            try:
                filename = os.path.join(
                    self._project_path + self._subfolders['training_data'],
                    'categories.pkl')
                with open(filename, 'rb') as f:
                    self.categories = pickle.load(f)
            except:
                filename = os.path.join(
                    self._project_path + self._subfolders['export'],
                    'categories.pkl')
                with open(filename, 'rb') as f:
                    self.categories = pickle.load(f)
                pass

        if field == 'models2evaluate':
            try:
                filename = os.path.join(
                    self._project_path + self._subfolders['training_data'],
                    'models2evaluate.pkl')
                with open(filename, 'rb') as f:
                    self.models2evaluate = pickle.load(f)
            except:
                filename = os.path.join(
                    self._project_path + self._subfolders['export'],
                    'models2evaluate.pkl')
                with open(filename, 'rb') as f:
                    self.models2evaluate = pickle.load(f)
                pass

        if field == 'bow':
            try:
                filename = os.path.join(
                    self._project_path + self._subfolders['bow'],
                    'bow.pkl')
                with open(filename, 'rb') as f:
                    self.bow = pickle.load(f)
            except:
                filename = os.path.join(
                    self._project_path + self._subfolders['export'],
                    'bow.pkl')
                with open(filename, 'rb') as f:
                    self.bow = pickle.load(f)
                pass
        return

    def predict(self, Xtfidf, verbose=True):
        '''
        Computes predictions for every independent classifier
        Inputs:
            - Xtfidf:    tfidf matrix
        '''
        if self.ths_dict is None:
            self._recover('ths_dict')

        if self.categories is None:
            self._recover('categories')

        if self.best_models is None:
            self._recover('best_models')

        if self.models2evaluate is None:
            self._recover('models2evaluate')

        umbrales = []
        for cat in self.categories:
            umbrales.append(self.ths_dict[cat])
        umbrales = np.array(umbrales)
        Ncats = len(self.categories)
        NP = Xtfidf.shape[0]

        Preds = np.zeros(shape=(NP, Ncats))

        for k in range(0, len(self.categories)):
            cat = self.categories[k]
            model_name = self.best_models[cat]
            model_params = self.models2evaluate[model_name]
            try:
                filename = os.path.join(
                    self._get_folder('models'),
                    model_params['model_name'] + '_' + str(cat) +
                    '_classifier.pkl')
                with open(filename, 'rb') as f:
                    model = pickle.load(f)
            except:
                filename = os.path.join(
                    self._get_folder('export'),
                    model_params['model_name'] + '_' + str(cat) +
                    '_classifier.pkl')
                with open(filename, 'rb') as f:
                    model = pickle.load(f)
                pass
            p = model.predict(Xtfidf)
            Preds[:, k] = p

        # Estimating the class with softmax
        # winners = np.argmax(Preds, axis=1)
        # tag_pred = [categories[w] for w in winners]

        tag_pred = []
        # Obtaining multilabel prediction
        for k in range(0, NP):
            cuales = Preds[k, :] > umbrales
            tags = list(np.array(self.categories)[cuales])
            p = Preds[k, :][cuales]
            difs = Preds[k, :][cuales] - umbrales[cuales]
            # print(len(tags))
            tmp_dict = {'tags_pred': tags, 'difs': difs, 'preds': p}
            tag_pred.append(tmp_dict)
        return Preds, tag_pred

    def obtain_multilabel_preds(self, Preds, sort_option, final_threshold,
                                verbose=True):
        '''
        computes predicted output labels
        Inputs:
            - Preds:                binary predictions
            - sort_option:          unused, check...
            - final_threshold:      global final threshold to include labels in the prediction
        '''
        if self.categories is None:
            self._recover('categories')

        if self.ths_dict is None:
            self._recover('ths_dict')

        NP = Preds.shape[0]
        Ncats = len(self.categories)

        multilabel_predicts = []
        for n in range(0, NP):
            labels = {}
            for kcat in range(0, Ncats):
                cat = self.categories[kcat]
                if Preds[n, kcat] > self.ths_dict[cat]:
                    #        Preds[n, kcat] >= threshold):
                    #if Preds[n, kcat] >= threshold:
                    # incluimos las predicciones que superan los umbrales individuales
                    labels.update(
                        {cat: {'p': Preds[n, kcat],
                               'diff': Preds[n, kcat] - self.ths_dict[cat]}})
            if len(labels) == 0:
                # buscamos el de maximo valor
                max_pred = -99
                max_pred_kcat = ''

                for kcat in range(0, Ncats):
                    cat = self.categories[kcat]
                    if Preds[n, kcat] > max_pred:
                        max_pred = Preds[n, kcat]
                        max_pred_kcat = kcat

                cat = self.categories[max_pred_kcat]
                labels.update(
                    {cat: {'p': Preds[n, max_pred_kcat],
                           'diff': Preds[n, max_pred_kcat] - self.ths_dict[cat]}})

            # Normalizing values
            values = []
            for p in labels:
                values.append(labels[p]['p'])
            if len(values) > 0:
                M = max(values)
                for p in labels:
                    labels[p]['p'] = labels[p]['p'] / M

            multilabel_predicts.append(labels)

        # filename = os.path.join(
        #     self._project_path + self._subfolders['results'],
        #     'multilabel_predicts.pkl')
        # with open(filename, 'wb') as f:
        #     pickle.dump(multilabel_predicts, f)

        # obtaining the labels_pred in order of importance
        labels_pred = []
        for k in range(0, len(multilabel_predicts)):
            values = []
            for key in list(multilabel_predicts[k].keys()):
                if sort_option == 'maximum_response':
                    values.append((key, multilabel_predicts[k][key]['p']))
                if sort_option == 'maximum_diff':
                    values.append((key, multilabel_predicts[k][key]['diff']))
            values.sort(key=lambda x: x[1], reverse=True)
            tags = [v[0] for v in values if v[1] >= final_threshold]

            labels_pred.append(tags)

        return multilabel_predicts, labels_pred

    def predict_new_document(self, docs, option='maximum_response',
                             final_threshold=0.7,
                             path2export='', verbose=True):
        '''
        computes predictions for a new document
        Inputs:
            - docs:                 docs to be labelled
            - final_threshold:      global final threshold to include labels in the prediction
            - path2export:          path to the exported information (optional)
        '''

        # recovering data from export path
        if path2export != '':
            filename = os.path.join(path2export, 'ths_dict.pkl')
            with open(filename, 'rb') as f:
                self.ths_dict = pickle.load(f)
        else:
            if self.ths_dict is None:
                self._recover('ths_dict')

        if path2export != '':
            filename = os.path.join(path2export, 'categories.pkl')
            with open(filename, 'rb') as f:
                self.categories = pickle.load(f)
        else:
            if self.categories is None:
                self._recover('categories')

        if path2export != '':
            filename = os.path.join(path2export, 'best_models.json')
            with open(filename, 'r') as f:
                self.best_models = json.load(f)
        else:
            if self.best_models is None:
                self._recover('best_models')

        if path2export != '':
            filename = os.path.join(path2export, 'models2evaluate.pkl')
            with open(filename, 'rb') as f:
                self.models2evaluate = pickle.load(f)
        else:
            if self.models2evaluate is None:
                self._recover('models2evaluate')

        if path2export != '':
            filename = os.path.join(path2export, 'bow.pkl')
            with open(filename, 'rb') as f:
                self.bow = pickle.load(f)
        else:
            if self.bow is None:
                self._recover('bow')

        if verbose:
            print('Predicting ', len(docs), 'projects ... please wait.')

        umbrales = []
        for cat in self.categories:
            umbrales.append(self.ths_dict[cat])

        umbrales = np.array(umbrales)
        Ncats = len(self.categories)

        # removing the separation in sentences
        docs = [doc.replace('\n ', ' ') for doc in docs]
        Xtfidf = self.bow.transform(docs)

        NP = Xtfidf.shape[0]

        Preds = np.zeros(shape=(NP, Ncats))

        for k in range(0, len(self.categories)):
            cat = self.categories[k]
            model_name = self.best_models[cat]
            model_params = self.models2evaluate[model_name]

            if path2export != '':
                filename = os.path.join(path2export, model_params['model_name'] + '_' + str(cat) + '_classifier.pkl')
                with open(filename, 'rb') as f:
                    model = pickle.load(f)
            else:
                filename = os.path.join(
                    self._get_folder('models'),
                    model_params['model_name'] + '_' + str(cat) +
                    '_classifier.pkl')
                with open(filename, 'rb') as f:
                    model = pickle.load(f)

            p = model.predict(Xtfidf)
            Preds[:, k] = p

        multilabel_pred, labels_pred = self.obtain_multilabel_preds(
            Preds, option, final_threshold, verbose=True)

        raw_preds = self.obtain_raw_preds(Preds, verbose=True)

        return multilabel_pred, labels_pred, raw_preds

    def obtain_raw_preds(self, Preds, verbose=True):
        # Preds:                binary predictions
        # Npreds:               number of sorted predicts to return 
        if self.categories is None:
            self._recover('categories')

        if self.ths_dict is None:
            self._recover('ths_dict')

        NP = Preds.shape[0]
        Ncats = len(self.categories)

        raw_predicts = []
        for n in range(0, NP):
            labels = []
            for kcat in range(0, Ncats):
                cat = self.categories[kcat]
                labels.append((cat, Preds[n, kcat]))
                
            labels = sorted(labels, key=lambda x: -x[1])

            raw_predicts.append(labels)

        return raw_predicts
