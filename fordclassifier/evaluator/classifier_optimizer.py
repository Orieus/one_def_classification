# -*- coding: utf-8 -*-
'''

@author:  Angel Navia Vázquez
June 2018

Included Models:

    Logistic Regression, with
        C parameter in {0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000}
    Linear Support Vector Machine, with
        C parameter in {0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000}
    Polynomial Support Vector Machine (degree 2), with
        C parameter in {0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000}
    Polynomial Support Vector Machine (degree 3), with
        C parameter in {0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000}
    Decission Tree Classifier
    Adaboost Classifier with Number of estimators in {1, 5, 10, 50, 100}
    Random Forest, with N in {1, 5, 10, 50, 100}
    Multilayer Perceptron (1 hidden layer with N neurons),
        N in {1, 5, 10, 50, 100},
        alpha in {0.0001, 0.001, 0.01, 0.1, 1, 10, 100}
    K-Nearest Neighbours, with K in {1, 5, 10, 50, 100}
    LASSO with L in {0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0}
    Elastic Net, with
        L1 in {0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0} and
        L2 in {0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0}
    Multinomial Naive Bayes, with
        alpha in {0, 0.5, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0}
    Bernoulli Naive Bayes, with
        alpha in {0, 0.5, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0}
'''

# import code
# code.interact(local=locals())

import os
import json
import pickle
import shutil

import numpy as np
from math import floor
import random

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GroupShuffleSplit

# Local imports
from fordclassifier.classifier.classifier import Classifier
from fordclassifier.corpusanalyzer.bow import Bow

import ipdb


class classifierOptimizer(object):
    '''
    Class to train and optimize hyperparameters of the classifiers

    ============================================================================
    Methods:
    ============================================================================
    _recover:                   if a variable is not on memory, it tries to
                                recover it from disk
    _get_folder:                retuns full path to a subfolder
    _exists_file:               check if the file exists in disk
    _defineClassifiers :        defines the list of classifiers to evaluate
    obtain_docs_from_df:        transforms the data in the dataframe to a list
                                of docs and references
    prepare_training_data:      receives as input a dataframe and produces the
                                tfidf bow and tags
    create_partition_train_test_MINECO:   splits the dataset into train and
                                test for the Mineco case
    create_Nfold_partition:     creates Nfold partitions
    xval_models :               evaluates every model/hyperparameters by
                                cross-validation
    find_best_models:           finds the best model/hyperparameters for every
                                category
    compute_thresholds:         computes the thresholds of the classifiers
    ============================================================================
    '''

    def __init__(self, project_path, classifier_types, subfolders,
                 min_df=2, verbose=True, cat_model='multilabel',
                 title_mul=1, copycf=True):
        '''
        Initialization:
        '''
        self._project_path = project_path      # working directory
        self._verbose = verbose      # messages are printed on screen when True
        self._classifier_types = classifier_types  # list of classifs to eval
        # Category model: multilabel, multiclass or weighted
        self.cat_model = cat_model
        self._subfolders = subfolders  # subfolders structure

        # Minimum document frequency for the bow computation
        self.min_df = min_df

        # Multiplier for the title.
        # The title of each project can be replicated to amplify the weight of
        # its words. title_mul must be a positive integer.
        # The words in the titles are multiplied title_mul times
        self.title_mul = title_mul

        self.best_auc = None         # Best AUC
        self.best_models = None      # Best models
        self.Xtfidf_tr = None        # Xtfidf for training
        self.tags_tr = None
        self.ths_dict = None
        self.categories = None
        self.partition = None

        # models to evaluate (classif, params)
        self.models2evaluate = self._defineClassifiers(self._classifier_types)

        filename = os.path.join(self._get_folder('training_data'),
                                'models2evaluate.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(self.models2evaluate, f)

        filename = os.path.join(self._get_folder('export'),
                                'models2evaluate.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(self.models2evaluate, f)

        if copycf:
            # Save a copy of the configuration file into export
            source_cf = os.path.join(self._project_path, 'config.cf')
            dest_cf = os.path.join(self._get_folder('export'), 'config.cf')
            shutil.copyfile(source_cf, dest_cf)

        return

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
        if field == 'best_models':
            try:
                input_file = os.path.join(self._get_folder('results'),
                                          'best_models.json')
                with open(input_file, 'r') as f:
                    self.best_models = json.load(f)
            except:
                input_file = os.path.join(self._get_folder('export'),
                                          'best_models.json')
                with open(input_file, 'r') as f:
                    self.best_models = json.load(f)
                pass

        if field == 'Xtfidf_tr':
            filetoload_Xtfidf = os.path.join(
                self._project_path + self._subfolders['training_data'],
                'train_data.pkl')
            with open(filetoload_Xtfidf, 'rb') as f:
                [self.Xtfidf_tr, self.tags_tr, refs_tr] = pickle.load(f)

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

        if field == 'partition':
            filename = os.path.join(
                self._project_path + self._subfolders['training_data'],
                'partition.pkl')
            with open(filename, 'rb') as f:
                self.partition = pickle.load(f)

        return

    def _defineClassifiers(self, classifier_types):
        '''
        Defines the list of classifiers to evaluate
        '''
        # classifier_types: list of types of classifiers to be included
        models2evaluate = {}
        # rangeC_string = ["00001", "0001", "001", "01", "1", "10", "100",
        #                  "1000"]
        # rangeC_value = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        rangeC_string = ["00001", "00003", "0001", "0003", "001", "003", "01",
                         "03", "1", "3", "10", "30", "100", "300", "1000"]
        rangeC_value = [0.0001, 0.00032, 0.001, 0.0032, 0.01, 0.032, 0.1, 0.32,
                        1, 3.2, 10, 32, 100, 320, 1000]

        rangeN_string = ["1", "5", "10", "50", "100"]
        rangeN_value = [1, 5, 10, 50, 100]

        rangeL_string = ['0', '01', '03', '05', '07', '09', '10']
        rangeL_value = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

        range_alpha_string = ['0', '05', '1', '10', '100', '1000', '10000',
                              '100000']
        range_alpha_value = [0, 0.5, 1.0, 10.0, 100.0, 1000.0, 10000.0,
                             100000.0]

        if self._verbose:
            print("The following models will be evaluated:")

        model_type = 'LR'
        if model_type in classifier_types:
            if self._verbose:
                print(model_type)
            for k in range(0, len(rangeC_value)):
                tmp_dict = {}
                model_name = model_type + 'C' + rangeC_string[k]
                tmp_dict.update({'model_type': model_type})
                tmp_dict.update({'model_name': model_name})
                tmp_dict.update({'C': rangeC_value[k]})
                models2evaluate.update({model_name: tmp_dict})

        model_type = 'LSVM'
        if model_type in classifier_types:
            if self._verbose:
                print(model_type)
            for k in range(0, len(rangeC_value)):
                tmp_dict = {}
                model_name = model_type + 'C' + rangeC_string[k]
                tmp_dict.update({'model_type': model_type})
                tmp_dict.update({'model_name': model_name})
                tmp_dict.update({'C': rangeC_value[k]})
                models2evaluate.update({model_name: tmp_dict})

        model_type = 'SVMpoly'
        if model_type in classifier_types:
            degree = 2
            if self._verbose:
                print(model_type + ', degree= ' + str(degree))
            for k in range(0, len(rangeC_value)):
                tmp_dict = {}
                model_name = (model_type + 'D' + str(degree) + 'C' +
                              rangeC_string[k])
                tmp_dict.update({'model_type': model_type})
                tmp_dict.update({'model_name': model_name})
                tmp_dict.update({'C': rangeC_value[k]})
                tmp_dict.update({'degree': degree})
                models2evaluate.update({model_name: tmp_dict})

        model_type = 'SVMpoly'
        if model_type in classifier_types:
            degree = 3
            if self._verbose:
                print(model_type + ', degree= ' + str(degree))
            for k in range(0, len(rangeC_value)):
                tmp_dict = {}
                model_name = (model_type + 'D' + str(degree) + 'C' +
                              rangeC_string[k])
                tmp_dict.update({'model_type': model_type})
                tmp_dict.update({'model_name': model_name})
                tmp_dict.update({'C': rangeC_value[k]})
                tmp_dict.update({'degree': degree})
                models2evaluate.update({model_name: tmp_dict})

        model_type = 'DT'
        if model_type in classifier_types:
            if self._verbose:
                print(model_type)
            tmp_dict = {}
            model_name = model_type
            tmp_dict.update({'model_type': model_type})
            tmp_dict.update({'model_name': model_name})
            models2evaluate.update({model_name: tmp_dict})

        model_type = 'AB'
        if model_type in classifier_types:
            if self._verbose:
                print(model_type)
            for k in range(0, len(rangeN_value)):
                tmp_dict = {}
                model_name = model_type + 'N' + rangeN_string[k]
                tmp_dict.update({'model_type': model_type})
                tmp_dict.update({'model_name': model_name})
                tmp_dict.update({'N': rangeN_value[k]})
                models2evaluate.update({model_name: tmp_dict})

        model_type = 'RF'
        if model_type in classifier_types:
            if self._verbose:
                print(model_type)
            for k in range(0, len(rangeN_value)):
                tmp_dict = {}
                model_name = model_type + 'N' + rangeN_string[k]
                tmp_dict.update({'model_type': model_type})
                tmp_dict.update({'model_name': model_name})
                tmp_dict.update({'N': rangeN_value[k]})
                models2evaluate.update({model_name: tmp_dict})

        model_type = 'MLP'
        if model_type in classifier_types:
            if self._verbose:
                print(model_type)
            for kc in range(0, len(rangeC_value[0:7])):
                for kn in range(0, len(rangeN_value)):
                    tmp_dict = {}
                    model_name = (model_type + 'C' + rangeC_string[kc] + 'N' +
                                  rangeN_string[kn])
                    tmp_dict.update({'model_type': model_type})
                    tmp_dict.update({'model_name': model_name})
                    tmp_dict.update({'C': rangeC_value[kc]})
                    tmp_dict.update({'N': rangeN_value[kn]})
                    models2evaluate.update({model_name: tmp_dict})

        model_type = 'KNN'
        if model_type in classifier_types:
            if self._verbose:
                print(model_type)
            for k in range(0, len(rangeN_value)):
                tmp_dict = {}
                model_name = model_type + 'N' + rangeN_string[k]
                tmp_dict.update({'model_type': model_type})
                tmp_dict.update({'model_name': model_name})
                tmp_dict.update({'N': rangeN_value[k]})
                models2evaluate.update({model_name: tmp_dict})

        model_type = 'LASSO'
        if model_type in classifier_types:
            if self._verbose:
                print(model_type)
            for k in range(0, len(rangeL_value)):
                tmp_dict = {}
                model_name = model_type + 'L' + rangeL_string[k]
                tmp_dict.update({'model_type': model_type})
                tmp_dict.update({'model_name': model_name})
                tmp_dict.update({'L': rangeL_value[k]})
                models2evaluate.update({model_name: tmp_dict})

        model_type = 'EN'
        if model_type in classifier_types:
            if self._verbose:
                print(model_type)
            for k1 in range(0, len(rangeL_value)):
                for k2 in range(0, len(rangeL_value)):
                    tmp_dict = {}
                    model_name = (model_type + 'L1' + rangeL_string[k1] +
                                  'L2' + rangeL_string[k2])
                    tmp_dict.update({'model_type': model_type})
                    tmp_dict.update({'model_name': model_name})
                    tmp_dict.update({'L1': rangeL_value[k1]})
                    tmp_dict.update({'L2': rangeL_value[k2]})
                    models2evaluate.update({model_name: tmp_dict})

        model_type = 'MNB'
        if model_type in classifier_types:
            if self._verbose:
                print(model_type)
            for k in range(0, len(range_alpha_string)):
                tmp_dict = {}
                model_name = model_type + 'alpha' + range_alpha_string[k]
                tmp_dict.update({'model_type': model_type})
                tmp_dict.update({'model_name': model_name})
                tmp_dict.update({'alpha': range_alpha_value[k]})
                models2evaluate.update({model_name: tmp_dict})

        model_type = 'BNB'
        if model_type in classifier_types:
            if self._verbose:
                print(model_type)
            for k in range(0, len(rangeL_value)):
                tmp_dict = {}
                model_name = model_type + 'alpha' + rangeL_string[k]
                tmp_dict.update({'model_type': model_type})
                tmp_dict.update({'model_name': model_name})
                tmp_dict.update({'alpha': rangeL_value[k]})
                models2evaluate.update({model_name: tmp_dict})

        return models2evaluate

    def obtain_docs_from_df(self, df):
        '''
        Transforms the data in the dataframe to a list of docs and references.

        Args:
            df:        Input dataframe
        '''

        # df: dataframe with the input data
        titulo_lang = list(df.loc[:, 'Titulo_lang'])
        resumen_lemas = list(df.loc[:, 'Resumen_lemas'])
        titulo_lemas = list(df.loc[:, 'Titulo_lemas'])
        refs = list(df.loc[:, 'Referencia'])

        docs = []
        selected_refs = []
        Nprojects = len(refs)

        for kdoc in range(0, Nprojects):
            # Add title
            text = ''
            if titulo_lang[kdoc] == 'es':
                doc = titulo_lemas[kdoc] + ' '
                doc = doc.lower()
                doc = doc.replace('\n', ' ')
                # Replicate title to amplify the weight of its words
                # self.title_mul is the multiplier of the title component.
                # It must be a positive integer.
                # The words in the titles are multiplied title_mul times
                text += self.title_mul * doc

            # Add abstract
            doc = resumen_lemas[kdoc] + ' '
            doc = doc.lower()
            text += doc.replace('\n', ' ')
            docs.append(text)

            selected_refs.append(refs[kdoc])

        return selected_refs, docs

    def prepare_training_data(self, df, task, seed_missing_categories=False,
                              verbose=True):
        """
        This method is specific for a particular task but the resulting files
        are general to any multilabel/multiclass problem

        Available tasks:

           UNESCO: loads tags from the UNESCO codes
           FORD:   loads tags mapped from Unesco to Ford

        Receives as input a dataframe and produces the list of documents for
        bow computation

            df:     dataframe with the input data
            task:   to choose between FORD and UNESCO
            seed_missing_categories: boolean to decide if some missing
                    categories are filled with a deterministic criterium
        """

        # ################
        # Corpus selection
        # ################

        # Filtering out those projects without summary in spanish
        df = df.loc[df['Resumen_lang'] == 'es']
        print(f'-- -- Selected {len(df)} projects in Spanish')

        if self.cat_model == 'multiclass':
            df = df.loc[df['GroupRef'].isin([None, 'nan'])]
            print(f'-- -- Selected {len(df)} non coordinated projects')

        # ##########
        # Get labels
        # ##########

        col = {'UNESCO': 'UNESCO_cd', 'FORD': 'Unesco2Ford'}
        tags = list(df.loc[:, col[task]])

        # Convert tags in list of strings
        new_tags = [t.split(',') for t in tags]
        tags = []

        if task == 'UNESCO':

            for l in new_tags:
                newlist = [tag[0:2] for tag in l if len(tag) >= 2]
                newlist = list(set(newlist))
                tags.append(newlist)

        if task == 'FORD':

            # Supercategories, identified by a final '_', are removed.
            tags = []
            for l in new_tags:
                newlist = [tag for tag in l
                           if len(tag) > 0 and not tag.endswith('_')]
                tags.append(newlist)

        # ###################################
        # Biotec and nanotec label generation
        #####################################

        # Add labels about categories not mapped from UNESCO
        if seed_missing_categories:
            titulos = list(df.loc[:, 'Titulo'])
            resumenes = list(df.loc[:, 'Resumen'])
            corpus = [(t[0] + ' ' + t[1]).lower()
                      for t in zip(titulos, resumenes)]

            N_BioTecAmb = 0
            N_NanoTec = 0
            N_BioTecSalud = 0
            N_BioTecAgr = 0

            # Representative keywords for the missing categories
            nanotec_def = [
                'nanotecnol', 'nanomaterial', 'nanotubo', 'nanosistema',
                'nanoescala', 'nanoparticula', 'nanorobot', 'nanorrobot',
                'bionanotecnol', 'nanoestructura']
            biotec_def = [
                'biotecnol', 'biofabric', 'biolixiv', 'bionanotec',
                'biorremed', 'mutagenesis', 'viroterapia', 'bioproduc',
                'bioplastico', 'bioprocesamiento', 'biocatalisis',
                'bioferment', 'biomaterial', 'transgenico']
            ambiental_def = ['ambiental', 'medio ambiente', 'biorremediacion']
            salud_def = ['salud', 'farmaco', 'medicina', 'viroterapia',
                         'enfermedad', 'terapia', 'biomedicina']
            agric_def = ['agricola', 'agricultura', 'transgenic', 'planta',
                         'semilla']

            if self.cat_model in ['multilabel', 'weighted']:
                for k, texto in enumerate(corpus):
                    if any(x in texto for x in biotec_def):
                        if any(x in texto for x in ambiental_def):
                            N_BioTecAmb += 1
                            tags[k] = ['BioTecAmb'] + tags[k]
                        if any(x in texto for x in salud_def):
                            N_BioTecSalud += 1
                            tags[k] = ['BioTecSalud'] + tags[k]
                        if any(x in texto for x in agric_def):
                            N_BioTecAgr += 1
                            tags[k] = ['BioTecAgr'] + tags[k]
                    if any(x in texto for x in nanotec_def):
                        N_NanoTec += 1
                        tags[k] = ['NanoTec'] + tags[k]
            else:
                # In the multiclass case. If a project has a unique label
                # and we believe it is a biotec or nanotec project, the
                # existing label is removed. Otherwise, all project would have
                # more than one label, and no one of the would enter the
                # training set
                for k, texto in enumerate(corpus):
                    if any(x in texto for x in biotec_def):
                        if any(x in texto for x in ambiental_def):
                            N_BioTecAmb += 1
                            if len(tags[k]) == 1:
                                tags[k] = ['BioTecAmb']
                            else:
                                tags[k] = ['BioTecAmb'] + tags[k]
                        if any(x in texto for x in salud_def):
                            N_BioTecSalud += 1
                            if len(tags[k]) == 1:
                                tags[k] = ['BioTecSalud']
                            else:
                                tags[k] = ['BioTecSalud'] + tags[k]
                        if any(x in texto for x in agric_def):
                            N_BioTecAgr += 1
                            if len(tags[k]) == 1:
                                tags[k] = ['BioTecAgr']
                            else:
                                tags[k] = ['BioTecAgr'] + tags[k]
                    if any(x in texto for x in nanotec_def):
                        N_NanoTec += 1
                        if len(tags[k]) == 1:
                            tags[k] = ['NanoTec']
                        else:
                            tags[k] = ['NanoTec'] + tags[k]

            print(' ')
            print('=' * 50)
            print('Seeding BioTecAmb with %d new labels' % N_BioTecAmb)
            print('Seeding NanoTec with %d new labels' % N_NanoTec)
            print('Seeding BioTecSalud with %d new labels' % N_BioTecSalud)
            print('Seeding BioTecAgr with %d new labels' % N_BioTecAgr)
            print('=' * 50)
            print(' ')

        # ###########
        # Compute Bow
        # ###########
        refs, docs = self.obtain_docs_from_df(df)

        if self.cat_model in ['multilabel', 'weighted']:
            ind = []
            self.refs = refs
            self.tags = tags
        elif self.cat_model == 'multiclass':
            # Select samples with one and only one tag
            ind = [i for i, t in enumerate(tags) if len(t) == 1]
            self.refs = [refs[i] for i in ind]
            self.tags = [tags[i] for i in ind]
            docs = [docs[i] for i in ind]
        else:
            exit(f'ERROR: Unknown category model: {self.cat_model}')

        flattened_tags = [y for x in self.tags for y in x]
        categories = list(set(flattened_tags))

        if verbose:
            print("---- Computing TFIDF...")
        # Now, compute bow.
        bow = Bow(self.min_df)
        self.Xtfidf = bow.fit(docs)
        self.vocab = bow.tfidf.vocabulary_
        self.inv_vocab = bow.obtain_inv_vocab()

        if verbose:
            print(f"---- BoW computed with {len(self.vocab)} tokens")

        # ############
        # Save results
        # ############
        filename = os.path.join(
            self._project_path, self._subfolders['bow'], 'bow.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(bow, f)

        filename = os.path.join(
            self._project_path, self._subfolders['export'], 'bow.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(bow, f)

        filetosave_Xtfidf = os.path.join(
            self._project_path, self._subfolders['training_data'],
            'Xtfidf.pkl')
        with open(filetosave_Xtfidf, 'wb') as f:
            pickle.dump([self.Xtfidf, self.vocab, self.inv_vocab], f)

        filetosave_tags = os.path.join(
            self._project_path, self._subfolders['training_data'], 'tags.pkl')
        with open(filetosave_tags, 'wb') as f:
            pickle.dump(self.tags, f)

        filetosave_tags = os.path.join(
            self._project_path, self._subfolders['training_data'], 'refs.pkl')
        with open(filetosave_tags, 'wb') as f:
            pickle.dump(self.refs, f)

        filetosave = os.path.join(
            self._project_path, self._subfolders['training_data'],
            'categories.pkl')
        with open(filetosave, 'wb') as f:
            pickle.dump(categories, f)

        print('Writing categories to export')

        filetosave = os.path.join(
            self._project_path, self._subfolders['export'], 'categories.pkl')
        with open(filetosave, 'wb') as f:
            pickle.dump(categories, f)

        return ind

    def create_partition_train_test_MINECO(
            self, fraction_train, coordinated_projects, verbose=True):
        '''
            fraction_train:       fraction to be used as train data
             coord:                list of coordinated proyects

        This function os specific for Mineco, and takes into account the
        coordinated projects to split between train and test
        '''

        filename = os.path.join(
            self._project_path, self._subfolders['training_data'],
            'Xtfidf.pkl')
        with open(filename, 'rb') as f:
            [Xtfidf, vocab, inv_vocab] = pickle.load(f)

        filename = os.path.join(
            self._project_path, self._subfolders['training_data'], 'tags.pkl')
        with open(filename, 'rb') as f:
            tags = pickle.load(f)

        filename = os.path.join(
            self._project_path, self._subfolders['training_data'], 'refs.pkl')
        with open(filename, 'rb') as f:
            refs = pickle.load(f)

        gss = GroupShuffleSplit(n_splits=1, train_size=fraction_train)
        sel_index_tr, sel_index_tst = list(gss.split(
            coordinated_projects, groups=coordinated_projects))[0]

        Xtfidf_tr = Xtfidf[sel_index_tr, :]
        tags_tr = [tags[i] for i in sel_index_tr]
        refs_tr = [refs[i] for i in sel_index_tr]

        Xtfidf_tst = Xtfidf[sel_index_tst, :]
        tags_tst = [tags[i] for i in sel_index_tst]
        refs_tst = [refs[i] for i in sel_index_tst]

        filename = os.path.join(
            self._project_path, self._subfolders['test_data'], 'test_data.pkl')
        with open(filename, 'wb') as f:
            pickle.dump([Xtfidf_tst, tags_tst, refs_tst], f)

        filename = os.path.join(
            self._project_path, self._subfolders['training_data'],
            'train_data.pkl')
        with open(filename, 'wb') as f:
            pickle.dump([Xtfidf_tr, tags_tr, refs_tr], f)
        return

    def create_Nfold_partition(self, Nfold, verbose=True):
        '''
        Creates Nfold partitions

            Nfold:  Number of partitions
        '''

        # Working folders
        folder = os.path.join(self._project_path,
                              self._subfolders['training_data'])

        # Read training data.
        filename = os.path.join(folder, 'train_data.pkl')
        with open(filename, 'rb') as f:
            # Load all data (though only tags_tr will be used)
            [Xtfidf_tr, tags_tr, refs_tr] = pickle.load(f)
        NPtrain = Xtfidf_tr.shape[0]

        # Read categories
        filename = os.path.join(folder, 'categories.pkl')
        with open(filename, 'rb') as f:
            categories = pickle.load(f)

        # Generate and save a common partition for all categories
        partition = [random.randint(0, Nfold-1) for p in range(NPtrain)]
        filename = os.path.join(folder, 'partition.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(partition, f)

        for cat in categories:
            print(f"---- Positive class distribution for category {cat}")

            # Labels for cateogory cat
            y_orig = [1.0 if cat in taglist else -1.0 for taglist in tags_tr]

            # Testing the distribution of the positive class
            npos = np.zeros(Nfold)
            for k, y in enumerate(y_orig):
                npos[partition[k]] += (y == 1.0)

            print(npos)

            filename = os.path.join(folder, str(cat) + '_ytr.pkl')
            with open(filename, 'wb') as f:
                pickle.dump(y_orig, f)
        return

    def xval_models(self, verbose=True):
        '''
        evaluates every model/hyperparameters by cross-validation
        '''

        if self.Xtfidf_tr is None:
            self._recover('Xtfidf_tr')

        if self.categories is None:
            self._recover('categories')

        if self.partition is None:
            self._recover('partition')

        models = list(self.models2evaluate.keys())
        execution_pairs = []
        for cat in self.categories:
            for model_name in models:
                execution_pairs.append([cat, model_name])

        random.shuffle(execution_pairs)

        # Input samples. This is just to abbreviate
        Xtr = self.Xtfidf_tr
        # Sample weights
        if self.cat_model == 'weighted':
            wtr = np.array([1/(len(x) + (len(x) == 0)) for x in self.tags_tr])
        else:
            wtr = None

        for pair in execution_pairs:

            cat = pair[0]
            model_name = pair[1]

            filename = os.path.join(
                self._get_folder('training_data'), str(cat) + '_ytr.pkl')
            with open(filename, 'rb') as f:
                y = pickle.load(f)

            ytr = np.array(y)

            model_params = self.models2evaluate[model_name]
            file_output_ROC = os.path.join(
                self._get_folder('eval_ROCs'),
                f"ROC_{model_params['model_name']}_{cat}.pkl")

            if not self._exists_file(file_output_ROC):

                try:

                    if verbose:
                        print(cat, model_name)
                    model = Classifier(model_params, verbose=False)
                    NPtr = len(ytr)
                    preds_loo = [0] * NPtr
                    y_loo = [0] * NPtr
                    cuales_particiones = list(set(self.partition))
                    rango = range(NPtr)
                    for cual in cuales_particiones:

                        # Get training samples, labels and sample weights
                        cuales_tr = [i for i, c in enumerate(self.partition)
                                     if c != cual]
                        x_tr_ = Xtr[cuales_tr, :]
                        y_tr_ = ytr[cuales_tr]
                        if self.cat_model == 'weighted':
                            w_tr_ = wtr[cuales_tr]
                        else:
                            w_tr_ = None

                        # Training
                        model.fit(x_tr_, y_tr_, w_tr_)

                        # Get test samples and labels
                        cuales_tst = [i for i, c in enumerate(self.partition)
                                      if c == cual]
                        x_tst_ = Xtr[cuales_tst, :]
                        y_tst_ = ytr[cuales_tst]
                        p = model.predict(x_tst_)

                        # Store cv predictions in preds_loo and y_loo
                        for kk, c in enumerate(cuales_tst):
                            preds_loo[c] = p[kk]
                            y_loo[c] = y_tst_[kk]

                    fpr_loo, tpr_loo, thresholds = roc_curve(ytr, preds_loo)
                    roc_auc_loo = auc(fpr_loo, tpr_loo)
                    mdict = {'fpr_loo': list(fpr_loo),
                             'tpr_loo': list(tpr_loo),
                             'roc_auc_loo': roc_auc_loo, 'y_tr': list(ytr),
                             'preds_loo': list(preds_loo),
                             'thresholds': thresholds}
                    with open(file_output_ROC, 'wb') as f:
                        pickle.dump(mdict, f)

                    # Training final model
                    model.fit(Xtr, ytr, wtr)

                    filename = os.path.join(
                        self._get_folder('models'),
                        f"{model_params['model_name']}_{cat}_classifier.pkl")
                    with open(filename, 'wb') as f:
                        pickle.dump(model, f)
                except:
                    if verbose:
                        print("Error in execution: ", cat, model_name)
                    pass

        return

    def find_best_models(self, keys_classifier, verbose=True):
        '''
        finds the best model/hyperparameters for every category
        '''
        # keys_classifier:      list of classifier types to analyze
        if verbose:
            print("Comparing models...")

        if self.categories is None:
            self._recover('categories')

        classifiers = list(self.models2evaluate.keys())
        Nclass = len(classifiers)
        Ncats = len(self.categories)
        X = np.zeros(shape=(Ncats, Nclass))
        for kcat in range(0, Ncats):
            for kclass in range(0, Nclass):
                try:
                    model_type = classifiers[kclass]
                    found = False
                    for key in keys_classifier:
                        if key in model_type:
                            found = True
                    if found:
                        cat = self.categories[kcat]
                        file_input_ROC = os.path.join(
                            self._get_folder('eval_ROCs'), 'ROC_' +
                            model_type + '_' + cat + '.pkl')
                        with open(file_input_ROC, 'rb') as f:
                            mdict = pickle.load(f)
                        auc = mdict['roc_auc_loo']
                        X[kcat, kclass] = auc
                except:
                    pass

        self.best_models = {}
        self.best_auc = {}
        data = ["Best models: \r\n"]
        if verbose:
            print("Best models:")
        for kcat in range(0, Ncats):
            x = X[kcat]
            cual = np.argmax(x)
            maximo = np.max(x)
            cadena = (self.categories[kcat] + ' \t ' + classifiers[cual] +
                      ' \t AUC = ' + str(maximo))
            data .append(cadena + '\r\n')
            if verbose:
                print(cadena)
            self.best_models.update({self.categories[kcat]: classifiers[cual]})
            self.best_auc.update({self.categories[kcat]: maximo})

        output_file = os.path.join(
            self._get_folder('results'), 'best_models.json')
        with open(output_file, 'w') as f:
            json.dump(self.best_models, f)

        print('Writing best_models to export')
        output_file = os.path.join(
            self._get_folder('export'), 'best_models.json')
        with open(output_file, 'w') as f:
            json.dump(self.best_models, f)

        output_file = os.path.join(
            self._get_folder('results'), 'best_auc.json')
        with open(output_file, 'w') as f:
            json.dump(self.best_auc, f)

        output_file = os.path.join(
            self._get_folder('results'), 'best_models.txt')
        with open(output_file, 'w') as f:
            f.writelines(data)

        # writing the best models in export
        for kcat in range(0, Ncats):
            cat = self.categories[kcat]
            model_name = self.best_models[cat]
            model_params = self.models2evaluate[model_name]

            filename = os.path.join(
                self._get_folder('models'), model_params['model_name'] +
                '_' + str(cat) + '_classifier.pkl')
            with open(filename, 'rb') as f:
                model = pickle.load(f)

            filename = os.path.join(
                self._get_folder('export'), model_params['model_name'] +
                '_' + str(cat) + '_classifier.pkl')
            with open(filename, 'wb') as f:
                pickle.dump(model, f)

        return

    def compute_thresholds(self, verbose=True):
        '''
        computes the thresholds of the classifiers
        '''

        if self.categories is None:
            self._recover('categories')

        if self.best_models is None:
            self._recover('best_models')

        Ncats = len(self.categories)
        ths_dict = {}

        for kcat in range(0, Ncats):
            try:
                cat = self.categories[kcat]
                model_name = self.best_models[cat]
                file_input_ROC = os.path.join(
                    self._get_folder('eval_ROCs'),
                    'ROC_' + model_name + '_' + cat + '.pkl')
                with open(file_input_ROC, 'rb') as f:
                    mdict = pickle.load(f)
                fpr = mdict['fpr_loo']
                tpr = mdict['tpr_loo']
                ths = mdict['thresholds']

                mix = []
                for k in range(0, len(fpr)):
                    mix.append(tpr[k] + (1 - fpr[k]))

                cual = np.argmax(mix)
                th = ths[cual]
                ths_dict.update({cat: th})
                print(cat, th, cual, tpr[cual], fpr[cual])
            except:
                print("Error in cat ", cat)
                pass

        filename = os.path.join(
            self._project_path + self._subfolders['results'], 'ths_dict.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(ths_dict, f)

        filename = os.path.join(
            self._project_path + self._subfolders['export'], 'ths_dict.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(ths_dict, f)

        return
