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
import pandas as pd
from sklearn.metrics import roc_curve, auc
import json
import matplotlib.pyplot as plt
import operator
import itertools
from sklearn.metrics import confusion_matrix
from collections import OrderedDict

import pyemd

# Local imports
from fordclassifier.evaluator.predictorClass import Predictor
from fordclassifier.evaluator.rbo import *

import pdb


class Evaluator(object):
    '''
    Class to evaluate the performance of the classifiers

    ============================================================================
    Methods:
    ============================================================================
    _recover:     if a variable is not in memory, tries to recover it from disk
    _get_folder:               retuns full path to a subfolder
    _exists_file:              check if the file exists in disk
    draw_rocs:                 draws the Xval ROCs and saves them as png files
    load_Xtfidf:               Loads from disk Xtfidf and tags
    load_test_data:            Loads from disk test Xtfidf and tags
    load_train_data:           Loads from disk train Xtfidf and tags
    compute_average_xval_AUC:  computes the average AUC on xval
    compute_average_test_AUC:  computes the average AUC on test
    obtain_labels_from_Preds:  Produces the multilabel tag prediction from
                               individual predictions of every classifier

    compute_confussion_matrix: computes the confusion matrix on test
                               (multiclass case)
    compute_confusion_matrix_multilabel: computes the confussion matrix for a
                               multilabel set (multilabel case)
    draw_confussion_matrix:    draws the CM and saves it as a png file
    draw_ROCS_tst:             draws the ROC curves for the test data
    draw_anyROC:               draws the ROC curves
    compute_thresholds:        computes the thresholds
    compute_cardinality:       computes the cardinality of the tags
    compute_label_density:     Computes the label density
    JaccardIndex:              Computes the Jaccard index
    compute_multilabel_threshold:   Computes the multilabel threshold
    draw_costs_on_test:        draws the multilabel cost for the test data
    load_multilabel_threshold: Loads the multilabel thresholds
    Jaccard_RBO_cost:          Computes a convex combination of the Jaccard and
                               RBO costs
    align_strings:             Aligns strings into columns
    get_pred_weights:          Returns the normalized predictions
    write_prediction_report:   writes a simple prediction report in text format
    ============================================================================
    '''

    def __init__(self, project_path, subfolders, categories=None, verbose=True):
        '''
        Initialization: Creates the initial object data
        Inputs:
            - project_path: path to the working project
            - subfolders: subfolder structure
        '''

        self._project_path = project_path   # working directory
        self._verbose = verbose     # messages are printed on screen when True
        self.models2evaluate = None  # models to evaluate (classif, params)
        self._subfolders = None      # subfolders structure
        self.best_auc = None         # Best AUC
        self.best_models = None      # Best models
        self.Xtfidf_tr = None        # Xtfidf for training

        self.tags_tr = None          # Training tags
        self.tags = None             # All tags

        self.ths_dict = None    # dict with the thresholds for every classifier
        self.Preds = None       # Prediction matrix, one column per category
        self.Preds_tr = None    # Pred. matrix, one column per category, train
        self.Preds_tst = None   # Pred. matrix, one column per category, test
        self.index_tst = None   # Index for tags test
        self.categories = categories  # List of categories
        self.Xtfidf_tst = None        # Xtfidf for test
        self.tags_tst = None          # Test tags
        self.CONF = None              # Confusion matrix
        self.multilabel_th = None     # Multilabel Threshold

        self._subfolders = subfolders

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

        if field == 'best_auc':
            input_file = os.path.join(self._get_folder('results'),
                                      'best_auc.json')
            with open(input_file, 'r') as f:
                self.best_auc = json.load(f)

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
                [self.Xtfidf_tr, tags_tr, self.tags_tr,
                 refs_tr] = pickle.load(f)

        if field == 'Xtfidf_tst':
            filetoload_Xtfidf = os.path.join(
                self._project_path + self._subfolders['test_data'],
                'test_data.pkl')
            with open(filetoload_Xtfidf, 'rb') as f:
                [self.Xtfidf_tst, tags_tst, self.tags_tst,
                 refs_tst] = pickle.load(f)

        if field == 'tags':
            filetoload_tags = os.path.join(
                self._project_path + self._subfolders['training_data'],
                'tags.pkl')
            with open(filetoload_tags, 'rb') as f:
                self.tags = pickle.load(f)

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

        if field == 'Preds':
            filename = os.path.join(
                self._project_path + self._subfolders['results'], 'Preds.pkl')
            with open(filename, 'rb') as f:
                self.Preds = pickle.load(f)

        if field == 'Preds_tr':
            filename = os.path.join(
                self._project_path, self._subfolders['results'],
                'Preds_tr.pkl')
            with open(filename, 'rb') as f:
                self.Preds_tr = pickle.load(f)

        if field == 'Preds_tst':
            filename = os.path.join(
                self._project_path, self._subfolders['results'],
                'Preds_test.pkl')
            with open(filename, 'rb') as f:
                self.Preds_tst = pickle.load(f)

        if field == 'CONF':
            filename = os.path.join(
                self._project_path + self._subfolders['results'], 'CONF.pkl')
            with open(filename, 'rb') as f:
                self.CONF = pickle.load(f)

        if field == 'tags_index':
            filename = os.path.join(
                self._project_path + self._subfolders['test_data'],
                'tags_index.pkl')
            with open(filename, 'rb') as f:
                [self.tags_tst, self.index_tst] = pickle.load(f)

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

        if field == 'multilabel_th':
            try:
                filename = os.path.join(
                    self._project_path + self._subfolders['training_data'],
                    'multilabel_th.pkl')
                with open(filename, 'rb') as f:
                    self.multilabel_th = pickle.load(f)
            except:
                filename = os.path.join(
                    self._project_path + self._subfolders['export'],
                    'multilabel_th.pkl')
                with open(filename, 'rb') as f:
                    self.multilabel_th = pickle.load(f)
                pass

        return

    def draw_rocs(self, verbose=True):
        '''
        Draws the Xval ROCs and saves them as png files
        Inputs:
            - None, it operates on self values
        '''
        if verbose:
            print("Saving ROC figures ...")

        if self.categories is None:
            self._recover('categories')

        if self.models2evaluate is None:
            self._recover('models2evaluate')

        # get the evaluated models
        models = list(self.models2evaluate.keys())
        Nclass = len(models)
        Ncats = len(self.categories)

        for kcat in range(0, Ncats):
            plt.figure(figsize=(15, 12))
            aucs = []
            cat = self.categories[kcat]
            for kclass in range(0, Nclass):
                try:
                    model_name = models[kclass]
                    file_input_ROC = os.path.join(
                        self._get_folder('eval_ROCs'),
                        'ROC_' + model_name + '_' + cat + '.pkl')
                    with open(file_input_ROC, 'rb') as f:
                        mdict = pickle.load(f)
                    auc = mdict['roc_auc_loo']
                    aucs.append((model_name, auc))
                except:
                    pass

            # Sorting by AUC
            aucs.sort(key=operator.itemgetter(1), reverse=True)

            colors = ['k', 'r', 'g', 'b', 'm', 'c', 'r--', 'g--', 'b--', 'm--',
                      'c--', 'k--']

            # drawing the best 10 models
            for k in range(0, 10):
                try:
                    model_name = aucs[k][0]
                    auc = aucs[k][1]
                    file_input_ROC = os.path.join(
                        self._get_folder('eval_ROCs'),
                        'ROC_' + model_name + '_' + cat + '.pkl')
                    with open(file_input_ROC, 'rb') as f:
                        mdict = pickle.load(f)
                    fpr = mdict['fpr_loo']
                    tpr = mdict['tpr_loo']
                    text = model_name + ', AUC= ' + str(auc)[0:6]
                    if auc > 0.6:
                        if k == 0:
                            # drawing the best model with thicker line
                            plt.plot(fpr, tpr, colors[k], label=text,
                                     linewidth=6.0)
                        else:
                            plt.plot(fpr, tpr, colors[k], label=text,
                                     linewidth=2.0)
                except:
                    pass

            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('ROC curves for category ' + cat)
            plt.grid(True)
            plt.legend(loc="lower right")
            filename = os.path.join(self._get_folder('ROCS_tr'),
                                    cat + '_ROC_xval.png')
            plt.savefig(filename)
            plt.close()
            if verbose:
                print(cat, )
        return

    def load_Xtfidf(self, verbose=True):
        '''
        Loads from disk Xtfidf and tags
        Inputs:
            - None, it operates on self values
        '''

        if self.Xtfidf is None:
            self._recover('Xtfidf')

        if self.tags is None:
            self._recover('tags')

        return self.Xtfidf, self.tags

    def load_test_data(self, verbose=True):
        '''
        Loads from disk test Xtfidf and tags
        Inputs:
            - None, it operates on self values
        '''

        filename = os.path.join(
            self._project_path + self._subfolders['test_data'],
            'test_data.pkl')
        with open(filename, 'rb') as f:
            [self.Xtfidf_tst, self.tags_tst, refs_tst] = pickle.load(f)

        new_tags_tst = []
        for tags in self.tags_tst:
            unique_tags = sorted(set(tags), key=tags.index)
            new_tags_tst.append(unique_tags)

        return self.Xtfidf_tst, new_tags_tst, refs_tst

    def load_train_data(self, verbose=True):
        '''
        Loads from disk train Xtfidf and tags
        Inputs:
            - None, it operates on self values
        '''
        filename = os.path.join(
            self._project_path + self._subfolders['training_data'],
            'train_data.pkl')
        with open(filename, 'rb') as f:
            [self.Xtfidf_tr, self.tags_tr, refs_tr] = pickle.load(f)

        new_tags_tr = []
        for tags in self.tags_tr:
            unique_tags = sorted(set(tags), key=tags.index)
            new_tags_tr.append(unique_tags)

        return self.Xtfidf_tr, new_tags_tr, refs_tr

    def compute_average_xval_AUC(self, verbose=True):
        '''
        Computes the average AUC on xval
        Inputs:
            - None, it operates on self values
        '''

        if self.best_auc is None:
            self._recover('best_auc')

        aucs = list(self.best_auc.values())
        average_auc = np.mean(aucs)
        return average_auc

    def obtain_labels_from_Preds(self, Preds, threshold,
                                 categories=None, verbose=True):

        '''
        Produces the multilabel tag prediction from individual predictions of
        every classifier
        Inputs:
            - Preds: predictions matrix, one column per category, as many rows
                     as patterns
            - threshold: multilabel threshold
        '''

        if self.categories is None:
            self._recover('categories')

        labels_preds = []
        Ndocs = Preds.shape[0]
        for kdoc in range(0, Ndocs):
            l = []
            p = Preds[kdoc, :]
            # Normalize individual predictions, the maximum becomes 1.0 in all
            # cases
            if max(p) > 0:
                p = p / max(p)
            orden = np.argsort(-p)
            for index in orden:
                if p[index] > threshold:
                    l.append(self.categories[index])
            labels_preds.append(l)

        return labels_preds

    def compute_confusion_matrix(self, orig_tags, best_pred_tags, filename,
                                 sorted_categories=[], verbose=True):
        '''
        computes the confussion matrix on test (multiclass case)
        Inputs:
            - orig_tags: original labels
            - best_pred_tags: predicted labels
            - filename: file to save results
            - sorted_categories: categories to take into account, respecting
              the order
        '''
        if self.categories is None:
            self._recover('categories')

        if len(sorted_categories) > 0:
            labels_categories = sorted_categories
        else:
            labels_categories = self.categories

        self.CONF = confusion_matrix(orig_tags, best_pred_tags,
                                     labels=labels_categories)

        pathfilename = os.path.join(
            self._project_path + self._subfolders['results'], filename)
        with open(pathfilename, 'wb') as f:
            pickle.dump(self.CONF, f)

        return self.CONF

    def compute_confusion_matrix_multilabel(self, orig_tags, best_pred_tags,
                                            filename, sorted_categories=[],
                                            verbose=True):
        '''
        computes the confussion matrix for a multilabel set (multilabel case)
        Inputs:
            - orig_tags: original labels
            - best_pred_tags: predicted labels
            - filename: file to save results
            - sorted_categories: categories to take into account, respecting
              the order
        '''

        if self.categories is None:
            self._recover('categories')

        if len(sorted_categories) > 0:
            labels_categories = sorted_categories
        else:
            labels_categories = self.categories

        Ncats = len(labels_categories)
        self.CONF = np.zeros((Ncats, Ncats))
        NP = len(orig_tags)

        for k in range(0, NP):
            cats_orig = orig_tags[k]
            cats_pred = best_pred_tags[k]
            for m in range(0, Ncats):
                for n in range(0, Ncats):
                    cat_orig = labels_categories[m]
                    cat_pred = labels_categories[n]

                    if cat_orig in cats_orig and cat_pred in cats_pred:
                        self.CONF[m, n] += 1.0

        # self.CONF = confusion_matrix(orig_tags, best_pred_tags,
        #                             labels=labels_categories)

        pathfilename = os.path.join(
            self._project_path + self._subfolders['results'], filename)
        with open(pathfilename, 'wb') as f:
            pickle.dump(self.CONF, f)

        return self.CONF

    def compute_confusion_matrix_multilabel_v2(
           self, orig_tags, best_pred_tags, filename, sorted_categories=[],
            order_sensitive=False, verbose=True):
        '''
        computes the confusion matrix for a multilabel set
        Inputs:
            - orig_tags: original labels
            - best_pred_tags: predicted labels
            - filename: file to save results
            - sorted_categories: categories to take into account, respecting
              the order
            - order_sensitive: indicates if the computation is order sensitive
              or not
        '''

        # Set dump factor
        if order_sensitive:
            dump_factor = 0.5
        else:
            dump_factor = 1.0

        # Take categories from the input arguments. If not, from the object.
        # If not, from a file using the recover method.
        if len(sorted_categories) > 0:
            categories = sorted_categories
        else:
            # Get list of categories
            if self.categories is None:
                self._recover('categories')
            categories = self.categories

        # Validate true labels
        n = len([x for x in orig_tags if len(x) == 0])
        if n > 0:
            print('---- WARNING: {} samples without labels '.format(n) +
                  'will be ignored.')

        # Validate predicted labels
        n = len([x for x in best_pred_tags if len(x) == 0])
        if n > 0:
            print('---- WARNING: {} samples without predictions '.format(n) +
                  'will be ignored.')

        # Loop over the true and predicted labels
        Ncats = len(categories)
        self.CONF = np.zeros((Ncats, Ncats))
        for cats_orig, cats_pred in zip(orig_tags, best_pred_tags):

            if len(cats_orig) > 0 and len(cats_pred) > 0:

                # Compute numerical true label vector
                value_orig = 1.0
                p = np.zeros(Ncats)
                for c in cats_orig:
                    p[categories.index(c)] = value_orig
                    value_orig *= dump_factor
                p = p / np.sum(p)

                # Compute numerical prediction label vector
                value_pred = 1.0
                q = np.zeros(Ncats)
                for c in cats_pred:
                    q[categories.index(c)] = value_pred
                    value_pred *= dump_factor
                q = q / np.sum(q)

                # Compute diagonal elements
                min_pq = np.minimum(p, q)
                M = np.diag(min_pq)

                # Compute non-diagonal elements
                p_ = p - min_pq
                q_ = q - min_pq
                z = 1 - np.sum(min_pq)
                if z > 0:
                    M += (p_[:, np.newaxis] * q_) / z

                self.CONF += M

        pathfilename = os.path.join(
            self._project_path, self._subfolders['results'], filename)
        with open(pathfilename, 'wb') as f:
            pickle.dump(self.CONF, f)

        return self.CONF

    def compute_EMD_error(self, orig_tags, best_pred_tags, fpath,
                          order_sensitive=False):
        '''
        computes the confusion matrix for a multilabel set
        Args:
            - orig_tags:      original labels
            - best_pred_tags: predicted labels
            - fpath:          path to the file with the similarity matrix
            - order_sensitive: indicates if the computation is order sensitive
                              or not
        '''

        # ######################
        # Load similarity values
        if type(fpath) is str:
            df_S = pd.read_excel(fpath)
            # Compute cost matrix
            C = 1 - df_S[df_S.columns].values
            # WARNING: For later versions of pandas, you might need to use:
            #          Note that df_S.columnst shooud be taken from 1, because
            #          The first column is taken as the index column.
            # C = 1 - df_S[df_S.columns[1:]].to_numpy()
        else:
            # This is a combination of cost matrices that takes the
            # component-wise minimum of the costs
            C = 1
            for fp in fpath:
                df_S = pd.read_excel(fp)
                # Compute cost matrix
                Cf = 1 - df_S[df_S.columns].values
                C = np.minimum(C, Cf)
            
            # This combination of cost matrices takes each cost matrix with a
            # different weights. Only for two Cost matrices.
            # df_S = pd.read_excel(fpath[0])            
            # C1 = 1 - df_S[df_S.columns].values
            # df_S = pd.read_excel(fpath[1])
            # Cs = 1 - df_S[df_S.columns].values
            # ncat = Cs.shape[0]
            # C = np.minimum(1 - np.eye(ncat),
            #              np.minimum(0.25 + 0.75 * Cs,  0.5 + 0.5 * C1)) 

        # This is to make sure that C is "C-contitguos", a requirement of pyemd
        C = np.ascontiguousarray(C, dtype=np.float64)

        # Set dump factor
        if order_sensitive:
            dump_factor = 0.5
        else:
            dump_factor = 1.0

        # Take categories in the order of the cost matrix
        categories = df_S.columns.tolist()

        # Validate true labels
        n = len([x for x in orig_tags if len(x) == 0])
        if n > 0:
            print(f'---- WARNING: {n} samples without labels will be ignored')

        # Validate predicted labels
        n = len([x for x in best_pred_tags if len(x) == 0])
        if n > 0:
            print(f'---- WARNING: {n} samples without preds will be ignored')

        # ##################
        # Compute EMD errors

        # Loop over the true and predicted labels
        Ncats = len(categories)
        self.emd = 0
        count = 0
        for cats_orig, cats_pred in zip(orig_tags, best_pred_tags):

            if len(cats_orig) > 0 and len(cats_pred) > 0:

                # Compute numerical true label vector
                value_orig = 1.0
                p = np.zeros(Ncats)
                for c in cats_orig:
                    p[categories.index(c)] = value_orig
                    value_orig *= dump_factor
                p = p / np.sum(p)

                # Compute numerical prediction label vector
                value_pred = 1.0
                q = np.zeros(Ncats)
                for c in cats_pred:
                    q[categories.index(c)] = value_pred
                    value_pred *= dump_factor
                q = q / np.sum(q)

                # Compute EMD distance for the given sample
                emd_i = pyemd.emd(p, q, C)
                self.emd += emd_i
                count += 1

        self.emd /= count

        return self.emd

    def compute_sorted_errors(self, CONF, categories):

        eps = 1e-20

        # Sample size per category
        n_cat = len(categories)
        ns_cat = CONF.sum(axis=1, keepdims=True)
        # Total sample size
        ns_tot = CONF.sum()

        # Compute all-normalized confusion matrix
        CONF_a = CONF / ns_tot
        # Compute row-normalized confusion matrix
        CONF_r = ((CONF.astype('float') + eps) /
                  (ns_cat + n_cat*eps))

        # Sort errors by
        unsorted_values = [(categories[i], categories[j], 100*CONF_a[i, j],
                            100*CONF_r[i, j], 100*ns_cat[i][0]/ns_tot)
                           for i in range(n_cat) for j in range(n_cat)]
        sorted_values_a = sorted(unsorted_values, key=lambda x: -x[2])
        sorted_values_r = sorted(unsorted_values, key=lambda x: -x[3])

        # Remove diagonal elements
        sorted_values_a = [x for x in sorted_values_a if x[0] != x[1]]
        sorted_values_r = [x for x in sorted_values_r if x[0] != x[1]]

        # Remove relative errors of categories with zero samples
        sorted_values_r = [x for x in sorted_values_r
                           if ns_cat[categories.index(x[0])] > 0]

        cols = ['Cat. real', 'Clasif', 'Err/total (%)', 'Error/cat (%)',
                'Peso muestral']
        df_ranked_abs = pd.DataFrame(sorted_values_a, columns=cols)
        df_ranked_rel = pd.DataFrame(sorted_values_r, columns=cols)

        f_path = os.path.join(self._project_path, self._subfolders['results'],
                              'ranked_abs_errors.xlsx')
        df_ranked_abs.to_excel(f_path)

        f_path = os.path.join(self._project_path, self._subfolders['results'],
                              'ranked_rel_errors.xlsx')
        df_ranked_rel.to_excel(f_path)

        return df_ranked_abs, df_ranked_rel

    def compute_error_confusion_matrix(self, CONF, normalize=True,
                                       verbose=True):

        # Returns the ratio of elements outside the diagonal
        allsum = np.sum(CONF)
        diagsum = np.sum(np.diagonal(CONF))
        offdiagsum = allsum - diagsum
        error = offdiagsum / allsum
        return error

    def draw_confusion_matrix(self, CONF, filename, sorted_categories=[],
                              verbose=True, normalize=True):
        '''
        draws the CM and saves it as a png file
        Inputs:
            - CONF: conf matrix to be stored
            - filename: filename
            - sorted_categories: list of sorted categories
            - normalize: indicates to normalize CONF
        '''

        # An extemelly small value to avoid zero division
        eps = 1e-20
        n_cat = len(sorted_categories)

        if len(sorted_categories) > 0:
            labels_categories = sorted_categories
        else:
            if self.categories is None:
                self._recover('categories')
            labels_categories = self.categories

        if normalize:
            # Normalize
            CONF = ((CONF.astype('float') + eps) /
                    (CONF.sum(axis=1, keepdims=True) + n_cat*eps))
        else:
            CONF = CONF.astype('float')

        plt.figure(figsize=(15, 12))
        cmap = plt.cm.Blues
        plt.imshow(CONF, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        tick_marks = np.arange(len(labels_categories))
        plt.xticks(tick_marks, labels_categories, rotation=90)
        plt.yticks(tick_marks, labels_categories)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        pathfilename = os.path.join(self._get_folder('figures'), filename)
        print(f"SALVADO EN {pathfilename}")
        plt.savefig(pathfilename)
        plt.clf()

        return

    def draw_ROCS_tst(self, Preds_tst, tags_tst):
        '''
        draws the ROC curves for the test data
        Inputs:
            - Preds_tst: predicted labels
            - tags_tst: true labels
        '''

        if self.best_models is None:
            self._recover('best_models')

        if self.categories is None:
            self._recover('categories')

        colors = ['k', 'r', 'g', 'b', 'm', 'c', 'r--', 'g--', 'b--', 'm--',
                  'c--', 'k--']

        # retain the first tag in the labels
        tags = [t[0] if len(t) > 0 else '' for t in tags_tst]

        for k in range(0, len(self.categories)):
            cat = self.categories[k]

            y_tst = [1.0 if p == cat else -1.0 for p in tags]
            preds_tst = list(Preds_tst[:, k])

            fpr_tst, tpr_tst, thresholds = roc_curve(y_tst, preds_tst)
            roc_auc_tst = auc(fpr_tst, tpr_tst)
            model_name = self.best_models[cat]
            file_output_ROC = os.path.join(
                self._get_folder('ROCS_tst'),
                'ROC_' + model_name + '_' + cat + '.pkl')
            mdict = {'fpr_tst': list(fpr_tst), 'tpr_tst': list(tpr_tst),
                     'roc_auc_tst': roc_auc_tst, 'y_tst': list(y_tst),
                     'preds_tst': list(preds_tst)}
            with open(file_output_ROC, 'wb') as f:
                pickle.dump(mdict, f)
            plt.figure(figsize=(15, 12))
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('ROC test curve for category ' + cat)
            text = model_name + ', AUC= ' + str(roc_auc_tst)[0:6]
            plt.plot(fpr_tst, tpr_tst, colors[3], label=text, linewidth=6.0)
            plt.grid(True)
            plt.legend(loc="lower right")
            filename = os.path.join(
                self._get_folder('ROCS_tst'), cat + '_ROC_test.png')
            plt.savefig(filename)
            plt.close()
        return

    def draw_anyROC(self, Preds_tst, tags_tst, case):
        '''
        draws the ROC curves
        Inputs:
            - Preds_tst: predicted labels
            - tags_tst: true labels
        '''

        if self.categories is None:
            self._recover('categories')

        colors = ['k', 'r', 'g', 'b', 'm', 'c', 'r--', 'g--', 'b--', 'm--',
                  'c--', 'k--']

        # retain the first tag in the labels
        tags = [t[0] if len(t) > 0 else '' for t in tags_tst]

        aucs = []
        for k in range(0, len(self.categories)):
            cat = self.categories[k]

            y_tst = [1.0 if p == cat else -1.0 for p in tags]
            preds_tst = list(Preds_tst[:, k])

            fpr_tst, tpr_tst, thresholds = roc_curve(y_tst, preds_tst)
            roc_auc_tst = auc(fpr_tst, tpr_tst)
            aucs.append(roc_auc_tst)
            file_output_ROC = os.path.join(
                self._get_folder('ROCS_tst'),
                cat + '_' + 'ROC_' + case + '.pkl')
            mdict = {'fpr_tst': list(fpr_tst), 'tpr_tst': list(tpr_tst),
                     'roc_auc_tst': roc_auc_tst, 'y_tst': list(y_tst),
                     'preds_tst': list(preds_tst)}
            with open(file_output_ROC, 'wb') as f:
                pickle.dump(mdict, f)

            plt.figure(figsize=(15, 12))
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('ROC test curve for category ' + cat)
            text = case + ', AUC= ' + str(roc_auc_tst)[0:6]
            plt.plot(fpr_tst, tpr_tst, colors[3], label=text, linewidth=6.0)
            plt.grid(True)
            plt.legend(loc="lower right")
            filename = os.path.join(self._get_folder('ROCS_tst'),
                                    cat + '_' + 'ROC_' + case + '.png')
            plt.savefig(filename)
            plt.close()
        average_auc = np.nanmean(aucs)
        return average_auc

    def compute_average_test_AUC(self, verbose=True):
        '''
        computes the average AUC on test
        Inputs:
            - None, it operates on self values
        '''
        if self.best_models is None:
            self._recover('best_models')

        if self.categories is None:
            self._recover('categories')

        aucs = []
        for k in range(0, len(self.categories)):
            cat = self.categories[k]
            model_name = self.best_models[cat]
            filename = os.path.join(
                self._get_folder('ROCS_tst'),
                'ROC_' + model_name + '_' + cat + '.pkl')
            with open(filename, 'rb') as f:
                mdict = pickle.load(f)
            auc = mdict['roc_auc_tst']
            aucs.append(auc)
        average_auc = np.nanmean(aucs)
        return average_auc

    def compute_thresholds(self, verbose=True):
        '''
        computes the thresholds
        Inputs:
            - None, it operates on self values
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
                    # We select the threshold maximizing this convex combinat
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
        return

    def compute_cardinality(self, tags):
        '''
        computes the cardinality of the tags
        Inputs:
            - tags: labels
        '''
        C = np.mean([len(set(l)) for l in tags])
        return C

    def compute_label_density(self, tags):
        '''
        Computes the label density
        Inputs:
            - tags: labels
        '''
        # total number of possible labels
        NL = len(set(itertools.chain.from_iterable(tags)))
        D = np.mean([len(set(l)) / NL for l in tags])
        return D

    def JaccardIndex(self, orig, pred):
        '''
        Computes the Jaccard index
        Inputs:
            - orig: original labels
            - pred: predicted labels
        '''
        accs = []
        for k in range(0, len(orig)):
            l_orig = orig[k]
            l_pred = pred[k]
            num = len(set(l_orig).intersection(l_pred))
            den = len(set(l_orig + l_pred))
            acc = num / den
            accs.append(acc)
        JI = np.mean(accs)
        return JI

    def compute_multilabel_threshold(self, p, alpha, option, th_values,
                                     verbose=True):
        '''
        Computes the multilabel threshold
        Inputs:
            - p: RBO parameter, ``p`` is the probability of looking for overlap
                 at rank k + 1 after having examined rank k
            - alpha: convex Jaccard-RBO combination parameter
            - option: sorting option for multilabel prediction
            - th_values: range of threshold values to be evaluated
        '''
        if self.Xtfidf_tr is None:
            self._recover('Xtfidf_tr')

        if self.Preds_tr is None:
            self._recover('Preds_tr')

        # Warning tags_tr may have duplicates...
        self.tags_tr = [list(OrderedDict.fromkeys(l)) for l in self.tags_tr]

        if verbose:
            print('-' * 50)
        COST = []
        DENS_pred = []
        DENS_true = []
        COST_dens = []
        density_true = self.compute_cardinality(self.tags_tr)

        # to normalize Jaccard_RBO_cost, depends on p
        baseline = [0]
        for k in range(2, 50):
            l = list(range(1, k))
            baseline.append(rbo(l, l, p)['min'])

        P = Predictor(self._project_path, self._subfolders, verbose=False)

        for threshold in th_values:
            multilabel_pred_tr, labels_pred_tr = P.obtain_multilabel_preds(
                self.Preds_tr, option, threshold, verbose=True)
            density_pred = self.compute_cardinality(labels_pred_tr)
            DENS_pred.append(density_pred)
            DENS_true.append(density_true)
            dens_error = (density_pred - density_true) ** 2
            COST_dens.append(dens_error)
            # Computing Jackard_RBO cost
            jrbos = []
            for k in range(0, len(self.tags_tr)):
                values = []
                for key in labels_pred_tr[k]:
                    values.append((key, multilabel_pred_tr[k][key]['p']))
                values.sort(key=lambda x: x[1], reverse=True)
                l_pred = []
                for v in values:
                    l_pred.append(v[0])
                jrbo = self.Jaccard_RBO_cost(
                    self.tags_tr[k], l_pred, baseline, p, alpha)
                jrbos.append(jrbo)

            cost_jrbo = np.mean(jrbos)
            print(threshold, cost_jrbo, density_true, density_pred, )
            COST.append(cost_jrbo)

        max_cost = max(COST)
        max_dens = max(COST_dens)

        COST_dens = [x / max_dens * max_cost for x in COST_dens]

        plt.figure(figsize=(15, 12))
        plt.xlabel('Th')
        plt.ylabel('Jackard-RBO cost')
        plt.title('Jackard-RBO and Label Density costs for p =' + str(p) +
                  ' and alpha= ' + str(alpha))
        plt.plot(th_values, COST, 'b', label='Jackard-RBO cost', linewidth=3.0)
        plt.plot(th_values, COST_dens, 'r', label='Labels Density cost',
                 linewidth=3.0)
        cual_min = np.argmin(COST)
        th_JRBO = th_values[cual_min]
        plt.plot(th_values[cual_min], COST[cual_min], 'bo',
                 label='Minimum Jackard-RBO cost', linewidth=3.0)
        cual_min = np.argmin(COST_dens)
        th_DENS = th_values[cual_min]
        plt.plot(th_values[cual_min], COST_dens[cual_min], 'ro',
                 label='Minimum Labels Density cost', linewidth=3.0)
        plt.legend(loc="upper right")
        plt.grid(True)
        filename = os.path.join(
            self._project_path + self._subfolders['results'],
            'JRBO_COST_tr_p_' + str(p) + '_alpha_' + str(alpha) + '.png')
        plt.savefig(filename)
        plt.close()

        self.multilabel_th = np.mean([th_JRBO, th_DENS])

        filename = os.path.join(
            self._project_path + self._subfolders['training_data'],
            'multilabel_th.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(self.multilabel_th, f)

        filename = os.path.join(
            self._project_path + self._subfolders['export'],
            'multilabel_th.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(self.multilabel_th, f)

        return self.multilabel_th

    def draw_costs_on_test(self, p, alpha, option, th_values, verbose=True):
        '''
        draws the multilabel cost for the test data
        Inputs:
            - p: RBO parameter, ``p`` is the probability of looking for
                 overlap at rank k + 1 after having examined rank k
            - alpha: convex Jaccard-RBO combination parameter
            - option: sorting option for multilabel prediction
            - th_values: range of threshold values to be evaluated
        '''

        if self.Xtfidf_tst is None:
            self._recover('Xtfidf_tst')

        if self.Preds_tst is None:
            self._recover('Preds_tst')

        if self.multilabel_th is None:
            self._recover('multilabel_th')

        # Warning tags_tst may have duplicates...
        self.tags_tst = [list(OrderedDict.fromkeys(l)) for l in self.tags_tst]

        if verbose:
            print('-' * 50)
        COST = []
        DENS_pred = []
        DENS_true = []
        COST_dens = []
        density_true = self.compute_cardinality(self.tags_tst)

        # to normalize Jaccard_RBO_cost, depends on p
        baseline = [0]
        for k in range(2, 50):
            l = list(range(1, k))
            baseline.append(rbo(l, l, p)['min'])

        P = Predictor(self._project_path, self._subfolders, verbose=False)

        for threshold in th_values:
            multilabel_pred_tst, labels_pred_tst = P.obtain_multilabel_preds(
                self.Preds_tst, option, threshold, verbose=True)
            density_pred = self.compute_cardinality(labels_pred_tst)
            DENS_pred.append(density_pred)
            DENS_true.append(density_true)
            dens_error = (density_pred - density_true) ** 2
            COST_dens.append(dens_error)
            # Computing Jackard_RBO cost
            jrbos = []
            for k in range(0, len(self.tags_tst)):
                values = []
                for key in labels_pred_tst[k]:
                    values.append((key, multilabel_pred_tst[k][key]['p']))
                values.sort(key=lambda x: x[1], reverse=True)
                l_pred = []
                for v in values:
                    l_pred.append(v[0])
                jrbo = self.Jaccard_RBO_cost(
                    self.tags_tst[k], l_pred, baseline, p, alpha)
                jrbos.append(jrbo)

            cost_jrbo = np.mean(jrbos)
            print(threshold, cost_jrbo, density_true, density_pred, )
            COST.append(cost_jrbo)

        max_cost = max(COST)
        max_dens = max(COST_dens)

        COST_dens = [x / max_dens * max_cost for x in COST_dens]

        plt.figure(figsize=(15, 12))
        plt.xlabel('Th')
        plt.ylabel('Jackard-RBO cost')
        plt.title('Jackard-RBO and Label Density costs for p =' + str(p) +
                  ' and alpha= ' + str(alpha))
        plt.plot(th_values, COST, 'b', label='Jackard-RBO cost', linewidth=3.0)
        plt.plot(th_values, COST_dens, 'r', label='Labels Density cost',
                 linewidth=3.0)
        cual_min = np.argmin(abs(th_values - self.multilabel_th))
        plt.plot(th_values[cual_min], COST[cual_min], 'bo',
                 label='Jackard-RBO cost at threshold', linewidth=3.0)
        plt.plot(th_values[cual_min], COST_dens[cual_min], 'ro',
                 label='Labels Density cost at threshold', linewidth=3.0)
        plt.legend(loc="upper right")
        plt.grid(True)
        filename = os.path.join(
            self._project_path + self._subfolders['results'],
            'JRBO_COST_tst_p_' + str(p) + '_alpha_' + str(alpha) + '.png')
        plt.savefig(filename)
        plt.close()
        return

    def load_multilabel_threshold(self, path2export=''):
        '''
        Loads the multilabel thresholds
        Inputs:
            - path2export: export path
        '''

        if path2export != '':
            print('Loading multilabel_th from export')
            filename = os.path.join(path2export, 'multilabel_th.pkl')
            with open(filename, 'rb') as f:
                self.multilabel_th = pickle.load(f)
        else:
            if self.multilabel_th is None:
                self._recover('multilabel_th')

        return self.multilabel_th

    def Jaccard_RBO_cost(self, l_orig, l_pred, baseline, p, alpha):
        '''
        Computes a convex combination of the Jaccard and RBO costs
        Inputs:
            - l_orig: original labels
            - l_pred: predicted labels
            - baseline: normalizing values
            - p: RBO parameter, ``p`` is the probability of looking for overlap
                 at rank k + 1 after having examined rank k
            - alpha: convex Jaccard-RBO combination parameter
        '''
        try:
            if len(l_orig) > 0:
                num = len(set(l_orig).intersection(l_pred))
                den = len(set(l_orig + l_pred))
                ji = 1.0 - num / den
            else:
                if len(l_pred) == 0:
                    # empty labels and empty predict means cost = 0.0
                    ji = 0
                else:
                    # empty labels and non-empty predict means cost = 1.0
                    ji = 1.0

            r = 0
            L = min((len(l_orig), len(l_pred)))
            if L > 0:
                r = 1 - rbo(l_orig, l_pred, p)['min'] / baseline[L]
            else:
                r = 1.0
            if len(l_orig) == 0 and len(l_pred) == 0:
                r = 0.0
        except:
            print('Error in Jaccard_RBO_cost ' +
                  '----------------------------------------------------')
            import code
            code.interact(local=locals())
            pass
        jrbo = (alpha * ji + (1 - alpha) * r) / 2.0
        return jrbo

    def align_strings(self, string0, string1, string2, string3, L, M, N, P):
        '''
        Aligns strings into columns
        '''
        empty = '                                                                                 '
        # if len(string1) > M or len(string2) > N or len(string3) > P:
        #    import code
        #    code.interact(local=locals())
        if L - len(string0) > 0:
            string0 = string0 + empty[0: L - len(string0)]
        if M - len(string1) > 0:
            string1 = string1 + empty[0: M - len(string1)]
        if N - len(string2) > 0:
            string2 = string2 + empty[0: N - len(string2)]
        if P - len(string3) > 0:
            string3 = string3 + empty[0: P - len(string3)]

        aligned_string = string0 + '| ' + string1 + '| ' + string2 + '| ' + string3 + '\r\n'
        return aligned_string

    def get_pred_weights(self, refs, label_preds, multilabel_preds):
        '''
        Returns the normalized predictions **Unused????****
        Inputs:
            - refs: *unused*
            - label_preds:
            - multilabel_preds:
        '''
        weights = []
        for k, labels in enumerate(label_preds):
            w0 = [multilabel_preds[k][key]['p'] for key in labels]
            scale = np.sum(w0)
            weights.append([w / scale for w in w0])

        return weights

    def write_prediction_report(self, refs_tst, tags_tst, labels_pred_tst,
                                multilabel_pred_tst, filename_out):
        '''
        writes a simple prediction report in text format
        Inputs:
            - refs_tst: references
            - tags_tst: original labels
            - labels_pred_tst: predicted labels
            - multilabel_pred_tst: multilabel predicted labels
            - filename_out: file to save results
        '''

        # writing report for the best threshold value
        string0 = 'PROJECT REFERENCE'
        string1 = 'TARGET LABELS'
        string2 = 'PREDICTED LABELS'
        string3 = ' '
        data = [self.align_strings(string0, string1, string2, string3, 20, 30,
                                   50, 10)]
        data.append('=' * 80 + '\r\n')

        for k in range(0, len(tags_tst)):

            string0 = refs_tst[k]

            string1 = ''
            for t in tags_tst[k]:
                string1 += t + ', '

            if len(tags_tst[k]) == 0:
                string1 += '--------------'

            values = labels_pred_tst[k]

            if len(values) == 0:
                string2 = '--------------'
            else:
                string2 = ''

            pesos = []
            for key in values:
                pesos.append(multilabel_pred_tst[k][key]['p'])

            for key in values:
                weight = multilabel_pred_tst[k][key]['p'] / np.sum(pesos)
                str_weight = str(weight)[0:5]
                string2 += key + '(' + str_weight + '), '

            string3 = ' '

            cadena = self.align_strings(string0, string1, string2, string3,
                                        20, 30, 50, 10)
            data.append(cadena)

        filename = os.path.join(self._project_path,
                                self._subfolders['results'], filename_out)
        with open(filename, 'w') as f:
            f.writelines(data)
        print('Saved ', filename)

        return
