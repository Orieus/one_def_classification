# -*- coding: utf-8 -*-
'''

@author:  Jesús Cid Sueiro and Angel Navia Vázquez
July 2018

'''
import os
import _pickle as pickle
import pandas as pd
import numpy as np
from tabulate import tabulate
from time import time

# Local imports
from fordclassifier.corpusanalyzer.textproc import Lemmatizer
from fordclassifier.evaluator.classifier_optimizer import classifierOptimizer
from fordclassifier.evaluator.predictorClass import Predictor
from fordclassifier.evaluator.evaluatorClass import Evaluator

import pdb


class FCSupClassifier(object):
    """
    This class contains those methods used to process categorical from the
    corpus database used in the MINECO project (which containts abstracts of
    research projects in a particular DB structure).

    Thus, they are specific for the particular data structure used in this
    project.
    """

    def __init__(self, categories=None, cf=None, path2project=None,
                 f_struct_cls=None, task='FORD', cat_model='multilabel',
                 min_df=2, title_mul=1):

        self.categories = categories
        self.cf = cf
        self.path2project = path2project
        self.f_struct_cls = f_struct_cls
        self.project_groups = None
        self.cat_model = cat_model

        # Minimum document frequency for the bow computation
        self.min_df = min_df

        # Multiplier of the title, to provide more influence to words in the
        # title
        self.title_mul = title_mul

        # task = 'FORD' or 'UNESCO'
        self.task = task

        return

    def partition(self, df, seed_missing_categories=True, fraction_train=0.8,
                  Nfold=10):
        '''
        Creates the data partition
        Inputs:
            df:    Dataframe with the data
            seed_missing_categories: Define labels for the missing categories
                   using a deterministic criterium
            fraction_train: Fraction of the data to be used for training
            Nfold: Number of partitions for crossvalidation
        '''

        # Defining the relative subfolder structure of the project
        subfolders = self.f_struct_cls

        filename = os.path.join(self.path2project, 'subfolders.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(subfolders, f)

        # For data preparation, defininig the classifier types is not relevant
        classifier_types = []

        # Initializing objects
        CO = classifierOptimizer(
            self.path2project, classifier_types, subfolders,
            min_df=self.min_df, verbose=False, cat_model=self.cat_model,
            title_mul=self.title_mul)

        # Filtering language.
        df = df.loc[df['Resumen_lang'] == 'es']
        # newlabels = list(df['Unesco2Ford'])
        print(df.count()[0])

        # Process the df to extract data, and complete missing categories
        ind = CO.prepare_training_data(
            df, self.task, seed_missing_categories=seed_missing_categories,
            verbose=True)

        # The group assigned to each project is taken from "GroupRefs"
        # Single projects (not coordinated) have a None value in "GroupRef"
        # For this reason, I name its group with the name of the project.
        if self.cat_model in ['multilabel', 'weighted']:
            self.project_groups = [
                x[0] if x[1] is None else x[1]
                for x in df[['Referencia', 'GroupRef']].values]
        elif self.cat_model == 'multiclass':
            all_project_groups = [
                x[0] if x[1] is None else x[1]
                for x in df[['Referencia', 'GroupRef']].values]
            self.project_groups = [all_project_groups[i] for i in ind]
        else:
            exit(f'ERROR: Unknown category model: {self.cat_model}')

        # Create train/test partition
        CO.create_partition_train_test_MINECO(
            fraction_train, self.project_groups, verbose=True)

        # Create Nfold partition
        CO.create_Nfold_partition(Nfold, verbose=True)

        return

    def optimizeSupClassifiers(self, classifier_types, classifiers2compare):
        '''
        Trains the classifiers, evaluates and selects the best models
        Inputs:
            -classifier_types : classifier families to be evaluated
            -classifiers2compare : classifier families to fiond the best one
        '''

        subfolders = self.f_struct_cls

        # Initializing objects
        CO = classifierOptimizer(self.path2project, classifier_types,
                                 subfolders, min_df=self.min_df, verbose=False,
                                 cat_model=self.cat_model)
        EV = Evaluator(self.path2project, subfolders, verbose=False)

        # newlabels = list(df['Unesco2Ford'])
        print('---- Cross validation model training ...')
        t0 = time()
        CO.xval_models(verbose=True)
        print(f'     ... done in {time()-t0} seconds')

        # Take the best model, compute thresholds and draw ROCS
        print('---- Selecting best model ...')
        t0 = time()
        CO.find_best_models(classifiers2compare, verbose=True)
        print(f'     ... done in {time()-t0} seconds')

        print('---- Computing thresholds ...')
        t0 = time()
        CO.compute_thresholds()
        print(f'     ... done in {time()-t0} seconds')

        print('---- Drawing ROCS ...')
        t0 = time()
        EV.draw_rocs(verbose=True)
        print(f'     ... done in {time()-t0} seconds')

        return

    def evaluateSupClassifier(self, p, alpha, option, th_values):
        '''
        Author:  Angel Navia Vázquez
        April 2018
        '''

        # Number of steps (only for printing purposes)
        n = 7
        print('---- Evaluating classifiers, please wait...')

        # Defining the relative subfolder structure of the project
        subfolders = self.f_struct_cls
        # This is just to abbreviate
        p2p = self.path2project
        p2res = os.path.join(p2p, subfolders['results'])

        # Take level 2 categories only
        sorted_categories = [x for x in self.categories if not x.endswith('_')]

        # Initializing objects
        EV = Evaluator(p2p, subfolders, verbose=False)
        P = Predictor(p2p, subfolders, verbose=False)

        # ############
        # Loading data
        Xtfidf_tr, tags_tr, refs_tr = EV.load_train_data()
        Xtfidf_tst, tags_tst, refs_tst = EV.load_test_data()

        # ######################
        # 1. Compute Predictions

        print('---- Computing predictions (1/{})...'.format(n))
        Preds_test, tag_pred_test = P.predict(Xtfidf_tst, verbose=True)
        Preds_tr, tag_pred_tr = P.predict(Xtfidf_tr, verbose=True)

        # grabamos fuera para que nos valga el predict para cualquier dato
        filetosave = os.path.join(p2res, 'Preds_test.pkl')
        with open(filetosave, 'wb') as f:
            pickle.dump(Preds_test, f)
        filetosave = os.path.join(p2res, 'tags_pred_test.pkl')
        with open(filetosave, 'wb') as f:
            pickle.dump(tag_pred_test, f)
        filetosave = os.path.join(p2res, 'Preds_tr.pkl')
        with open(filetosave, 'wb') as f:
            pickle.dump(Preds_tr, f)
        filetosave = os.path.join(p2res, 'tags_pred_tr.pkl')
        with open(filetosave, 'wb') as f:
            pickle.dump(tag_pred_tr, f)

        # #############################
        # 2. Compute or load thresholds

        # To compute the final threshold
        print('---- Computing multilabel thresholds (2/{})...'.format(n))
        final_threshold = EV.compute_multilabel_threshold(
            p, alpha, option, th_values)

        # To load a precomputed value
        # print('Loading threshold value...')
        # final_threshold = EV.load_multilabel_threshold()

        # ###################################
        # 3. Computing multilabel predictions

        print('---- Computing multilabel predictions (3/{})...'.format(n))
        multilabel_pred_tr, labels_pred_tr = P.obtain_multilabel_preds(
            Preds_tr, option, final_threshold, verbose=True)

        multilabel_pred_tst, labels_pred_tst = P.obtain_multilabel_preds(
            Preds_test, option, final_threshold, verbose=True)

        filename_out = 'multilabel_prediction_test.txt'
        EV.write_prediction_report(refs_tst, tags_tst, labels_pred_tst,
                                   multilabel_pred_tst, filename_out)

        EV.draw_costs_on_test(p, alpha, option, th_values, verbose=True)

        final_threshold = EV.load_multilabel_threshold()

        order_sensitive = True

        # ###############################
        # 4. Computing confusion matrices

        print('---- Computing confusion matrices (4/{})...'.format(n))

        # Train
        multilabel_pred_tr, labels_tr = P.obtain_multilabel_preds(
            Preds_tr, option, final_threshold, verbose=True)
        CONF_tr_multilabel = EV.compute_confusion_matrix_multilabel_v2(
            tags_tr, labels_tr, 'CONF_tr_multilabel.pkl',
            sorted_categories=sorted_categories,
            order_sensitive=order_sensitive)
        EV.draw_confusion_matrix(
            CONF_tr_multilabel, 'CONF_tr_multilabel.png', sorted_categories)
        EV.draw_confusion_matrix(
            CONF_tr_multilabel, 'CONF_tr_multilabel_un.png', sorted_categories,
            normalize=False)

        # Test
        multilabel_pred_tst, labels_test = P.obtain_multilabel_preds(
            Preds_test, option, final_threshold, verbose=True)
        CONF_tst_multilabel = EV.compute_confusion_matrix_multilabel_v2(
            tags_tst, labels_test, 'CONF_tst_multilabel.pkl',
            sorted_categories=sorted_categories,
            order_sensitive=order_sensitive)
        EV.draw_confusion_matrix(
            CONF_tst_multilabel, 'CONF_tst_multilabel.png', sorted_categories)
        EV.draw_confusion_matrix(
            CONF_tst_multilabel, 'CONF_tst_multilabel_un.png',
            sorted_categories, normalize=False)

        # Compute nanked list of the most confusing categoriy pairs.
        df_ranked_abs, df_ranked_rel = EV.compute_sorted_errors(
            CONF_tst_multilabel, sorted_categories)

        n_max = 30
        headers = ['Cat. real', 'Clasif', 'Err/total (%)',
                   'Error/cat (%)', 'Peso muestral']
        print('---- Top ranked absolute errors:')
        print(df_ranked_abs.head(10))
        # print(tabulate(sorted_a[:n_max], headers=headers, tablefmt='psql'))
        print('---- Top ranked relative errors:')
        print(df_ranked_rel.head(10))
        # print(tabulate(sorted_r[:n_max], headers=headers, tablefmt='psql'))

        # ############################
        # 5. Computing error measures

        print('---- Computing error measures (5/{})...'.format(n))

        # Train
        Er_tr_multilabel = (
            1 - np.trace(CONF_tr_multilabel) / np.sum(CONF_tr_multilabel))
        Er_tst_multilabel = (
            1 - np.trace(CONF_tst_multilabel) / np.sum(CONF_tst_multilabel))

        print('---- ---- Training error: {}'.format(Er_tr_multilabel))
        print('---- ---- Test error: {}'.format(Er_tst_multilabel))

        # Compute EMD errors
        # Level FORD 1 errors
        path2tax = '../MINECO2018/source_data/taxonomy'
        fpath = os.path.join(path2tax, 'S_Matrix_Ford1.xlsx')
        emd_F1_tr = EV.compute_EMD_error(tags_tr, labels_tr, fpath)
        print(f"---- ---- Level 1 train error rate: {emd_F1_tr}")
        emd_F1_tst = EV.compute_EMD_error(tags_tst, labels_test, fpath)
        print(f"---- ---- Level 1 test rate: {emd_F1_tst}")

        # Cost sentitive errors
        fpath = os.path.join(path2tax, 'S_Matrix_sim.xlsx')
        emd_CS_tr = EV.compute_EMD_error(tags_tr, labels_tr, fpath)
        print(f"---- ---- Cost-sensitive train error train rate: {emd_CS_tr}")
        emd_CS_tst = EV.compute_EMD_error(tags_tst, labels_test, fpath)
        print(f"---- ---- Cost-sensitive test error train rate: {emd_CS_tst}")

        # Cost sentitive errors
        fpath = [os.path.join(path2tax, 'S_Matrix_Ford1.xlsx'),
                 os.path.join(path2tax, 'S_Matrix_sim.xlsx')]
        emd_CC_tr = EV.compute_EMD_error(tags_tr, labels_tr, fpath)
        print(f"---- ---- Joint cs train error train rate: {emd_CC_tr}")
        emd_CC_tst = EV.compute_EMD_error(tags_tst, labels_test, fpath)
        print(f"---- ---- Joint cs train error train rate: {emd_CC_tst}")

        # ######################
        # 6. Compute average AUC

        # Compute the average AUC on the test set and draw the test ROCs
        print('---- Computing ROCs (6/{})...'.format(n))
        EV.draw_ROCS_tst(Preds_test, tags_tst)
        average_auc_tst = EV.compute_average_test_AUC()
        print("---- ---- The average AUC on the test set is", average_auc_tst)

        # #################################
        # 7. Save predictions into database
        print('---- Saving classification results to DB (7/{})...'.format(n))

        # Extract prediction weights from the emultilabel_pred dictionaries
        weights_tr = EV.get_pred_weights(refs_tr, labels_tr,
                                         multilabel_pred_tr)
        weights_test = EV.get_pred_weights(refs_tst, labels_test,
                                           multilabel_pred_tst)

        # Build the dataframe required to write into the DB
        refs_all = refs_tr + refs_tst
        labels_all = labels_tr + labels_test
        weights_all = weights_tr + weights_test

        labels_all_str = ['+'.join(x) for x in labels_all]
        weights_all_str = [str(x) for x in weights_all]

        x = list(zip(refs_all, labels_all_str, weights_all_str))
        df_labels = pd.DataFrame(
            x, columns=['Referencia', 'FORDclassif', 'FORDweights'])

        return df_labels

    def computeClassifierPredictions_old(self, df0):

        # Number of steps in this process. This is for print purposes only
        n = 8

        # #######################################
        # 1. Import project data from source file

        print('---- (1/{}) Reading data'.format(n))

        Nproy = df0.count()[0]
        print('---- ---- {} projects loaded.'.format(Nproy))

        # ################
        # 2. Lemmatization

        # Initialize lemmatizer object
        languages = self.cf.get('PREPROC', 'languages')
        lemmatizer_tool = self.cf.get('PREPROC', 'lemmatizer_tool')
        hunspelldic = self.cf.get('PREPROC', 'hunspelldic')
        stw_file = self.cf.get('PREPROC', 'stw_file')
        ngram_file = self.cf.get('PREPROC', 'ngram_file')
        dict_eq_file = self.cf.get('PREPROC', 'dict_eq_file')
        tilde_dictio = self.cf.get('PREPROC', 'tilde_dictio')

        lm = Lemmatizer(languages, lemmatizer_tool, hunspelldic, stw_file,
                        ngram_file, dict_eq_file, tilde_dictio)

        # Read all relevant data in a single dataframe
        refs = df0['Referencia'].values
        abstracts = df0['Resumen'].values
        titles = df0['Título'].values

        # Lemmatize titles
        print('---- (2/{}) Detecting language from titles'.format(n))
        titles_lang = lm.langDetection(titles)
        print('---- (3/{}) Lemmatizing titles'.format(n))
        titles_lemmas = lm.lemmatizeES(titles, chunksize=100)

        # Processing abstracts
        print('---- (4/{}) Detecting language from abstracts'.format(n))
        abstracts_lang = lm.langDetection(abstracts)
        print('---- (5/{}) Lemmatizing abstracts'.format(n))
        abstracts_lemmas = lm.lemmatizeES(abstracts, chunksize=100)

        # Compose dataframe with lemmatized projects
        lemmatizedRes = list(zip(refs, titles_lang, titles_lemmas,
                                 abstracts_lang, abstracts_lemmas))
        df_lemmas = pd.DataFrame(lemmatizedRes, columns=[
            'Referencia', 'Titulo_lang', 'Titulo_lemas', 'Resumen_lang',
            'Resumen_lemas'])

        # IMPORTANT: Language filtering
        #            Only projedct with the abstract in SPANISH are classified
        df_lemmas = df_lemmas.loc[df_lemmas['Resumen_lang'] == 'es']

        Nproy_ES = df_lemmas.count()[0]
        rate_filtered = (Nproy - Nproy_ES) / Nproy * 100.0
        print(('     ---- {} projects removed after language filtering'
               ).format(Nproy - Nproy_ES))
        print('          ({} %)'.format(str(rate_filtered)[0:3]))

        # #########################
        # 3. BOW computations

        # Defining the relative subfolder structure of the project
        subfolders = self.f_struct_cls
        # This is just to abbreviate the variable name
        p2p = self.path2project

        # Computing BOW
        print('---- (6/{}) Preprocessing (computing BoW)'.format(n))
        classifier_types = ['LR']
        CO = classifierOptimizer(p2p, classifier_types, subfolders,
                                 min_df=self.min_df, verbose=False)
        # DF must contain the lemmatized columns
        refs, docs = CO.obtain_docs_from_df(df_lemmas)

        # #######################
        # 4. Classification setup

        print('---- (7/{}) Setting classifiers up'.format(n))

        # Compute thresholds
        EV = Evaluator(p2p, subfolders, verbose=False)
        predict_new_document_from_export = 0
        if predict_new_document_from_export == 1:
            final_threshold = EV.load_multilabel_threshold(
                path2export='./export/')
        else:
            final_threshold = EV.load_multilabel_threshold()

        # #################
        # 5. Classification

        print('---- (8/{}) Computing classifier predictions'.format(n))
        P = Predictor(p2p, subfolders, verbose=False)
        p2export = os.path.join(p2p, self.f_struct_cls['export'])
        multilabel_weights, labels_pred, raw_preds = P.predict_new_document(
            docs, option='maximum_response', final_threshold=final_threshold,
            path2export=p2export, verbose=True)

        # refs contains the project references, labels_pred contains the
        # predicted labels the prediction weights can be obtained as
        # multilabel_weights[k][label][p]

        refs_preds = list(zip(refs, labels_pred))
        df_labelpreds = pd.DataFrame(refs_preds, columns=[
            'Referencia', 'Clasificacion'])

        refs_preds_raw = list(zip(refs, raw_preds))
        df_labelpreds_raw = pd.DataFrame(refs_preds_raw, columns=[
            'Referencia', 'Clasificacion'])

        return df_labelpreds, df_labelpreds_raw

    def computeClassifierPredictions(self, df0):

        # Number of steps in this process. This is for print purposes only
        n = 8

        # #######################################
        # 1. Import project data from source file

        print('---- (1/{}) Reading data'.format(n))

        Nproy = df0.count()[0]
        print('---- ---- {} projects loaded.'.format(Nproy))

        # ################
        # 2. Lemmatization

        # Initialize lemmatizer object
        languages = self.cf.get('PREPROC', 'languages')
        lemmatizer_tool = self.cf.get('PREPROC', 'lemmatizer_tool')
        hunspelldic = self.cf.get('PREPROC', 'hunspelldic')
        stw_file = self.cf.get('PREPROC', 'stw_file')
        ngram_file = self.cf.get('PREPROC', 'ngram_file')
        dict_eq_file = self.cf.get('PREPROC', 'dict_eq_file')
        tilde_dictio = self.cf.get('PREPROC', 'tilde_dictio')

        lm = Lemmatizer(languages, lemmatizer_tool, hunspelldic, stw_file,
                        ngram_file, dict_eq_file, tilde_dictio)

        # Read all relevant data in a single dataframe
        refs = df0['Referencia'].values
        abstracts = df0['Resumen'].values
        titles = df0['Título'].values

        # Lemmatize titles
        print('---- (2/{}) Detecting language from titles'.format(n))
        titles_lang = lm.langDetection(titles)
        print('---- (3/{}) Lemmatizing titles'.format(n))
        titles_lemmas = lm.lemmatizeES(titles, chunksize=100)

        # Processing abstracts
        print('---- (4/{}) Detecting language from abstracts'.format(n))
        abstracts_lang = lm.langDetection(abstracts)
        print('---- (5/{}) Lemmatizing abstracts'.format(n))
        abstracts_lemmas = lm.lemmatizeES(abstracts, chunksize=100)

        # Compose dataframe with lemmatized projects
        lemmatizedRes = list(zip(refs, titles_lang, titles_lemmas,
                                 abstracts_lang, abstracts_lemmas))
        df_lemmas = pd.DataFrame(lemmatizedRes, columns=[
            'Referencia', 'Titulo_lang', 'Titulo_lemas', 'Resumen_lang',
            'Resumen_lemas'])

        # Check how many projects are not in Spanish
        df_lemmas_es = df_lemmas.loc[df_lemmas['Resumen_lang'] == 'es']

        Nproy_ES = df_lemmas_es.count()[0]
        rate_filtered = (Nproy - Nproy_ES) / Nproy * 100.0
        print(f'     ---- {Nproy - Nproy_ES} projects not in Spanish')
        print(f'          ({str(rate_filtered)[0:3]} %)')

        # ###################
        # 3. BOW computations

        # Defining the relative subfolder structure of the project
        subfolders = self.f_struct_cls
        # This is just to abbreviate the variable name
        p2p = self.path2project

        # Computing BOW
        print('---- (6/{}) Preprocessing (computing BoW)'.format(n))
        classifier_types = ['LR']
        CO = classifierOptimizer(p2p, classifier_types, subfolders,
                                 min_df=self.min_df, verbose=False,
                                 title_mul=self.title_mul, copycf=False)
        # DF must contain the lemmatized columns
        refs, docs = CO.obtain_docs_from_df(df_lemmas)

        # #######################
        # 4. Classification setup

        print('---- (7/{}) Setting classifiers up'.format(n))

        # Compute thresholds
        EV = Evaluator(p2p, subfolders, verbose=False)
        predict_new_document_from_export = 0
        if predict_new_document_from_export == 1:
            final_threshold = EV.load_multilabel_threshold(
                path2export='./export/')
        else:
            final_threshold = EV.load_multilabel_threshold()

        # #################
        # 5. Classification

        print('---- (8/{}) Computing classifier predictions'.format(n))
        P = Predictor(p2p, subfolders, verbose=False)
        p2export = os.path.join(p2p, self.f_struct_cls['export'])
        multilabel_weights, labels_pred, raw_preds = P.predict_new_document(
            docs, option='maximum_response', final_threshold=final_threshold,
            path2export=p2export, verbose=True)

        # refs contains the project references, labels_pred contains the
        # predicted labels the prediction weights can be obtained as
        # multilabel_weights[k][label][p]

        # Language filtering: remove classification for data in other
        # languages
        lan = df_lemmas['Resumen_lang']
        for i, l in enumerate(lan):
            if l != 'es':
                labels_pred[i] = ['']
                raw_preds[i] = [(x[0], 0) for x in raw_preds[i]]

        refs_preds = list(zip(refs, labels_pred))
        df_labelpreds = pd.DataFrame(refs_preds, columns=[
            'Referencia', 'Clasificacion'])

        refs_preds_raw = list(zip(refs, raw_preds))
        df_labelpreds_raw = pd.DataFrame(refs_preds_raw, columns=[
            'Referencia', 'Clasificacion'])

        return df_labelpreds, df_labelpreds_raw
