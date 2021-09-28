# -*- coding: utf-8 -*-
'''

Author:  Angel Navia Vázquez
July 2018
'''

import pickle
import os
import configparser
import numpy as np

from fordclassifier.datamanager.datamanager import DataManager
from fordclassifier.corpusanalyzer.wordembed import WordEmbedding as WE
from fordclassifier.classifier.unsclassifier import UnsupClassifier
from fordclassifier.evaluator.evaluatorClass import Evaluator


class FCUnsupClassifier(object):
    """
    This class contains those methods used to evaluate unsupervised classifiers
    for the MINECO project (which containts abstracts of research projects in a
    particular DB structure).

    Thus, they are specific for the particular data structure used in this
    project.
    """

    def __init__(self, categories=None, cf=None, path2project=None,
                 f_struct_cls=None, task='FORD'):

        self.categories = categories
        self.cf = cf
        self.path2project = path2project
        self.f_struct_cls = f_struct_cls

        # task = 'FORD' or 'UNESCO'
        self.task = task

        return

    def evaluateUnsupClassifier(self, threshold):
        '''
        Author:  Angel Navia Vázquez
        April 2018
        '''

        # These are just abbreviations
        p2p = self.path2project    # Path to project folder
        sf = self.f_struct_cls    # Subfolders inside the project folder

        # Take level 2 categories only
        sorted_categories = [x for x in self.categories if not x.endswith('_')]

        # Data Manager
        compute_predicts_tr = 1
        # ISSUE: This is not used (?)
        # compute_predicts_tst = 1

        filename_DF = os.path.join(p2p, sf['training_data'],
                                   'DF_MINECO_lemas.pkl')
        filename_DF2 = os.path.join(p2p, sf['training_data'],
                                    'DF_MINECO_taxonomy.pkl')

        dim = 300
        nepochs = 2

        # ISSUE: Not used (?)
        # filename_WE = os.path.join(
        #     p2p, sf['we'], 'we_dim_{0}_nepochs_{1}.pkl'.format(dim, nepochs))
        we = WE(p2p, dim, nepochs, sf)

        # Extract data from the DB and save it.
        # The Datamanager will replace this.
        config_fname = 'config.cf'
        path2config = os.path.join(p2p, config_fname)
        cf = configparser.ConfigParser()
        cf.read(path2config)

        # Trying to connect to the database
        DM = DataManager(p2p, cf)
        print("---- Reading projects ...")
        df = DM.readDBtable('proyectos')

        # ISSUE: It seems that this dataframe is only used to be saved.
        print("---- Reading taxonomy ...")
        df_taxonomy = DM.readDBtable('taxonomy')

        # ISSUE: Not clear if this dataframes need to be saved
        with open(filename_DF2, 'wb') as f:
            pickle.dump(df_taxonomy, f)

        with open(filename_DF, 'wb') as f:
            pickle.dump(df, f)

        # Language filtering
        df = df.loc[df['Resumen_lang'] == 'es']
        print(df.columns)

        titulo_lemas = list(df.loc[:, 'Titulo_lemas'])
        resumen_lemas = list(df.loc[:, 'Resumen_lemas'])
        tags = list(df.loc[:, 'Unesco2Ford'])

        # We prepare the data to classify outside the method, so it is general
        docs = []
        for kproy in range(0, len(titulo_lemas)):
            text = titulo_lemas[kproy] + ' ' + resumen_lemas[kproy]
            text = text.replace('\n', ' ').lower()
            docs.append(text)

        # filename = os.path.join(p2p, sf['training_data'], 'categories.pkl')
        # with open(filename, 'rb') as f:
        #     categories = pickle.load(f)

        UC = UnsupClassifier(p2p, sf, verbose=True)
        print('Obtaining original labels...')
        orig_tags = UC.obtain_original_labels(tags, verbose=True)
        orig_tags_1 = [t[0] if len(t) > 0 else '' for t in orig_tags]

        # Creating category definitions
        # Execute this only when the definitions change
        #   OLD   defs_dict = UC.get_category_definitions(
        #             sorted_categories, df_taxonomy, verbose=True)
        weighted_dict = UC.get_category_definitions_tfidf(
            sorted_categories, df_taxonomy, verbose=True)
        # weighted_dict is not used. Its value is also stored in a file that
        # will be read by obtaon_predicts_wordcount

        # Warning, this step may take very long...
        filename = 'unssup_Preds.pkl'
        if compute_predicts_tr:
            sigma = 1.0
            print('Computing predictions...')
            # OLD Preds = UC.obtain_predicts(
            #     docs, we, sigma, filename, verbose=True)
            Preds_tr = UC.obtain_predicts_wordcount(docs, we, sigma, filename,
                                                    verbose=True)
        else:
            Preds_tr = UC.load_predicts(filename)

        EV = Evaluator(p2p, sf, categories=sorted_categories,
                       verbose=False)
        # labels_preds_tr = UC.obtain_predicted_labels(
        #     Preds_tr, threshold, verbose=True)
        labels_preds_tr = EV.obtain_labels_from_Preds(
            Preds_tr, threshold, verbose=True)

        density_preds_tr = EV.compute_label_density(labels_preds_tr)
        density_orig_tr = EV.compute_label_density(orig_tags)
        print(' ')
        print('Densities orig_tr and pred_tr')
        print(density_orig_tr, density_preds_tr)

        # Compute confusion matrix
        # Retain the first tag to compute CONF matrix
        labels_preds_1 = [t[0] if len(t) > 0 else '' for t in labels_preds_tr]
        CONF_tr_unsup = EV.compute_confusion_matrix(
            orig_tags_1, labels_preds_1, 'CONF_tr_unsup_tr.pkl',
            sorted_categories, verbose=True)
        CONF_tr_unsup = (CONF_tr_unsup.astype('float') /
                         CONF_tr_unsup.sum(axis=1)[:, np.newaxis])
        EV.draw_confusion_matrix(
            CONF_tr_unsup, 'CONF_tr_unsup.png', sorted_categories)

        # Compute ROC
        case = 'Unsup_tr'
        average_auc = EV.draw_anyROC(Preds_tr, orig_tags, case)

        print("The average AUC for the case " + case + " is", average_auc)
        print('End!')

        return

