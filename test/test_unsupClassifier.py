# -*- coding: utf-8 -*-
'''

Author:  Angel Navia Vázquez
July 2018

pip3 install --user mysqlclient

python3 test_unsupClassifier.py

'''
#from fordclassifier.corpusclassproject.corpusclassproject import (
#    CorpusClassProject)

import pickle
import os
import configparser
import numpy as np

from fordclassifier.corpusclassproject.datamanager import DataManager
from fordclassifier.corpusanalyzer.wordembed import WordEmbedding as WE
from fordclassifier.classifier.unsclassifier import UnsupClassifier
from fordclassifier.evaluator.evaluatorClass import Evaluator
from fordclassifier.evaluator.predictorClass import Predictor

# Path to the project
month_name = '2018_may'
month_name = '2018_jun_FORD_improved_labels'

project_path = '/export/g2pi/navia/mineco/' + month_name + '/'
#project_path = 'X:/navia/mineco/' + month_name + '/'

# Defining the relative subfolder structure of the project
# Defining the relative subfolder structure of the project
subfolders = {'training_data': 'classifiers/training_data/',
        'test_data': 'classifiers/test_data/',
              'models': 'classifiers/models/',
              'results': 'classifiers/results/',
              'figures': 'classifiers/figures/',
              'eval_ROCs': 'classifiers/eval_ROCs/',
              'ROCS_tr': 'classifiers/figures/ROCS_tr/',
              'ROCS_tst': 'classifiers/figures/ROCS_tst/',
              'tmp': 'classifiers/tmp/',
              'we': 'models/we/',
              'bow': 'models/bow/'
              }

sorted_categories = ['Mat', 'Comp', 'Fis', 'Quim', 'Tierra', 'Bio', 'OtherCN',
             'ICivil', 'IEE_Inf', 'IMec', 'IQuim', 'IMater', 'IMed', 
                        'IAmb', 'BioTecAmb', 'BioTecInd', 'NanoTec', 'OtherIT',
             'MBasica', 'MClinic', 'CSalud', 'BioTecSalud',
                       'OtherMedCS',
             'AgSilvPesc', 'CAnimLecher', 'Veterin', 'BioTecAgr',
                          'OtherAgVet',
             'Psico', 'EconNeg', 'CEduc', 'Sociolog', 'Derecho',
                      'Polit', 'GeogrSocEcon', 'PeriodCom', 'OtherCSoc',
             'HistorArqu', 'IdiomLiterat', 'FilosEticRelig',
                          'Arte', 'OtherHArt']

# Data Manager
load_from_SQL = 0
compute_predicts_tr = 0
compute_predicts_tst = 0
compute_confusion_matrix = 1

filename_DF = os.path.join(project_path + subfolders['training_data'], 'DF_MINECO_lemas_14_6_2018.pkl')   
filename_DF2 = os.path.join(project_path + subfolders['training_data'], 'DF_MINECO_taxonomy_14_6_2018.pkl')   

filename_DF = os.path.join(project_path + subfolders['training_data'], 'DF_MINECO_lemas_4_7_2018.pkl')   
filename_DF2 = os.path.join(project_path + subfolders['training_data'], 'DF_MINECO_taxonomy_4_7_2018.pkl')   

dim = 300
nepochs = 1000
filename_WE = os.path.join(project_path + subfolders['we'], 'we_dim_' + str(dim) + '_nepochs_' + str(nepochs) + '.pkl')
we = WE(project_path, dim, nepochs, subfolders)

if load_from_SQL == 1:   # To avoid opening the SQL database
    # Extracting data from the database, and save it in  clean_corpus, the Datamanager will replace this
    config_fname = 'config.cf'
    path2config = os.path.join(project_path, config_fname)
    cf = configparser.ConfigParser()
    cf.read(path2config)
    # Trying to connect to the database
    DM = DataManager(project_path, cf)
    df = DM.readDBtable('proyectos')
    df_categories = DM.readDBtable('taxonomy')

    with open(filename_DF2, 'wb') as f:
        pickle.dump(df_categories, f)

    with open(filename_DF, 'wb') as f:
        pickle.dump(df, f)

else:    # load

    with open(filename_DF2, 'rb') as f:
        df_taxonomy = pickle.load(f)

    with open(filename_DF, 'rb') as f:
        df = pickle.load(f)
# Language filtering
df = df.loc[df['Resumen_lang'] == 'es']
print(df.columns)

#titulos = list(df.loc[:, 'Titulo'])
#resumenes = list(df.loc[:, 'Resumen'])
titulo_lemas = list(df.loc[:, 'Titulo_lemas'])
resumen_lemas = list(df.loc[:, 'Resumen_lemas'])
tags = list(df.loc[:, 'Unesco2Ford'])
# We prepare the data to classify outside the method, so it is general
docs = []
for kproy in range(0, len(titulo_lemas)):
    text = titulo_lemas[kproy] + ' ' + resumen_lemas[kproy]
    text = text.replace('\n', ' ').lower()
    docs.append(text)

'''
if train_we == 1:
    we.obtain_sentences(df)
    print('Training WE with %d epochs...' % nepochs)
    we.train()  # Al entrenar ya guarda wedict y WEmatrices
    print('Fin training WE')
'''

filename = os.path.join(project_path + subfolders['training_data'], 'categories.pkl')
with open(filename, 'rb') as f:
    categories = pickle.load(f)

UC = UnsupClassifier(
    project_path, subfolders, verbose=True)
orig_tags = UC.obtain_original_labels(tags, verbose=True)
orig_tags_1 = [t[0] if len(t) > 0 else '' for t in orig_tags]

## creando definiciones de categorías
# Execute this only when the definitions change
#   OLD   defs_dict = UC.get_categories_definitions(categories, df_taxonomy, verbose=True)
#weighted_dict = UC.get_categories_definitions_tfidf(categories, df_taxonomy, verbose=True)

# Warning, this step may take very long...
filename = 'unsup_Preds.pkl'
if compute_predicts_tr:
    sigma = 1.0
    print('Computing predictions...')
    #  OLD Preds = UC.obtain_predicts(docs, we, sigma, filename, verbose=True)
    Preds_tr = UC.obtain_predicts_wordcount(docs, we, sigma, filename, verbose=True)
else:
    Preds_tr = UC.load_predicts(filename)

EV = Evaluator(project_path, subfolders, verbose=False)
threshold = 0.9
#labels_preds_tr = UC.obtain_predicted_labels(Preds_tr, threshold, verbose=True)
labels_preds_tr = EV.obtain_labels_from_Preds(Preds_tr, threshold, verbose=True)

density_preds_tr = EV.compute_label_density(labels_preds_tr)
density_orig_tr = EV.compute_label_density(orig_tags)
print(' ')
print('Densities orig_tr and pred_tr')
print(density_orig_tr, density_preds_tr)

if compute_confusion_matrix == 1:
    # retain the first tag to compute CONF matrix
    labels_preds_1 = [t[0] if len(t) > 0 else '' for t in labels_preds_tr]
    CONF_tr_unsup = EV.compute_confusion_matrix(orig_tags_1, labels_preds_1, 'CONF_tr_unsup_tr.pkl', sorted_categories, verbose=True)
    CONF_tr_unsup = CONF_tr_unsup.astype('float') / CONF_tr_unsup.sum(axis=1)[:, np.newaxis]
    EV.draw_confusion_matrix(CONF_tr_unsup, 'CONF_tr_unsup.png', sorted_categories)

case = 'Unsup_tr'
average_auc = EV.draw_anyROC(Preds_tr, orig_tags, case)
print("The average AUC for the case " + case + " is", average_auc)

print('End!')

import code
code.interact(local=locals())

