# -*- coding: utf-8 -*-
'''

Author:  Angel Navia Vázquez
April 2018

pip3 install --user mysqlclient

python3 test_classifiers.py

https://github.com/dlukes/rbo
http://delivery.acm.org/10.1145/1860000/1852106/a20-webber.pdf?ip=163.117.145.175&id=1852106&acc=ACTIVE%20SERVICE&key=DD1EC5BCF38B3699%2EAFCE2F3122C4D47C%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1528398754_1a4a6ed9718652d53e2d553dbdec3aec

\rm config_unix.cf
sed -e 's/\r//' config.cf > config_unix.cf
chmod 755 robotscheck_unix.py

# Ejecucion en windows
python mainFORDclassifier.py --load_p X:\navia\mineco\2018_jun_FORD_improved_labels\

python mainFORDclassifier.py --load_p X:\navia\mineco\2018_jul_final_test2\

python mainFORDclassifier.py --load_p X:\navia\mineco\2018_jul_seed_missing_categories\

'''

from fordclassifier.evaluator.evaluatorClass import Evaluator
from fordclassifier.evaluator.classifier_optimizer import classifierOptimizer
from fordclassifier.evaluator.predictorClass import Predictor
from fordclassifier.corpusclassproject.datamanager import DataManager
from fordclassifier.corpusanalyzer.wordembed import WordEmbedding as WE

import pickle
import os
import configparser
import numpy as np

from fordclassifier.evaluator.rbo import *
import matplotlib.pyplot as plt

# Path to the project
# Unesco
#month_name = '2018_may'
# Ford
month_name = '2018_jun_FORD'
month_name = '2018_jun_FORD_improved_labels'
month_name = '2018_jul_final_test'
month_name = '2018_jul_seed_missing_categories'


project_path = '/export/g2pi/navia/mineco/' + month_name + '/'
#project_path = 'X:/navia/mineco/' + month_name + '/'

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
              'bow': 'models/bow/',
              'export': 'classifiers/export/',
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



p = 0.9   # This parameter should also be selected by crossvalidation...
alpha = 0.9  # This parameter balances between Jackard and RBO, alpha = 1.0, only Jackard
option = 'maximum_response'
th_values = [0.999, 0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0, -0.05, -0.1, -0.15]
#final_threshold = 0.7
seed_missing_categories = True
task = 'FORD'

# Data Manager
load_from_SQL = 0

# Optimizer
prepare_training_data = 0
create_partitions = 0
create_Nfold_partition = 0

# classifier_optimizer
cross_validate_models = 0

# predict
predict = 0

# evaluator
compute_multilabel_costs = 0
draw_costs_on_test = 0

order_sensitive = True
compute_confusion_matrix = 1
compute_average_AUC_test = 0
predict_new_document = 0

predict_new_document_from_export = 0

classifier_types = ['LR', 'MNB', 'BNB', 'LSVM', 'RF', 'SVMpoly', 'MLP']
classifier_types = ['LR', 'MNB', 'BNB', 'LSVM', 'RF', 'SVMpoly', 'MLP']
classifier_types = ['LR', 'MNB', 'BNB', 'LSVM', 'RF', 'SVMpoly', 'MLP']

classifier_types = ['LR', 'MNB', 'BNB']
classifier_types = ['LR']

# We can use a different subset of classifiers to find the best, for instance classifiers2compare = ['LR', 'MNB']
# finds the best classifiers among the options LR and MNB, even when other models are available
#classifiers2compare = ['LR', 'MNB', 'BNB']
classifiers2compare = classifier_types

# Initializing objects
EV = Evaluator(project_path, subfolders, verbose=False)
CO = classifierOptimizer(project_path, classifier_types, subfolders, verbose=False)
P = Predictor(project_path, subfolders, verbose=False)


#filename = os.path.join(project_path + subfolders['training_data'], 'DF_MINECO_lemas_5_6_2018.pkl')
#filename = os.path.join(project_path + subfolders['training_data'], 'DF_MINECO_lemas_11_6_2018.pkl')   
filename = os.path.join(project_path + subfolders['training_data'], 'DF_MINECO_lemas_17_7_2018.pkl')   
if load_from_SQL == 1:   # To avoid opening the SQL database
    # Extracting data from the database, and save it in  clean_corpus, the Datamanager will replace this

    config_fname = 'config.cf'
    #config_fname = 'config_prueba.cf'   ### BBDD de pruebas
    #config_fname = 'config_unix.cf'   ### minimos datos
    path2config = os.path.join(project_path, config_fname)
    cf = configparser.ConfigParser()
    cf.read(path2config)

    # Trying to connect to the database
    DM = DataManager(project_path, cf)
    df = DM.readDBtable('proyectos')

    with open(filename, 'wb') as f:
        pickle.dump(df, f)
else:    # load
    print('Loading dataframe...')
    with open(filename, 'rb') as f:
        df = pickle.load(f)

print(df.columns)

Nproy = df.count()[0]
print('Total de proyectos = ', Nproy)
# filtrado idioma
df = df.loc[df['Resumen_lang'] == 'es']
Nproy_tras_filtro = df.count()[0]

porcentaje_filtrado = (Nproy - Nproy_tras_filtro) / Nproy * 100.0
print('Total de proyectos eliminados tras filtrar por idioma = ', Nproy - Nproy_tras_filtro, ', un ', str(porcentaje_filtrado)[0:3] , '%' )

if prepare_training_data == 1:
    print('Preparing training data')
    CO.prepare_training_data(df, task, seed_missing_categories=seed_missing_categories, verbose=True)

if create_partitions == 1:
    fraction_train = 0.8
    #CO.create_partition_train_test(fraction_train, verbose=True)
    # Load the coordinated projects list
    #coordinated_projects = []

    # Hasta saber cómo se almacena esta información, lo cargamos aquí y lo pasamos como parámetro
    filename = os.path.join(project_path + subfolders['tmp'], 'coord.pkl')
    with open(filename, 'rb') as f:
        coordinated_projects = pickle.load(f)
    CO.create_partition_train_test_MINECO(fraction_train, coordinated_projects, verbose=True)

if create_Nfold_partition == 1:
    Nfold = 10
    CO.create_Nfold_partition(Nfold, verbose=True)

if cross_validate_models == 1:
    print('Cross-validating models')
    CO.xval_models(verbose=True)
    CO.find_best_models(classifiers2compare, verbose=True)
    CO.compute_thresholds()
    EV.draw_rocs(verbose=True)

if predict == 1:
    Xtfidf_tst, tags_tst, refs_tst = EV.load_test_data()
    Xtfidf_tr, tags_tr, refs_tr = EV.load_train_data()

    print('Predicting test...')
    Preds_test, tag_pred_test = P.predict(Xtfidf_tst, verbose=True)

    print('Predicting train...')
    Preds_tr, tag_pred_tr = P.predict(Xtfidf_tr, verbose=True)

    # grabamos fuera para que nos valga el predict para cualquier dato
    filetosave = os.path.join(project_path + subfolders['results'], 'Preds_test.pkl')
    with open(filetosave, 'wb') as f:
        pickle.dump(Preds_test, f)
    filetosave = os.path.join(project_path + subfolders['results'], 'tags_pred_test.pkl')
    with open(filetosave, 'wb') as f:
        pickle.dump(tag_pred_test, f)
    filetosave = os.path.join(project_path + subfolders['results'], 'Preds_tr.pkl')
    with open(filetosave, 'wb') as f:
        pickle.dump(Preds_tr, f)
    filetosave = os.path.join(project_path + subfolders['results'], 'tags_pred_tr.pkl')
    with open(filetosave, 'wb') as f:
        pickle.dump(tag_pred_tr, f)

if compute_multilabel_costs == 1:

    # To compute the final threshold
    final_threshold = EV.compute_multilabel_threshold(p, alpha, option, th_values)

    # To load a precomputed value
    print('Loading threshold value...')
    final_threshold = EV.load_multilabel_threshold()

    Xtfidf_tst, tags_tst, refs_tst = EV.load_test_data()

    #EV.draw_costs_on_test(p, alpha, option, th_values, normalize_tags)

    # Final predictions
    filename = os.path.join(project_path + subfolders['results'], 'Preds_tr.pkl')
    with open(filename, 'rb') as f:
        Preds_tr = pickle.load(f)
    multilabel_pred_tr, labels_pred_tr = P.obtain_multilabel_preds(Preds_tr, option, final_threshold, verbose=True)

    filename = os.path.join(project_path + subfolders['results'], 'Preds_test.pkl')
    with open(filename, 'rb') as f:
        Preds_tst = pickle.load(f)
    multilabel_pred_tst, labels_pred_tst = P.obtain_multilabel_preds(Preds_tst, option, final_threshold, verbose=True)

    filename_out = 'multilabel_prediction_test.txt'
    EV.write_prediction_report(refs_tst, tags_tst, labels_pred_tst, multilabel_pred_tst, filename_out)

if draw_costs_on_test:
    EV.draw_costs_on_test(p, alpha, option, th_values, verbose=True)

# Computing the confussion matrix, with the main predicted category and first tag
if compute_confusion_matrix == 1:

    final_threshold = EV.load_multilabel_threshold()

    print('Computing confusion matrix on test data...')
    Xtfidf_tst, tags_tst, refs_tst = EV.load_test_data()
    filename = os.path.join(project_path + subfolders['results'], 'Preds_test.pkl')
    with open(filename, 'rb') as f:
        Preds_tst = pickle.load(f)

    multilabel_pred_tst, labels_test = P.obtain_multilabel_preds(Preds_tst, option, final_threshold, verbose=True)

    '''
    # retain the first tag in the labels
    tags_tst_1 = [t[0] if len(t) > 0 else '' for t in tags_tst]
    labels_test_1 = [t[0] if len(t) > 0 else '' for t in labels_test]
    CONF_tst = EV.compute_confusion_matrix(tags_tst_1, labels_test_1, 'CONF_tst.pkl', sorted_categories)   
    EV.draw_confusion_matrix(CONF_tst, 'CONF_tst.png', sorted_categories)

    CONF_tst_multilabel = EV.compute_confusion_matrix_multilabel(tags_tst, labels_test, 'CONF_tst_multilabel.pkl', sorted_categories)   
    EV.draw_confusion_matrix(CONF_tst_multilabel, 'CONF_tst_multilabel.png', sorted_categories)
    '''
    CONF_tst_multilabel_v2 = EV.compute_confusion_matrix_multilabel_v2(tags_tst, labels_test, 'CONF_tst_multilabel.pkl', sorted_categories=sorted_categories, order_sensitive = order_sensitive)   
    EV.draw_confusion_matrix(CONF_tst_multilabel_v2, 'CONF_tst_multilabel.png', sorted_categories)

    print('Computing confusion matrix on train data...')
    Xtfidf_tr, tags_tr, refs_tr = EV.load_train_data()
    filename = os.path.join(project_path + subfolders['results'], 'Preds_tr.pkl')
    with open(filename, 'rb') as f:
        Preds_tr = pickle.load(f)

    multilabel_pred_tr, labels_tr = P.obtain_multilabel_preds(Preds_tr, option, final_threshold, verbose=True)

    '''
    # retain the first tag in the labels
    tags_tr_1 = [t[0] if len(t) > 0 else '' for t in tags_tr]
    labels_tr_1 = [t[0] if len(t) > 0 else '' for t in labels_tr]
    CONF_tr = EV.compute_confusion_matrix(tags_tr_1, labels_tr_1, 'CONF_tr.pkl', sorted_categories)
    EV.draw_confusion_matrix(CONF_tr, 'CONF_tr.png', sorted_categories)

    CONF_tr_multilabel = EV.compute_confusion_matrix_multilabel(tags_tr, labels_tr, 'CONF_tr_multilabel.pkl', sorted_categories)   
    EV.draw_confusion_matrix(CONF_tr_multilabel, 'CONF_tr_multilabel.png', sorted_categories)
    '''

    CONF_tr_multilabel_v2 = EV.compute_confusion_matrix_multilabel_v2(tags_tr, labels_tr, 'CONF_tr_multilabel.pkl', sorted_categories=sorted_categories, order_sensitive = order_sensitive)   
    EV.draw_confusion_matrix(CONF_tr_multilabel_v2, 'CONF_tr_multilabel.png', sorted_categories)

# Compute the average AUC on the test set and draw the test ROCs
if compute_average_AUC_test == 1:
    Xtfidf_tst, tags_tst, refs_tst = EV.load_test_data()
    filename = os.path.join(project_path + subfolders['results'], 'Preds_test.pkl')
    with open(filename, 'rb') as f:
        Preds_test = pickle.load(f)

    EV.draw_ROCS_tst(Preds_test, tags_tst)
    average_auc_tst = EV.compute_average_test_AUC()
    print("The average AUC on the test set is", average_auc_tst)

if predict_new_document == 1:
    print('Loading threshold value...')
    final_threshold = EV.load_multilabel_threshold()

    # DF must contain the lemmatized columns
    refs, docs = CO.obtain_docs_from_df(df)

    multilabel_weights, labels_pred = P.predict_new_document(docs, option='maximum_response', normalize_tags=True, final_threshold=final_threshold, verbose=True)
    # refs contains the project references, labels_pred contains the predicted labels
    # the prediction weights can be obtained as multilabel_weights[k][label][p]

    '''
    tags = list(df.loc[:, 'Unesco2Ford'])
    print('Sample tags vs. predictions:')
    for kdoc in range(0, 50):
        print('Orig tags = ', tags[kdoc], '\t', 'Predicted tags = ', labels_pred[kdoc])
    '''

if predict_new_document_from_export == 1:
    print('Loading threshold value...')
    final_threshold = EV.load_multilabel_threshold(path2export='./export/')

    # DF must contain the lemmatized columns
    refs, docs = CO.obtain_docs_from_df(df)

    multilabel_weights, labels_pred = P.predict_new_document(docs, option='maximum_response', final_threshold=final_threshold, path2export='./export/', verbose=True)

    tags = list(df.loc[:, 'Unesco2Ford'])
    print('Sample tags vs. predictions:')
    for kdoc in range(0, 50):
        print('Orig tags = ', tags[kdoc], '\t', 'Predicted tags = ', labels_pred[kdoc])

print('=' * 50)
print('Done!')
print('=' * 50)

'''
titulos = list(df.loc[:, 'Titulo'])
refs = list(df.loc[:, 'Referencia'])
resumenes = list(df.loc[:, 'Resumen'])

titulos = [t.lower() for t in titulos]
resumenes = [t.lower() for t in resumenes]

N_BioTecAmb = 0
N_Nanotec = 0
N_BiotecSalud = 0
N_BiotecAgr = 0

dict_BioTecAmb = {}
dict_Nanotec = {}
dict_BiotecSalud = {}
dict_BiotecAgr = {}


for k in range(0, len(titulos)):
    texto = titulos[k] + ' ' + resumenes[k]
    if 'biotecnologia' in texto or 'biotecnologica' in texto or 'biotecnologico' in texto:
        if 'ambiental' in texto or 'medio ambiente' in texto:
            N_BioTecAmb += 1
            dict_BioTecAmb.update({refs[k]: titulos[k]})
        if 'salud' in texto:
            N_BiotecSalud += 1
            dict_BiotecSalud.update({refs[k]: titulos[k]})
        if 'agricola' in texto or 'agricultura' in texto:
            N_BiotecAgr += 1
            dict_BiotecAgr.update({refs[k]: titulos[k]})
        if 'nanotecnologia' in texto:
            N_Nanotec += 1
            dict_Nanotec.update({refs[k]: titulos[k]})

print('N_BioTecAmb', N_BioTecAmb)
print('N_Nanotec', N_Nanotec)
print('N_BiotecSalud', N_BiotecSalud)
print('N_BiotecAgr', N_BiotecAgr)

data = []

data.append('=======================================================================================\r\n')
data.append('=============================    BioTecAmb    ===========================\r\n')
data.append('=======================================================================================\r\n')
for key in list(dict_BioTecAmb.keys()):
    data.append(str(key) + ': ' + dict_BioTecAmb[key] + '\r\n')
data.append('=======================================================================================\r\n')
data.append('=============================    Nanotec    ===========================\r\n')
data.append('=======================================================================================\r\n')
for key in list(dict_Nanotec.keys()):
    data.append(str(key) + ': ' + dict_Nanotec[key] + '\r\n')
data.append('=======================================================================================\r\n')
data.append('=============================    BiotecSalud    ===========================\r\n')
data.append('=======================================================================================\r\n')
for key in list(dict_BiotecSalud.keys()):
    data.append(str(key) + ': ' + dict_BiotecSalud[key] + '\r\n')
data.append('=======================================================================================\r\n')
data.append('=============================    BiotecAgr    ===========================\r\n')
data.append('=======================================================================================\r\n')
for key in list(dict_BiotecAgr.keys()):
    data.append(str(key) + ': ' + dict_BiotecAgr[key] + '\r\n')


import codecs
filename = 'missing_cats.txt'
file = codecs.open(filename, "w", "utf-8")
for line in data:
    file.write(line)

file.close()

with open('missing_cats.pkl', 'wb') as f:
    pickle.dump([dict_BioTecAmb, dict_Nanotec, dict_BiotecSalud, dict_BiotecAgr], f)


print('Saved ')
'''

'''
N_BioTecAmb 168
N_Nanotec 77
N_BiotecSalud 157
N_BiotecAgr 106


'''


#import code
#code.interact(local=locals())



