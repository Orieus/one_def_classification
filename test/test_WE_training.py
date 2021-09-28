# -*- coding: utf-8 -*-
'''

Author:  Angel Navia Vázquez
April 2018

pip3 install --user mysqlclient

python3 test_WE_training.py

Gensim functions
we.model.wv.most_similar(positive=['woman', 'king'], negative=['man'])
we.model.wv.most_similar_cosmul(positive=['woman', 'king'], negative=['man'])
we.model.wv.doesnt_match("breakfast cereal dinner lunch".split())
we.model.wv.similarity('woman', 'man')
we.model.score(["The fox jumped over a lazy dog".split()])
we.model.wv.evaluate_word_pairs(os.path.join(module_path, 'test_data','wordsim353.tsv'))
we.model.wv.accuracy(os.path.join(module_path, 'test_data', 'questions-words.txt'))


'''
#from fordclassifier.corpusclassproject.corpusclassproject import (
#    CorpusClassProject)

import pickle
import os
import configparser
import time
import numpy as np

from fordclassifier.corpusclassproject.datamanager import DataManager
from fordclassifier.corpusanalyzer.wordembed import WordEmbedding as WE
#from fordclassifier.corpusanalyzer.textproc import Lemmatizer
from fordclassifier.classifier.unsclassifier import UnsupClassifier

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

# Creating subfolder structure
for subfolder in list(subfolders.keys()):
    folder = os.path.join(project_path, subfolders[subfolder])
    if not os.path.exists(folder):
        os.makedirs(folder)
filename = os.path.join(project_path, 'subfolders.pkl')
with open(filename, 'wb') as f:
    pickle.dump(subfolders, f)

# Data Manager
load_from_SQL = 0
obtain_predicts = 0

filename_DF = os.path.join(project_path + subfolders['training_data'], 'DF_MINECO_lemas_14_6_2018.pkl')   
filename_DF2 = os.path.join(project_path + subfolders['training_data'], 'DF_MINECO_taxonomy_14_6_2018.pkl')   

# Training a WE model
train_we = 0
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
print(df.columns)

titulos = list(df.loc[:, 'Titulo'])
resumenes = list(df.loc[:, 'Resumen'])
titulo_lemas = list(df.loc[:, 'Titulo_lemas'])
resumen_lemas = list(df.loc[:, 'Resumen_lemas'])
tags = list(df.loc[:, 'Unesco2Ford'])

if train_we == 1:
    we.obtain_sentences(df)
    print('Training WE with %d epochs...' % nepochs)
    we.train()  # Al entrenar ya guarda wedict y WEmatrices
    print('Fin training WE')

## cargando definiciones de categorías

filename = os.path.join(project_path + subfolders['training_data'], 'categories.pkl')
with open(filename, 'rb') as f:
    categories = pickle.load(f)

UC = UnsupClassifier(project_path, subfolders, verbose=True)
#defs_dict = UC.get_categories_definitions(categories, df_taxonomy)

docs = []
for kproy in range(0, len(titulo_lemas)):
    text = titulo_lemas[kproy] + ' ' + resumen_lemas[kproy]
    text = text.replace('\n', ' ').lower()
    docs.append(text)

sigma = 1.0
#print('Computing predictions...')
#preds = UC.obtain_predicts(docs, we, sigma)

threshold = 0.8
labels_preds = UC.obtain_predicted_labels(threshold, verbose=True)

# Convertimos los tags en lista de strings
new_tags = [t.split(',') for t in tags]

# quitamos las supercategorías, acaban en _
tags = []
for l in new_tags:
    newlist = []
    for tag in l:
            if len(tag) > 0:
                if tag[-1] != '_':
                    newlist.append(tag)

    newlist = newlist
    tags.append(newlist)

# remove duplicates in the list
from collections import OrderedDict
tags_without_duplicates = []
for t in tags:
    unique_list = list(OrderedDict((element, None) for element in t))
    tags_without_duplicates.append(unique_list)

# retain the first tag
tags_1 = [t[0] if len(t) > 0 else '' for t in tags_without_duplicates]

labels_preds_1 = [t[0] if len(t) > 0 else '' for t in labels_preds]

from sklearn.metrics import confusion_matrix
CONF = confusion_matrix(tags_1, labels_preds_1, labels=categories)

import matplotlib.pyplot as plt
plt.figure(figsize=(15, 12))
cmap = plt.cm.Blues
plt.imshow(CONF, interpolation='nearest', cmap=cmap)
plt.colorbar()
tick_marks = np.arange(len(categories))
plt.xticks(tick_marks, categories, rotation=90)
plt.yticks(tick_marks, categories)
plt.tight_layout()
plt.xlabel('True label')
plt.ylabel('Predicted label')
filename = os.path.join(project_path + subfolders['figures'], 'CONF_unsup.png')
plt.savefig(filename)
plt.close()

print('End!')

import code
code.interact(local=locals())



'''
cual_ref = 7872
ref_text = resumen_lemas[cual_ref].replace('\n', ' ').lower().split(' ')
tit = titulos[cual_ref]
body = resumenes[cual_ref]
tokens = (titulo_lemas[cual_ref] + ' ' + resumen_lemas[cual_ref]).replace('\n', ' ').lower().split(' ')
'''
import code
code.interact(local=locals())


'''
config_fname = 'config.cf'
path2config = os.path.join(project_path, config_fname)
cf = configparser.ConfigParser()
cf.read(path2config)

defs_dict = {}
LEM = Lemmatizer(cf)
for cat in categories:
    aux = df_categories.loc[df_categories['Reference'] == cat]
    definition = aux['Definition'].iloc[0]
    definition = definition.replace('\n', ' ').replace('+', ' ')
    tokens = definition.split(' ')
    tokens = [t for t in tokens if len(t) > 0]
    cadena = ' '.join(tokens)
    LEM.processESstr(text=cadena)

'''

'''
#we.most_similar(['codo'], [])
similar = we.most_similar(['lechuga'], [], 20)
similar_cosmul = we.most_similar_cosmul(['lechuga'], [], 20)

cercanas = we.closest_words('lechuga', 15)
#cercanas_cosine = we.closest_words_cosine('lechuga', 15)


tokens1 = ['ley', 'lechuga', 'perro', 'pimiento']
tokens2 = ['juicio', 'casa', 'pepino', 'libro', 'brazo', 'colgante', 'collar', 'amapola', 'repollo', 'ley']



tokens_ref = ['pepino', 'lechuga', 'control']

tokens2 = ['control', 'pepino', 'remolacha', 'repollo', 'lechuga']
tokens3 = ['ley', 'juicio', 'sentencia', 'demanda', 'lechuga']
sigma = 0.1

K2, D, words1, words2 = we.tokens_dist(tokens_ref, tokens2, sigma)
K3, D, words1, words2 = we.tokens_dist(tokens_ref, tokens3, sigma)

v2 = np.sum(np.max(K2, axis=0)) + np.sum(np.max(K2, axis=1))
v3 = np.sum(np.max(K3, axis=0)) + np.sum(np.max(K3, axis=1))
'''

#import code
#code.interact(local=locals())



'''
tokens1 = titulo_lemas[1232].replace('\n', ' ').lower().split(' ')
tokens2 = titulo_lemas[3232].replace('\n', ' ').lower().split(' ')

tokens1 = resumen_lemas[1232].replace('\n', ' ').lower().split(' ')
tokens2 = resumen_lemas[3232].replace('\n', ' ').lower().split(' ')
'''

#time_ini = time.time()
#D, dmin = we.tokens_dist(tokens1, tokens2)
#print(time.time() - time_ini)
'''
array([[356.94458008,  52.71513367, 538.86187744, 373.36904907],
       [382.5440979 , 235.993927  , 690.07458496, 397.22836304],
       [438.12530518,  84.42300415, 618.95172119, 453.31921387]])

'''
#time_ini = time.time()
#D2, dmin1, dmin2, words1, words2 = we.tokens_dist2(tokens1, tokens2)
#print(time.time() - time_ini)
'''
cual_ref = 67832
ref_text = resumen_lemas[cual_ref].replace('\n', ' ').lower().split(' ')
titulos[cual_ref]
tokens1 = resumen_lemas[cual_ref].replace('\n', ' ').lower().split(' ')

proys = []
best_max = 0
best_K = 0
best_k = 0
sigma = 0.1
sigma = 1.0
best_tokens2 = []

for k in range(0, len(resumen_lemas)):
    try:
        tokens2 = resumen_lemas[k].replace('\n', ' ').lower().split(' ')
        if len(tokens2) > 5 and k != cual_ref:
            K, D, words1, words2 = we.tokens_dist(tokens1, tokens2, sigma)
            if len(words2) > 0:
                v1 = np.max(np.round(K), axis=0)
                v2 = np.max(np.round(K), axis=1)
                v = np.sum(v1) + np.sum(v2)
                if v > best_max:
                    best_max = v
                    best_K = K
                    best_k = k
                    best_tokens2 = tokens2
            else:
                v = 0
        else:
            v = 0
        proys.append(v)
    except:
        print("Error 1")
        import code
        code.interact(local=locals())
        pass
    #if k == 2246:
    #    import code
    #    code.interact(local=locals())
    #if np.max(K) > 0.99:
    #    import code
    #    code.interact(local=locals())

index = np.argsort(-1 * np.array(proys))

print('=' * 50)
print('Titulo de referencia:')
titulos[cual_ref]
print('=' * 50)
print('=' * 50)
print('10 Titulos mas cercanos:')
for k in range(0, 10):
    titulos[index[k]]
    print('-' * 50)

print('=' * 50)


print('Done!')

import code
code.interact(local=locals())
titulos[cual_ref]

if True:
    we = WE(project_path, dim, subfolders)
    we.obtain_sentences(df)
    print('Training WE with %d epochs...' % nepochs)
    we.train(nepochs)
    #we.obtain_wedict()
    # Saving the we dict
    #we.save_wedict('wedict_dim_' + str(dim) + '_nepochs_' + str(nepochs) + '.pkl')

    # Saving the we object for future use
    with open(filename, 'wb') as f:
        pickle.dump(we, f)
    print('Fin training WE')
else:
    print('Loading WE model')
    with open(filename, 'rb') as f:
        we = pickle.load(f)
'''