# -*- coding: utf-8 -*-
'''

Author:  Angel Navia Vázquez
April 2018

pip3 install --user mysqlclient

python3 test_ANV.py

import code
code.interact(local=locals())

git commit --amend --author="Angel Navia Vazquez <navia@tsc.uc3m.es>" -m "Correcting author email"
git commit --amend --author="angelnaviavazquez <navia@tsc.uc3m.es>" -m "Correcting author email"

git config --global user.name "angelnaviavazquez"
git config --global user.email angel.navia@uc3m.es
git commit --amend --reset-author -m "Updating author"

git config --global credential.helper cache
git config --global credential.helper 'cache --timeout=360000000'

# Python3 environment in Windows:
conda create -n py36 python=3.6 anaconda
activate py36
deactivate

# View environments
conda info --envs

Metrics
https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html#label-based-metrics

Script to test the new objects

# to find dependencies
pip install pipreqs

'''
#from fordclassifier.corpusclassproject.corpusclassproject import (
#    CorpusClassProject)

from fordclassifier.evaluator.evaluatorClass import Evaluator
import pickle
import os
import configparser
import numpy as np

from fordclassifier.corpusclassproject.datamanager import DataManager
from fordclassifier.corpusanalyzer.wordembed import WordEmbedding as WE

month_name = '2018_may'
project_path = '/export/g2pi/navia/mineco/' + month_name + '/'
project_path = 'X:/navia/mineco/' + month_name + '/'

# Defining the subfolder structure of the project
subfolders = {'training_data': 'classifiers/training_data/',
			  'test_data': 'classifiers/test_data/',
              'models': 'classifiers/models/',
              'results': 'classifiers/results/',
              'figures': 'classifiers/figures/',
              'eval_ROCs': 'classifiers/eval_ROCs/',
              'ROCS_tr': 'classifiers/figures/ROCS_tr/',
              'ROCS_tst': 'classifiers/figures/ROCS_tst/',
              'tmp': 'classifiers/tmp/',
              'we': 'models/we/'
              }

load_SQL = 1


if load_SQL == 1:
    # Extracting data from the database, and save it in  clean_corpus, the Datamanager will replace this
    # Trying to connect to the database
    config_fname = 'config.cf'
    #config_fname = 'config_prueba.cf'   ### BBDD de pruebas
    path2config = os.path.join(project_path, config_fname)
    cf = configparser.ConfigParser()
    cf.read(path2config)

    DM = DataManager(project_path, cf)
    df = DM.readDBtable('proyectos')

    filename = os.path.join(project_path + subfolders['training_data'], 'DF_MINECO_lemas_5_6_2018.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(df, f)
else:    # load
    filename = os.path.join(project_path + subfolders['training_data'], 'DF_MINECO_lemas_5_6_2018.pkl')
    with open(filename, 'rb') as f:
        df = pickle.load(f)

    print(df.columns)
    print(df.count()[0])
    # filtrado idioma
    df = df.loc[df['Resumen_lang'] == 'es']
    print(df.count()[0])


# Evaluar y elegir el mejor clasificador
if True:
    classifier_types = ['LR', 'MNB', 'BNB', 'LSVM', 'RF', 'SVMpoly', 'MLP']
    classifier_types = ['LR', 'MNB', 'BNB', 'LSVM', 'RF', 'SVMpoly', 'MLP']
    classifier_types = ['LR', 'MNB', 'BNB', 'LSVM', 'RF', 'SVMpoly']
    #classifiers2compare = ['LR', 'MNB', 'BNB']
    classifiers2compare = classifier_types
    #classifier_types = ['SVMpoly']
    EV = Evaluator(project_path, classifier_types, subfolders, verbose=True)

if False:
    print('Computing TFIDF')
    EV.compute_Xtfidf(df, subfolders, verbose=True)

if False:
    EV.xval_models(verbose=True)
    EV.find_best_models(classifiers2compare, verbose=True)
    EV.draw_rocs(verbose=True)

# Obtener las predicciones en test
if False:
    #EV.compute_thresholds()
    filename = os.path.join(project_path + 'classifiers/test_data/', 'test_data.pkl')
    with open(filename, 'rb') as f:
        [Xtfidf_tst, tags_tst] = pickle.load(f)

    print('Predicting test...')
    Preds, tag_pred = EV.predict(Xtfidf_tst, verbose=True)

    # grabamos fuera para que nos valga el predict para cualquier dato
    filetosave = os.path.join(project_path + subfolders['results'], 'Preds.pkl')
    with open(filetosave, 'wb') as f:
        pickle.dump(Preds, f)

    filetosave = os.path.join(project_path + subfolders['results'], 'tags_pred.pkl')
    with open(filetosave, 'wb') as f:
        pickle.dump(tag_pred, f)


if False:
    filename = os.path.join(project_path + 'classifiers/training_data/', 'test_data.pkl')
    with open(filename, 'rb') as f:
        [Xtfidf_tst, tags_tst] = pickle.load(f)
    filename = os.path.join(project_path + subfolders['results'], 'tags_pred.pkl')
    with open(filename, 'rb') as f:
        tags_pred = pickle.load(f)
    CONF = EV.compute_confussion_matrix(tags_tst, tags_pred)
    # Normalize
    CONF = CONF.astype('float') / CONF.sum(axis=1)[:, np.newaxis]
    EV.draw_confussion_matrix(CONF, categories)

if False:
    EV.draw_ROCS_tst()
    average_auc_tst = EV.compute_average_test_AUC()
    print(average_auc_tst)

if False:
    EV.compute_thresholds()

if False:
    EV.predict_multilabel()


import code
code.interact(local=locals())




#EV.xval_models(categories)

#EV.find_best_models(categories, classifier_types, verbose=True)

#EV.draw_rocs(models, categories)

#average_auc = EV.compute_average_xval_AUC()
#print(average_auc)

'''
# Elegimos aleatoriamente un subconjunto de test de 1000 proyectos
N = Xtfidf.shape[0]
sel_index = np.random.permutation(N)[0:5200]
index_ok = []
tags_ok = []
for k in range(0, len(sel_index)):
	try:
		tag = str(tags[sel_index[k]])
		if len(tag) >= 4:
			if len(tag) > 4:
				tag = tag[0:4]
				index_ok.append(sel_index[k])
				tags_ok.append(tag)
	except:
		print("Error")
		import code
		code.interact(local=locals())
		pass

index_ok = index_ok[0:1000]
tags_ok = tags_ok[0:1000]

import code
code.interact(local=locals())

Xtfidf_test = Xtfidf[index_ok, :]
y_tst = tags_ok

filename = os.path.join(project_path + subfolders['tmp'], 'test_data.pkl')
with open(filename, 'wb') as f:
	pickle.dump([Xtfidf_test, y_tst], f)
'''

filename = os.path.join(project_path + subfolders['tmp'], 'test_data.pkl')
with open(filename, 'rb') as f:
	[Xtfidf_test, y_tst] = pickle.load(f)

CM = EV.compute_confussion_matrix_xval(Xtfidf_test, y_tst)



print('END!')
import code
code.interact(local=locals())

'''
simple TEST

classifier_types = ['LR', 'MNB']
EV = Evaluator(project_path, classifier_types, True)

model_type = 'LRC01'
model = Classifier(EV._models2evaluate[model_type], True)
X = np.random.rand(5,50)
y =  np.sign(np.random.rand(5,1)-0.5).ravel()
model.fit(X, y)
p = model.predict(X)
'''



if False:

    def align_strings(string0, string1, string2, string3, L, M, N, P):
        empty = '                                                                                 '
        #if len(string1) > M or len(string2) > N or len(string3) > P:
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

    # writing report for the best threshold value
    jrbos = []
    string0 = 'PROJECT REFERENCE'
    string1 = 'TARGET LABELS'
    string2 = 'PREDICTED LABELS'
    string3 = 'JRBO COST'
    data = [align_strings(string0, string1, string2, string3, 20, 30, 50, 10)]
    data.append('=======================================================================================\r\n')
    for k in range(0, len(tags_tr)):

        string0 = refs_tr[k]

        string1 = ''
        for t in tags_tr[k]:
            string1 += t + ', '

        if len(tags_tr[k]) == 0:
            string1 += '--------------'

        values = []
        for key in list(multilabel_pred_tr[k].keys()):
            values.append((key, multilabel_pred_tr[k][key]['p']))

        values.sort(key=lambda x: x[1], reverse=True)
        l_pred = []
        string2 = ''
        for v in values:
            string2 += v[0] + '(' + str(v[1])[0:5] + '), '
            l_pred.append(v[0])

        if len(values) == 0:
            string2 += '--------------'

        jrbo = Jaccard_RBO_cost(tags_tr[k], l_pred, baseline, p, alpha)
        jrbos.append(jrbo)

        string3 = str(jrbo)[0:7]

        cadena = align_strings(string0, string1, string2, string3, 20, 30, 50, 10)
        data.append(cadena)

    if option == 'maximum_response':
        filename = os.path.join(project_path + subfolders['results'], 'multilabel_prediction_maximum_response_train.txt')
    if option == 'maximum_diff':
        filename = os.path.join(project_path + subfolders['results'], 'multilabel_prediction_maximum_diff_train.txt')
    with open(filename, 'w') as f:
        f.writelines(data)
    print(np.mean(jrbos))



if False == 1:
    Xtfidf_tst, tags_tst, refs_tst = EV.load_test_data()

    filename = os.path.join(project_path + subfolders['results'], 'Preds_test.pkl')
    with open(filename, 'rb') as f:
        Preds_test = pickle.load(f)

    print('-' * 50)
    print(option)
    print('-' * 50)

    if barrido:
        COST = []
        DENS_pred = []
        DENS_true = []
        density_true = EV.compute_cardinality(tags_tst)

        for threshold in th_values:
            multilabel_pred_test, labels_pred = P.obtain_multilabel_preds(Preds_test, option, threshold, normalize_tags, verbose=True)
            density_pred = EV.compute_cardinality(labels_pred)
            DENS_pred.append(density_pred / 5.0)
            DENS_true.append(density_true / 5.0)

            # Computing Jackard_RBO cost
            jrbos = []
            for k in range(0, len(tags_tst)):
                values = []
                for key in list(multilabel_pred_test[k].keys()):
                    values.append((key, multilabel_pred_test[k][key]['p']))
                values.sort(key=lambda x: x[1], reverse=True)
                l_pred = []
                for v in values:
                    l_pred.append(v[0])
                jrbo = Jaccard_RBO_cost(tags_tst[k], l_pred, baseline, p, alpha)
                jrbos.append(jrbo)

            cost_jrbo = np.mean(jrbos)
            print(threshold, cost_jrbo, density_true, density_pred)
            COST.append(cost_jrbo)

        plt.figure()
        plt.xlabel('Th')
        plt.ylabel('Jackard-RBO cost')
        plt.title('Jackard-RBO cost on test for p =' + str(p) + ' and alpha= ' + str(alpha))
        plt.plot(th_values, COST, 'b', label='Jackard-RBO cost', linewidth=3.0)
        plt.plot(th_values, DENS_true, 'r', label='Label Density true labels (1/5)', linewidth=3.0)
        plt.plot(th_values, DENS_pred, 'r--', label='Label Density predictions (1/5)', linewidth=3.0)
        cual_min = np.argmin(COST)
        plt.plot(th_values[cual_min], COST[cual_min], 'bo', label='Minimum Jackard-RBO cost', linewidth=3.0)
        plt.legend(loc="upper right")
        plt.grid(True)
        filename = os.path.join(project_path + subfolders['figures'], 'JRBO_COST_tst_p_' + str(p) + '_alpha_' + str(alpha) + '.png')
        plt.savefig(filename)
        plt.close()

    multilabel_pred_test, labels_pred = P.obtain_multilabel_preds(Preds_test, option, final_threshold, normalize_tags, verbose=True)

    # writing report for the best threshold value
    jrbos = []
    string0 = 'PROJECT REFERENCE'
    string1 = 'TARGET LABELS'
    string2 = 'PREDICTED LABELS'
    string3 = 'JRBO COST'
    data = [align_strings(string0, string1, string2, string3, 20, 30, 50, 10)]
    data.append('=======================================================================================\r\n')
    for k in range(0, len(tags_tst)):

        string0 = refs_tst[k]

        string1 = ''
        for t in tags_tst[k]:
            string1 += t + ', '

        if len(tags_tst[k]) == 0:
            string1 += '--------------'

        values = []
        for key in list(multilabel_pred_test[k].keys()):
            values.append((key, multilabel_pred_test[k][key]['p']))

        values.sort(key=lambda x: x[1], reverse=True)
        l_pred = []
        string2 = ''
        for v in values:
            string2 += v[0] + '(' + str(v[1])[0:5] + '), '
            l_pred.append(v[0])

        if len(values) == 0:
            string2 += '--------------'

        jrbo = Jaccard_RBO_cost(tags_tst[k], l_pred, baseline, p, alpha)
        jrbos.append(jrbo)

        string3 = str(jrbo)[0:7]

        cadena = align_strings(string0, string1, string2, string3, 20, 30, 50, 10)
        data.append(cadena)

    if option == 'maximum_response':
        filename = os.path.join(project_path + subfolders['results'], 'multilabel_prediction_maximum_response_test.txt')
    if option == 'maximum_diff':
        filename = os.path.join(project_path + subfolders['results'], 'multilabel_prediction_maximum_diff_test.txt')
    with open(filename, 'w') as f:
        f.writelines(data)
    print(np.mean(jrbos))


print('END!')
#import code
#code.interact(local=locals())


'''
multilabel_ights, labels_pred = P.obtain_multilabel_preds(Preds_tr, option, final_threshold, normalize_tags, verbose=True)


# probamos a predecir la lista completa de proyectos, se puede usar cualquier otra lematizada
resumen_lemas = list(df.loc[:, 'Resumen_lemas'])
#docs = [resumen_lemas[3], resumen_lemas[56], resumen_lemas[2356], resumen_lemas[5667], resumen_lemas[3657], resumen_lemas[1543]]
docs = [u'literatura española siglo x poesía prosa análisis', u'desarrollo cultivo tomate patata agricultura deforestación zona desértica', u'derecho penal juicio juez pena multa', u'desarrollo software python programación código  abierto', u'trasplante corazón medicina sangre fármaco']
#Preds, tag_pred = P.predict_new_document(docs, verbose=True)
#multilabel_pred, labels_pred = P.obtain_multilabel_preds(Preds, option, final_threshold, normalize_tags, verbose=True)

print('=================================')
print(labels_pred)
print('=================================')
'''
