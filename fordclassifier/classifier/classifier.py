# -*- coding: utf-8 -*-
'''
This class unifies all possible classifiers and parameters under a common
structure

Author:  Angel Navia

Example of use:

import numpy as np
from common.lib.classifier import Classifier
model = Classifier('SVMpoly2C001')
model.display()
X = np.random.rand(5,50)
y =  np.sign(np.random.rand(5,1)-0.5).ravel()
model.fit(X, y)
p = model.predict(X)

'''
from sklearn import linear_model
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

import numpy as np
try:
    import cPickle as pickle
except:
    import pickle


class Classifier(object):
    '''
    Class to wrap all the classifiers

    ============================================================================
    Methods:
    ============================================================================
    fit(X, y)  : trains the model using X and y (numpy matrices, possibly
                 sparse)
    predict(X) : obtains predictions for data in X
    ============================================================================

    '''

    def __init__(self, model_params, verbose=True):

        # The classifiers are initializaed with the parameters in model_params
        self.model_type = model_params['model_type']
        self._verbose = verbose   # if True, messages are printed on screen

        # initializing models of every type
        if self.model_type == 'LR':
            C = model_params['C']
            self.model = linear_model.LogisticRegression(
                C=C, class_weight='balanced')
            if self._verbose:
                print('Initializing classifier of type ' + self.model_type +
                      ' with parameter C = ' + str(C))

        if self.model_type == 'LSVM':
            C = model_params['C']
            self.model = svm.SVC(kernel='linear', C=C, probability=True,
                                 class_weight='balanced')
            if self._verbose:
                print('Initializing classifier of type ' + self.model_type +
                      ' with parameter C = ' + str(C))

        if self.model_type == 'SVMpoly':
            C = model_params['C']
            degree = model_params['degree']
            self.model = svm.SVC(kernel='poly', degree=degree, C=C,
                                 probability=True, class_weight='balanced')
            if self._verbose:
                print('Initializing classifier of type ' + self.model_type +
                      ' with parameter C = ' + str(C) + ' and degree = ' +
                      str(degree))

        if self.model_type == 'DT':
            self.model = DecisionTreeClassifier()
            if self._verbose:
                print('Initializing classifier of type ' + self.model_type)

        if self.model_type == 'RF':
            N = model_params['N']
            self.model = RandomForestClassifier(n_estimators=N)
            if self._verbose:
                print('Initializing classifier of type ' + self.model_type +
                      ' with parameter N = ' + str(N))

        if self.model_type == 'MLP':
            C = model_params['C']
            N = model_params['N']
            self.model = MLPClassifier(alpha=C, hidden_layer_sizes=(N,))
            if self._verbose:
                print('Initializing classifier of type ' + self.model_type +
                      ' with parameter C = ' + str(C))

        if self.model_type == 'KNN':
            N = model_params['N']
            self.model = KNeighborsClassifier(n_neighbors=N)
            if self._verbose:
                print('Initializing classifier of type ' + self.model_type +
                      ' with parameter N = ' + str(N))

        if self.model_type == 'AB':
            N = model_params['N']
            self.model = AdaBoostClassifier(n_estimators=N)
            if self._verbose:
                print('Initializing classifier of type ' + self.model_type)

        if self.model_type == 'LASSO':
            L = model_params['L']
            self.model = Lasso(alpha=L)
            if self._verbose:
                print('Initializing classifier of type ' + self.model_type +
                      ' with parameter L = ' + str(L))

        if self.model_type == 'EN':
            L1 = model_params['L1']
            L2 = model_params['L2']
            self.model = ElasticNet(alpha=L1, l1_ratio=L2)
            if self._verbose:
                print('Initializing classifier of type ' + self.model_type +
                      f' with parameter L1 = {L1} and L2 = {L2}')

        if self.model_type == 'MNB':
            alpha = model_params['alpha']
            self.model = MultinomialNB(alpha=alpha)
            if self._verbose:
                print('Initializing classifier of type ' + self.model_type)

        if self.model_type == 'BNB':
            alpha = model_params['alpha']
            self.model = BernoulliNB(alpha=alpha)
            if self._verbose:
                print('Initializing classifier of type ' + self.model_type)
        return

    def fit(self, X, y, w=None):
        """
           X: input data (numpy matrix, admits sparse)
           y: target data (binary)
           w: Sample weights

        # returns the trained model in self.model
        """
        try:
            if self.model_type in ['LR', 'LSVM', 'SVMpoly', 'DT', 'RF', 'MLP',
                                   'KNN', 'AB', 'LASSO', 'EN', 'MNB', 'BNB']:
                if self.model_type in ['LR', 'LSVM', 'SVMpoly', 'DT', 'RF',
                                       'MLP', 'AB', 'MNB', 'BNB']:
                    self.model.fit(X, y, w)
                else:
                    if w is not None:
                        print("WARNING: sample weights not available for" +
                              f"classifiers of type {self.model_type}. They " +
                              "will be ignored")
                    self.model.fit(X, y)
                if self._verbose:
                    print('Trained classifier of type ' + self.model_type)
        except:
            if self._verbose:
                print('Something went wrong training classifier of type ' +
                      self.model_type)
            pass

        return

    def predict(self, X):
        # X: input data (numpy matrix, possibly sparse)
        # returns a vector with the predicions for X

        if self.model_type in ['LR', 'LSVM', 'SVMpoly', 'DT', 'RF', 'MLP',
                               'KNN', 'AB', 'MNB', 'BNB']:
            p = self.model.predict_proba(X)

        if self.model_type in ['LASSO', 'EN']:
            p = self.model.predict(X)

        newp = []
        for x in p:
            if self.model_type in ['LR', 'DT', 'RF', 'MLP', 'KNN', 'AB', 'MNB',
                                   'BNB', 'LSVM', 'SVMpoly']:
                try:
                    value = x[1]
                except:
                    value = x[0]
                    pass
            # mapping prediction value to (-1, 1)
            value = value * 2.0 - 1.0
            if value < -1.0:
                value = -1.0
            if value > 1.0:
                value = 1.0
            newp.append(value)
        p = np.array(newp)
        return p
