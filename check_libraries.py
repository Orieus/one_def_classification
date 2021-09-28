#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sept 2018

Angel Navia

python check_libraries.py

@author: navia
"""

from subprocess import call

# Esto tarda mucho, ¿dejar fuera???
# call('conda update conda -y'.split())
# call('conda update anaconda -y'.split())

# import os
# import platform
# pathlist = os.environ["PATH"].split(';')
# print(pathlist)

# if platform.system() == 'Windows':

#     for p in ["C:\\anaconda3\\Scripts", "C:\\anaconda3",
#               "C:\\anaconda3\\Library\\bin"]:

#         if p not in pathlist:
#             print("Setting path " + p)
#             os.environ["PATH"] = os.environ["PATH"] + ";" + p

#     print("New environment: " + os.environ["PATH"])

try:
    import ipdb
    print('ipdb OK')
except:
    call('pip install ipdb'.split())

try:
    import MySQLdb
    print('MySQLdb OK')
except:
    call('conda install -y mysqlclient'.split())

try:
    import tabulate
    print('tabulate OK')
except:
    call('conda install -y tabulate'.split())

try:
    import gensim
    print('gensim OK')
except:
    call('conda install -y gensim'.split())

try:
    import langid
    print('langid OK')
except:
    call('pip install langid'.split())

try:
    import progress
    print('progress OK')
except:
    call('pip install progress'.split())

try:
    import wikipedia
    print('wikipedia OK')
except:
    call('pip install wikipedia'.split())

try:
    import pandas
    print('pandas OK')
except:
    call('conda install -y pandas'.split())

try:
    import sklearn
    print('sklearn OK')
except:
    call('conda install -y scikit-learn'.split())

try:
    import sqlalchemy
    print('sqlalchemy OK')
except:
    call('conda install -y sqlalchemy'.split())

try:
    import matplotlib
    print('matplotlib OK')
except:
    call('conda install -y matplotlib'.split())

try:
    import nltk
    print('nltk OK')
except:
    call('conda install -y nltk'.split())

try:
    import xlrd
    print('xlrd OK')
except:
    call('conda install -y xlrd'.split())

try:
    nltk.data.find('tokenizers/punkt')
    print('punkt OK')
except LookupError:
    nltk.download('punkt')