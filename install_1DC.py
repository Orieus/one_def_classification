
# PYTHON

# CALL conda update conda
# CALL conda update anaconda

from subprocess import call

try: 
    import ipdb
except:
    call('pip install ipdb'.split())

try:
    import MySQLdb
except:
    call('conda install -y mysqlclient'.split())

try:
    import tabulate
except:
    call('conda install -y tabulate'.split())

try:
    import gensim
except:
    call('conda install -y gensim'.split())

try:
    import langid
except:
    call('pip install langid'.split())

try:
    import progress
except:
    call('pip install progress'.split())

try:
    import wikipedia
except:
    call('pip install wikipedia'.split())

try: 
    import pandas
except:
    call('conda install -y pandas'.split())

try: 
    import sklearn
except:
    call('conda install -y scikit-learn'.split())

try:
    import sqlalchemy
except:
    call('conda install -y sqlalchemy'.split())

try:
    import matplotlib
except:
    call('conda install -y matplotlib'.split())

try:
    import nltk
except:
    call('conda install -y nltk'.split())

try:
    import xlrd
except:
    call('conda install -y xlrd'.split())

# call(['set', 'PATH=%PATH%;C:\\anaconda3\\Scripts;C:\\anaconda3;C:\\anaconda3\\Library\bin'])