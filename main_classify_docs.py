# -*- coding: utf-8 -*-
"""
Main program for the FORD classifier

Created on Apr 18 2018

@authors: Ángel Navia Vázquez, Jerónimo Arenas García, Jesús Cid Sueiro

"""

from __future__ import print_function    # For python 2 copmatibility
import argparse
import os
import platform
import configparser

from fordclassifier.fordclassifier import FORDclassifier


def clear():

    """Cleans terminal window
    """
    # Checks if the application is running on windows or other OS
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')


# ################################
# Main body of application

clear()
print('*************************')
print('*** CLASIFICADOR FORD ***')
print('*************************')
print('')

# ####################
# Read input arguments
# ####################

# settings
parser = argparse.ArgumentParser()
parser.add_argument(
    '--input', type=str, default=None,
    help="ruta al fichero xls o xlsx que contiene los proyectos a clasificar")
parser.add_argument('--output', type=str, default=None,
                    help="ruta al directorio de salida")
args = parser.parse_args()

# Read argumens
project_path = './'
source_path = args.input
target_path = args.output

# ##############################
# Compute classifier predictions
print("\n*** CLASIFICANDO PROYECTOS...")

FC = FORDclassifier(project_path)

# This is the subfolder structure for the classification modules.
FC.f_struct['import'] = './'
FC.f_struct['import_taxonomy'] = 'taxonomy/'
FC.f_struct_cls = {'training_data': 'best_models/',
                   'export': 'best_models/'}

# Create configparser object insite the FC object
cf_fname = os.path.join('best_models', 'config.cf')
FC.cf = configparser.ConfigParser()
FC.cf.read(cf_fname)

# Read minimum document frequency from the config
if FC.cf.has_option('PREPROC', 'min_df'):
    FC.min_df = int(FC.cf.get('PREPROC', 'min_df'))
else:
    # For backward compatibility (min_df may not exist in the config file)
    FC.min_df = 2

# Read minimum document frequency from the config
if FC.cf.has_option('PREPROC', 'title_mul'):
    FC.title_mul = int(FC.cf.get('PREPROC', 'title_mul'))
else:
    # For backward compatibility (title_mul may not exist in the config file)
    FC.title_mul = 1

FC._computePredictions(source_path, target_path)

