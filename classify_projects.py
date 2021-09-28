#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sept 2018

Angel Navia

python classify_projects.py -i filename -o path

python classify_projects.py -i ./Europa.xls -o ./

@author: navia
"""

import sys
import os

from fordclassifier.fordclassifier import FORDclassifier
import configparser


def main(argv):

    for i in range(0, len(argv)):
        if argv[i] == '-i':
            input_filename = argv[i + 1]
        if argv[i] == '-o':
            output_path = argv[i + 1]

    print("Leyendo fichero: ", input_filename)
    print("Guardando resultados en: ", output_path)

    project_path = './'
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
        # For backward compatib. (title_mul may not exist in the config file)
        FC.title_mul = 1

    FC._computePredictions(input_filename, output_path)
    print("Fin proceso")

if __name__ == "__main__":
    main(sys.argv[1:])
