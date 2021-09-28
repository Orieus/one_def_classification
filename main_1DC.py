# -*- coding: utf-8 -*-
"""
Main program for the FORD classifier

Created on Apr 18 2018

@authors: Ángel Navia Vázquez, Jerónimo Arenas García, Jesús Cid Sueiro

"""

from __future__ import print_function    # For python 2 copmatibility
import os
import copy
import platform
import argparse

from fordclassifier.fordclassifier import FORDclassifier


def clear():

    """ Cleans terminal window
    """
    # Checks if the application is running on windows or other OS
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')


def query_options(options, active_options=None, msg=None, zero_option='exit'):
    """
    Prints a heading mnd the subset of options indicated in the list of
    active_options, and returns the one selected by the used

    Args:
        options        : A dictionary of options
        active_options : List of option keys indicating the available options
                         print. If None, all options are selected.
        msg            : Heading mesg to be printed befor the list of
                         available options
        zero_option    : If 'exit', an exit option is shown
                         If 'up', an option to go back to the main menu
    """

    # Print the heading messsage
    if msg is None:
        print('\n')
        print('**************')
        print('*** MAIN MENU.')
        print('Available options:')
    else:
        print(msg)

    # Print the active options
    if active_options is None:
        # If no active options ar specified, all of them are printed.
        active_options = list(options.keys())

    for n, opt in enumerate(active_options):
        print(' {0}. {1}'.format(n + 1, options[opt]))

    n_opt = len(active_options)
    if zero_option == 'exit':
        print(' 0. Exit the application\n')
        n_opt += 1
    elif zero_option == 'up':
        print(' 0. Back to the main menu\n')
        n_opt += 1

    range_opt = range(n_opt)

    n_option = None
    while n_option not in range_opt:
        n_option = input('What would you like to do? [{0}-{1}]: '.format(
            str(range_opt[0]), range_opt[-1]))
        try:
            n_option = int(n_option)
        except:
            print('Write a number')
            n_option = None

    if n_option == 0:
        option = 'zero'
    else:
        option = active_options[n_option - 1]
    return option


def request_confirmation(msg="     Are you sure?"):

    # Iterate until an admissible response is got
    r = ''
    while r not in ['yes', 'no']:
        r = input(msg + ' (yes | no): ')

    return r == 'yes'


# ################################
# Main body of application

clear()
print('***********************')
print('*** FORD CLASSIFIER ***')
print('***********************')

var_exit = False
time_delay = 3

# ########################
# Configurable parameters:
# ########################

# Seed for the random generation
seed = 201712

# ####################
# Read input arguments
# ####################

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--new_p', type=str, default=None,
                    help="path to a new evaluation project")
parser.add_argument('--load_p', type=str, default=None,
                    help="path to an existing evaluation project")
parser.add_argument('--import_path', type=str, default=None,
                    help="path to the folder containing all the source data")
args = parser.parse_args()

# ########################
# Prepare user interaction
# ########################

# This is the complete list of level-0 options.
# The options that are shown to the user will depend on the project state
options_L0 = {'newP': 'Create new classification project',
              'loadP': 'Load existing project',
              'activateP': 'Activate configuration file',
              'showDB': 'Show database',
              'resetDB': 'Reset database tables',
              'importData': 'Import data sources',
              'genCorpus': 'Generate corpus model',
              'genSupData': 'Get category definitions or labels',
              'optimizeClas': 'Optimize Classifiers',
              'evaluateClas': 'Evaluate best classifiers',
              'applyClas': 'Compute classifier predictions',
              'exportData': 'Export tables'}

# This is the default list of active options.
default_opt = ['showDB', 'resetDB', 'importData', 'genCorpus', 'genSupData',
               'optimizeClas', 'evaluateClas', 'applyClas', 'exportData']

n0 = len(options_L0)

# Read argumens
if args.new_p is None and args.load_p is None:
    # If no project is specified in args.p, only the project selection or
    # creation options are available
    project_path = None
    active_options = ['newP', 'loadP']
    query_needed = True
elif args.new_p is not None and args.load_p is None:
    project_path = args.new_p
    option1 = 'newP'
    query_needed = False
elif args.new_p is None and args.load_p is not None:
    project_path = args.load_p
    option1 = 'loadP'
    query_needed = False
else:
    exit('ERROR: You cannot specify both a new project and a project to ' +
         'load. Do not use options --new_p and --load_p in the same call.')

if args.import_path is None:
    f_struct = None
else:
    f_struct = {'import': args.import_path}

# ################
# Interaction loop
# ################

while not var_exit:

    if query_needed:
        option1 = query_options(options_L0, active_options)
    else:
        # This is for the next time
        query_needed = True

    if option1 == 'zero':

        # Activate flag to exit the application
        var_exit = True

    elif option1 == 'newP':

        # ###############################
        # Create a new classification project
        print("\n*** CREATING NEW CLASSIFICATION PROJECT")

        if project_path is None:
            while project_path is None or project_path == "":
                project_path = input('-- Write the path to the new project: ')

        # Create classification project
        FC = FORDclassifier(project_path)
        FC.create(f_struct)

        print("-- Project {0} created.".format(project_path))
        print("---- Project metadata saved in {0}".format(FC.metadata_fname))
        print("---- A default config file has been located in the project " +
              "folder.")
        print("---- Open it and set your configuration variables properly.")
        print("---- Once the config file is ready, activate it.")

        # Update list of active menu options
        active_options = ['activateP']

    elif option1 == 'loadP':

        # ########################
        # Load an existing project
        print("\n*** LOADING CLASSIFICATION PROJECT")

        if project_path is None:
            while project_path is None or project_path == "":
                project_path = input('-- Write the path to the project: ')

        # Load classification project
        FC = FORDclassifier(project_path)
        msg = FC.load(f_struct)

        if not FC.state['isProject']:
            # The project could not be loaded. Only the options to load
            # ore create a new one will be active.
            print(msg)
            project_path = None
            active_options = ['newP', 'loadP']
        elif not FC.state['configReady']:
            # The projec has been loaded, but the config file has not been
            # activated. This must be done at least once since the project
            # is created.
            print("-- Project {} loaded.".format(project_path))
            print("---- Config file has not been activated.")
            print("---- Revise the config.cf file and activate it.")
            active_options = ['activateP']
        else:
            print('Project {} loaded.'.format(project_path))
            # Optimistic list of active options. Some of them might fail
            active_options = copy.copy(default_opt)

    elif option1 == 'activateP':

        # #################################################
        # Activate configuration file and load data Manager
        print("\n*** ACTIVATING CONFIGURATION FILE")

        FC.setup()
        if not FC.state['configReady']:
            print("---- Config file could not be activated.")
            print("---- Revise the config.cf file and activate it.")
            active_options = ['activateP']
        else:
            print('Config file activated.'.format(project_path))
            # Optimistic list of active options. Some of them might fail
            active_options = copy.copy(default_opt)

    elif option1 == 'showDB':

        # #######################
        # Generate a corpus model
        msg = '\nSelect what do you want to view:'
        options_L1 = {'overview': 'Database Summary',
                      **{'T_' + x: 'Table ' + x  for x in FC.db_tables},
                      'sample': 'Random sample'}
        option2 = query_options(options_L1, msg=msg, zero_option='up')

        if option2 != 'zero':
            FC.showDBdata(option2)

    elif option1 == 'resetDB':

        # #####################################
        # Reset the selected tables from the DB
        print("\n*** RESETTING DATABASE TABLES")

        msg = '\nSelect the table to reset:'
        options_L1 = {**{'T_' + x: 'Table ' + x  for x in FC.db_tables},
                      'resetLab': 'Reset labeler tables: labels_label, ' +
                                  'labels_info & labelhistory',
                      'resetAll': 'Reset all tables'}
        option2 = query_options(options_L1, msg=msg, zero_option='up')

        if option2 != 'zero':
            if option2 == 'resetAll':
                # Warn:
                print("---- WARNING: This will reset the entire database")
                print("              All data will be lost")
                if request_confirmation():
                    FC.resetDBtables()
                active_options = ['showDB', 'resetDB', 'importData']
            elif option2 == 'resetLab':
                # Warn:
                print("---- WARNING: All data in all tables will be lost")
                if request_confirmation():
                    tables = ['labels_label', 'labels_info', 'labelhistory']
                    FC.resetDBtables(tables)
            else:
                # Warn:
                print("---- WARNING: All data in the table will be lost")
                if request_confirmation():
                    table = option2[2:]
                    FC.resetDBtables(table)

    elif option1 == 'importData':

        # ###########
        # Import data
        print("\n*** IMPORTING DATA SOURCES")
        dir_corpus = FC.f_struct['import_corpus']
        dir_taxonomy = FC.f_struct['import_taxonomy']
        dir_labels = FC.f_struct['import_labels']

        if args.import_path is None:
            print("You should provide a path to the data sources.")
            print("It will be assumed that all data will be located in a " +
                  "specific subfolder of the given data path: ")
            txt = " - [SOURCE_PATH]/{}: path to "
            print((txt + "Text Corpus files").format(dir_corpus))
            print((txt + "Taxonomy files").format(dir_taxonomy))
            print((txt + "Label set files").format(dir_labels))
            default_path = os.path.join(project_path, FC.f_struct['import'])
            data_path = input('\nWrite the SOURCE_PATH (or press enter to ' +
                              'use the default path, {}):'.format(
                                  default_path))
            if data_path == "":
                data_path = None
        else:
            data_path = args.import_path
            print("You have already provided a path to the data sources.")
            print("It will be assumed that all data will be located in a " +
                  "specific subfolder of the given path: ")
            txt = " - {}: path to ".format(data_path)
            print((txt + "Text Corpus files").format(dir_corpus))
            print((txt + "Taxonomy files").format(dir_taxonomy))
            print((txt + "Label set files").format(dir_labels))

        msg = '\nSelect the type of data to import:'
        options_L1 = {'corpus': 'Text corpus',
                      'lemmas': 'Lemmatized corpus',
                      'c_projects': 'Coordinated project indicators',
                      'taxonomy': 'Taxonomy',
                      'unesco2ford': 'Unesco-Ford mapped labels',
                      'labels': 'Manual labels',
                      'all': 'Import all available data in the given folder'}
        option2 = query_options(options_L1, msg=msg, zero_option='up')

        if option2 != 'zero':
            FC.importData(data_path, option2)
            # Optimistic list of active options. Some of them might fail
            active_options = copy.copy(default_opt)

    elif option1 == 'genCorpus':

        # #######################
        # Generate a corpus model
        msg = '\nSelect the task to do:'
        options_L1 = {'detectC': 'Identify coordinated projects',
                      'lemmas': 'Compute lemmatized corpus',
                      'BoW': 'Compute BoW',
                      'WE': 'Compute Word Embedding'}
        option2 = query_options(options_L1, msg=msg, zero_option='up')

        if option2 != 'zero':
            # Generate the appropriate corpus model
            if option2 == 'detectC':
                print("\n*** IDENTIFYING COORDINATED PROYECTS")
            else:
                print("\n*** GENERATING A CORPUS MODEL BASED ON " + option2)
            FC.generateCorpusModel(option2)
            print("Task completed.")

            # Optimistic list of active options. Some of them might fail
            active_options = copy.copy(default_opt)

    elif option1 == 'genSupData':

        # #########################
        # Generate a category model

        msg = '\nSelect the task to do:'
        options_L1 = {'labeler': 'Run labeling tool',
                      'showLabels': 'Show current labels',
                      'testU2F': 'Test unesco2ford labels vs manual labels',
                      'labelStats': 'Show some label statistics',
                      'labelsPerCall': 'Show label stats per call',
                      'lemmatizeDefs': 'Lemmatize definitions',
                      'expand': 'Expand categories'}
        option2 = query_options(options_L1, msg=msg, zero_option='up')

        if option2 != 'zero':
            FC.generateCategoryModel(option2)
            print("Category Model generated.")
            # Optimistic list of active options. Some of them might fail
            active_options = copy.copy(default_opt)
            print("Done.")

    elif option1 == 'optimizeClas':

        # ####################
        # Optimize classifiers

        msg = '\nSelect the classiffier to use:'
        options_L1 = {'partition': 'Prepare data partitions',
                      'sup': 'Optimize supervised classification'}
        option2 = query_options(options_L1, msg=msg, zero_option='up')

        if option2 != 'zero':
            FC.optimizeClassifiers(option2)
            # Optimistic list of active options. Some of them might fail
            active_options = copy.copy(default_opt)
            print("Done.")

    elif option1 == 'evaluateClas':

        # ###################
        # Evaluate classifier

        msg = '\nSelect the classiffier to use:'
        options_L1 = {'sup': 'Evaluate supervised classification',
                      'uns': 'Evaluate unsupervised classification',
                      'man_sup': 'Evaluate supervised classifier with ' +
                                 'manual labels'}
        option2 = query_options(options_L1, msg=msg, zero_option='up')
        if option2 != 'zero':
            FC.evaluateClassifier(option2)
            # Optimistic list of active options. Some of them might fail
            active_options = copy.copy(default_opt)
            print("Done.")

    elif option1 == 'applyClas':

        # ##############################
        # Compute classifier predictions
        print("\n*** COMPUTING PREDICTIONS FOR NEW PROJECTS")
        FC.computePredictions()
        # Optimistic list of active options. Some of them might fail
        active_options = copy.copy(default_opt)
        print("Done.")

    elif option1 == 'exportData':

        # ###########
        # Export data
        print("\n*** EXPORTING DATA TO FILES")
        msg = '\nSelect the data to export:'
        options_L1 = {'lemmas': 'Lemmatized corpus',
                      'c_projects': 'Coordinated project indicators',
                      'labels': 'Manual labels',
                      'taxonomy': 'Taxonomy',
                      'all': 'Export all these data'}
        option2 = query_options(options_L1, msg=msg, zero_option='up')
        if option2 != 'zero':
            FC.exportData(option2)
            print("Done.")

    else:

        print("Unknown option")

