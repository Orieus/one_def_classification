import os
import ast
import platform
import shutil
import _pickle as pickle

import configparser

# Imports for the evaluator
import pandas as pd
import numpy as np
from collections import Counter

try:
    import tkinter as tk
    from tkinter import filedialog

    from time import time

    # Local imports
    # import fordclassifier.labelfactory.labelfactory as labelfactory
    import labelfactory.labelfactory as labelfactory

except:
    pass

from fordclassifier.datamanager.datamanager import DataManager
from fordclassifier.corpusanalyzer.wordembed import WordEmbedding as WE

# This is to solve a known incompatibility issue between matplotlib and
# tkinter on mac os.
if platform.system() == 'Darwin':     # Darwin is the system name for mac os.
    # IMPORTANT: THIS CODE MUST BE LOCATED BEFORE ANY OTHER IMPORT TO
    #            MATPLOTLIB OR TO A LIBRARY IMPORTING FROM MATPLOTLIB
    import matplotlib
    matplotlib.use('TkAgg')

# This modules are from matplotlib or they import matplotlib modules, and
# must be below the matplotlib.use() command.
from fordclassifier.corpusanalyzer.fc_corpusanalyzer import FCCorpusAnalyzer
from fordclassifier.categoryanalyzer.fc_taxonomyanalyzer import (
    FCTaxonomyAnalyzer)
from fordclassifier.classifier.fc_supclassifier import FCSupClassifier
from fordclassifier.classifier.fc_unsupclassifier import FCUnsupClassifier

import matplotlib.pyplot as plt


def addSlash(p):
    """
    If string p does not end with the folder separator ('/' in OS Linux,
    '\\' in Windows, it is added.
    """

    if platform.system() == 'Windows':
        sep = '\\'
    else:
        sep = '/'

    if p.endswith(sep):
        return p
    else:
        return p + sep


def removeSlash(p):
    """
    If string p ends with character '/' or backslash, it is removed.
    """

    if p.endswith('/') or p.endswith('\\'):
        return p[:-1]
    else:
        return p


class FORDclassifier(object):
    """
    Main class of a corpus classification project.

    The behavior of this class depends on the state of the project, in
    dictionary self.state, with the followin entries:

    - 'isProject'   : If True, project created. Metadata variables loaded
    - 'configReady' : If True, config file succesfully loaded. Datamanager
                      activated.
    """

    def __init__(self, path2project, metadata_fname='metadata.pkl',
                 config_fname='config.cf'):
        """
        Opens a corpus classification project.
        """

        # This is the minimal information required to start with a project
        self.path2project = addSlash(path2project)
        self.path2metadata = os.path.join(self.path2project, metadata_fname)
        self.path2config = os.path.join(self.path2project, config_fname)
        self.metadata_fname = metadata_fname

        # We assign a variable to this file name because it is used in
        # different parts of the code
        self.taxonomy_fname = 'Taxonomy.xlsx'

        # These are the default file and folder names for the folder
        # structure of the project. It can be modified by entering other
        # names as arguments of the create or the load method.
        self.f_struct = {
            'import': 'import/',               # Folder of input data sources
            'import_cprojects': 'c_projects',  # - Subfolder of coord. projs
            'import_corpus': 'corpus/',        # - Subfolder of text corpora
            'import_labels': 'labels/',        # - Subfolder of manual labels
            'import_lemmas': 'lemmas/',        # - Subfolder of manual labels
            'import_taxonomy': 'taxonomy/',    # - Subfolder of category defs
            'import_unesco2ford': 'unesco2ford/',  # - S. of unesco2ford labels
            'labels': 'labels/',               # Folder of new labels
            'export': 'export/',
            'models': 'models/',
            'bow': 'bow/',
            'we': 'we/',
            'classifiers': 'classifiers/',
            'output': 'output/'
            }

        self.f_struct['path2labels'] = os.path.join(self.path2project,
                                                    self.f_struct['labels'])

        # This is the subfolder structure for the classification modules.
        self.f_struct_cls = {'training_data': 'classifiers/training_data/',
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
                             'export': 'classifiers/export/'
                             }

        # State variables that will be laoded from the metadata file when
        # when the project was loaded.
        self.state = {
            'isProject': False,     # True if the project exist.
            'configReady': False,   # True if config file has been processed
            'dbReady': False}       # True if db exists and can be connected

        # Other class variables
        self.fs_class = None   # A specific file substructure is used for the
        # folder containing the classifier models.
        self.DM = None         # Data manager object
        self.cf = None         # Handler to the config file
        self.db_tables = None  # List of (empty or not) tables in the database
        self.ready2setup = False  # True after create() or load() are called

        # Categoriy set.
        self.categories = None
        self.cat_model = None

        # Bow parameters
        self.min_df = None
        self.title_mul = None

        return

    def _make_subfolders(self, path, subfolders, tags=None):
        """
        Completes a subfolder structure: it checks if the subfolders exist in
        path. If not, they are created.

        Args:
            path       :Path where the subfolders will be located
            subfolders :A dictionary of subfolders. The subfolder names are
                        the values of the dictionary
            tags       :A list of dictionary keys specifyng which subfolders
                        must be created. If None, all of them are creatted
        """

        if tags is None:
            tags = list(subfolders.keys())

        for t in tags:
            path2sf = os.path.join(path, subfolders[t])
            if not os.path.exists(path2sf):
                os.makedirs(path2sf)

        return

    def create(self, f_struct=None, f_struct_cls=None):
        """
        Creates a corpus classification project.
        To do so, it defines the main folder structure, and creates (or cleans)
        the project folder, specified in self.path2project

        Args:
            f_struct :Contains all information related to the structure of
                      project files and folders:
                        - paths (relative to ppath)
                        - file names
                        - suffixes, prefixes or extensions that could be used
                          to define other files or folders.
                      (default names are used when not given)
                      If None, default names are given to the whole folder tree
            f_struct_cls :Contains all information related to the structure of
                          project files and folders for classification modules
        """

        # This is just to abbreviate
        p2p = self.path2project

        # Check and clean project folder location
        if os.path.exists(p2p):
            print('Folder {} already exists.'.format(p2p))

            # Remove current backup folder, if it exists
            old_p2p = removeSlash(p2p) + '_old/'
            if os.path.exists(old_p2p):
                shutil.rmtree(old_p2p)

            # Copy current project folder to the backup folder.
            shutil.move(p2p, old_p2p)
            print('Moved to ' + old_p2p)

        # Create project folder and subfolders
        os.makedirs(self.path2project)

        self.update_folders(f_struct, f_struct_cls)

        # Place a copy of a default configuration file in the project folder.
        # This file should be adapted to the new project settings.
        shutil.copyfile('config.cf.default', os.path.join(p2p, 'config.cf'))
        shutil.copyfile('config_labeler.cf.default',
                        os.path.join(self.f_struct['path2labels'],
                                     'config.cf'))

        # Update the state of the project.
        self.state['isProject'] = True

        # Save metadata
        self.save_metadata()

        # The project is ready to setup, but the user should edit the
        # configuration file first
        self.ready2setup = True

    def update_folders(self, f_struct=None, f_struct_cls=None):
        """
        Updates the project folder structure using the file and folder
        names in f_struct.

        Args:
            f_struct :Contains all information related to the structure of
                      project files and folders:
                        - paths (relative to ppath)
                        - file names
                        - suffixes, prefixes or extensions that could be used
                          to define other files or folders.
                      (default names are used when not given)
                      If None, default names are given to the whole folder tree
            f_struct_cls :Contains all information related to the structure of
                          project files and folders for classification modules
        """

        # ######################
        # Project file structure

        # Overwrite default names in self.f_struct dictionary by those
        # specified in f_struct
        if f_struct is not None:
            self.f_struct.update(f_struct)

        # This is just to abbreviate
        p2p = self.path2project

        # Now, we have to do something different with folders and subfolders.
        # for this reason, I define a list specifying which are the main
        # folders. This is not very nice...
        main_folder_tags = ['import', 'export', 'labels', 'models',
                            'classifiers', 'output', 'models']
        self._make_subfolders(p2p, self.f_struct, main_folder_tags)

        # Import subfolders
        path2import = os.path.join(p2p, self.f_struct['import'])

        tags = ['import_cprojects', 'import_corpus', 'import_labels',
                'import_lemmas', 'import_taxonomy', 'import_unesco2ford']
        self._make_subfolders(path2import, self.f_struct, tags)

        # Corpus model subfolders
        path2corpus_models = os.path.join(p2p, self.f_struct['models'])
        tags = ['bow', 'we']
        self._make_subfolders(path2corpus_models, self.f_struct, tags)

        # ####################################
        # Supervised classification subfolders

        # Overwrite default names by those specified in f_struct
        if f_struct_cls is not None:
            self.f_struct_cls.update(f_struct_cls)

        # Create supervised classification subfolders
        self._make_subfolders(p2p, self.f_struct_cls)

        return

    def save_metadata(self):

        # Save metadata
        metadata = {'f_struct': self.f_struct, 'fs_class': self.fs_class,
                    'f_struct_cls': self.f_struct_cls, 'state': self.state}
        with open(self.path2metadata, 'wb') as f:
            pickle.dump(metadata, f)

    def load(self, f_struct={}):
        """
        Loads an existing project, by reading the metadata file in the project
        folder.

        It can be used to modify file or folder names, or paths, by specifying
        the new names/paths in the f_struct dictionary.
        """

        # Check and clean project folder location
        if not os.path.exists(self.path2project):
            msg = ('Folder {} does not exist.\n'.format(self.path2project) +
                   'You must create the project first')

        # Check metadata file
        elif not os.path.exists(self.path2metadata):
            msg = ('Metadata file {} does not exist.\n'.format(
                   self.path2metadata) +
                   'This is likely not a project folder. \n' +
                   'Select another project or create a new one.')

        else:
            # Load project metadata
            with open(self.path2metadata, 'rb') as f:
                metadata = pickle.load(f)

            # Store state and fils structure
            self.state = metadata['state']

            # This if for backward compatibility
            if 'f_struct_cls' not in metadata:
                metadata['f_struct_cls'] = None

            # The following is used to automatically update any changes in the
            # keys of the self.f_struct dictionary. This will be likely
            # unnecesary once a stable version of the code is reached.
            self.update_folders(metadata['f_struct'],
                                metadata['f_struct_cls'])

            # Update folder with the new folder names provided in f_struct
            self.update_folders(f_struct)

            # Save updated folder structure in the metadata file.
            self.save_metadata()

            if self.state['configReady']:
                self.cf = configparser.ConfigParser()
                self.cf.read(self.path2config)
                self.DM = DataManager(self.path2project, self.cf)
                msg = 'Project succesfully loaded.'
            else:
                msg = 'Project loaded, but the config file is not activated.'

        self.ready2setup = True
        self.setup()

        return msg

    def setup(self):
        """
        Set up the classification projetc. To do so:
            - Loads the configuration file and initialize the data manager.
            - Creates a DB table.
        """

        if self.ready2setup is False:
            print("---- Error: you cannot setup a project that has not been " +
                  "created or loaded")
            return

        # Create configparser object
        self.cf = configparser.ConfigParser()
        self.cf.read(self.path2config)

        # Read category model
        # The "if" is for backward compatibility
        if self.cf.has_option('CATEGORIES', 'cat_model'):
            self.cat_model = self.cf.get('CATEGORIES', 'cat_model')
        else:
            self.cat_model = 'multilabel'

        # Transfer data from the config file of the project to the
        # configuration file of the labelling application
        self.config_labeler()

        # Read minimum document frequency from the config
        if self.cf.has_option('PREPROC', 'min_df'):
            self.min_df = int(self.cf.get('PREPROC', 'min_df'))
        else:
            # This is for backward compatibility, because this variable was not
            # initially included in the config file
            self.min_df = 2

        # Read title multiplier from the config
        if self.cf.has_option('PREPROC', 'title_mul'):
            self.title_mul = int(self.cf.get('PREPROC', 'title_mul'))
        else:
            # This is for backward compatibility, because this variable was not
            # initially included in the config file
            self.title_mul = 1

        # Create datamanager objetc
        self.DM = DataManager(self.path2project, self.cf)
        self.state['dbReady'] = self.DM.dbON

        # Update the project state
        if self.state['dbReady']:

            self.db_tables = self.DM.getTableNames()

            if self.db_tables == []:
                print("---- No tables in the DB. Creating basic tables.")
                self.DM.createDBtables()
                # Update table list
                self.db_tables = self.DM.getTableNames()
            else:
                print("---- Some tables already exist in the DB")

            self.state['configReady'] = True
        else:
            print("---- ERROR: The database could not be connected")
            self.state['configReady'] = False

        # Save the state of the project.
        self.save_metadata()

        return

    def config_labeler(self):
        """
        Transfer data from the configuration file to the configuration file in
        the labeling application.

        This is done to make config_labeler transparent to the user. The user
        do not need to edit it.
        """

        # Read data from the target config file
        cf_labeler = configparser.ConfigParser()
        path2labelcf = os.path.join(self.f_struct['path2labels'], 'config.cf')
        cf_labeler.read(path2labelcf)

        # #########
        # Read data

        # Read all variables that must be transferred to the labeler cf file
        db_connector = self.cf.get('DB', 'DB_CONNECTOR')
        db_name = self.cf.get('DB', 'DB_NAME')
        db_server = self.cf.get('DB', 'DB_SERVER')
        db_user = self.cf.get('DB', 'DB_USER')
        db_password = self.cf.get('DB', 'DB_PASSWORD')
        db_cats_str = self.cf.get('CATEGORIES', 'categories')
        # The default valure here is for backward compatibility.
        # Older versions of the config file did not include the parentcat
        # variable into the main file
        db_parentcat_str = self.cf.get('CATEGORIES', 'parentcat',
                                       fallback=None)

        # Read the corresponding values in the target variables
        db2_connector = cf_labeler.get('DataPaths', 'db_connector')
        db2_name = cf_labeler.get('DataPaths', 'db_name')
        db2_server = cf_labeler.get('DataPaths', 'db_server')
        db2_user = cf_labeler.get('DataPaths', 'db_user')
        db2_password = cf_labeler.get('DataPaths', 'db_password')
        db2_cats_str = cf_labeler.get('Labeler', 'categories')
        db2_parentcat_str = cf_labeler.get('Labeler', 'parentcat')

        # For backward compatibility. If there is no parentcat in the main
        # config file, the value in the labeler config file is ok.
        if db_parentcat_str is None:
            db_parentcat_str = db2_parentcat_str

        # ####################
        # Transfer DB settings

        # The labeler config file is rewritten only iff any of the current
        # values of the above variables have changed.
        if (db_connector != db2_connector or
                db_name != db2_name or
                db_server != db2_server or
                db_user != db2_user or
                db_password != db2_password or
                db_cats_str != db2_cats_str or
                db_parentcat_str != db2_parentcat_str):

            cf_labeler.set('DataPaths', 'db_connector', db_connector)
            cf_labeler.set('DataPaths', 'db_name', db_name)
            cf_labeler.set('DataPaths', 'db_server', db_server)
            cf_labeler.set('DataPaths', 'db_user', db_user)
            cf_labeler.set('DataPaths', 'db_password', db_password)
            cf_labeler.set('Labeler', 'categories', db_cats_str)
            cf_labeler.set('Labeler', 'parentcat', db_parentcat_str)

            # Update changes in the labelconfig file.
            with open(path2labelcf, 'w') as f:
                cf_labeler.write(f)

        return

    def showDBdata(self, option):

        if option == 'overview':
            # Show a list of all tables in the DB and their attributes
            self.DM.showDBview()
        elif option == 'sample':
            # Take a labeled project at random and show all data relating to
            # it in the DB.
            self.DM.showSample()
        elif option.startswith('T_'):
            # The option must contain the name of a table in the DB
            tablename = option[2:]
            self.DM.showTable(tablename)
        else:
            print('---- Option not available')
        return

    def resetDBtables(self, tables=None):
        """
        Reset (drop and create emtpy) tables from the database.

        Args:

            tables: If string, name of the table to reset.
                    If list, list of tables to reset
                    If None, all tables in the DB will be reset
        """

        self.DM.resetDBtables(tables)
        self.db_tables = self.DM.getTableNames()

        if tables is None:
            print("\n---- Tables {} have been reset.".format(self.db_tables))
        else:
            print("\n---- Tables {} have been reset.".format(tables))

        return

    def importData(self, data_path=None, datatypes='all'):
        """
        Import source data.
        This task is fully in charge of the data manager, so it is transferred
        to it, datatype by datatype.
        """

        if data_path is None or data_path == "":
            # If no datapath is specified, the default path is used
            data_path = self.f_struct['import']
        else:
            # Update path to the import data
            self.f_struct['import'] = data_path

        if datatypes == 'all':
            datatypes = ['corpus', 'lemmas', 'c_projects', 'taxonomy',
                         'unesco2ford', 'labels']
        else:
            datatypes = [datatypes]

        for dt in datatypes:
            print('---- Importing {}... This may take a while.'.format(
                dt))
            filesdir = None

            if dt == 'corpus':
                subdata_path = os.path.join(
                    data_path, self.f_struct['import_corpus'])
                fname = 'Convocatorias.xls'
                filesdir = 'projectfiles'

            elif dt == 'lemmas':
                subdata_path = os.path.join(
                    data_path, self.f_struct['import_lemmas'])
                fname = 'corpus_lemmas.xlsx'

            elif dt == 'c_projects':
                subdata_path = os.path.join(
                    data_path, self.f_struct['import_cprojects'])
                # fname = 'coordinated_projects.pkl'
                fname = 'c_projects.xlsx'

            elif dt == 'taxonomy':
                subdata_path = os.path.join(
                    data_path, self.f_struct['import_taxonomy'])
                fname = self.taxonomy_fname

            elif dt == 'unesco2ford':
                subdata_path = os.path.join(
                    data_path, self.f_struct['import_unesco2ford'])
                fname = 'mapUnescoFord.xlsx'

            elif dt == 'labels':
                subdata_path = os.path.join(
                    data_path, self.f_struct['import_labels'])
                fname = {'labels': 'labels_label.xlsx',
                         'info': 'labels_info.xlsx',
                         'labelhistory': 'labelhistory.xlsx'}

            self.DM.importData(subdata_path, dt, fname, filesdir)

        # Print reports
        for dt in datatypes:
            if dt == 'corpus':
                cols, n_rows = self.DM.getTableInfo('proyectos')
                print("---- Imported projects. There are " +
                      "{} projects in the database.".format(n_rows))
            elif dt == 'lemmas':
                print("---- Lemmas imported and integrated into the database.")
            elif dt == 'c_projects':
                print("---- Coordinated project tag imported and integrated" +
                      " into the database")
            elif dt == 'taxonomy':
                cols, n_rows = self.DM.getTableInfo('taxonomy')
                print("---- Imported taxonomy with {} categories".format(
                    n_rows))
            elif dt == 'unesco2ford':
                cols, n_rows = self.DM.getTableInfo('labels_label')
                print("---- Labels generated from a Unesco->Ford map, for " +
                      "all projects with a UNESCO code")
            elif dt == 'labels':
                cols, n_rows = self.DM.getTableInfo('labels_label')
                print("---- Imported labels. There are " +
                      "{} labels in the database" .format(n_rows))

        return

    def generateCorpusModel(self, modeltype):
        """
        Launches all task related to the processing of the input corpus
        """

        if modeltype == 'detectC':
            self.detectCoord()
        elif modeltype == 'lemmas':
            self.lemmatizeCorpus()
        elif modeltype == 'BoW':
            self.computeBoW()
        elif modeltype == 'WE':
            self.computeWE(dim=300, nepochs=2)
        return

    def detectCoord(self):

        # #############
        # Read projects
        print('---- Reading source data.')
        df = self.DM.readDBtable('proyectos',
                                 selectOptions='Referencia, Titulo, Resumen')
        # By default, we assign each project to itself
        refs = self.DM.readDBtable('proyectos', selectOptions='Referencia')

        # ###########################
        # Detect coordinated projects
        print('---- Detecting coordinated projects.')
        CA = FCCorpusAnalyzer()
        refcoord, coord_list2 = CA.detectCoord(df, refs)

        # ########################
        # Save results to database

        # Save to DB
        print('---- Saving results')
        if 'GroupRef' not in self.DM.getColumnNames('proyectos'):
            self.DM.addTableColumn('proyectos', 'GroupRef', 'TEXT')
        self.DM.setField('proyectos', 'Referencia', 'GroupRef', refcoord)

        # Export to file
        pcoords_fpath = os.path.join(
            self.path2project, self.f_struct['export'],
            'coordinated_projects.pkl')
        with open(pcoords_fpath, 'wb') as fout:
            pickle.dump(coord_list2, fout)

        return

    def lemmatizeCorpus(self, chunksize=100):
        """
        Computes a lemmatized corpus by applying basic NLP tools to all
        documents in the corpus database.
        """

        # #########
        # Read data

        # Read all relevant data in a single dataframe
        print('---- (1/5) Reading data from DB')
        df = self.DM.readDBtable('proyectos', limit=None,
                                 selectOptions='Referencia, Titulo, Resumen')
        refs = df['Referencia'].values
        abstracts = df['Resumen'].values
        titles = df['Titulo'].values

        # ################
        # Lemmatize titles
        print('---- (2/5) Computing lemmas')
        CA = FCCorpusAnalyzer()
        df_lemas_tit, df_lemas_abs = CA.computeLemmas(self.cf, refs,
                                                      abstracts, titles)

        # ######################
        # Save lemmatized corpus

        # Prepare DB tables
        if self.cf.get('DB', 'DB_CONNECTOR') == 'mysql':
            self.DM.addTableColumn('proyectos', 'Titulo_lang',
                                   'VARCHAR(5) CHARACTER SET utf8')
            self.DM.addTableColumn('proyectos', 'Resumen_lang',
                                   'VARCHAR(5) CHARACTER SET utf8')
        else:
            self.DM.addTableColumn('proyectos', 'Titulo_lang', 'TEXT')
            self.DM.addTableColumn('proyectos', 'Resumen_lang', 'TEXT')
        self.DM.addTableColumn('proyectos', 'Titulo_lemas', 'TEXT')
        self.DM.addTableColumn('proyectos', 'Resumen_lemas', 'TEXT')

        # Save tables into DB
        print('---- (3/5) Saving lemmatized titles into DB')
        self.DM.upsert('proyectos', 'Referencia', df_lemas_tit)
        print('---- (4/5) Saving lemmatized abstracts into DB')
        self.DM.upsert('proyectos', 'Referencia', df_lemas_abs)

        # Export results to file
        print("---- (5/5) Exporting results to file")
        opt = ('Referencia, Resumen_lang, Resumen_lemas, Titulo_lang, ' +
               'Titulo_lemas')
        # Read fields computed by the lemmatizer
        df = self.DM.readDBtable('proyectos', selectOptions=opt)
        out_path = os.path.join(self.path2project, self.f_struct['export'],
                                'corpus_lemmas.xlsx')
        df.to_excel(out_path)

        return

    def computeBoW(self, verbose=True):
        """
        Computes the BoW for a given corpu
        """

        # ##############################################
        # Read all docs from DB with abstract in Spanish
        df = self.DM.readDBtable('proyectos', selectOptions=None,
                                 filterOptions="Resumen_lang='es'")

        # ############################
        # Compute BoW and vocabularies
        CA = FCCorpusAnalyzer()
        Xtfidf, vocab, inv_vocab = CA.computeBow(
            df, min_df=self.min_df, title_mul=self.title_mul, verbose=True)

        # ########
        # Save BoW
        path2bow = os.path.join(self.path2project, self.f_struct['models'],
                                self.f_struct['bow'], 'Xtfidf.pkl')
        with open(path2bow, 'wb') as f:
            pickle.dump([Xtfidf, vocab, inv_vocab], f)

        return

    def computeWE(self, dim=300, nepochs=10, verbose=True):
        """
        Computes the BoW for a given corpu
        """

        # ##############################################
        # Read all docs from DB with abstract in Spanish
        df = self.DM.readDBtable('proyectos', selectOptions=None,
                                 filterOptions="Resumen_lang='es'")

        # ##################
        # Compute embeddings
        we = WE(self.path2project, dim, nepochs, self.f_struct_cls)
        we.obtain_sentences(df)
        print('---- Training WE with {} epochs...'.format(nepochs))
        we.train()

        # we.obtain_wedict()
        # Saving the we dict
        # we.save_wedict(
        #     'wedict_dim_' + str(dim) + '_nepochs_' + str(nepochs) + '.pkl')

        # ################
        # Saving WE object
        filename = os.path.join(
            self.path2project, self.f_struct_cls['we'],
            'we_dim_{0}_nepochs_{1}.pkl'.format(dim, nepochs))
        with open(filename, 'wb') as f:
            pickle.dump(we, f)

        print('---- End of WE training')

        return

    def generateCategoryModel(self, cat_type):

        if cat_type == 'labeler':

            self._getLabels()

        elif cat_type == 'showLabels':

            self.DM.showLabels()

        elif cat_type == 'testU2F':

            self._compareLabels('Unesco2Ford', sep=',')

        elif cat_type == 'labelStats':

            self._showLabelStats('manual', sep='++')
            self._showLabelStats('Unesco2Ford', sep=',')
            self._showLabelStats('FORDclassif', sep='+')

        elif cat_type == 'labelsPerCall':

            self._showLabelsPerCall()

        elif cat_type == 'lemmatizeDefs':

            self._lemmatizeDefs()

        elif cat_type == 'expand':

            self._expandCategories()

        return

    def _getLabels(self):

        def text2label(ref):
            """
            Returns the text to show in the labeling application in order to
            label the item with reference ref.
            """

            def todisplay(caps_str):
                """
                This function capitalizes the first letter of each sentence
                in caps_str
                """
                sentence = caps_str.lower().split('.')
                sentence = [el.strip().capitalize() for el in sentence]

                return '. '.join(sentence)

            df = self.DM.readDBtable('proyectos', limit=None,
                                     selectOptions='Titulo, Resumen',
                                     filterOptions="Referencia='" + ref + "'",
                                     orderOptions=None)

            text = (todisplay(df['Titulo'].values[0]) + '\n\n' +
                    todisplay(df['Resumen'].values[0]))

            return text

        # Launch labelling application
        labelfactory.run_labeler(
            project_path=self.f_struct['path2labels'], url=None,
            transfer_mode='expand', user=None, export_labels=None, num_urls=5,
            type_al='random', ref_class=None, alth=0, p_al=0, p_relabel=0,
            tourneysize=40, text2label=text2label)

        return

    def _compareLabels(self, label_field, sep=','):
        """
        Computes the confusion matrix for the input labels with respect to the
        true labels taken from the human labeling tool

        Args:
            label_field: Columns in DB 'proyectos' containing the labels to
                         evaluate. Available options are:
                            - Unesco2Ford
                            - FORDclassif
            sep: Separator used to separate categories in the DB
        """

        # ##################
        # Read manual labels
        print('---- Testing ' + label_field + 'labels against manual labels:')
        df_labels = self.DM.readDBtable(
            'labels_info', selectOptions='Referencia, labelstr, userId')
        self.categories = ast.literal_eval(self.cf.get('CATEGORIES',
                                                       'categories'))

        df_projs = self.DM.readDBtable(
            'proyectos', selectOptions='Referencia, Titulo, Resumen, ' +
            'Unesco2Ford, FORDclassif')

        # ##############
        # Compare labels
        TA = FCTaxonomyAnalyzer(
            categories=self.categories, path2project=self.path2project,
            f_struct_cls=self.f_struct_cls)

        Er_Order, Er_NoOrder = TA.compareLabels(
            df_labels, df_projs, label_field, sep, cat_model=self.cat_model)

        single_u2f = (self.cat_model == 'multiclass')
        df_out = TA.saveLabelPairs(df_labels, df_projs, single_u2f)
        fpath = os.path.join(
            self.path2project, self.f_struct['output'], 'samples2test.xls')
        df_out.to_excel(fpath)
        # Save label pairs in xls file for manual exploration

        print('---- ---- Order-dependent error: {}'.format(Er_Order))
        print('---- ---- Order-independent error: {}'.format(Er_NoOrder))

        return

    def _showLabelStats(self, label_field, sep=','):
        """
        Computes some statistics about the labels contained in labels_field

        Args:
            label_field: Columns in DB 'proyectos' containing the labels to
                         evaluate. Available options are:
                            - Unesco2Ford
                            - FORDclassif
            sep: Separator used to separate categories in the DB
        """

        # ##################
        # Read manual labels
        print('---- Some statistics about ' + label_field)

        if label_field == 'manual':
            df = self.DM.readDBtable('labels_info')
            labelstrs = df['labelstr'].values
        else:
            df = self.DM.readDBtable(
                'proyectos', selectOptions='Referencia, ' + label_field)
            labelstrs = df[label_field].values

        # Remove samples without class labels
        idx = [i for i, x in enumerate(labelstrs)
               if x is not None]
        labelstrs_clean = [labelstrs[i] for i in idx]

        # Extract labels:
        labels = [x.split(sep) for x in labelstrs_clean]

        lab_1st = [x[0] for x in labels]
        filename = 'hist_' + label_field + '_1st.png'
        msg = '1st class distribution of {} labels'.format(label_field)
        self.drawLabelDistrib(lab_1st, filename, msg)

        lab_flat = [item for x in labels for item in x]
        filename = 'hist_' + label_field + '_flat.png'
        msg = 'Class distribution of {} labels'.format(label_field)
        self.drawLabelDistrib(lab_flat, filename, msg)

        # Plot the distribution of the number of labels per project
        h_labels = [len(x) for x in labels]
        plt.figure(figsize=(5, 4))
        nbins = np.arange(np.max(h_labels)) + 0.5
        n, bins, patches = plt.hist(
            x=h_labels, bins=nbins, align='mid', rwidth=0.9)
        plt.grid(axis='y', zorder=0)
        plt.xlabel('Number of {} labels'.format(label_field))
        plt.ylabel('Number of projects')
        plt.title('Distribution of the number of labels per project')
        f_path = os.path.join(self.path2project, self.f_struct_cls['figures'],
                              'n' + filename)
        plt.savefig(f_path)

        # Print average an maximum number of labels.
        print(('---- ---- Average number of {} labels per project: {}'
               ).format(label_field, np.mean(h_labels)))
        print(('---- ---- Maximum number of {} labels in a single project: {}'
               ).format(label_field, np.max(h_labels)))

        print('---- Label statistics saved in file ' + filename)

    def _showLabelsPerCall(self):
        """
        Computes some statistics about the labels, call by call

        Args:
        """

        # ##################
        # Read manual labels
        label_field = 'Unesco2Ford'
        sep = ','

        print(f'---- Some statistics per call. Labels {label_field}')

        if label_field == 'manual':
            df = self.DM.readDBtable('labels_info')
            labelstrs = df['labelstr'].values
        else:
            df = self.DM.readDBtable(
                'proyectos', selectOptions='Referencia, TConv, ' + label_field)
            labelstrs = df[label_field].values
            convs_all = df['TConv'].values

        # Remove samples without class labels
        idx = [i for i, x in enumerate(labelstrs)
               if x is not None]
        labelstrs_clean = [labelstrs[i] for i in idx]
        convs_clean = [convs_all[i] for i in idx]

        # Extract labels:
        labels = [x.split(sep) for x in labelstrs_clean]

        # Plot the distribution of the number of labels per project
        h_labels = [len(x) for x in labels]

        n_Conv = Counter(convs_clean)

        df_conv = self.DM.readDBtable('convocatorias')
        convs, n_projs = zip(*n_Conv.items())

        # Per conv processing
        n_projs_1label = []
        for conv in convs:
            n_projs_1label.append(len([1 for x in zip(convs_clean, h_labels)
                                       if x[0] == conv and x[1] == 1]))

        d = {'Convocatoria': convs,
             'No. de proyectos': n_projs,
             'No. proys. 1 cat': n_projs_1label}

        df = pd.DataFrame(data=d)
        fpath = os.path.join(self.path2project, self.f_struct['output'],
                             'proys_convs.xlsx')
        df.to_excel(fpath)

        return

    def drawLabelDistrib(self, labs, filename, msg):
        '''
        Plots a histogram of the labels in labs, and store them in a file
        '''

        h_lab1st = dict(Counter(labs))

        self.categories = ast.literal_eval(self.cf.get('CATEGORIES',
                                                       'categories'))

        h = list(h_lab1st.items())
        for x in self.categories:
            if x not in h_lab1st and not x.endswith('_'):
                h.append((x, 0))

        #
        h_sorted = sorted(h, key=lambda x: x[1])
        x_sorted = [el[0] for el in h_sorted]
        y_sorted = [el[1] for el in h_sorted]

        # Example data
        x_pos = np.arange(len(x_sorted))

        plt.figure(figsize=(15, 12))
        plt.barh(x_pos, y_sorted, align='center', alpha=0.4)
        plt.yticks(x_pos, x_sorted)
        plt.xlabel('Number of samples')
        plt.title(msg)

        f_path = os.path.join(self.path2project, self.f_struct_cls['figures'],
                              filename)
        plt.savefig(f_path)

        return

    def _lemmatizeDefs(self):
        """
        Computes a lemmatized set of definitions for the given categories
        """

        # #########
        # Read data

        df = self.DM.readDBtable('taxonomy', limit=None,
                                 selectOptions='Reference, Name, Definition')

        # #########
        # Lemmatize

        print('---- Lemmatizing category definitions:')
        TA = FCTaxonomyAnalyzer()
        df_lemas = TA.lemmatizeDefs(self.cf, df)

        # # Lemmatize the concatenation of Name and Definition columns
        # lm = Lemmatizer(self.cf)
        # # str(x[2]) is used because some cell maye be NoneType
        # data = list(df.values)
        # lematizedRes = list(map(lambda x: (x[0], lm.processESstr(
        #     x[1] + ' ' + str(x[2]), keepsentence=True, removenumbers=True)),
        #     data))
        # df_lemas = pd.DataFrame(lematizedRes,
        #                         columns=['Reference', 'Def_lemmas'])

        # ############
        # Save results

        print('---- Saving lemmatized definitions into database:')
        if 'Def_lemmas' in self.DM.getColumnNames('taxonomy'):
            print('---- WARNING: Column Def_lemmas already exists in the ' +
                  'taxonomy table. Current content will be overwritten.')
        else:
            self.DM.addTableColumn('taxonomy', 'Def_lemmas', 'TEXT')
        self.DM.upsert('taxonomy', 'Reference', df_lemas)

        return

    def _expandCategories(self):
        '''
        Expands category definitions using text taken from the Wikipedia.

        It assumes that the taxonomy table contains pointers to the
        wikipedia pages defining each category
        '''

        # #########
        # Read data
        df = self.DM.readDBtable('taxonomy')

        # #################
        # Expand categories
        TA = FCTaxonomyAnalyzer(cf=self.cf)
        df_wdefs = TA.expandCategories(df)

        # ######
        # Saving
        if 'WikiDefs' in self.DM.getColumnNames('taxonomy'):
            print('---- WARNING: Column WikiDefs already exists in the ' +
                  'taxonomy table.')
            print('              Current content will be overwritten.')
        else:
            self.DM.addTableColumn('taxonomy', 'WikiDefs', 'TEXT')
        self.DM.upsert('taxonomy', 'Reference', df_wdefs)

        return

    def optimizeClassifiers(self, option):
        """
        Optimize classifier.
        Fits a collection of classifiers and selects the best one by cross
        validation

        Args:

            option : Data preparation or type of classifier
                'partition': Prepare data for cross validations: makes the
                       data partitions for training and evaluation.
                'uns': Unsupervised
                'sup': Supervised
                'ssc': Semi-supervised
        """

        if option == 'partition':
            self._partition()
        elif option == 'sup':
            self._optimizeSupClassifiers()
        elif option == 'uns':
            print('---- Option not available yet')
        return

    def _partition(self):

        # #########
        # Read data

        # Read projects from database
        print('---- Reading projects')
        df = self.DM.readDBtable('proyectos')
        print(df.columns)
        print(df.count()[0])

        # Creating subfolder structure
        self._make_subfolders(self.path2project, self.f_struct_cls)

        # ##############
        # Make partition

        SC = FCSupClassifier(
            f_struct_cls=self.f_struct_cls, path2project=self.path2project,
            cat_model=self.cat_model, min_df=self.min_df,
            title_mul=self.title_mul)
        SC.partition(df, seed_missing_categories=True,
                     fraction_train=0.8, Nfold=10)

        return

    def _optimizeSupClassifiers(self):

        # #########
        # Read data

        # We can use a different subset of classifiers to find the best, e.g.
        #     classifiers2compare = ['LR', 'MNB']
        # finds the best classifiers among the options LR and MNB, even when
        # other models are available
        # classifier_types = ['LR', 'MNB', 'BNB', 'LSVM', 'RF', 'SVMpoly',
        #                     'MLP']
        classifier_types = ['LR']
        classifiers2compare = ['LR']

        # ####################
        # Optimize classifiers
        print('---- Optimizing classifiers, please wait...')
        SC = FCSupClassifier(
            f_struct_cls=self.f_struct_cls, path2project=self.path2project,
            cat_model=self.cat_model, min_df=self.min_df,
            title_mul=self.title_mul)

        SC.optimizeSupClassifiers(classifier_types, classifiers2compare)

        return

    def evaluateClassifier(self, option):
        """
        Train classifier.

        Args:

            option : Data preparation or type of classifier
                'partition': Prepare data for cross validations: makes the
                       data partitions for training and evaluation.
                'uns': Unsupervised
                'sup': Supervised
                'ssc': Semi-supervised
        """

        if option == 'sup':
            self._evaluateSupClassifier()
        elif option == 'uns':
            self._evaluateUnsupClassifier()
        elif option == 'man_sup':
            self._compareLabels('FORDclassif', sep='+')
        return

    def _evaluateSupClassifier(self):
        '''
        Author:  Angel Navia Vázquez
        April 2018
        '''

        # ###############################
        # Evaluate Supervised Classifiers

        # Configurable parameters
        p = 0.9   # This parameter should also be selected by crossvalidation
        alpha = 0.9  # It balances Jackard & RBO, alpha=1.0, only Jackard
        option = 'maximum_response'
        th_values = [0.999, 0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6,
                     0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1,
                     0.05, 0, -0.05, -0.1, -0.15]

        self.categories = ast.literal_eval(self.cf.get('CATEGORIES',
                                                       'categories'))
        SC = FCSupClassifier(categories=self.categories,
                             f_struct_cls=self.f_struct_cls,
                             path2project=self.path2project,
                             cat_model=self.cat_model, min_df=self.min_df)
        df_labels = SC.evaluateSupClassifier(p, alpha, option, th_values)

        # ############
        # Save results

        # Write to Database
        self.DM.addTableColumn('proyectos', 'FORDclassif', 'TEXT')
        self.DM.addTableColumn('proyectos', 'FORDweights', 'TEXT')
        self.DM.upsert('proyectos', 'Referencia', df_labels)

        return

    def _evaluateUnsupClassifier(self):
        '''
        Author:  Angel Navia Vázquez
        April 2018
        '''

        # #################################
        # Evaluate Unsupervised Classifiers

        # Configurable parameters
        threshold = 0.9

        self.categories = ast.literal_eval(self.cf.get('CATEGORIES',
                                                       'categories'))

        SC = FCUnsupClassifier(categories=self.categories,
                               f_struct_cls=self.f_struct_cls,
                               path2project=self.path2project)
        SC.evaluateUnsupClassifier(threshold)

        # ############
        # Save results

        # No results are saved in the DB because the final selected classifier
        # was the supervised one.

        return

    def computePredictions(self):

        # Set the 'output' directory to save the classified predictions
        target_path = os.path.join(self.path2project, self.f_struct['output'])
        # Call the predictor.
        self._computePredictions(target_path=target_path)
        return

    def _computePredictions(self, source_path=None, target_path=None,
                            lang='EN'):

        # ##############################
        # Select input and outputs paths

        # Messages. We include the option to write messages in Spanish
        # because the final classification will be used by non-experts
        if lang == 'EN':
            msg_sf = ('---- Select the source file with the projects to ' +
                      'be classified:')
            msg_g0 = 'Select the Excel file containing the project data'
            msg_tf = '---- Select the target folder:'
            msg_g1 = 'Select the folder to save the classified projects'
            msg_err = ('---- Error: File could not be loaded. The source ' +
                       'file must have extension ".xls" or ".xlsx"')
            msg_out = '----- FORD categories saved in: {}'

        else:
            msg_sf = ('---- Seleccione el fichero fuente con los proyectos ' +
                      'a clasificar:')
            msg_g0 = ('Seleccione el fichero Excel que contiene los ' +
                      'proyectos a clasificar')
            msg_tf = '---- Selecciones la carpeta de salida'
            msg_g1 = ('Seleccione la carpeta donde se guardarán los ' +
                      'resultados de la clasificación')
            msg_err = ('---- Error: no se ha podido cargar el fichero. El ' +
                       'fichero fuente debe tener extensión ".xls" o ".xlsx"')
            msg_out = '---- Clasificaciones FORD guardadas en {}'

        # Input path
        if source_path is None:
            # Select source file from a GUI
            print(msg_sf)
            root = tk.Tk()    # Create the window interface
            root.withdraw()   # This is to avoid showing the tk window
            # Show the filedialog window (only)
            root.overrideredirect(True)
            root.geometry('0x0+0+0')
            root.deiconify()
            root.lift()
            root.focus_force()
            source_path = filedialog.askopenfilename(initialdir=os.getcwd(),
                                                     title=msg_g0)
            root.update()     # CLose the filedialog window
            root.destroy()    # Destroy the tkinter object.

        # Output path
        if target_path is None:
            # Select source file from a GUI
            print(msg_tf)
            root = tk.Tk()    # Create the window interface
            root.withdraw()   # This is to avoid showing the tk window
            # Show the filedialog window (only)
            root.overrideredirect(True)
            root.geometry('0x0+0+0')
            root.deiconify()
            root.lift()
            root.focus_force()
            target_path = filedialog.askdirectory(
                initialdir=os.path.dirname(source_path), title=msg_g1)
            root.update()     # CLose the filedialog window
            root.destroy()    # Destroy the tkinter object.

        # #########
        # Read data

        # Complete subfolder structure
        # self._make_subfolders(self.path2project, self.f_struct_cls)

        # Process only excel files not starting with tilde character
        if (source_path.endswith('.xls') or source_path.endswith('.xlsx')):
            # Read all data. But, only Referencia, Titulo and Resumen are
            # needed
            df0 = pd.read_excel(source_path).replace(np.nan, '', regex=True)
        else:
            print(msg_err)
            df0 = None

        # #################
        # Classify data

        start = time()
        if df0 is not None:
            SC = FCSupClassifier(
                categories=self.categories, cf=self.cf,
                f_struct_cls=self.f_struct_cls, path2project=self.path2project,
                cat_model=self.cat_model, min_df=self.min_df,
                title_mul=self.title_mul)
            df_preds, df_preds_all = SC.computeClassifierPredictions(df0)

        print("Solved in {} seconds.".format(time()-start))

        # #######################################
        # Read FORD codes and full category names

        # Load dataframe with taxonomy data
        path2taxonomy = os.path.join(
            self.f_struct['import'], self.f_struct['import_taxonomy'],
            self.taxonomy_fname)
        df_t = pd.read_excel(path2taxonomy, dtype={'Code': str})
        # Replace nan's by None's
        df_t = df_t.where((pd.notnull(df_t)), None)
        # Index by the abbreviated category names
        df_t.set_index(['Reference'], inplace=True)

        # ########################################################
        # Fomat table of all categories and weights (df_preds_all)

        def format_weights(x):

            # Sort categories by weight
            x_sorted = sorted(x, key=lambda kv: -kv[1])

            # Replace abbreviated category names by full names
            x_scaled = [(df_t.loc[z[0], 'Name'],
                         (z[1]+1)/2) for z in x_sorted]

            # Normalize weight
            tot = np.sum([z[1] for z in x_scaled])
            x_rescaled = [(z[0], z[1]*100/tot) for z in x_scaled]

            # Convert to list of strings
            y = list(sum(
                [(z[0], '{:4.2f}'.format(z[1])) for z in x_rescaled], ()))

            return y

        # Generate new column with ordered lists of categories and weights
        df_preds_all['Clasificacion'] = list(map(
            format_weights, df_preds_all['Clasificacion']))
        # Split column with categories and weight into a single
        df_preds_all = df_preds_all.assign(
            **pd.DataFrame(df_preds_all.Clasificacion.values.tolist(
                )).add_prefix('Col_'))

        #  Rename columns
        cols = df_preds_all.columns
        cols = [x for x in cols if x.startswith('Col_')]
        newcols = ['Peso ' + str(eval(x[-1])//2) if eval(x[-1]) % 2 else
                   'Categoria ' + str(eval(x[-1])//2) for x in cols]
        df_preds_all.rename(index=str, columns=dict(zip(cols, newcols)),
                            inplace=True)

        # Remove unnecessary columns
        df_preds_all.drop(columns=['Clasificacion'], inplace=True)

        # ####################################################
        # Generate table of assigned categories (df_preds_cat)

        def expand_names(x):
            # Replace abbreviated category names by full names
            y = [df_t.loc[z, 'Name'] if z != '' else '' for z in x]
            return y

        # Format table of categories and weights
        df_preds['Clasificacion'] = list(map(
            expand_names, df_preds['Clasificacion']))

        # Split column with categories and weight into a single
        df_preds = df_preds.assign(
            **pd.DataFrame(df_preds.Clasificacion.values.tolist(
                )).add_prefix('Cat_'))
        # Remove unnecessary columns
        df_preds_cat = df_preds.drop(columns=['Clasificacion'])

        # Split column with categories and weight into a single
        df_preds_1st = df_preds_cat.copy()
        df_preds_1st = df_preds_1st[['Referencia', 'Cat_0']]
        df_t.set_index(['Name'], inplace=True)

        df_preds_1st.loc[:, 'Code'] = list(map(
            lambda x: df_t.loc[x, 'Code'] if x != '' else '',
            df_preds_1st['Cat_0']))

        # ############
        # Save results

        # This encodings are selected in order to make the csv files readable
        # in excel for mac and windows.
        if platform.system() == 'Darwin':
            enc = 'mac_roman'
        elif platform.system() == 'Linux':
            enc = 'utf-8'
        else:
            enc = 'cp1252'

        if df0 is not None:

            # Export to file
            out_fname = os.path.join(target_path, os.path.splitext(
                os.path.basename(source_path))[0] + '_1aCategoria.csv')
            df_preds_1st.to_csv(out_fname, encoding=enc)

            # Export to file
            out_fname = os.path.join(target_path, os.path.splitext(
                os.path.basename(source_path))[0] + '_Categorias.csv')
            df_preds_cat.to_csv(out_fname, encoding=enc)

            # Export to file
            out_fname = os.path.join(target_path, os.path.splitext(
                os.path.basename(source_path))[0] + '_pesos.csv')
            df_preds_all.to_csv(out_fname, encoding=enc)

            print(msg_out.format(out_fname))

        return

    def exportData(self, datatypes='all'):
        """
        Export data from the database.
        """

        path = addSlash(os.path.join(
            self.path2project, self.f_struct['export']))

        if datatypes is 'all':
            datatypes = ['lemmas', 'c_projects', 'labels', 'taxonomy']
        elif type(datatypes) is str:
            datatypes = [datatypes]

        for dt in datatypes:

            if dt == 'lemmas':
                self.DM.exportTable(
                    'proyectos', 'xlsx', path, 'corpus_lemmas.xlsx',
                    cols=('Referencia, Resumen_lang, Resumen_lemas, ' +
                          'Titulo_lang, Titulo_lemas'))
                print('---- Lemmas exported to {}lemmas.xlsx.'.format(path))

            elif dt == 'c_projects':
                self.DM.exportTable(
                    'proyectos', 'xlsx', path, 'c_projects.xlsx',
                    cols=('Referencia, GroupRef'))
                print('---- Coordinated project assignments exported to ' +
                      ' {}c_projects.xlsx.'.format(path))

            elif dt == 'labels':
                # Export label metadata
                self.DM.exportTable(
                    'labels_label', 'xlsx', path, 'labels_label.xlsx')
                # Export label metadata
                self.DM.exportTable(
                    'labels_info', 'xlsx', path, 'labels_info.xlsx')
                # Export label metadata
                self.DM.exportTable(
                    'labelhistory', 'xlsx', path, 'labelhistory.xlsx')
                print('---- All human labels exported to files ' +
                      'labels_label.xlsx, labels_info.xlsx and ' +
                      'labelhistory.xlsx in folder {}'.format(path))

            elif dt == 'taxonomy':
                self.DM.exportTable('taxonomy', 'xlsx', path, 'taxonomy.xlsx')
                print('---- Taxonomy exported to {}taxonomy.xlsx.'.format(
                    path))

        return
