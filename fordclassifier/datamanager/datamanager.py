"""
This class provides functionality for:

* importing project data provided by MINECO
* reading specific fields (with the possibility to filter by field values)
* storing calculated values in the dataset

Created on May 11 2018

@author: Jerónimo Arenas García

"""

from __future__ import print_function    # For python 2 copmatibility
import os
import pandas as pd
import numpy as np
import copy
import ast
from tabulate import tabulate

# Local imports
from fordclassifier.datamanager import base_dm_sql

# Temporary imports
import ipdb


class DataManager(base_dm_sql.BaseDMsql):
    """
    Text.
    """

    def __init__(self, path2project, cf):
        """
        Initializes a DataManager object

        Args:
            path2project :Path to the project folder
            cf           :A ConfigParser object
        """

        # Store paths to the main project folders and files
        self._cf = cf
        self._path2project = copy.copy(path2project)

        db_name = cf.get('DB', 'DB_NAME')
        db_connector = cf.get('DB', 'DB_CONNECTOR')
        db_server = cf.get('DB', 'DB_SERVER')
        db_user = cf.get('DB', 'DB_USER')
        db_password = cf.get('DB', 'DB_PASSWORD')

        super(DataManager, self).__init__(
            db_name, db_connector, path2project, db_server, db_user,
            db_password)

        # List of categories.
        self.categories = ast.literal_eval(cf.get('CATEGORIES', 'categories'))

    def createDBtables(self, tables=None):
        """
        Create DB table structure
        Args:
            tables: If string, name of the table to reset.
                    If list, list of tables to reset
                    If None (default), all tables are created.
        """

        # If table is string, convert to list.
        if type(tables) is str:
            tables = [tables]

        # Fit characters to the allowed format
        if self.connector == 'mysql':
            # We need to enforze utf8 for mysql
            fmt = ' CHARACTER SET utf8'
        else:
            # For sqlite 3 no formatting is required.
            fmt = ''

        # #######################
        # Tables for project data
        if tables is None or 'proyectos' in tables:
            # Table of projects
            print("---- Creating table: proyectos")
            sql_cmd = """CREATE TABLE proyectos(
                        Cconv INTEGER,
                        TConv TEXT{0},
                        Referencia VARCHAR(30){0} PRIMARY KEY,
                        Titulo TEXT{0},
                        Resumen TEXT{0},
                        UNESCO_cd TEXT{0}
                        )""".format(fmt)
            self._c.execute(sql_cmd)

        if tables is None or 'UNESCO_codes' in tables:
            # Table for UNESCO data
            print("---- Creating table: UNESCO_codes")
            sql_cmd = """CREATE TABLE UNESCO_codes(
                        UNESCO_cd INTEGER,
                        Descr TEXT{0}
                        )""".format(fmt)
            self._c.execute(sql_cmd)

        if tables is None or 'convocatorias' in tables:
            # Table for MINECO project calls
            print("---- Creating table: convocatorias")
            sql_cmd = """CREATE TABLE convocatorias(
                        year INTEGER,
                        Cconv INTEGER,
                        Ctitulo TEXT{0}
                        )""".format(fmt)
            self._c.execute(sql_cmd)

        # ###########################
        # Tables to define categories
        if tables is None or 'taxonomy' in tables:
            print("---- Creating table: taxonomy")
            sql_cmd = """CREATE TABLE taxonomy(
                        Reference VARCHAR(30){0} PRIMARY KEY,
                        Name TEXT{0},
                        Code TEXT{0},
                        Parent TEXT{0},
                        Definition TEXT{0},
                        WikiRef TEXT{0}
                        )""".format(fmt)
            self._c.execute(sql_cmd)

        # ######################
        # Tables for predictions
        if tables is None or 'predictions' in tables:
            # Table for the predictions
            print("---- Creating table: predictions")

            # The name 'url' has historical reasons, but it keeps the meaning:
            # it is aimed to specify a kind of uniform resource location.
            sql_cmd = ("""CREATE TABLE predictions(
                        Referencia VARCHAR(30){0} PRIMARY KEY,
                        url TEXT{0}
                        )""").format(fmt)
            self._c.execute(sql_cmd)

            # One prediction column per category
            for cat in self.categories:
                self.addTableColumn('predictions', cat, 'DOUBLE')

        # #################
        # Tables for Labels
        if tables is None or 'labels_label' in tables:
            # Table for the label values
            print("---- Creating table: labels_label")
            sql_cmd = """CREATE TABLE labels_label(
                        Referencia VARCHAR(30){0} PRIMARY KEY
                        )""".format(fmt)
            self._c.execute(sql_cmd)
            # One label column per category
            for cat in self.categories:
                self.addTableColumn('labels_label', cat, 'INTEGER')

        if tables is None or 'labels_info' in tables:
            # Table for the label metadata
            print("---- Creating table: labels_info")
            sql_cmd = """CREATE TABLE labels_info(
                        Referencia VARCHAR(30){0} PRIMARY KEY,
                        marker INTEGER,
                        relabel INTEGER,
                        weight INTEGER,
                        userId TEXT{0},
                        datestr TEXT{0},
                        labelstr TEXT{0}
                        )""".format(fmt)
            self._c.execute(sql_cmd)

        if tables is None or 'labelhistory' in tables:
            # Table for the historic record of labelling events
            print("---- Creating table: labelhistory")
            sql_cmd = """CREATE TABLE labelhistory(
                        Referencia VARCHAR(30){0} PRIMARY KEY,
                        datestr TEXT{0},
                        url TEXT{0},
                        marker INTEGER,
                        relabel INTEGER,
                        label TEXT{0},
                        userId TEXT{0}
                        )""".format(fmt)
            self._c.execute(sql_cmd)

        # Commit changes
        self._conn.commit()

    def importData(self, data_path, datatypes, fname, filesdir=None):
        """
        Reads the original data sources and integrates them in the project data
        structure.

        This method is just a gate to several importing methods. It is useful
        to specify the importer through the datatypes variable form the main
        file

        Args:

            datatypes: Type of data ('corpus'|'taxonomy'|'labels')

        """

        if datatypes == 'corpus':
            self._importCorpus(data_path)
            # self._importCorpus(data_path, fname, filesdir)
        elif datatypes == 'lemmas':
            self._importLemmas(data_path, fname)
        elif datatypes == 'c_projects':
            self._importCProjects(data_path, fname)
        elif datatypes == 'taxonomy':
            self._importTaxonomy(data_path, fname)
        elif datatypes == 'unesco2ford':
            self._importUnesco2Ford(data_path, fname)
        elif datatypes == 'labels':
            self._importLabels(data_path, fname)
        else:
            exit('---- ERROR: Data not imported. Unknown data type.')

        return

    def _importCorpus(self, data_path='import',
                      callfilename='Convocatorias.xls',
                      filesdir='projectfiles'):
        """
        Reads the text corpus in the original format and integrates it in the
        project data structure

        Args:

            data_path: Path to the data source
        """

        # Load dataframe with call information (table "convocatorias")
        # Since no sorting is necessary, we simply save everything in table
        # convocatorias
        df = pd.read_excel(os.path.join(data_path, callfilename))
        self.insertInTable(
            'convocatorias', self.getColumnNames('convocatorias'), df.values)

        # Next, we need to load and save all data corresponding to all
        # available projects
        projectdir = os.path.join(data_path, 'projectfiles')
        df = pd.DataFrame()

        print('Processing all excel files in directory', projectdir)
        for root, dirs, files in os.walk(projectdir):
            for file in files:
                # Process only excel files not starting with tilde character
                if file.endswith('.xls') or file.endswith('.xlsx'):
                    if not file.startswith('~'):
                        tmp_df = pd.read_excel(
                            os.path.join(root, file)).replace(np.nan, '',
                                                              regex=True)
                        df = df.append(tmp_df, ignore_index=True)

        # We remove duplicates keeping only one entry per reference
        # !!!!!!!!! We need to check why there are so many duplicates
        # corresponding to different UNESCO codes
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # This line would keep only first project with given reference
        # dfnoduplicates = df.drop_duplicates(subset='Referencia')

        # Instead, we will merge all UNESCO codes for that, we convert first
        # that field to string type
        df['Código UNESCO'] = df['Código UNESCO'].astype(str)
        dfnoduplicates = df.groupby(
            df['Referencia'], sort=False, as_index=False).aggregate(
                {'Cconv': 'first', 'Conv. Título': 'first',
                 'Referencia': 'first', 'Título': 'first', 'Resumen': 'first',
                 'Código UNESCO': lambda x: ','.join(x)})

        dfnoduplicates = dfnoduplicates[[
            'Cconv', 'Conv. Título', 'Referencia', 'Título', 'Resumen',
            'Código UNESCO']]

        self.insertInTable('proyectos', self.getColumnNames('proyectos'),
                           dfnoduplicates.values)

        # Al proyects in table 'proyectos' must be translated to the
        # predictions table, because the labeler takes the project references
        # from this table
        dfpreds = pd.DataFrame()
        dfpreds.loc[:, 'Referencia'] = dfnoduplicates['Referencia']
        dfpreds.loc[:, 'url'] = dfpreds['Referencia']

        self.insertInTable('predictions', ['Referencia', 'url'],
                           dfpreds.values)

        # We insert also UNESCO codes removing duplicates first
        df = df[['Código UNESCO', 'Descripción UNESCO']].drop_duplicates()

        self.insertInTable(
            'UNESCO_codes', self.getColumnNames('UNESCO_codes'), df.values)

        # And commit changes before finishing
        self._conn.commit()

    def displayProjectInfo(self, limit=None, filterOptions=None):
        """
        Provides information to display for each project.

        *** This function is completely specific to the MINECO project ****

        Args:
            tablename    :  Table to read from
            filterOptions:  string with filtering options for the SQL query
                            (e.g., 'WHERE UNESCO_cd=23')
            limit:          The maximum number of records to retrieve

        """

        df = self.readDBtable('proyectos', limit, 'Referencia, Resumen',
                              filterOptions)

        # This function capitalizes the first letter of each sentence
        # Needs to be taken outside this class, to some container for string
        # manipulation funcs
        def todisplay(caps_str):
            sentence = caps_str.lower().split('.')
            sentence = [el.strip().capitalize() for el in sentence]
            return '. '.join(sentence)

        return list(
            map(lambda x: (x[0], x[0]+'\n'+todisplay(x[1])), df.values))

    def showDBview(self):

        title = "= Database {} ====================".format(
            self._cf.get('DB', 'DB_NAME'))

        print("="*len(title))
        print(title)
        print("="*len(title))
        print("")

        for table in self.getTableNames():

            print("---- Table: {}".format(table))
            cols, n_rows = self.getTableInfo(table)
            print("---- ---- Attributes: {}".format(', '.join(cols)))
            print("---- ---- No. of rows: {}".format(n_rows))
            print("")

        return

    def showLabels(self):
        """
        This method shows jointly the manual labesl and those taken from
        Unesco2Ford projects.
        Note that this method is different from those that visualize table
        labels_info, which only containg the manual labels.
        """

        # Read labels
        df_labels = self.readDBtable('labels_info')
        refs = df_labels['Referencia'].tolist()

        # Read unesco2ford labels
        df_projs = self.readDBtable(
            'proyectos',
            selectOptions='Referencia, UNESCO2Ford, UNESCO_cd')
        df_projs = df_projs[df_projs['Referencia'].isin(refs)]

        # Join in a single dataframe the df with the manual labels and that
        # with the Unesco2Ford labels. Only those projects with a manual label
        # are taken into account.
        df_out = df_labels.join(
            df_projs.set_index('Referencia'), on='Referencia')

        print('\n---- LABELS: ')
        txt = tabulate(df_out, headers='keys', tablefmt='psql')
        print(txt)

        return

    def showSample(self):

        # This function is defined also in _getlabels. It should not be
        # defined twice (pending work)
        def todisplay(caps_str):
            """
            This function capitalizes the first letter of each sentence
            in caps_str
            """
            sentence = caps_str.lower().split('.')
            sentence = [el.strip().capitalize() for el in sentence]

            return '. '.join(sentence)

        title = "= Random example from {} ====================".format(
            self._cf.get('DB', 'DB_NAME'))

        print("="*len(title))
        print(title)
        print("="*len(title))
        print("")

        # Read labeled project.
        if self.connector == 'mysql':
            order_opt = 'RAND()'
        else:     # if sqlite
            order_opt = 'RANDOM()'
        df = self.readDBtable('labels_info', limit=1, selectOptions=None,
                              filterOptions=None, orderOptions=order_opt)
        ref = df.loc[0]['Referencia']

        dfProjs = self.readDBtable(
            'proyectos', limit=1, selectOptions=None,
            filterOptions='Referencia="'+ref+'"',
            orderOptions=None)

        print('---- PROJECT TITLE: ')
        print(todisplay(dfProjs['Titulo'].tolist()[0]))

        print('\n---- ABSTRACT: ')
        print(todisplay(dfProjs['Resumen'].tolist()[0]))

        print('\n---- LABELS: ')
        df_labels = pd.DataFrame(columns=[
            'Referencia', 'UNESCO_cd', 'Unesco2Ford', 'userId', 'labelstr',
            'FORDclassif', 'FORDweights'])
        df_labels[['Referencia', 'userId', 'labelstr']] = df[[
            'Referencia', 'userId', 'labelstr']]
        df_labels[['UNESCO_cd']] = dfProjs[['UNESCO_cd']]

        # if 'Unesco2Ford' in dfProjs.columns:
        #     df_labels[['Unesco2Ford']] = dfProjs[['Unesco2Ford']]
        for col in ['Unesco2Ford', 'FORDclassif', 'FORDweights']:
            if col in dfProjs.columns:
                df_labels[[col]] = dfProjs[[col]]
        txt = tabulate(df_labels, headers='keys', tablefmt='psql')
        print(txt)

        return

    def _importLemmas(self, data_path='import', filename='corpus_lemmas.xlsx'):

        # Load lemmas from file
        print('---- ---- Reading lemmas from file')
        df = pd.read_excel(os.path.join(data_path, filename))

        # Add new columns to database if they do not exist
        if self.connector == 'mysql':
            self.addTableColumn('proyectos', 'Resumen_lang', 'VARCHAR(5)')
            self.addTableColumn('proyectos', 'Titulo_lang', 'VARCHAR(5)')
        else:
            self.addTableColumn('proyectos', 'Resumen_lang', 'TEXT')
            self.addTableColumn('proyectos', 'Titulo_lang', 'TEXT')
        self.addTableColumn('proyectos', 'Resumen_lemas', 'TEXT')
        self.addTableColumn('proyectos', 'Titulo_lemas', 'TEXT')

        # Save lemmas in database.
        print('---- ---- Saving lemmas into database')
        self.upsert('proyectos', 'Referencia', df)

        return

    def _importCProjects(self, data_path='import', fname='c_projects.xlsx'):
        '''
        Imports the groups of coordinated projects.
        Each group will be identified by the name of one of its members (the
        first in the list)
        A new column of table proyectos (named GroupRef) is added, assigning
        each project to its corresponding group.
        '''

        # Load lemmas from file
        print('---- ---- Reading coordination data from files')
        df = pd.read_excel(os.path.join(data_path, fname))

        # Add new columns to database if they do not exist
        if 'GroupRef' not in self.getColumnNames('proyectos'):
            self.addTableColumn('proyectos', 'GroupRef', 'TEXT')
        else:
            print('---- ---- Column GroupRef already exist. Current values ' +
                  'will be overwritten.')

        # Save lemmas in database.
        print('---- ---- Saving cordination data into database')
        self.upsert('proyectos', 'Referencia', df)

        # ## OLD VERSION
        # ## This is the old version, based on code by Jeronimo Arenas.
        # ## It is no longer used because imporing from xlsx files exported
        # ## from the database is simpler.
        #
        # # Read the file containing the list of grups of coordinated projects
        # fpath = os.path.join(data_path, filename)
        # with open(fpath, 'rb') as f:
        #     coords = pickle.load(f)

        # # By default, we assign each project to itself
        # refs = self.readDBtable('proyectos', selectOptions='Referencia')
        # ref2coord = dict(zip(refs, refs))

        # # Assign each project in a group to its group representative
        # for group in coords:
        #     for p in group[0]:
        #         # Each group is represented by the first project in the list
        #         ref2coord[p] = group[0][0]
        # refcoord = list(ref2coord.items())

        # # Create column in table if it does not exist and fill it
        # if 'GroupRef' not in self.getColumnNames('proyectos'):
        #     self.addTableColumn('proyectos', 'GroupRef', 'TEXT')
        # self.setField('proyectos', 'Referencia', 'GroupRef', refcoord)

        return

    def _importTaxonomy(self, data_path='MINECOdata', fname='Taxonomy.xlsx'):

        """
        Reads the original taxonomy definition sources and integrates it in the
        project data structure

        Args:

            data_path: Path to the source data
            filename: Name of file containing the Taxonomy information

        """

        # Load dataframe with taxonomy data
        df = pd.read_excel(os.path.join(data_path, fname),
                           dtype={'Code': str})
        # Replace nan's by None's
        df = df.where((pd.notnull(df)), None)

        # Specify column ordering.
        cols = ['Code', 'Name', 'Reference', 'Parent', 'Definition', 'WikiRef']
        self.insertInTable('taxonomy', cols, df.values)

        # And commit changes before finishing
        self._conn.commit()

        return

    def _importUnesco2Ford(self, data_path=None, fname='mapUnescoFord.xlsx'):
        """
        Read labels from an external data source and integrates it in the
        project data structure.

        This method could be required, at least while the labelling application
        was not completely integrated in the data structure.

        Args:

            data_path: Path to the source data

        """

        def from_u2f(x, u2f, code2ref):
            """
            Transforms a string of comma separated UNESCO codes into a string
            of comma separated equivalent FORD codes

            :Args:
                :u2f:       A dictionary mapping UNESCO codes to FORD
                :code2ref:  A dictionary mapping FORD codes into the
                            corresponding reference value in the taxonomy table
            """

            try:

                # Tranform string into list removing repetitions but preserving
                # order
                # list(dict.fromkeys()) is used to take unique elements from
                # the list while preserving the order of appearance
                x0 = list(dict.fromkeys(x.split(',')))

                # All (6-digit) unesco codes in z that are not keys from
                # code2ref are transformed into a 4-digit code
                # x_norm = ([code2ref[u2f[int(u)]] if int(u) in u2f else
                #            code2ref[u2f[int(u) // 100]] for u in x2])

                x1 = [int(u) if int(u) in u2f else int(u) // 100 for u in x0]

                # Map UNESCO values to FORD-reference.
                y0 = [code2ref[u2f[u]] for u in x1]

                # Remove repetitions
                y1 = list(dict.fromkeys(y0))
                z = ','.join(y1)

            except:
                ipdb.set_trace()
                z = ''

            return z

        # Load dataframe with call information (table "convocatorias")
        # Since no sorting is necessary, we simply save everything in table
        # convocatorias
        df = pd.read_excel(
            os.path.join(data_path, fname), index_col='UNESCO_cd',
            dtype={'FORD_cd': str})[['FORD_cd']]

        arguments = list(zip(df.index, df['FORD_cd'].values))

        # This is for backward compatibility: table "FORD_eq" may not exist
        # if the table was created with an old version of this code.
        if 'FORD_eq' not in self.getColumnNames('UNESCO_codes'):
            self.addTableColumn('UNESCO_codes', 'FORD_eq', 'TEXT')

        self.setField('UNESCO_codes', 'UNESCO_cd', 'FORD_eq', arguments)

        # Load projects and UNESCO codes from DB
        dfp = self.readDBtable('proyectos',
                               selectOptions='Referencia, UNESCO_cd')

        # Load taxonomy table
        dft = self.readDBtable('taxonomy', selectOptions='Code, Reference')
        code2ref = dict(dft.values)

        # Transform unesco values to ford
        u2f = dict(arguments)
        fordvalues = [(p, from_u2f(v, u2f, code2ref)) for p, v in dfp.values]

        # Save ford values into a new DB columns
        self.addTableColumn('proyectos', 'Unesco2Ford', 'TEXT')
        self.setField('proyectos', 'Referencia', 'Unesco2Ford', fordvalues)

        return

    def _importLabels(self, data_path='MINECOdata', fname=None):
        """"
        Import a set of manual labels taken from the labelling application.
        Reads label values, label metadata and the label history record from
        the 3 output files of the labeler, and stores them in the corresponding
        tables of the project database.
        """

        # Defaul file names.
        if fname is None:
            fname = {'labels': 'labels_label.xlsx',
                     'info': 'labels_info.xlsx',
                     'labelhistory': 'labelhistory.xlsx'}

        # Import label values
        df = pd.read_excel(os.path.join(data_path, fname['labels']))
        self.upsert('labels_label', 'Referencia', df)

        # Import label values
        df = pd.read_excel(os.path.join(data_path, fname['info']))
        self.upsert('labels_info', 'Referencia', df)

        # Import label values
        df = pd.read_excel(os.path.join(data_path, fname['labelhistory']))
        self.upsert('labelhistory', 'datestr', df)
