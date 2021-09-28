#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python libraries
from __future__ import print_function
import os
import sys
import pandas as pd

import MySQLdb
import sqlite3
from sqlalchemy import create_engine

import ipdb
import numpy as np

# Local imports
from labelfactory.labeling import baseDM

import time

# Services from the project
# sys.path.append(os.getcwd())


class DM_SQL(baseDM.BaseDM):

    """
    DataManager is the class providing read and write facilities to access and
    update the dataset of labels and predictions

    It assumes that data will be stored in SQL databases, though it have some
    facilities to import data from files

    It assumes a SQL database structure with 4 tables:

        preds
        label_info
        label_values
        labelhistory

    (the specific name of these tables is read from the configuratio file)

    The class provides facilities to:

        - Read and write data in a SQL database
        - ?
    """

    def __init__(self, source_type, dest_type, file_info, db_info,
                 categories, parentcat, ref_class, alphabet,
                 compute_wid='yes', unknown_pred=0):

        super(DM_SQL, self).__init__(
            source_type, dest_type, file_info, db_info, categories, parentcat,
            ref_class, alphabet, compute_wid='yes', unknown_pred=0)

        # Subclass-specific variables
        self.db_name = self.db_info['name']
        self.server = self.db_info['server']
        self.user = self.db_info['user']
        self.connector = self.db_info['connector']
        self.password = self.db_info['password']
        self.preds_tablename = self.db_info['preds_tablename']
        self.label_values_tablename = self.db_info['label_values_tablename']
        self.label_info_tablename = self.db_info['label_info_tablename']
        self.history_tablename = self.db_info['history_tablename']
        self.ref_name = self.db_info['ref_name']

        # Private variables
        self._c = None
        self._conn = None

    def __del__(self):
        """
        When destroying the object, it is necessary to commit changes
        in the database and close the connection
        """

        try:
            self._conn.commit()
            self._conn.close()
        except:
            print("---- Error closing database")

    def _getTableNames(self):
        """
        Returns a list with the names of all tables in the database
        (taken from a code by Jeronimo Arenas)
        """

        # The specific command depends on whether we are using mysql or sqlite
        if self.connector == 'mysql':
            sqlcmd = ("SELECT table_name FROM INFORMATION_SCHEMA.TABLES " +
                      "WHERE table_schema='" + self.db_name + "'")
        else:
            sqlcmd = "SELECT name FROM sqlite_master WHERE type='table'"

        self._c.execute(sqlcmd)
        tbnames = [el[0] for el in self._c.fetchall()]

        return tbnames

    def _getColumnNames(self, tablename):
        """
        Returns a list with the names of all columns in the indicated table
        (taken from a code by Jeronimo Arenas)

        Args:
            tablename: the name of the table to retrieve column names
        """

        # Check if tablename exists in database
        if tablename in self._getTableNames():
            # The specific command depends on whether we are using mysql or
            #  sqlite
            if self.connector == 'mysql':
                sqlcmd = "SHOW COLUMNS FROM " + tablename
                self._c.execute(sqlcmd)
                columnnames = [el[0] for el in self._c.fetchall()]
            else:
                sqlcmd = "PRAGMA table_info(" + tablename + ")"
                self._c.execute(sqlcmd)
                columnnames = [el[1] for el in self._c.fetchall()]

            return columnnames

        else:
            print('Error retrieving column names: Table does not exist on ' +
                  'database')
            return []

    def _addTableColumn(self, tablename, columnname, columntype):
        """
        Add a new column to the specified table

        Args:
            tablename    :Table to which the column will be added
            columnname   :Name of new column
            columntype   :Type of new column

        """

        # Check if the table exists
        if tablename in self._getTableNames():

            # Check that the column does not already exist
            if columnname not in self._getColumnNames(tablename):

                sqlcmd = ('ALTER TABLE ' + tablename + ' ADD COLUMN ' +
                          columnname + ' ' + columntype)
                self._c.execute(sqlcmd)

                # Commit changes
                self._conn.commit()

            else:
                print(
                    'Error adding column to table. The column already exists')
                print(tablename, columnname)

        else:
            print('Error adding column to table. Please, select a valid ' +
                  'table name from the list')
            print(self._getTableNames())

    def _createDBtable(self, tablenames=None):
        """
        Creates any of the project tables.

            Args:
                tablenames :Name of any of the project tables. If None, all
                            tables required by a labeling project are created.
        """

        # Fit characters to the allowed format
        if self.connector == 'mysql':
            # We need to enforze utf8 for mysql
            fmt = ' CHARACTER SET utf8'
        else:
            # For sqlite 3 no formatting is required.
            fmt = ''

        for name in tablenames:

            if name == self.preds_tablename:

                # Table for predictions
                # The name 'url' has historical reasons, but it keeps the
                # meaning: it specifies a kind of uniform resource location.
                sql_cmd = ("""CREATE TABLE {0}(
                                {1} VARCHAR(30){2} PRIMARY KEY,
                                url TEXT{2}
                                )""").format(name, self.ref_name, fmt)
                self._c.execute(sql_cmd)

                # One prediction column per category
                for cat in self.categories:
                    self._addTableColumn(name, cat, 'DOUBLE')

            elif name == self.label_values_tablename:

                # Table for the label values
                sql_cmd = """CREATE TABLE {0}(
                                {1} VARCHAR(30){2} PRIMARY KEY,
                                )""".format(name, self.ref_name, fmt)
                self._c.execute(sql_cmd)

                # One label column per category
                for cat in self.categories:
                    self._addTableColumn(name, cat, 'INTEGER')

            elif name == self.label_info_tablename:

                # Table for the label metadata
                sql_cmd = ("""CREATE TABLE {0}(
                                {1} VARCHAR(30){2} PRIMARY KEY,
                                marker INTEGER,
                                relabel INTEGER,
                                weight INTEGER,
                                userId TEXT{2},
                                datestr TEXT{2}
                                labelstr TEXT{0}
                                )""").format(name, self.ref_name, fmt)
                self._c.execute(sql_cmd)

            elif name == self.history_tablename:

                # Table for the historic record of labelling events
                sql_cmd = ("""CREATE TABLE {0}(
                                {1} VARCHAR(30){2} PRIMARY KEY,
                                datestr TEXT{2},
                                url TEXT{2},
                                marker INTEGER,
                                relabel INTEGER,
                                label TEXT{2},
                                userId TEXT{2}
                                )""").format(name, self.ref_name, fmt)
                self._c.execute(sql_cmd)

            else:
                sys.exit('---- ERROR: Wrong table name')

        # Commit changes
        self._conn.commit()

    def _insertInTable(self, tablename, columns, arguments):
        """
        Insert new records into table

        Args:
            tablename:  Name of table in which the data will be inserted
            columns:    Name of columns for which data are provided
            arguments:  A list of lists or tuples, each element associated
                        to one new entry for the table

        Taken from a code by Jeronimo Arenas
        """

        # If the list of values to insert is empty, return
        if len(arguments) == 0:
            return

        # Make sure columns is a list, and not a single string
        if not isinstance(columns, (list,)):
            columns = [columns]

        ncol = len(columns)

        if len(arguments[0]) == ncol:
            # Make sure the tablename is valid
            if tablename in self._getTableNames():
                # Make sure we have a list of tuples; necessary for mysql
                arguments = list(map(tuple, arguments))

                sqlcmd = ('INSERT INTO ' + tablename +
                          '(' + ','.join(columns) + ') VALUES (')
                if self.connector == 'mysql':
                    sqlcmd += '%s' + (ncol-1)*',%s' + ')'
                else:
                    sqlcmd += '?' + (ncol-1)*',?' + ')'

                print("---- ---- Saving {0}".format(tablename))
                start = time.clock()
                self._c.executemany(sqlcmd, arguments)
                print(str(time.clock() - start) + ' seconds')

                # Commit changes
                self._conn.commit()
        else:
            print('Error inserting data in table: number of columns mismatch')

    def _setField(self, tablename, valueflds, values):
        """
        Update records of a DB table

        Args:
            tablename:  Table that will be modified
            valueflds:  list with the names of the columns that will be updated
                        (e.g., 'Lemas')
            values:     A list of tuples in the format
                           (index_colvalue, valuefldvalue)
                        (e.g., [('Ref1', 'gen celula'),
                               ('Ref2', 'big_data, algorithm')])

        Taken from a code by Jeronimo Arenas
        """

        # Auxiliary function to circularly shift a tuple one position to the
        # right
        def circ_right_shift(tup):
            ls = list(tup[1:]) + [tup[0]]
            return tuple(ls)

        # If the list of values to insert is empty, return
        if len(values) == 0:
            return

        # Make sure valueflds is a list, and not a single string
        if not isinstance(valueflds, (list,)):
            valueflds = [valueflds]
        ncol = len(valueflds)

        if len(values[0]) == (ncol + 1):
            # Make sure the tablename is valid
            if tablename in self._getTableNames():
                # Make sure we have a list of tuples; necessary for mysql
                # Put key value last in the tuples
                values = list(map(circ_right_shift, values))

                sqlcmd = 'UPDATE ' + tablename + ' SET '
                if self.connector == 'mysql':
                    sqlcmd += ', '.join([el+'=%s' for el in valueflds])
                    sqlcmd += ' WHERE ' + self.ref_name + '=%s'
                else:
                    sqlcmd += ', '.join([el+'=?' for el in valueflds])
                    sqlcmd += ' WHERE ' + self.ref_name + '=?'

                self._c.executemany(sqlcmd, values)

                # Commit changes
                self._conn.commit()

        else:
            print('Error updating table values: number of columns mismatch')

    def _upsert(self, tablename, df):

        """
        Update records of a DB table with the values in the df
        This function implements the following additional functionality:

        - If there are columns in df that are not in the SQL table,
          columns will be added
        - New records will be created in the table if there are rows
          in the dataframe without an entry already in the table. For this,
          index_col indicates which is the column that will be used as an
          index

        Args:
            tablename: Table that will be modified
            df:        Dataframe to be saved in table tablename

        Adapted from a code by Jeronimo Arenas.
        """

        # Check that table exists and index_col exists both in the Table and
        # the Dataframe
        if tablename not in self._getTableNames():
            sys.error('Upsert function failed: Table does not exist')
        elif self.ref_name not in self._getColumnNames(tablename):
            sys.error("Upsert function failed: Key field does not exist" +
                      "in the selected table")

        # Create new columns if necessary
        for clname in df.columns:
            if clname not in self._getColumnNames(tablename):
                if df[clname].dtypes == np.float64:
                    self._addTableColumn(tablename, clname, 'DOUBLE')
                elif df[clname].dtypes == np.int64:
                    self._addTableColumn(tablename, clname, 'INTEGER')
                else:
                    self._addTableColumn(tablename, clname, 'TEXT')

        # This is to update changes made by other processes.
        self._conn.commit()

        # Check which values are already in the table, and split the dataframe
        # into records that need to be updated, and records that need to be
        # inserted
        sqlQuery = 'SELECT ' + self.ref_name + ' FROM ' + tablename
        keyintable = pd.read_sql(sqlQuery, con=self._conn,
                                 index_col=self.ref_name)
        keyintable = keyintable.index.tolist()

        # Replace NaN with None, because mysql raises an error if nan is
        # sent to a non numeric column.
        df = df.where(pd.notnull(df), None)
        # if np.any(pd.isnull(df)):
        #     df[pd.isnull(df)] = None
        # values = [tuple(x) for x in df_ext.values]

        # Split dataframe in the entries related to existing references
        # and entries about new references
        refs = set(df.index)
        refs_old = list(refs & set(keyintable))  # Refs that exist in the db
        refs_new = list(refs - set(refs_old))    # New refs

        df_old = df.loc[refs_old]
        df_new = df.loc[refs_new]

        # Convert reference index into a new column
        df_ext_old = df_old.reset_index().rename(
            columns={'index': self.ref_name})
        df_ext_new = df_new.reset_index().rename(
            columns={'index': self.ref_name})

        # Convert dataframe values into a list of tuples
        values_update = [tuple(x) for x in df_ext_old.values]
        values_insert = [tuple(x) for x in df_ext_new.values]

        print("---- ---- ---- Updating {}".format(tablename))
        t0 = time.clock()
        self._setField(tablename, df.columns.tolist(), values_update)
        print(time.clock() - t0)
        print("---- ---- ---- Inserting {}".format(tablename))
        t0 = time.clock()
        self._insertInTable(tablename, df_ext_new.columns.tolist(),
                            values_insert)
        print(time.clock() - t0)

        return

    def loadData(self):

        """ Load data and label history from file.
            This is the basic method to read the information about labels, urls
            and predictions from files in the standard format.

            If the dataset file or the labelhistory file does not exist, no
            error is returned, though empty data variables are returned.

            :Returns:
                :df_labels:  Multi-index Pandas dataframe containing labels.
                             Fields are:
                    'info':  With columns marker', 'relabel', 'weight',
                             'userId', 'date'
                    'label': One column per categorie, containing the labels
                :df_preds: Pandas dataframa indexed by the complete list of
                           wids, with one column of urls and one addicional
                           column per category containing predictions.
                :labelhistory: Dataframe containing, for each wid, a record of
                        the labeling events up to date.

            IMPORTANT NOTE: labelhistory has a different structure in this
                            module than in other subclasses of baseDM, which
                            use a dictionary instead of a pandas dataframe to
                            record the label history.
        """

        # Connect to the database
        try:
            if self.connector == 'mysql':
                self._conn = MySQLdb.connect(
                    self.server, self.user, self.password, self.db_name)
                self._c = self._conn.cursor()
            elif self.connector == 'sqlalchemy':
                engine_name = ('mysql://' + self.user + ':' + self.password +
                               '@' + self.server + '/' + self.db_name)
                print('---- Creating engine {}'.format(engine_name))
                engine = create_engine(engine_name)
                self._conn = engine.connect()
            else:
                # sqlite3
                # sqlite file will be in the root of the project, we read the
                # name from the config file and establish the connection
                db_path = os.path.join(self.directory,
                                       self.dataset_fname + '.db')
                print("---- Connecting to {}".format(db_path))
                self._conn = sqlite3.connect(db_path)
                self._c = self._conn.cursor()
            self.dbON = True
        except:
            print("---- Error connecting to the database")

        try:
            # #####################
            # Create missing tables

            # Create all tables that do not exist in the database yet.
            tablenames = self._getTableNames()
            alltables = [self.preds_tablename, self.label_values_tablename,
                         self.label_info_tablename, self.history_tablename]
            missing_tables = [t for t in alltables if t not in tablenames]

            self._createDBtable(missing_tables)

            # ################
            # Load predictions
            sqlQuery = 'SELECT * FROM ' + self.preds_tablename
            df_preds = pd.read_sql(
                sqlQuery, con=self._conn, index_col=self.ref_name)

            # ###########
            # Load labels

            # Load label metadata
            sqlQuery = 'SELECT * FROM ' + self.label_info_tablename
            df_labelinfo = pd.read_sql(
                sqlQuery, con=self._conn, index_col=self.ref_name)
            # Rename column 'datestr' to 'date':
            df_labelinfo.rename(columns={'datestr': 'date'}, inplace=True)
            # Convert column names into tuples
            df_labelinfo.columns = (
                [('info', c) for c in df_labelinfo.columns])

            # Load label values
            sqlQuery = 'SELECT * FROM ' + self.label_values_tablename
            df_labelvalues = pd.read_sql(
                sqlQuery, con=self._conn, index_col=self.ref_name)
            # Convert column names into tuples
            df_labelvalues.columns = (
                [('label', c) for c in df_labelvalues.columns])

            # Joing label metadata and label values into a single dataframe
            # df_labels = pd.concat([df_labelinfo, df_labelvalues])
            df_labels = df_labelinfo.join(df_labelvalues)

            # Convert tuple column names to multi-index
            df_labels.columns = pd.MultiIndex.from_tuples(
                df_labels.columns)

            # ##################
            # Load label history
            sqlQuery = 'SELECT * FROM ' + self.history_tablename
            # Read dataframe. Note that I do not take any reference columns
            labelhistory = pd.read_sql(sqlQuery, con=self._conn)
            # Rename columns datestr to date
            # (this is required because 'date' is a reserved word in sql)
            labelhistory.rename(
                columns={'datestr': 'date', self.ref_name: 'wid'},
                inplace=True)

        except Exception as E:
            print('Exception {}'.format(str(E)))

        return df_labels, df_preds, labelhistory

    def get_df(self, data, labelhistory):

        """ Converts the data dictionary used in former versions of the web
            labeler into the label and prediction dataframes.

            THERE IS NO VERSION OF THIS METHOD FOR SQL DATAMANAGER.

            The version of get_df in the base clase doesw not work for dmSQL
            because it uses a dicionary form of labelhistory.

            I do not create a versio nof this function for SQL because it is
            likely unnecesary
        """

        sys.exit(
            '---- ERROR: There is no version of get_df method for sql data')

        return

    def saveData(self, df_labels, df_preds, labelrecord, save_preds=True):

        """ Save label and prediction dataframes and labelhistory pickle files.

            :Args:
                :df_labels:   Pandas dataframe of labels
                :df_preds:    Pandas dataframe of predictions
                :labelrecord: Dataframe with the labelling events of this
                              session
                :save_preds:  If False, predictions are not saved.
s        """

        # Connect to the database
        try:

            if self._conn is None:
                if self.connector == 'mysql':
                    self._conn = MySQLdb.connect(
                        self.server, self.user, self.password, self.db_name)
                    self._c = self._conn.cursor()
                elif self.connector == 'sqlalchemy':
                    engine_name = (
                        'mysql://' + self.user + ':' + self.password + '@' +
                        self.server + '/' + self.db_name)
                    print('---- Creating engine {}'.format(engine_name))
                    engine = create_engine(engine_name)
                    self._conn = engine.connect()
                else:
                    # sqlite3
                    # sqlite file will be in the root of the project, we read
                    # the name from the config file and establish connection
                    db_path = os.path.join(self.directory,
                                           self.dataset_fname + '.db')
                    print("---- Connecting to {}".format(db_path))
                    self._conn = sqlite3.connect(db_path)
                    self._c = self._conn.cursor()
                self.dbON = True
        except:
            print("---- Error connecting to the database")

        # Save labels
        self._upsert(self.label_info_tablename,
                     df_labels['info'].rename(columns={'date': 'datestr'}))
        self._upsert(self.label_values_tablename, df_labels['label'])
        # Save label history
        self._upsert(self.history_tablename,
                     labelrecord.rename(columns={'date': 'datestr'}))
        #                                            'wid': self.ref_name}))

        # Save predictions (unles otherwise stated)
        if save_preds:
            self._upsert(self.preds_tablename, df_preds)

        if self.connector != 'sqlalchemy':
            self._conn.commit()
