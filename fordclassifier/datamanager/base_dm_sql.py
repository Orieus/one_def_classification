"""
This class provides functionality for managing a generig sqlite or mysql
database:

* reading specific fields (with the possibility to filter by field values)
* storing calculated values in the dataset

Created on May 11 2018

@author: Jerónimo Arenas García

"""

from __future__ import print_function    # For python 2 copmatibility
import os
import pandas as pd
import MySQLdb
import sqlite3
import numpy as np
from tabulate import tabulate
import copy

import ipdb


class BaseDMsql(object):
    """
    Data manager base class.
    """

    def __init__(self, db_name, db_connector, path2project=None,
                 db_server=None, db_user=None, db_password=None):
        """
        Initializes a DataManager object

        Args:
            db_name      :Name of the DB
            db_connector :Connector. Available options are mysql or sqlite
            path2project :Path to the project folder (sqlite only)
            db_server    :Server (mysql only)
            db_user      :User (mysql only)
            db_password  :Password (mysql only)
        """

        # Store paths to the main project folders and files
        self._path2project = copy.copy(path2project)
        self.dbname = db_name
        self.connector = db_connector
        self.server = db_server
        self.user = db_user
        self.password = db_password

        # Other class variables
        self.dbON = False    # Will switch to True when the db was connected.
        # Connector to database
        self._conn = None
        # Cursor of the database
        self._c = None

        # Try connection
        try:
            if self.connector == 'mysql':
                self._conn = MySQLdb.connect(self.server, self.user,
                                             self.password, self.dbname)
                self._c = self._conn.cursor()
                print("MySQL database connection successful")
                self.dbON = True
                self._conn.set_character_set('utf8')
            elif self.connector == 'sqlite3':
                # sqlite3
                # sqlite file will be in the root of the project, we read the
                # name from the config file and establish the connection
                db_fname = os.path.join(self._path2project,
                                        self.dbname + '.db')
                print("---- Connecting to {}".format(db_fname))
                self._conn = sqlite3.connect(db_fname)
                self._c = self._conn.cursor()
                self.dbON = True
            else:
                print("---- Unknown DB connector {}".format(self.connector))
        except:
            print("---- Error connecting to the database")

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

    def resetDBtables(self, tables=None):
        """
        Delete existing database, and regenerate empty tables

        Args:
            tables: If string, name of the table to reset.
                    If list, list of tables to reset
                    If None (default), all tables are deleted, and all tables
                    (inlcuding those that might not exist previously)
        """

        # If tables is None, all tables are deleted an re-generated
        if tables is None:
            # Delete all existing tables
            for table in self.getTableNames():
                self._c.execute("DROP TABLE " + table)
            # Create tables. No tables as specifies in order to create tables
            # that did not exist previously also.
            self.createDBtables()

        else:

            # It tables is not a list, make the appropriate list
            if type(tables) is str:
                tables = [tables]

            # Remove all selected tables (if exist in the database).
            for table in set(tables) & set(self.getTableNames()):
                self._c.execute("DROP TABLE " + table)

            # All deleted tables are created again
            self.createDBtables(tables)

        self._conn.commit()

        return

    def resetDB(self):
        """
        Deletes existing database, and regenerate empty tables
        """

        if self.connector == 'mysql':
            # In mysql we simply drop all existing tables
            for tablename in self.getTableNames():
                self._c.execute("DROP TABLE " + tablename)
            self._conn.commit()

        else:
            # If sqlite3, we need to delete the file, and start over
            try:
                self._conn.commit()
                self._conn.close()
            except:
                print("Error closing database")

            # Delete sqlite3 file
            db_fname = os.path.join(self._path2project, self.dbname + '.db')
            os.remove(db_fname)

            try:
                self._conn = sqlite3.connect(db_fname)
                self._c = self._conn.cursor()
            except:
                print("Error connecting to the database")

        self.createDBtables()

    def addTableColumn(self, tablename, columnname, columntype):
        """
        Add a new column to the specified table.

        Args:
            tablename  :Table to which the column will be added
            columnname :Name of new column
            columntype :Type of new column.

        Note that, for mysql, if type is TXT or VARCHAR, the character set if
        forzed to be utf8.
        """

        # Check if the table exists
        if tablename in self.getTableNames():

            # Check that the column does not already exist
            if columnname not in self.getColumnNames(tablename):

                # Fit characters to the allowed format if necessary
                fmt = ''
                if (self.connector == 'mysql' and
                    ('TEXT' in columntype or 'VARCHAR' in columntype) and
                    not ('CHARACTER SET' in columntype or
                         'utf8' in columntype)):

                    # We need to enforze utf8 for mysql
                    fmt = ' CHARACTER SET utf8'

                sqlcmd = ('ALTER TABLE ' + tablename + ' ADD COLUMN ' +
                          columnname + ' ' + columntype + fmt)
                self._c.execute(sqlcmd)

                # Commit changes
                self._conn.commit()

            else:
                print(("WARNING: Column {0} already exist in table {1}."
                       ).format(columnname, tablename))

        else:
            print('Error adding column to table. Please, select a valid ' +
                  'table name from the list')
            print(self.getTableNames())

    def dropTableColumn(self, tablename, columnname):
        """
        Remove column from the specified table

        Args:
            tablename    :Table to which the column will be added
            columnname   :Name of column to be removed

        """

        # Check if the table exists
        if tablename in self.getTableNames():

            # Check that the column does not already exist
            if columnname in self.getColumnNames(tablename):

                # ALTER TABLE DROP COLUMN IS ONLY SUPPORTED IN MYSQL
                if self.connector == 'mysql':

                    sqlcmd = ('ALTER TABLE ' + tablename + ' DROP COLUMN ' +
                              columnname)
                    self._c.execute(sqlcmd)

                    # Commit changes
                    self._conn.commit()

                else:
                    print('Column drop not yet supported for SQLITE')

            else:
                print('Error deleting column. The column does not exist')
                print(tablename, columnname)

        else:
            print('Error deleting column. Please, select a valid table name' +
                  ' from the list')
            print(self.getTableNames())

        return

    def readDBtable(self, tablename, limit=None, selectOptions=None,
                    filterOptions=None, orderOptions=None):
        """
        Read data from a table in the database can choose to read only some
        specific fields

        Args:
            tablename    :  Table to read from
            selectOptions:  string with fields that will be retrieved
                            (e.g. 'REFERENCIA, Resumen')
            filterOptions:  string with filtering options for the SQL query
                            (e.g., 'WHERE UNESCO_cd=23')
            orderOptions:   string with field that will be used for sorting the
                            results of the query
                            (e.g, 'Cconv')
            limit:          The maximum number of records to retrieve

        """

        try:

            # Check that table name is valid

            if tablename in self.getTableNames():

                sqlQuery = 'SELECT '
                if selectOptions:
                    sqlQuery = sqlQuery + selectOptions
                else:
                    sqlQuery = sqlQuery + '*'

                sqlQuery = sqlQuery + ' FROM ' + tablename + ' '

                if filterOptions:
                    sqlQuery = sqlQuery + ' WHERE ' + filterOptions

                if orderOptions:
                    sqlQuery = sqlQuery + ' ORDER BY ' + orderOptions

                if limit:
                    sqlQuery = sqlQuery + ' LIMIT ' + str(limit)

                # This is to update the connection to changes by other
                # processes.
                self._conn.commit()

                # Return the pandas dataframe. Note that numbers in text format
                # are not converted to
                return pd.read_sql(sqlQuery, con=self._conn,
                                   coerce_float=False)

            else:
                print('Error in query. Please, select a valid table name ' +
                      'from the list')
                print(self.getTableNames())

        except Exception as E:
            print(str(E))

    def getTableNames(self):
        """
        Returns a list with the names of all tables in the database
        """

        # The specific command depends on whether we are using mysql or sqlite
        if self.connector == 'mysql':
            sqlcmd = ("SELECT table_name FROM INFORMATION_SCHEMA.TABLES " +
                      "WHERE table_schema='" + self.dbname + "'")
        else:
            sqlcmd = "SELECT name FROM sqlite_master WHERE type='table'"

        self._c.execute(sqlcmd)
        tbnames = [el[0] for el in self._c.fetchall()]

        return tbnames

    def getColumnNames(self, tablename):
        """
        Returns a list with the names of all columns in the indicated table

        Args:
            tablename: the name of the table to retrieve column names
        """

        # Check if tablename exists in database
        if tablename in self.getTableNames():
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

    def getTableInfo(self, tablename):

        # Get columns
        cols = self.getColumnNames(tablename)

        # Get number of rows
        sqlcmd = "SELECT COUNT(*) FROM " + tablename
        self._c.execute(sqlcmd)
        n_rows = self._c.fetchall()[0][0]

        return cols, n_rows

    def showTable(self, tablename, max_rows=500, max_width=200):
        """ A simple method to display the content of a single table.

            Args:
                max_rows:  Maximum number of rows to display. It the size of
                           the table is higher, only the first max_rows rows
                           are shown
                max_width: Maximum with of the table to display. If the size
                           of the table is higher, the tabulate environment
                           is not used and only a table heading is shown
        """

        title = "= Database {} ====================".format(self.dbname)
        print("="*len(title))
        print(title)
        print("="*len(title))
        print("")
        print("==== Table {} ".format(tablename))

        cols, n_rows = self.getTableInfo(tablename)

        df = self.readDBtable(tablename, limit=max_rows, selectOptions=None,
                              filterOptions=None, orderOptions=None)

        txt = tabulate(df, headers='keys', tablefmt='psql')
        txt_width = max(len(z) for z in txt.split('\n'))

        if txt_width > max_width:
            print('---- The table is too wide (up to {}'.format(txt_width) +
                  ' characters per line). Showing a portion of the table ' +
                  'header only')
            print(df.head(25))
        else:
            print(txt)

        return

    def insertInTable(self, tablename, columns, arguments):
        """
        Insert new records into table

        Args:
            tablename:  Name of table in which the data will be inserted
            columns:    Name of columns for which data are provided
            arguments:  A list of lists or tuples, each element associated
                        to one new entry for the table
        """

        # Make sure columns is a list, and not a single string
        if not isinstance(columns, (list,)):
            columns = [columns]

        ncol = len(columns)

        if len(arguments[0]) == ncol:
            # Make sure the tablename is valid
            if tablename in self.getTableNames():
                # Make sure we have a list of tuples; necessary for mysql
                arguments = list(map(tuple, arguments))

                # # Update DB entries one by one.
                # for arg in arguments:
                #     # sd
                #     sqlcmd = ('INSERT INTO ' + tablename + '(' +
                #               ','.join(columns) + ') VALUES(' +
                #               ','.join('{}'.format(a) for a in arg) + ')'
                #               )

                #     try:
                #         self._c.execute(sqlcmd)
                #     except:
                #         import ipdb
                #         ipdb.set_trace()

                sqlcmd = ('INSERT INTO ' + tablename +
                          '(' + ','.join(columns) + ') VALUES (')
                if self.connector == 'mysql':
                    sqlcmd += '%s' + (ncol-1)*',%s' + ')'
                else:
                    sqlcmd += '?' + (ncol-1)*',?' + ')'

                self._c.executemany(sqlcmd, arguments)

                # Commit changes
                self._conn.commit()
        else:
            print('Error inserting data in table: number of columns mismatch')

        return

    def setField(self, tablename, keyfld, valueflds, values):
        """
        Update records of a DB table

        Args:
            tablename:  Table that will be modified
            keyfld:     string with the column name that will be used as key
                        (e.g. 'REFERENCIA')
            valueflds:  list with the names of the columns that will be updated
                        (e.g., 'Lemas')
            values:     A list of tuples in the format
                            (keyfldvalue, valuefldvalue)
                        (e.g., [('Ref1', 'gen celula'),
                                ('Ref2', 'big_data, algorithm')])

        """

        # Make sure valueflds is a list, and not a single string
        if not isinstance(valueflds, (list,)):
            valueflds = [valueflds]
        ncol = len(valueflds)

        if len(values[0]) == (ncol+1):
            # Make sure the tablename is valid
            if tablename in self.getTableNames():

                # Update DB entries one by one.
                # WARNING: THIS VERSION MAY NOT WORK PROPERLY IF v
                #          HAS A STRING CONTAINING "".
                for v in values:
                    sqlcmd = ('UPDATE ' + tablename + ' SET ' +
                              ', '.join(['{0} ="{1}"'.format(f, v[i + 1])
                                         for i, f in enumerate(valueflds)]) +
                              ' WHERE {0}="{1}"'.format(keyfld, v[0]))
                    self._c.execute(sqlcmd)

                # This is the old version: it might not have the problem of
                # the above version, but did not work properly with sqlite.
                # # Make sure we have a list of tuples; necessary for mysql
                # # Put key value last in the tuples
                # values = list(map(circ_left_shift, values))

                # sqlcmd = 'UPDATE ' + tablename + ' SET '
                # if self.connector == 'mysql':
                #     sqlcmd += ', '.join([el+'=%s' for el in valueflds])
                #     sqlcmd += ' WHERE ' + keyfld + '=%s'
                # else:
                #     sqlcmd += ', '.join([el+'=?' for el in valueflds])
                #     sqlcmd += ' WHERE ' + keyfld + '=?'

                # self._c.executemany(sqlcmd, values)

                # Commit changes
                self._conn.commit()
        else:
            print('Error updating table values: number of columns mismatch')

        return

    def upsert(self, tablename, keyfld, df):

        """
        Update records of a DB table with the values in the df
        This function implements the following additional functionality:
        * If there are coumns in df that are not in the SQL table,
          columns will be added
        * New records will be created in the table if there are rows
          in the dataframe without an entry already in the table. For this,
          keyfld indicates which is the column that will be used as an
          index

        Args:
            tablename:  Table that will be modified
            keyfld:     string with the column name that will be used as key
                        (e.g. 'REFERENCIA')
            df:         Dataframe that we wish to save in table tablename

        """

        # Check that table exists and keyfld exists both in the Table and the
        # Dataframe
        if tablename in self.getTableNames():
            if not ((keyfld in df.columns) and
               (keyfld in self.getColumnNames(tablename))):
                print("Upsert function failed: Key field does not exist",
                      "in the selected table and/or dataframe")
                return
        else:
            print('Upsert function failed: Table does not exist')
            return

        # Reorder dataframe to make sure that the key field goes first
        flds = [keyfld] + [x for x in df.columns if x != keyfld]
        df = df[flds]

        # Create new columns if necessary
        for clname in df.columns:
            if clname not in self.getColumnNames(tablename):
                if df[clname].dtypes == np.float64:
                    self.addTableColumn(tablename, clname, 'DOUBLE')
                else:
                    if df[clname].dtypes == np.int64:
                        self.addTableColumn(tablename, clname, 'INTEGER')
                    else:
                        self.addTableColumn(tablename, clname, 'TEXT')

        # Check which values are already in the table, and split
        # the dataframe into records that need to be updated, and
        # records that need to be inserted
        keyintable = self.readDBtable(tablename, limit=None,
                                      selectOptions=keyfld)
        keyintable = keyintable[keyfld].tolist()
        values = [tuple(x) for x in df.values]
        values_insert = list(filter(lambda x: x[0] not in keyintable, values))
        values_update = list(filter(lambda x: x[0] in keyintable, values))

        if len(values_update):
            self.setField(tablename, keyfld, df.columns[1:].tolist(),
                          values_update)
        if len(values_insert):
            self.insertInTable(tablename, df.columns.tolist(), values_insert)

        return

    def exportTable(self, tablename, fileformat, path, filename, cols=None):
        """
        Export columns from a table to a file.

        Args:
            :tablename:  Name of the table
            :fileformat: Type of output file. Available options are
                            - 'xlsx'
                            - 'pkl'
            :filepath:   Route to the output folder
            :filename:   Name of the output file
            :columnames: Columns to save. It can be a list or a string
                         of comma-separated columns.
                         If None, all columns saved.
        """

        # Path to the output file
        fpath = os.path.join(path, filename)

        # Read data:
        if cols is list:
            options = ','.join(cols)
        else:
            options = cols

        df = self.readDBtable(tablename, selectOptions=options)

        # ######################
        # Export results to file
        if fileformat == 'pkl':
            df.to_pickle(fpath)

        else:
            df.to_excel(fpath)

        return

