#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python libraries
from __future__ import print_function
import os
import sys
# import cPickle as pickle
import pickle
import shutil
import pandas as pd

# import copy
import time
from datetime import datetime

from pymongo import MongoClient

# Local imports
from labelfactory.labeling import baseDM

# Services from the project
# sys.path.append(os.getcwd())


class DM_Files(baseDM.BaseDM):

    """
    DataManager is the class providing read and write facilities to access and
    update the dataset of labels and predictions

    It assumes that data will be stored in files or in a databes.

    If files, the following data structure is assumed

        project_path/.
                    /label_dataset_fname.pkl
                    /preds_dataset_fname.pkl
                    /labelhistory_fname.pkl
                    /input/.
                          /labels_fname
                          /preds_fname
                    /output/.
                    /used_/.

    (the specific file and folder names can be configured)

    If project_path does not exist, an error is returned.

    The class provides facilities to:

        - Read and write data in .pkl files or a mongo database
        - Read labels from the /input/ folder in csv format
        - Read preds from the /input/ folder in pkl files
        - Write outputs (tipically, new labels) in the desired format in
          /ouput/
    """

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
                :labelhistory: Dictionary containing, for each wid, a record of
                        the labeling events up to date.
        """

        dbName = self.db_info['name']
        hostname = self.db_info['hostname']
        user = self.db_info['user']
        pwd = self.db_info['pwd']
        label_coll_name = self.db_info['label_coll_name']
        history_coll_name = self.db_info['history_coll_name']
        port = self.db_info['port']

        try:
            print("Trying connection...")
            client = MongoClient(hostname)
            client[dbName].authenticate(user, pwd)
            db = client[dbName]
            print("Connected to mongodb @ {0}:[{1}]".format(
                hostname, port))
        except Exception as E:
            print("Fail to connect mongodb @ {0}:{1}, {2}".format(
                hostname, port, E))
            exit()

        # Read label collection
        collection = db[label_coll_name]
        num_urls = collection.count()
        data = {}
        if num_urls > 0:
            dataDB = collection.find({})
            for i in range(num_urls):
                wid = dataDB[i]['idna']
                data[wid] = dataDB[i]['value']
                if 'url' not in data[wid]:
                    data[wid]['url'] = wid

        # Read history
        collection = db[history_coll_name]
        num_events = collection.count()
        labelhistory = {}
        if num_events > 0:
            dataDB = collection.find({})
            for i in range(num_events):
                wid = dataDB[i]['idna']
                labelhistory[wid] = dataDB[i]['value']

        df_labels, df_preds = self.get_df(data, labelhistory)

        # In the current version, predictions are not being stored in the
        # mongo db. They must be loaded from files.
        if os.path.isfile(self.datapreds_file):
            # Load prediction dataframes stored in pickle files
            df_preds = pd.read_pickle(self.datapreds_file)

        return df_labels, df_preds, labelhistory

    def saveData(self, df_labels, df_preds, labelhistory, dest='mongodb',
                 save_preds=True):

        """ Save label and prediction dataframes and labelhistory pickle files.
            If dest='mongodb', they are also saved in a mongo database.

            If dest='mongodb', the dataframes are store in the mode specified
            in self.db_info['mode'].
                'rewrite' :The existing db collection is removed and data are
                           saved in a new one
                'update'  :The data are upserted to the existing db.

            :Args:
                :df_labels: Pandas dataframe of labels
                :df_preds:  Pandas dataframe of predictions
                :labelhistory:
                :dest: Type of destination: 'file' (data is saved in files) or
                 'mongodb'
                :save_preds:  If False, predictions are not saved.
        """

        if dest == 'file':
            # Keep a copy of the original datasets, just in case some
            # mistakes are made during labelling
            date_str = datetime.now().strftime("%Y%m%d%H%M%S%f")
            if os.path.isfile(self.dataset_file):
                dest_file = (self.used_path + self.dataset_fname + '_' +
                             date_str + '.pkl')
                shutil.move(self.dataset_file, dest_file)
            if os.path.isfile(self.datalabels_file):
                dest_file = (self.used_path + self.datalabels_fname + '_' +
                             date_str + '.pkl')
                shutil.move(self.datalabels_file, dest_file)
            if os.path.isfile(self.datapreds_file):
                dest_file = (self.used_path + self.datapreds_fname + '_' +
                             date_str + '.pkl')
                shutil.move(self.datapreds_file, dest_file)
            if os.path.isfile(self.labelhistory_file):
                dest_file = (self.used_path + self.labelhistory_fname + '_' +
                             date_str + '.pkl')
                shutil.move(self.labelhistory_file, dest_file)

            # Save label history
            with open(self.labelhistory_file, 'wb') as f:
                pickle.dump(labelhistory, f)

            # Save dataframes to files
            df_labels.to_pickle(self.datalabels_file)
            if save_preds:
                df_preds.to_pickle(self.datapreds_file)

        else:

            # Start a db connection
            dbName = self.db_info['name']
            hostname = self.db_info['hostname']
            user = self.db_info['user']
            pwd = self.db_info['pwd']
            label_coll_name = self.db_info['label_coll_name']
            mode = self.db_info['mode']

            # history_coll_name = self.db_info['history_coll_name']
            port = self.db_info['port']

            try:
                print("Trying db connection...")
                client = MongoClient(hostname)
                client[dbName].authenticate(user, pwd)
                db = client[dbName]
                # history_collection = db[history_coll_name]
                print("Connected to mongodb @ {0}:[{1}]".format(
                    hostname, port))
            except Exception as E:
                print("Fail to connect mongodb @ {0}:{1}, {2}".format(
                    hostname, port, E))
                exit()

            start_time = time.time()
            print("Saving database. This might take a while...")
            if mode == 'rewrite':
                # The database is deleted completely and the whole set of
                # labels and predictions in data are loaded
                label_collection = db[label_coll_name]
                label_collection.drop()

            # Open collection, or create it, if it does not exist.
            label_collection = db[label_coll_name]

            for i, w in enumerate(df_labels.index):
                # For each wid, create the corresponding data dictionary to
                # send to the db
                dataw = {}
                dataw['relabel'] = df_labels.loc[w, ('info', 'relabel')]
                dataw['marker'] = df_labels.loc[w, ('info', 'marker')]
                dataw['userId'] = df_labels.loc[w, ('info', 'userId')]
                dataw['date'] = df_labels.loc[w, ('info', 'date')]
                dataw['weight'] = df_labels.loc[w, ('info', 'weight')]
                dataw['label'] = {}
                for c in self.categories:
                    dataw['label'][c] = df_labels.loc[w, ('label', c)]

                # Store in db.
                if mode == 'rewrite':
                    # Insert data in the database
                    label_collection.insert({'idna': w, 'value': dataw})
                else:    # mode == 'update'
                    # The database is updated. Only the wids in dataw are
                    # modified.
                    label_collection.replace_one(
                        {'idna': w}, {'idna': w, 'value': dataw}, upsert=True)

                print(("\rSaving entry {0} out of {1}. Speed {2} entries" +
                       "/min").format(i + 1, len(df_labels), 60 * (i+1) /
                                      (time.time() - start_time)), end="")

    def migrate2DB(self, df_labels):

        """ Migrate all labeled  urls in data to a mongo db.
            The db collection, if it exists, is droped.

            This function is deprecated because the 'data' dictionary is no
            longer used.

            :Args:
                :data: Data dictionary containing the labels to save.
        """

        # Start a db connection
        dbName = self.db_info['name']
        hostname = self.db_info['hostname']
        user = self.db_info['user']
        pwd = self.db_info['pwd']
        label_coll_name = self.db_info['label_coll_name']
        file2db_mode = self.db_info['file2db_mode']

        # history_coll_name = self.db_info['history_coll_name']
        port = self.db_info['port']

        try:
            print("Trying db connection...")
            client = MongoClient(hostname)
            client[dbName].authenticate(user, pwd)
            db = client[dbName]
            print("Connected to mongodb @ {0}:[{1}]".format(hostname, port))
        except Exception as E:
            sys.exit("Fail to connect mongodb @ {0}:{1}, {2}".format(
                hostname, port, E))

        print("Saving database. This might take a while...")
        start_time = time.time()
        if file2db_mode == 'rewrite':
            # The database is deleted completely and the whole set of
            # labels and predictions in data are loaded
            label_collection = db[label_coll_name]
            label_collection.drop()
        label_collection = db[label_coll_name]

        for i, w in enumerate(df_labels.index):
            # For each wid, create the corresponding data dictionary to
            # send to the db
            dataw = {}
            dataw['relabel'] = df_labels.loc[w, ('info', 'relabel')]
            dataw['marker'] = df_labels.loc[w, ('info', 'marker')]
            dataw['userId'] = df_labels.loc[w, ('info', 'userId')]
            dataw['date'] = df_labels.loc[w, ('info', 'date')]
            dataw['weight'] = df_labels.loc[w, ('info', 'weight')]
            dataw['label'] = {}
            for c in self.categories:
                dataw['label'][c] = df_labels.loc[w, ('label', c)]

            # Store in db.
            if file2db_mode == 'rewrite':
                # Insert data in the database
                label_collection.insert({'idna': w, 'value': dataw})
            else:    # mode == 'update'
                # The database is updated. Only the wids in dataw are
                # modified.
                label_collection.replace_one(
                    {'idna': w}, {'idna': w, 'value': dataw}, upsert=True)

            print(("\rSaving entry {0} out of {1}. Speed {2} entries" +
                   "/min").format(i + 1, len(df_labels), 60 * (i+1) /
                                  (time.time() - start_time)), end="")

    def migrate2file(self):

        ''' Migrate all labeled urls in the mongo db to a pickle data file
            WARNING: THIS METHOD IS UNDER CONSTRUCTION.
        '''

        # Start a db connection
        dbName = self.db_info['name']
        hostname = self.db_info['hostname']
        user = self.db_info['user']
        pwd = self.db_info['pwd']
        label_coll_name = self.db_info['label_coll_name']

        # history_coll_name = self.db_info['history_coll_name']
        port = self.db_info['port']

        try:
            print("Trying db connection...")
            client = MongoClient(hostname)
            client[dbName].authenticate(user, pwd)
            db = client[dbName]
            print("Connected to mongodb @ {0}:[{1}]".format(hostname, port))
        except Exception as E:
            sys.exit("Fail to connect mongodb @ {0}:{1}, {2}".format(
                hostname, port, E))

        print("Open collection...")
        label_collection = db[label_coll_name]
        start_time = time.time()

        # count = 0
        # for i, w in enumerate(data):
        #     # Only the wids with at least some label are migrated to the db.
        #     lab_list = [data[w]['label'][c] for c in self.categories]

        #     if self._yes in lab_list or self._no in lab_list:
        #         count += 1
        #         label_collection.insert({'idna': w, 'value': data[w]})

        #         print ("\rSaving entry {0} out of {1}. Speed {2} entries" +
        #                "/min").format(i + 1, len(data), 60 * (i+1) /
        #                               (time.time() - start_time)),

        # print ""
        # print "Migration finished: {0} out of {1} wids saved in DB".format(
        #     count, len(data))
