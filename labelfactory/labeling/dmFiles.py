#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python libraries
from __future__ import print_function
import os
import sys
# import cPickle as pickle
import pickle
import json
import shutil
import pandas as pd

# import copy
from datetime import datetime

import ipdb

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

        # Read label history
        if os.path.isfile(self.labelhistory_file):
            if sys.version_info.major == 3:
                try:
                    with open(self.labelhistory_file, 'rb') as f:
                        labelhistory = pickle.load(f)
                except:
                    print("---- Cannot read a pkl file version for " +
                          "Python 2. Trying to load a json file.")
                    print("---- An ERROR will arise if json file does " +
                          "not exist.")
                    print("---- IF THIS IS THE CASE, YOU SHOULD DO \n" +
                          "         python run_pkl2json.py [path to "
                          "labelhistory]\n" +
                          "     FROM PYTHON 2 BEFORE RUNNING THIS SCRIPT.")
                    fname = self.labelhistory_file.replace('.pkl', '.json')
                    with open(fname, 'r', encoding='latin1') as f:
                        labelhistory = json.load(f)

                    # Convert date field, which is in string format, to
                    # datetime format.
                    for url, events in labelhistory.items():
                        for idx, record in events.items():
                            labelhistory[url][idx]['date'] = (
                                datetime.strptime(
                                    labelhistory[url][idx]['date'],
                                    "%Y-%m-%dT%H:%M:%S.%f"))
            else:
                with open(self.labelhistory_file, 'r') as f:
                    labelhistory = pickle.load(f)

        else:
            labelhistory = {}

        # Load dataset files.
        if (os.path.isfile(self.datalabels_file) and
                os.path.isfile(self.datapreds_file)):
            # Load label and prediction dataframes stored in pickle files
            df_labels = pd.read_pickle(self.datalabels_file)
            df_preds = pd.read_pickle(self.datapreds_file)
        elif os.path.isfile(self.dataset_file):
            # If there is an old dataset structure, read data there and
            # convert it into the label and preds dataframes
            with open(self.dataset_file, 'r') as handle:
                data = pickle.load(handle)
            df_labels, df_preds = self.get_df(data, labelhistory)
        else:
            # Warning: the next 4 commands are duplicated in importData.
            # Make sure taht any changes here are also done there
            # (I know, this is not a good programming style..)
            info = ['marker', 'relabel', 'weight', 'userId', 'date']
            arrays = [len(info)*['info'] + len(self.categories)*['label'],
                      info + self.categories]
            tuples = list(zip(*arrays))
            mindex = pd.MultiIndex.from_tuples(tuples)
            # Create empty pandas dataframe
            df_labels = pd.DataFrame(self._unknown, index=[],
                                     columns=mindex)
            # df_labels = None
            # df_preds = None
            cols = ['url'] + self.categories
            df_preds = pd.DataFrame(index=[], columns=cols)
            print(df_preds)

        return df_labels, df_preds, labelhistory

    def saveData(self, df_labels, df_preds, labelhistory, save_preds=True):

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
        df_preds.to_pickle(self.datapreds_file)

