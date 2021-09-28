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
import numpy as np

# import copy
from datetime import datetime

import ipdb

# Services from the project
# sys.path.append(os.getcwd())


def addSlash(p):
    """
    If string p does not end with character '/', it is added.
    """
    if p.endswith('/'):
        return p
    else:
        return p + '/'


class BaseDM(object):

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

    def __init__(self, source_type, dest_type, file_info, db_info,
                 categories, parentcat, ref_class, alphabet,
                 compute_wid='yes', unknown_pred=0):

        """
        Stores files, folder and path names into the data struture of the
        DataManager object.

        :Args:
            :source_type: 'file' if the data sources are in files
                          'db' if data is stored in a database
            :dest_type: 'file' if the data sources are in files
                        'db' if data is stored in a database
            :file_info: a dictionary containing strings about the names of
                        path and folders. Fields are:
                :project_path: string with the path to all input and ouput
                               files
                :input_folder: Name of folder containing input data
                :output_folder: Name of folder containing output data
                :used_folder: Name of the folder containing copies of old data
                              files
                :dataset_fname: Name of file containing the data
                :labelhistory_fname: Name of the label history file
                :labels_endname: Suffix of the label file
                :preds_endname: Suffix of the prediction files
                :urls_fname: Name of the file containing urls only
            :db_info: a dictionary containing information about the database.
                      Fields depend on the typ of database. For instance,
                      for mongo db, they are:
                :name: Name of the database
                :hostname: Name of the database host
                :user: User name
                :pwd: Password
                :label_coll_name: Name of the label collection
                :history_coll_name: Name of the history collection
                :port: Port
                :mode: mode of saving data in the database.
            :categories: Set of categories
            :parentcat: Dictionary defining the hyerarchical category structure
            :ref_class: Name of the category that the predictions refer to
            :alphabet: Possible labels for each category
            :compute_wid: Type of wid. If yes, the wid is a transformed url.
                          In no, the wid is equal to the url.
            :unknown_pred: Default value for unknown predictions.
        """

        self.source_type = source_type

        # Set variables about files and folders
        if file_info is not None:
            project_path = file_info['project_path']
            input_folder = file_info['input_folder']
            output_folder = file_info['output_folder']
            used_folder = file_info['used_folder']

            # Revise path and folder terminations
            project_path = addSlash(project_path)
            input_folder = addSlash(input_folder)
            output_folder = addSlash(output_folder)
            used_folder = addSlash(used_folder)

            # Folder containing all files related to labeling.
            self.directory = project_path

            # Check input and output folders
            self.input_path = os.path.join(self.directory, input_folder)
            if not os.path.isdir(self.input_path):
                os.makedirs(self.input_path)

            self.output_path = os.path.join(self.directory, output_folder)
            if not os.path.isdir(self.output_path):
                os.makedirs(self.output_path)

            self.used_path = os.path.join(self.directory, used_folder)
            if not os.path.isdir(self.used_path):
                os.makedirs(self.used_path)

            # Store names of files in the input folder
            self.labels_endname = file_info['labels_endname']
            self.preds_endname = file_info['preds_endname']
            self.urls_fname = file_info['urls_fname']
            self.export_labels_fname = file_info['export_labels_fname']

            # Store file names
            self.dataset_fname = file_info['dataset_fname']
            self.datalabels_fname = self.dataset_fname + self.labels_endname
            self.datapreds_fname = self.dataset_fname + self.preds_endname
            self.dataset_file = os.path.join(
                self.directory, self.dataset_fname + '.pkl')
            self.datalabels_file = os.path.join(
                self.directory, self.datalabels_fname + '.pkl')
            self.datapreds_file = os.path.join(
                self.directory, self.datapreds_fname + '.pkl')
            self.labelhistory_fname = file_info['labelhistory_fname']
            self.labelhistory_file = os.path.join(
                self.directory, self.labelhistory_fname + '.pkl')
            self.exportlabels_file = os.path.join(
                self.output_path, self.export_labels_fname + '.csv')

        # Store info about the database
        self.db_info = db_info

        # Type of wid
        self.compute_wid = compute_wid

        # Store category names
        self.categories = categories
        self.ref_class = ref_class

        self._yes = alphabet['yes']
        self._no = alphabet['no']
        self._unknown = alphabet['unknown']
        self._error = alphabet['error']

        # Default value for predictions
        self._unknown_p = unknown_pred

    def get_df(self, data, labelhistory):

        """ Converts the data dictionary used in former versions of the web
            labeler into the label and prediction dataframes.

            :Args:
                :data: Data dictionary of labels and predicts.
                :labelhistory: The labelhistory is used to get the date of the
                       last labelling event for each wid.

            :Returns:
                :df_labels:
                :df_preds:
        """

        # #######################
        # Compute preds dataframe

        # Create pandas dataframe structure
        wids = data.keys()
        cols = ['url'] + self.categories
        df_preds = pd.DataFrame(index=wids, columns=cols)
        # Fill urls
        urls = [data[w]['url'] for w in wids]
        df_preds['url'] = urls
        for w in wids:
            if 'pred' in data[w]:
                for c in self.categories:
                    df_preds[c] = [data[w]['pred'][c] for w in wids]

        # ########################
        # Compute labels dataframe

        # Create multiindex for the label dataframe
        info = ['marker', 'relabel', 'weight', 'userId', 'date']
        arrays = [len(info)*['info'] + len(self.categories)*['label'],
                  info + self.categories]
        tuples = list(zip(*arrays))
        mindex = pd.MultiIndex.from_tuples(tuples)

        # Create empty pandas dataframe
        df_labels = pd.DataFrame(columns=mindex)

        # Fill dataframe with wids that have some known label
        for w in data:
            # Only the wids with some label are stored.
            lab_list = data[w]['label'].values()
            if self._yes in lab_list or self._no in lab_list:
                # Fill row with category labels
                for c in self.categories:
                    if c in data[w]['label']:
                        df_labels.loc[w, ('label', c)] = data[w]['label'][c]
                    else:
                        df_labels.loc[w, ('label', c)] = self._unknown
                # Fill rows with category info
                for i in info:
                    if i != 'date':
                        df_labels.loc[w, ('info', i)] = data[w][i]
                    else:
                        # Read the date, if it exists, in the label history
                        if w in labelhistory:
                            # Take the most recent labeling date
                            record = max(labelhistory[w])
                            df_labels.loc[w, ('info', i)] = (
                                labelhistory[w][record]['date'])

        return df_labels, df_preds

    def df2data(self, df_labels, df_preds):

        """ Converts the label and prediction dataframes into the data
            dictionary used in former versions of the web labeler.

            :Args:
                :df_labels:
                :df_preds:

            :Returns:
                :data: Data dictionary of labels and predicts.
                :labelhistory: The labelhistory is used to get the date of the
                       last labelling event for each wid.
        """

        data = {}

        wids = df_preds.index.values
        for i, w in enumerate(wids):
            print('Processing {0} wids out of {1}\r'.format(i, len(wids)),
                  end="")
            dataw = df_preds.loc[w]
            data[w] = {}
            data[w]['url'] = dataw['url']
            data[w]['pred'] = {}
            for c in self.categories:
                data[w]['pred'][c] = dataw[c]

        wids = df_labels.index.values
        for w in wids:
            print ('Processing {0} wids out of {1}\r').format(i, len(wids)),
            dataw = df_preds.loc[w]
            data[w]['marker'] = dataw['marker']
            data[w]['relabel'] = dataw['relabel']
            data[w]['weight'] = dataw['weight']
            data[w]['userId'] = dataw['userId']
            data[w]['date'] = dataw['date']
            for c in self.categories:
                data[w]['pred'][c] = dataw['label'][c]

        return data

    def importData(self):

        """ Read data from the input folder.
            Only labels with positive or negative label are loaded

            :Args:
                :-: None. File locations and the set of categories are taken
                          from the class attributes

            :Returns:
                :df_labels: Pandas dataframe of labels
                :df_preds: Pandas dataframe of predictions
        """

        # Warning.
        if 'url' in self.categories or 'uid' in self.categories:
            sys.exit("ERROR: url and uid are reserved words. They cannot be " +
                     "used as category names.")

        # Initialize dictionaries
        labels = {}
        preds = {}

        # ###############
        # Read data files

        # Read new labels and predictions
        for cat in self.categories:
            labels[cat] = self.importLabels(cat)
            preds[cat] = self.importPredicts(cat)

        # Read predictions for a csv file directly in a pandas dataframe
        df_imported_preds = self.importPredicts()
        if df_imported_preds is not None:
            wid_list_csv = df_imported_preds.index.tolist()
        else:
            wid_list_csv = []

        # # Transfer the dictionary [wid][cat] in predsall into preds, which is
        # # a dictionary [cat][wid]
        # for wid in predsall:
        #     for cat in predsall[wid]:
        #         if cat in self.categories:
        #             # Note that these predictions override those in the
        #             # 'pkl' files
        #             preds[cat][wid] = predsall[wid][cat]
        #         else:
        #             print("---- WARNING: The prediction file contains " +
        #                   "unknown category " + cat + ", ignored")

        # Read new urls (without preditions or labels)
        urls_dict = self.importURLs()

        # ####################################
        # Import predictions from pickle files

        # Capture all wids in pred or label files
        print("---- Capturing wids from prediction files ")
        wid_set = set(urls_dict.keys())
        for cat in self.categories:
            wid_set = wid_set | set(preds[cat].keys())
        wid_set = wid_set | set(wid_list_csv)
        wid_set = list(wid_set)

        # Create the dictionary structure for data2
        print("---- Building predictions structure ")
        cols = ['url'] + self.categories
        df2_preds = pd.DataFrame(index=wid_set, columns=cols)

        # Join all urls and predictions in a dataset struture
        # First, insert data from the dictionary of urls
        df2_preds['url'].update(pd.Series(urls_dict))

        # Second, insert data from dictionaries of predictions
        ntot = len(wid_set)
        for cat in self.categories:
            urls_dict = {}
            pred_dict = {}
            for nk, wid in enumerate(preds[cat]):
                if ntot > 10000 and nk % 100 == 0:
                    print(('---- ---- Processing {0} wids out of {1} from ' +
                           'category {2}       \r').format(nk, ntot, cat),
                          end="")

                urls_dict[wid] = preds[cat][wid]['url']
                pred_dict[wid] = preds[cat][wid]['pred']

            df2_preds['url'].update(pd.Series(urls_dict))
            df2_preds[cat].update(pd.Series(pred_dict))

        # ################################
        # Import predictions from csv file

        if df_imported_preds is not None:

            # Capture all wids in the dataframe
            print("---- Capturing wids from prediction csv file ")

            # Categories to import:
            cat_list = list(set(self.categories) &
                            set(df_imported_preds.columns.tolist()))

            cat_unk = (set(df_imported_preds.columns.tolist()) -
                       set(self.categories))
            if len(cat_unk) > 0:
                print("WARNING: There as unknown categories in the " +
                      "prediction files: {}".format(cat_unk))

            # Insert predictions imported from csv file in to df2_preds
            # ntot = len(wid_list_csv)
            # TO-DO: maybe a loop is not necessary to merge these dataframes.
            print('---- ---- Processing {} wids...'.format(ntot))
            df2_preds.loc[wid_list_csv, cat_list] = (
                df_imported_preds.loc[wid_list_csv, cat_list])
            # for cat in cat_set:
            #     ipdb.set_trace()
            #     df2_preds.loc[wid_list_csv, cat_list] = (
            #         df_imported_preds.loc[wid_list_csv, cat])
            print('... done.')
            # for n, w in enumerate(wid_set):

            #     if ntot > 10000 and n % 100 == 0:
            #         print('Processing {0} wids out of {1}\r'.format(n, ntot),
            #               end="")

            #     for cat in cat_set:
            #         df2_preds.loc[w, cat] = df_imported_preds.loc[w, cat]

        # #############
        # Import labels

        # Capture all wids in pred or lfiles
        print("---- Capturing wids from label files ")
        wid_set = set()
        for cat in self.categories:
            wid_set = wid_set | set(labels[cat].keys())
        wid_set = list(wid_set)

        # Create the dictionary structure for data2
        print("---- Building dataset structure ")

        # Warning: the next 4 commands are duplicated in loadData.
        # Make sure taht any changes here are also done there
        # (I know, this is not a good programming style..)
        info = ['marker', 'relabel', 'weight', 'userId', 'date']
        arrays = [len(info)*['info'] + len(self.categories)*['label'],
                  info + self.categories]
        tuples = list(zip(*arrays))
        mindex = pd.MultiIndex.from_tuples(tuples)
        # Create empty pandas dataframe
        df2_labels = pd.DataFrame(self._unknown, index=wid_set,
                                  columns=mindex)

        # Second, insert data from dictionaries of predictions and labels
        ntot = len(wid_set)
        for cat in self.categories:

            for nk, w in enumerate(labels[cat]):
                if ntot > 10000 and nk % 100 == 0:
                    print(('Processing {0} wids out of {1} from ' +
                           'category {2}       \r').format(nk, ntot, cat),
                          end="")

                df2_labels.loc[w, ('label', cat)] = labels[cat][wid]

        print("---- End of import ")

        return df2_labels, df2_preds

    def importLabels(self, category):

        """ Get dictionary of labels relative to a given category

            :Args:
                :category: The category to load.

            :Returns:
                :labels: Dictionary of labels
        """

        # If there are no labels to return, an empty dict is returned.
        labels = {}

        # Read raw data from file, if it exists
        labels_file = self.input_path + category + self.labels_endname + '.csv'
        if os.path.isfile(labels_file):
            print("---- Importing labels from category " + category)

            data = []
            with open(labels_file, "r") as f:
                data = f.readlines()
                f.close()

            # Structure raw data into labels_dict dictionary
            data = [d.replace("\n", "") for d in data]

            for d in data:

                d = d.split(";")
                # Remove \r appearing in some labels.
                d[1] = d[1].replace("\r", "")

                # Store the label. Note that if some wid is duplicated.
                # The latter label records overwrite the former ones.
                labels.update({d[0]: d[1]})
                # WARNING: In former versions, labels were converted to
                # integers using
                # labels.update({d[0]: int(d[1])})
                # Now they are stored as strings.

            # Move the label file to the "used" folder.
            date_str = datetime.now().strftime("%Y%m%d%H%M%S%f")
            dest_file = self.used_path + category + self.labels_endname + \
                '_in' + date_str + '.csv'
            shutil.move(labels_file, dest_file)

        return labels

    def importPredicts(self, category=None):

        """ Get dictionary of predictions relative to a given category

            :Args:
                :category: The category to load (from a pkl file)
                           If None, all categories are read from a unique
                           csv file

            :Returns:
                :preds: Dictionary of predictions
                        - If category is not None, preds[wid] has the
                          prediction for url wid about the given category.
                        - If category is not None, preds is a dataframe
                          with the wid as uid column and one column with
                          predictions for each category.
        """

        # The default category is the reference class used by the
        # active learning algorithm
        if category is None:

            # Default output
            preds = None

            # Read raw data
            # preds_file = self.input_path + self.preds_endname + '.json'
            preds_file = self.input_path + self.preds_endname + '.csv'

            # Load predictions from file, if it exists
            if os.path.isfile(preds_file):
                print("---- Importing multicategory predictions")
                with open(preds_file, 'r') as f:
                    # preds = json.load(f)
                    preds = pd.read_csv(f)

                preds.set_index('uid', inplace=True)

                # Move the preditions file to the "used" folder.
                date_str = datetime.now().strftime("%Y%m%d%H%M%S%f")
                dest_file = self.used_path + self.preds_endname + \
                    '_in' + date_str + '.csv'
                #     '_in' + date_str + '.json'
                shutil.move(preds_file, dest_file)

        else:

            # Default output
            preds = {}

            # Read raw data
            preds_file = (self.input_path + category + self.preds_endname +
                          '.pkl')

            # Load predictions from file, if it exists
            if os.path.isfile(preds_file):
                print("---- Importing predictions from category " + category)
                with open(preds_file, 'r') as f:
                    preds = pickle.load(f)

                # Move the preditions file to the "used" folder.
                date_str = datetime.now().strftime("%Y%m%d%H%M%S%f")
                dest_file = self.used_path + category + self.preds_endname + \
                    '_in' + date_str + '.pkl'
                shutil.move(preds_file, dest_file)

        return preds

    def importURLs(self):

        """ Reads a list of urls from a file, computes a wid (web identifier)
            for each one of them and returns it in a dictionary {wid:url}.
            If self.computeWID is None, the wid is equal to the url.

            :Returns:
                :url_dict: Dictionary of urls
        """

        # Initialize ouput dictionary (this is the default output if no
        # urlfile exists)
        url_dict = {}

        # Read raw data from file, if it exists
        urls_file = self.input_path + self.urls_fname + '.csv'

        if os.path.isfile(urls_file):
            print("---- Reading new URLs")
            # data = []
            # with open(urls_file, "r") as f:
            #     data = f.readlines()
            #     f.close()
            data = pd.read_csv(urls_file, header=None)
            data = data[0].tolist()

            data = [d.replace("\r", "") for d in data]
            data = [d.replace("\n", "") for d in data]

            for url in data:

                # Transform url into wid
                if self.compute_wid in ['yes', 'www']:
                    wid = self.computeWID(url, mode=self.compute_wid)
                else:
                    wid = url

                # Store the pair wid:url in the ouput dictionary.
                # The latter wid records overwrite the former ones.
                url_dict.update({wid: url})

            # Move the urls file to the "used" folder.
            date_str = datetime.now().strftime("%Y%m%d%H%M%S%f")
            dest_file = self.used_path + self.urls_fname + \
                '_in' + date_str + '.csv'
            shutil.move(urls_file, dest_file)

        return url_dict

    def computeWID(self, url, mode='yes'):

        """ Computes a web identifier for a given url
            The computations are based in a code used by another software
            project (IaD2015_B2C)

            :Args:
                url:   Input url
                mode:  If 'yes', a complete transformation is done, by removing
                          the part of 'http://www.' that exists in the original
                          url
                       If 'www', only an initial 'www.' is removed

            WARNING:
            The url-to-wid transformation is not one-to-one: in some bad-luck
            cases, two different urls could be transformed into the same wid.
        """

        if mode == 'yes':
            wid = url.lower()
            wid = wid.replace("http://", '')
            wid = wid.replace("//", '')
            wid = wid.replace("www.", '')
            wid = wid.replace(".", '_')

            # This replacement does not affect the url if it is a domain site.
            # But it may transform the url of specific web pages.
            wid = wid.replace("/", "__")
        elif mode == 'www':

            if url[0:4] == 'www.':
                wid = url[4:]
            else:
                wid = url

        else:

            print('---- WARNING: The transformation mode is unknown.')
            print('----          The wid is taken as the url without changes')
            wid = url

        return wid

    def getDataset(self, df_labels, df_preds):

        """ Read the whole dataset from dataframes containing predictions,
            labels and the labeling history.

            :Args:
                :df_labels: Pandas dataframe of labels
                :df_preds:  Pandas dataframe of predictions

            :Returns:
                :preds:    Dict of predictions
                :labels:   Dict of labels
                :urls:     Dict of urls
                :markers:  Dict of markers
                :relabels: Dict of relabels
                :weights:  Dict of weights
        """

        # Initialize dictionaries
        preds = dict((c, {}) for c in self.categories)
        labels = dict((c, {}) for c in self.categories)

        # Read labels and predictions for all categories
        for cat in self.categories:

            preds[cat] = df_preds[cat].to_dict()

            for wid in preds[cat]:
                # Get prediction
                if preds[cat][wid] is None or np.isnan(preds[cat][wid]):
                    # Default value for none predictions. Not clear if this
                    # is a good options.
                    preds[cat][wid] = self._unknown_p

            labels[cat] = df_labels[('label', cat)].to_dict()

        # Get urls, markers and relabels
        urls = df_preds['url'].to_dict()
        markers = df_labels[('info', 'marker')].to_dict()
        relabels = df_labels[('info', 'relabel')].to_dict()
        weights = df_labels[('info', 'weight')].to_dict()

        # The following assignment can cause an error because dataset files
        # from older versions of this sw did not include a 'userId' entry.
        if 'userId' in df_labels['info']:
            userIds = df_labels[('info', 'userId')].to_dict()
        else:
            userIds = None

        return preds, labels, urls, markers, relabels, weights, userIds

    def getHistory(self):

        """ Loads the history file, and creates a dictionary recording the last
            labelling event for each url.

            If the history file does not exist, an empty dictionary is returned

            :Returns:
                :hdict: A dictionary containing, for every url identifier (wid)
                        the record of the last time it was labeled

            WARNING: I THINK THIS FUNCTION IS NO LONGER USED, BECAUSE THE DATA
                     ABOUT THE LAST LABEL EVENT IS STORED IN THE LABELS
                     DATAFRAME. THUS, IT CAN BE LIKELY REMOVED
        """

        #################################
        # Read the whole labeling history

        # Name of the fie containing the recorded labeling history
        file_labelh = self.directory + self.labelh_filename + '.pkl'

        if os.path.isfile(file_labelh):

            # Read data from history pickle file.
            with open(file_labelh, 'r') as handle:
                labelh_list = pickle.load(handle)

        else:

            print('Histórico de etiquetado no disponible.')

            # Create an incomplete history list from the current label
            # dictionary
            labelh_list = []

        ######################################################
        # Create dictionary with the last record for every wid
        hdict = {}
        for record in labelh_list:
            wid = record['wid']
            hdict[wid] = {'url': record['url'],
                          'label': record['label'],
                          'marker': record['marker'],
                          'date': record['date']}

            if 'userId' in record:
                hdict[wid]['userId'] = record['userId']
            else:
                hdict[wid]['userId'] = None

        return hdict

    def exportLabels(self, df_labels, category):

        """ Export labels to a csv file.

            :Args:
                :df_labels:
                :category: Category to be exported.

        :Returns:
            :-: None. The result is the saved csv file.
        """

        # path_labels :Keeps web identifiers and labels only.
        path_labels = self.output_path + category + self.labels_endname + \
            '.csv'

        # Keep a copy of the original file of labels, just in case some
        # mistakes are made during labelling
        # date_str = datetime.now().strftime("%Y%m%d%H%M%S%f")
        # dest_file = self.used_path + category + self.labels_endname + \
        #     '_out' + date_str + '.csv'
        # if os.path.isfile(path_labels):
        #     shutil.move(path_labels, dest_file)

        # Copy the data to be exported in a list.
        # Note that wids with an unknown label are not saved.
        data_out = []
        for wid in df_labels.index:
                data_out.append(
                    wid + ";" + str(df_labels.loc[wid, ('label', category)]))

        # Export data
        with open(path_labels, "w") as f:
            f.writelines(list("%s\r\n" % item for item in data_out))

    def exportHistory(self, labelhistory):

        """ Saves label history in a pickle and a csv files.

        Args:
            :labelhistory:

        Returns:
            Nothing
        """

        # Three label files will be stored:
        #    path_labelh (.csv): stores a whole record for each label.
        path_labelh = self.output_path + self.labelhistory_fname + '.csv'

        # Keep a copy of the original file of labels, just in case some
        # mistakes are made during labelling
        date_str = datetime.now().strftime("%Y%m%d%H%M%S%f")
        dest_file = self.used_path + self.labelhistory_fname + '_out' + \
            date_str + '.csv'
        if os.path.isfile(path_labelh):
            shutil.move(path_labelh, dest_file)

        # Append new label record to the history file
        print('Updating history file')
        data_out = []

        for wid in labelhistory:

            for tid in labelhistory[wid]:

                tags = list(labelhistory[wid][tid])
                tags.sort()

                text = ''.join(str(t) + ";" + str(labelhistory[wid][tid][t]) +
                               ";" for t in tags)
                data_out.append(text)

        with open(path_labelh, "w") as f:
            f.writelines(list("%s\r\n" % item for item in data_out))

