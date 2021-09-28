# -*- coding: utf-8 -*-
""" run_labeller is the main script for the labelling application.

    It must be invoked using

        python run_labeler.py [project_folder]

    Created on July, 2015
    Last update: Jan, 2017

    @autor: Jesus Cid.
"""

import ast
import time
import sys
import os
import shutil

# Local imports
from labelfactory.labeling.LabelGUIController import LabelGUIController
from labelfactory.labeling.urlsampler import URLsampler
from labelfactory.ConfigCfg import ConfigCfg as Cfg
from labelfactory.Log import Log
from labelfactory.labeling.dmFiles import DM_Files
from labelfactory.labeling.dmSQL import DM_SQL
from labelfactory.labeling.labelprocessor import LabelProcessor

if sys.version_info.major == 3:
    import tkinter as tk
else:
    import Tkinter as tk

# The following is to capture a user tag automatically.
try:
    import pwd
except ImportError:
    import getpass
    pwd = None

CF_FNAME = "config.cf"
CF_DEFAULT_PATH = "./config.cf.default"


def run_labeler(project_path, url, transfer_mode, user, export_labels,
                num_urls, type_al, ref_class, alth, p_al, p_relabel,
                tourneysize, text2label=None):
    """
    Runs the labelling application for a given labelling project.

    The application reads some input parameters from the config.cf file.

    Args:

        project_path: Path to the lebelling project
        url:       Specific url to label. If None, a batch os labels is
                   selected from the available urls
        transfer_mode: Mode to transfer the new data:
                     - expand (default)
                     - project
                     - contract
        user:      username of the human annotator
        export_labels:
                     - all (export all labels)
                     - rs (export random sampling labels only)
                     - al (export active learning labels only)
        num_urls:   Number of items to label
        ref_class:  Name of the class used as reference for active learning
        alth:       Active learning threshold
        p_al:       Active learning probability
        p_relabel:  Probability of selecting an already labeld sample to
                    relabel
        tourneysize:
        text2label: Method to transform a url into a text to show in the
                    labeling windows
    """

    # To complete the migration to python 3, I should replace all "raw_input"
    # by "input". Transitorily, to preserve compatibility with python 2, I
    # simply rename inut to raw_input
    if sys.version_info.major == 3:
        raw_input2 = input
    else:
        raw_input2 = raw_input

    #######
    # Start
    #######

    tm_options = ['project', 'expand', 'contract']
    if transfer_mode not in tm_options:
        sys.exit("Unrecognized transfer_mode. Available modes are {}".format(
            tm_options))

    # # Check if project folder exists. Otherwise create a default one.
    # if len(sys.argv) > 1:
    #     project_path = sys.argv[1]
    # else:
    #     project_path = raw_input2("Select the (absolute or relative) path " +
    #                               "to the labeling project folder: ")
    if project_path is None:
        project_path = raw_input2("Select the (absolute or relative) path " +
                                  "to the labeling project folder: ")

    if not project_path.endswith('/'):
        project_path = project_path + '/'

    # Check if project folder exists. This is necessary to follow
    if not os.path.isdir(project_path):
        createfolder = raw_input2("Folder " + project_path +
                                  "does not exist. Create? (y/n)")

        if createfolder == "y":
            os.makedirs(project_path)    # Create folder
        else:
            sys.exit("A route to the project folder must be provided.\n")

    #########################
    # Read configuration data
    #########################

    # Check if configuration file existe
    config_path = project_path + CF_FNAME
    if not os.path.isfile(config_path):
        shutil.copy(CF_DEFAULT_PATH, config_path)
        print("The project folder must contain a configuration file.")
        print("A default configuration file has been copied.")
        print("Please check and modify it if necessary before continuing.")

    # Read data from the configuation file
    cf = Cfg(config_path)

    # Data source and destination (options: file, mongodb)
    source_type = cf.get('DataPaths', 'source_type')
    dest_type = cf.get('DataPaths', 'dest_type')
    # This is for backward compatibility
    if source_type is None:
        source_type = 'file'
    if dest_type is None:
        dest_type = 'file'

    # Mongo DB settings
    if source_type == 'sql':
        db_info = {'name': cf.get('DataPaths', 'db_name'),
                   'user': cf.get('DataPaths', 'db_user'),
                   'server': cf.get('DataPaths', 'db_server'),
                   'password': cf.get('DataPaths', 'db_password'),
                   'connector': cf.get('DataPaths', 'db_connector'),
                   'preds_tablename': cf.get(
                        'DataPaths', 'db_preds_tablename'),
                   'label_values_tablename': cf.get(
                        'DataPaths', 'db_label_values_tablename'),
                   'label_info_tablename': cf.get(
                        'DataPaths', 'db_label_info_tablename'),
                   'history_tablename': cf.get(
                        'DataPaths', 'db_history_tablename'),
                   'ref_name': cf.get(
                        'DataPaths', 'db_ref_name'),
                   'mode': cf.get('DataPaths', 'db_mode')}
    elif source_type == 'mongodb' or dest_type == 'mongodb':
        db_info = {'name': cf.get('DataPaths', 'db_name'),
                   'hostname': cf.get('DataPaths', 'db_hostname'),
                   'user': cf.get('DataPaths', 'db_user'),
                   'pwd': cf.get('DataPaths', 'db_pwd'),
                   'label_coll_name': cf.get(
                        'DataPaths', 'db_label_coll_name'),
                   'history_coll_name': cf.get(
                        'DataPaths', 'db_history_coll_name'),
                   'port': cf.get('DataPaths', 'db_port'),
                   'mode': cf.get('DataPaths', 'db_mode')}
    else:
        db_info = None

    # Read file and folder names related to the dataset files.
    if source_type == 'file' or dest_type == 'file':
        file_info = {'project_path': project_path,
                     'input_folder': cf.get('DataPaths', 'input_folder'),
                     'output_folder': cf.get('DataPaths', 'output_folder'),
                     'used_folder': cf.get('DataPaths', 'used_folder'),
                     'dataset_fname': cf.get('DataPaths', 'dataset_fname'),
                     'labelhistory_fname': cf.get('DataPaths',
                                                  'labelhistory_fname'),
                     'labels_endname': cf.get('DataPaths', 'labels_endname'),
                     'preds_endname': cf.get('DataPaths', 'preds_endname'),
                     'urls_fname': cf.get('DataPaths', 'urls_fname'),
                     'export_labels_fname': cf.get('DataPaths',
                                                   'export_labels_fname')}
        # 'recycle_endname': cf.get('DataPaths', 'recycle_endname')
    else:
        # The prediction file is stored in files, so this is still required
        file_info = {'project_path': project_path,
                     'input_folder': cf.get('DataPaths', 'input_folder'),
                     'dataset_fname': cf.get('DataPaths', 'dataset_fname'),
                     'preds_endname': cf.get('DataPaths', 'preds_endname'),
                     'export_labels_fname': cf.get('DataPaths',
                                                   'export_labels_fname')}

    # Type of wid: if 'yes', the wid is computed as a transformed url.
    #              if 'no', the wid is taken equal to the url.
    compute_wid = cf.get('Labeler', 'compute_wid')

    # Type of data: if 'url': the data is a url shown in a browser
    #               if 'txt': the data is printed.
    datatype = cf.get('Labeler', 'datatype')

    # Taxonomy model: 'monolabel' or 'multilabel'
    cat_model = cf.get('Labeler', 'cat_model')

    # List of categories to label.
    categories = ast.literal_eval(cf.get('Labeler', 'categories'))
    # Taxonomy (dependency structure beween categories)
    parentcat = ast.literal_eval(cf.get('Labeler', 'parentcat'))

    fill_with_Other = cf.get('Labeler', 'fill_with_Other')

    # Possible labels for each category
    # yes_label = int(cf.get('Labeler', 'yes_label'))
    # no_label = int(cf.get('Labeler', 'no_label'))
    # unknown_label = int(cf.get('Labeler', 'unknown_label'))
    # error_label = int(cf.get('Labeler', 'error_label'))
    yes_label = cf.get('Labeler', 'yes_label')
    no_label = cf.get('Labeler', 'no_label')
    unknown_label = cf.get('Labeler', 'unknown_label')
    error_label = cf.get('Labeler', 'error_label')
    alphabet = {'yes': yes_label, 'no': no_label, 'unknown': unknown_label,
                'error': error_label}

    unknown_pred = int(cf.get('Labeler', 'unknown_pred'))

    # Check if a user identifier must be requested
    track_user = cf.get('Labeler', 'track_user')

    # WARNING: THE FOLLOWING PARAMETERS HAVE BEEN REMOVED FROM THE CONFIG FILE
    #          BECAUSE THEY ARE NOT SPECIFIC OF A LABELING PROJECT, BUT OF A
    #          LABELING SESSION.

    # # Max. no. of urls to be labeled at each labeling step
    # num_urls = int(cf.get('ActiveLearning', 'num_urls'))

    # # Type of active learning algorithms
    # type_al = cf.get('ActiveLearning', 'type_al')

    # # In multiclass cases, the reference class is the class used by the
    # # active learning algorithm to compute the sample scores.
    # ref_class = cf.get('ActiveLearning', 'ref_class')

    # # Parameters of the active learning algorithm
    # # WARNING: an error may arise if any of these values is not in the config
    # # file.
    # # AL threshold.
    # alth = float(cf.get('ActiveLearning', 'alth'))

    # # Probability of AL sampling. Samples are selected using AL with
    # # probability p, and using random sampling otherwise.
    # p_al = float(cf.get('ActiveLearning', 'p_al'))

    # # Probability of selecting a sample already labelled
    # p_relabel = float(cf.get('ActiveLearning', 'p_relabel'))

    # # Size of each tourney (only for 'tourney' AL algorithm)
    # tourneysize = int(cf.get('ActiveLearning', 'tourneysize'))

    ##########
    # Log file
    ##########

    # Create the log object
    log = Log(project_path + 'log')

    # Starting message for the log output.
    log.info('*****************************')
    log.info('****** WEB LABELER **********')

    #####################################
    # User (human labeler) identification
    #####################################
    if export_labels is None:

        if track_user == 'yes':

            if user is None:
                # Get a default user name taken as the captured user name from
                # the OS.
                if pwd:
                    user0 = pwd.getpwuid(os.geteuid()).pw_name  # For Unix, OS
                else:
                    user0 = getpass.getuser()  # For Windows

                # Ask for a labeler name.
                userId = raw_input2(
                    "Please write you user name [{0}]: ".format(user0))
                if userId == "":
                    userId = user0
                    print("Your User Name is {0}".format(userId))
            else:
                userId = user
                print("Your User Name is {0}".format(userId))
        else:
            userId = None
            if user is not None:
                print(("WARNING: User name {0} will be ignored. Change " +
                       "'track_user' to yes in the config file to use a " +
                       "user name effectively").format(user))

    ############################
    # Pre-processing config data
    ############################

    # Verify that parentclass names are in the category set
    for c in parentcat:
        if c not in categories:
            log.error("Unknown category {} ".format(c) + "in the parentcat " +
                      "dictionary. Revise the configuration file")
            sys.exit()
        elif parentcat[c] not in categories:
            log.error("Unknown category {} ".format(parentcat[c]) + "in the" +
                      " parentcat dictionary. Revise the configuration file")
            sys.exit()

    # Verify that ref_class is one of the given categories
    if ref_class is None and type_al == 'random':
        # This is an arbitrary choice, just to avoid an execution error, but
        # random sampling does not actually require any reference class.
        # Maybe the code should be modified to avoid this artificial assignment
        ref_class = categories[0]
    elif type_al == 'tourney' and ref_class not in categories:
        log.error("The reference class is not in the set of categories. " +
                  "Active learning algorithms require a reference class")
        sys.exit()

    # Include the negative class in the category tree.
    if fill_with_Other.lower() == 'yes':
        extra_class = 'Other'
        while extra_class.lower() in [c.lower() for c in categories]:
            extra_class += 'X'
        categories.append(extra_class)

    # Include categories without parents in the parent dictionary.
    for cat in categories:
        if cat not in parentcat:
            parentcat[cat] = None

    #####################
    # Create main objects
    #####################

    # Data manager object
    if source_type == 'file' or source_type == 'mongodb':
        data_mgr = DM_Files(source_type, dest_type, file_info, db_info,
                            categories, parentcat, ref_class, alphabet,
                            compute_wid, unknown_pred)
    elif source_type == 'sql':
        data_mgr = DM_SQL(source_type, dest_type, file_info, db_info,
                          categories, parentcat, ref_class, alphabet,
                          compute_wid, unknown_pred)

    # Label processing object.
    labelproc = LabelProcessor(categories, parentcat, log, alphabet, cat_model)

    ###################
    # Read all datasets
    ###################

    log.info('Loading data')

    # Load data from the standard datasets.
    df_labels, df_preds, labelhistory = data_mgr.loadData()

    # Load new labels and predictions from the input folder
    log.info("-- Loading new data from the input folder")
    start = time.clock()
    df2_labels, df2_preds = data_mgr.importData()
    save_preds = (len(df2_preds) > 0)
    print(str(time.clock() - start) + ' seconds')

    # Clean and format label dataframe.
    # Replace strange labels by 'unknown'
    df2_labels = labelproc.cleanLabels(df2_labels)
    # Replace 'None' or empty labels by 'unknown'
    df2_labels = labelproc.formatData(df2_labels)

    # Integrate imported labels and predictions into df_preds and df_labels.
    df_preds = labelproc.transferPreds(df2_preds, df_preds, transfer_mode)
    df_labels = labelproc.transferLabels(df2_labels, df_labels)

    # Read dataset
    log.info("-- Loading data from the repository")
    preds, labels, urls, markers, relabels, weights, userIds = \
        data_mgr.getDataset(df_labels, df_preds)

    if len(preds) == 0 and len(labels) == 0:
        log.error("The data repository is empty")
        sys.exit()

    if len(urls) == 0:
        log.error("The data repository has no urls. Add a file " +
                  file_info['urls_fname'] + ".csv to project subfolder " +
                  file_info['input_folder'])
        sys.exit()

    if export_labels is None:

        ##########################
        # Select urls for labeling
        ##########################

        # Start the sampler.
        # In this process, the controller creates the active learning objects
        sampler = URLsampler(
            ref_class, preds, labels, urls, markers, weights, alphabet,
            type_al, alth, p_al, p_relabel, tourneysize)

        if url is None:
            # Get the first batch of urls to be labelled.
            # The selecction of urls is driven by the active learning algorithm
            newurls, newwids, newqueries = sampler.get_urls_batch(
                max_urls=num_urls)
        else:
            # Compute wid of the target url
            if compute_wid in ['yes', 'www']:
                wid = data_mgr.computeWID(url, mode=compute_wid)
            else:
                wid = url
            newurls, newwids, newqueries = sampler.get_single_url(
                target_wid=wid)

        ########################
        # Show first url and GUI
        ########################
        # Start the controler.
        controller = LabelGUIController(newurls, newwids, newqueries, preds,
                                        labels, urls, categories, alphabet,
                                        datatype, cat_model, parentcat,
                                        text2label)

        # Take the first url to work with and show it in a web browser
        controller.takeandshow_sample()

        # Build  and launch GUI
        log.info("Starting GUI ....")
        log.info("Type of Active Learning algorithm: " + type_al)
        root = tk.Tk()
        root.title('The Label Factory')

        controller.init_view(root)

        # At this point the GUI is running.
        # The code that follows only runs when the GUI is closed.

        ################################
        # Update labels and save results
        ################################

        # Remove unlabeled urls
        controller.update_label_vector()

        # Transfer new labels to the label dataframe
        log.info('No more urls to label. Saving new labels...')
        labelproc.transferNewLabels2(controller.newlabels, df_labels, userId)
        labelproc.transferNewWeights2(sampler.weights, df_labels)
        # Record new labeling events into the label history
        if source_type == 'sql':
            # Making a difference with labelhistory (1) labelrecord is a
            # pandas dataframe, not a dict, and (2) it contains the labelling
            # events of this session only
            labelrecord = labelproc.formatLabelRecords(controller.newlabels,
                                                       userId)
        else:
            labelproc.transferLabelRecords(controller.newlabels, labelhistory,
                                           userId)

        # Save results in a mongo DB.
        if dest_type == 'mongodb':
            # Save data and label history into db
            log.info("-- Saving data in mongodb")
            start = time.clock()
            if db_info['mode'] == 'rewrite':
                data_mgr.saveData(df_labels, df_preds, labelhistory,
                                  'mongodb')
            elif db_info['mode'] == 'update':
                # Create a new dataframe contatining the new labels only
                # - Create empty label dataframe (df_newpreds won't be used)
                df_newlabels, df_newpreds = data_mgr.get_df({}, {})
                # - Transfer newlabels
                labelproc.transferNewLabels2(
                    controller.newlabels, df_newlabels, userId)
                labelproc.transferNewWeights2(sampler.weights, df_newlabels)
                # - Save newlabels in the DB
                data_mgr.saveData(df_newlabels, df_newpreds, labelhistory,
                                  'mongodb')
            log.info(str(time.clock() - start) + ' seconds')

        # Save data and label history into file or sql
        log.info("-- Saving data in files...")
        start = time.clock()

        if source_type == 'sql':
            # Add the new labelling events in the sql db history
            data_mgr.saveData(df_labels, df_preds, labelrecord, save_preds)
        else:
            # Store all label history.
            data_mgr.saveData(df_labels, df_preds, labelhistory,
                              save_preds=save_preds)
        log.info(str(time.clock() - start) + ' seconds')

        # ### Export data to other formats (csv files)
        for cat in categories:
            log.info("-- Exporting labels from category {0}.".format(cat))
            start = time.clock()
            data_mgr.exportLabels(df_labels, cat)
            log.info(str(time.clock() - start) + ' seconds')

        # Export history
        if source_type != 'sql':
            log.info("-- Exporting history")
            start = time.clock()
            data_mgr.exportHistory(labelhistory)
            log.info(str(time.clock() - start) + ' seconds')

    else:

        # Export labels to a csv file
        f = data_mgr.exportlabels_file
        print(f)
        if export_labels == 'all':
            df_labels['label'].to_csv(f)
        elif export_labels == 'rs':
            df_labels[df_labels[('info', 'marker')] == 0]['label'].to_csv(f)
        elif export_labels == 'al':
            df_labels[df_labels[('info', 'marker')] == 1]['label'].to_csv(f)
        else:
            log.error('-- Unknown label export option. See help to check' +
                      ' available options')
