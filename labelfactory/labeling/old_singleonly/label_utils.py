#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Python libraries
import os
import sys
# import numpy as np
import cPickle as pickle
import copy

from datetime import datetime

# Services from the project
sys.path.append(os.getcwd())

# Label coding
_YES_LABEL = 1
_NO_LABEL = -1
_ERROR_LABEL = -99


def curate_url(url):

    """ Put the url into a somewhat standard manner.

    Removes ".txt" extension that sometimes has, special characters and http://

    Args:
        url: String with the url to curate

    Returns:
        curated_url
    """

    curated_url = url
    curated_url = curated_url.replace(".txt", "")
    curated_url = curated_url.replace("\r", "")  # remove \r and \n
    curated_url = curated_url.replace("\n", "")  # remove \r and \n
    curated_url = curated_url.replace("_", ".")

    # remove "http://" and "/" (probably at the end of the url)
    curated_url = curated_url.replace("http://", "")
    curated_url = curated_url.replace("www.", "")
    curated_url = curated_url.replace("/", "")

    return curated_url

# def transfer_labels(directory, base_name,  newlabels, categories):

#     """ transfer_labels from the new labeled category to the other categories
#     """

#     source_class = base_name

#     categories.remove(base_name)
#     print 'Origin category = ', base_name

#     # Change markers.
#     # All samples are marked with 2 to show that they have been taken by class
#     # transfer.
#     for wid in newlabels.keys():
#         newlabels[wid]['marker'] = 2

#     for dest_class in categories:

#         print 'dest_class = ', dest_class

#         # Load input files
#         # dest_dir = directory + base_name + '/'
#         # _, dest_labels_dict, dest_labelhistory = load_input_files(
#         #    dest_dir, base_name)
#         dest_dir = directory + dest_class + '/'
#         _, dest_labels_dict, dest_labelhistory = load_input_files(dest_dir,
#                                                                   dest_class,
#                                                                   False)

#         print 'dest_dir = ', dest_dir

#         # Label transfer criterium
#         if source_class == 'negocio' and dest_class == 'b2c':
#             orig_label = _NO_LABEL
#             dest_label = _NO_LABEL
#         elif source_class == 'b2c' and dest_class == 'negocio':
#             orig_label = _YES_LABEL
#             dest_label = _YES_LABEL
#         else:
#             orig_label = _YES_LABEL
#             dest_label = _NO_LABEL

#         # Take labels from the origin category
#         labelstoload = {}
#         print len(newlabels.keys())

#         for wid in newlabels.keys():

#             if newlabels[wid]['label'] == orig_label:

#                 # Transfer the label with the new label value
#                 labelstoload[wid] = copy.copy(newlabels[wid])
#                 labelstoload[wid]['label'] = dest_label

#         # Save the new labels
#         save_labels(labelstoload, dest_dir, dest_class, '_labels_dict.csv')


def backup_labels(dir_in, dir_out=None):

    """Save a backup of current labels in a pickle file

    The dictionary is saved in a pickle file with
    the name "backup_labels_datetime.pkl" inside dir_out.
    If dir_out is undefined then dir_out = dir_in


    Args:
        dir_in: string with the path of the directory contaning the labels.
        dir_out: string with the path of the directory to put the pickle file.
    Returns:
        Boolean, false if there is no labels to backup, true if correct

    Raises:
        OSError: if problem reading the directories.
    """

    if dir_out is None:
        dir_out = dir_in

    try:
        current_label_keys = os.listdir(dir_in)
    except OSError:

        if not os.path.isdir(str(dir_in)):
            print 'No hay etiquetas para hacer copia de seguridad'
            return False
        else:
            raise

    current_label_values = []

    for url in current_label_keys:
        with open(dir_in + '/' + url, 'r') as f:
            for line in f:
                current_label_values.append(line)

    labels = dict(zip(current_label_keys, current_label_values))

    # Make the output dir if it does not exists
    try:
        os.makedirs(dir_out)
    except OSError:
        if not os.path.isdir(dir_out):
            raise

    path_out = (dir_out + '/backup_labels_' +
                datetime.today().isoformat() + ".pkl")

    with open(path_out, 'wb') as fp:
        pickle.dump(labels, fp)

    print 'Copia de seguridad salvada en ' + path_out

    return True


def text_file_to_urls_list(path):

    """Reads an url text file with one url per line.

    Args:
        path. String with the path of the urls file

    Returns:
        urls. List with the "curated urls"
    """
    urls = []
    with open(str(path), 'r') as f:
        for line in f:
            url = curate_url(line)  # Save in an standard manner
            urls.append(url)
    return urls


def load_csv_file(path):

    with open(path, "r") as f:
        data = f.readlines()

    data = [dd.replace("\n", "") for dd in data]

    labels_dict = {}

    for d in data:
        d = d.split(";")
        labels_dict.update({d[0].replace(".", "_"): int(d[1])})

    return labels_dict


def save_csv_file(data, path):

    data_out = []

    for key in data.keys():
        value = data[key]
        # The next command is obsolete. Replacements are no longer applied
        # data_out.append(key.replace("_", ".") + ";" + str(value))
        data_out.append(key + ";" + str(value))

    f = open(path, "wb")
    f.writelines(list("%s\r\n" % item for item in data_out))
    f.close()


def load_pickle_file(path):
    """Loads data from a pickle file
    Args:
        path: sting with the path of the  backup file in pickle format

    Returns:
        data: Structure containing the data in the pickle file.

    """

    data = None

    try:
        pkl_file = open(path, 'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()

    except IOError:

        print "Fichero Pickle en " + path + " no existe"

    return data


def save_pickle_file(varible, path):
    """Saves data into a pickle file

    Args:
        varible: data to save
        path: sting with the path of the  backup file in pickle format


    """

    pkl_file = open(path, 'wb')
    pickle.dump(varible, pkl_file)
    pkl_file.close()
