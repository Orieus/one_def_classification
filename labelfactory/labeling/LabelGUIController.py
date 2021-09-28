# -*- coding: utf-8 -*-
from __future__ import print_function
import webbrowser

from datetime import datetime
from functools import partial

# My own libraries
from .LabelViewGeneric import LabelViewGeneric

import sys
if sys.version_info.major == 3:
    import tkinter as tk
else:
    import Tkinter as tk

import ipdb


class LabelGUIController(object):

    """ Controls the logic of the labeling application.

        It contains the methods to listen the GUI and update it.
        In addition it takes the urls from the list one by one and opens a
        new browser window/tab when needed.

    """

    def __init__(self, newurls, newwids, newqueries, preds, labels, urls,
                 categories, alphabet, datatype='url', cat_model='single',
                 parent_cat={}, text2label=None):

        """ This method initialize the sampler object. As part of this process
        it creates the AL objects required for the sample generation.

        :Attributes:
            :newurls:    A list of urls to label
            :newwids:    Its corresponding wids
            :newqueries: The dict of queries to label
            :preds:      Dictionary containing, for each class and each wid,
                         the current predictions
            :labels:     Dict containing, for each wid, the new labels.
            :urls:       Dict containing, for each wid, its url
            :categories: List of categories.
            :alphabet:   Alphabet for positive, negative, etc, classes
            :cat_model:  Category model (single or multi -label)
        """

        ####################
        # Initial attributes

        # Attributes taken from the argument
        self.urls_AL = newurls
        self.wids_AL = newwids
        self.newlabels = newqueries
        self.preds = preds
        self.labels = labels

        # WARNING: self.urls is a list if datatype is 'url', but it is a
        # dictionary otherwise.
        self.urls = urls
        self.categories = categories
        self.parent_cat = parent_cat
        self.alphabet = alphabet
        self.datatype = datatype
        self.cat_model = cat_model

        # Initialization of other attributes.
        self.view = None
        self.text2label = text2label

    def takeandshow_sample(self):

        """ Gets next sample id from the list and visualize the sample.
            The type of visualization depends of the type of sample:
                if sample id is a url, a browser is opened.
                if sample id is not a url, data is printed.

            Note that the sample identifiers are stored in variable self.url
            for historical reasons. This variable does not necessarily stores
            urls.
        """

        self.url = None
        self.wid = None

        if self.urls_AL:  # We have urls to label in the list

            self.url = self.urls_AL.pop()
            self.wid = self.wids_AL.pop()

            if self.url:

                if self.datatype == 'url':
                    # Add the proper prefix to the given url
                    if (self.url[0:7] == 'http://' or
                            self.url[0:8] == 'https://'):
                        aux_url = self.url
                    elif self.url[0:4] == 'www.':
                        aux_url = 'http://' + self.url
                    else:
                        aux_url = 'http://www.' + self.url

                    webbrowser.open(aux_url, new=0)

                else:
                    print('---- **** LABEL THIS TEXT:')
                    print(self.url)
                    print('')

                # If this is the first url to label, there is no previous label
                # to update. But, if self.view is not None, the update method
                # is called.
                if self.view:
                    # self.view.master.wm_attributes("-topmost",1)
                    self.view.update()

        else:  # url list is empty

            # Update label vector
            # This in no longer needed here, because the method is invoked
            # in run_labeler.py
            # self.update_label_vector()

            # SIMPLY DESTROY THE LABELING WINDOW
            print("Ronda de etiquetado acabada.")
            self.view.master.destroy()

    def takeandshow_url(self, params=None):

        """ Gets next url in the list and opens the browser if needed
            This is the ancestor of takeandshow_sample. It is expected to be
            replaced by takeanshow_sample. Then it can be removed
        """

        self.url = None
        self.wid = None

        if self.urls_AL:  # We have urls to label in the list

            self.url = self.urls_AL.pop()
            self.wid = self.wids_AL.pop()

            if self.url:

                # Add the proper prefix to the given url
                if self.url[0:7] == 'http://' or self.url[0:8] == 'https://':
                    aux_url = self.url
                elif self.url[0:4] == 'www.':
                    aux_url = 'http://' + self.url
                else:
                    aux_url = 'http://www.' + self.url

                webbrowser.open(aux_url, new=0)

                # If this is the first url to label, there is no previous label
                # to update. But, if self.view is not None, the update method
                # is called.
                if self.view:
                    # self.view.master.wm_attributes("-topmost",1)
                    self.view.update()

        else:  # url list is empty

            # Update label vector
            # This in no longer needed here, because the method is invoked
            # in run_labeler.py
            # self.update_label_vector()

            # SIMPLY DESTROY THE LABELING WINDOW
            print("Ronda de etiquetado acabada.")
            self.view.master.destroy()

    def update_label_vector(self):

        """ Shows updated labels
            NOTE: The original version of this method updated the label vector
                  y. In the current version, the only goal is to remove the
                  nonlabeled urls from self.newlabels and print some results.
        """

        if self.newlabels:

            # Remove elements without labels
            # WARNING!!: DO NOT REMOVE .keys() BECAUSE self.newlabels IS
            #            MODIFIED IN THE LOOP!
            wid_list = list(self.newlabels.keys())
            for wid in wid_list:
                if 'label' not in self.newlabels[wid]:
                    del self.newlabels[wid]
                elif self.newlabels[wid]['label'] == []:
                    # No category has been assigned. The url is ignored
                    del self.newlabels[wid]

            # This is just to abbreviate the code below
            yes = self.alphabet['yes']
            no = self.alphabet['no']

            for wid in self.newlabels:

                # Note that self.urls is a dictionary of urls indexed by wids.
                # wid in self.urls checks if the wid is in the wids of the
                # dictionary
                if wid in self.urls:

                    # Identify categories with previous label
                    islabel = {}
                    for c in self.categories:
                        islabel[c] = wid in self.labels[c]

                    if any(islabel.values()):

                        if self.cat_model == 'single':
                            # Warn if the new label is in conflict with the old
                            # ones. If so, print the conflicting labels
                            newlabel = self.newlabels[wid]['label']
                            print("New label in {0}: {1}".format(wid,
                                  newlabel))
                            if newlabel != 'error' and islabel[newlabel]:
                                if self.labels[newlabel][wid] == no:
                                    print("--> In conflict with the old " +
                                          "labels: ", end="")
                                    for c in self.categories:
                                        if self.labels[c][wid] == yes:
                                            print("{0}, ".format(c), end="")
                                    print(" ")
                        elif self.cat_model == 'multi':
                            # Warn if the new label is in conflict with the old
                            # ones. If so, print the conflicting labels
                            print("New label in {0}: {1}".format(wid,
                                  self.newlabels[wid]['label']))
                            for newlabel in self.newlabels[wid]['label']:
                                if (newlabel in self.categories and
                                        islabel[newlabel]):
                                    if self.labels[newlabel][wid] == no:
                                        print("--> In conflict with the old " +
                                              "labels: ", end="")
                                        for c in self.categories:
                                            if self.labels[c][wid] == yes:
                                                print("{0}, ".format(c),
                                                      end="")
                                        print(" ")
                        else:
                            exit("---- Unknown category model {}".format(
                                 self.cat_model))

                        # Also, print the current (not None) predictions
                        flag = 0
                        for c in self.categories:
                            if wid in self.preds[c]:
                                if self.preds[c][wid] is not None:
                                    if flag == 0:
                                        print("Current (nonzero) scores are:")
                                        flag = 1
                                    if self.preds[c][wid] != 0:
                                        print("    {0}: {1}".format(
                                            c, self.preds[c][wid]))

                    # The new label replaces the old one in any case.
                    # print "Saving of labels in vector y omitted"
                    # self.y[idx] = self.newlabels[wid]['label']

                else:

                    print("{0} is not in dictionary".format(wid))

    #
    # EVENT HANDLING FUNCTIONS (Actualy executed in GUI thread,
    # KEEP THEM SIMPLE!!!!)
    #

    def init_view(self, root):

        """ Initializes GUI view

            In addition it bindes the Buttons with the callback methods.

            Args:
                option: parameter that serves to select the
                main label of the window. Can be "comercio" or "empleo"
                root: where to open the view
        """

        self.view = LabelViewGeneric(self.categories, master=root,
                                     cat_model=self.cat_model,
                                     parent_cat=self.parent_cat,
                                     datatype=self.datatype,
                                     text2label=self.text2label)

        # Set the action for all categories in list 'categories'
        for class_name in self.categories:
            self.view.mybuttons[class_name]["command"] = partial(
                self.labeling_event, class_name)

        # Set the action for the special category 'error'
        self.view.mybuttons['error']["command"] = partial(
            self.labeling_event, 'error')

        # Set the action for the special button 'end'
        if self.cat_model == 'multi':
            self.view.mybuttons['end']["command"] = partial(
                self.labeling_event, 'end')

        # Start the gui is anything to label if (url is None) does not start it
        self.view.start_gui(self.url)

    def labeling_event(self, class_name):

        """
        Defines the commands to execut after each button is pressed or
        unpressed.
        """

        if self.cat_model == 'single':

            # Save label
            self.newlabels[self.wid]['label'] = class_name
            self.newlabels[self.wid]['date'] = datetime.now()

            self.takeandshow_sample()

            # # Change GUI label
            if self.url:
                self.view.update_guilabel(self.url)
            # The following is no longer required, because if self.url is None,
            # then takeandshow_sample() destroys the GUI.
            # else:
            #     self.view.master.destroy()

        elif self.cat_model == 'multi':

            # If this is the first pushed button, initialize the label list
            if 'label' not in self.newlabels[self.wid]:
                self.newlabels[self.wid]['label'] = []

            relief = self.view.mybuttons[class_name]['relief']

            if class_name in self.categories:
                if relief == 'raised':

                    # If error button is pressed, unpress.
                    if self.view.mybuttons['error']['relief'] == 'sunken':
                        self.view.mybuttons['error'].config(
                            relief=tk.RAISED, text='error')

                    # Add label to the list of labeled categories
                    self.newlabels[self.wid]['label'].append(class_name)
                    self.view.mybuttons[class_name].config(
                        relief=tk.SUNKEN, text='['+class_name+']')
                elif relief == 'sunken':
                    # Unpress button and remove category from the list
                    self.newlabels[self.wid]['label'].remove(class_name)
                    self.view.mybuttons[class_name].config(
                        relief=tk.RAISED, text=class_name)
                else:
                    exit('---- OOPS: Unexpected relief value in button')

            elif class_name == 'error':

                if relief == 'raised':
                    # Unpress all buttons
                    for c in self.categories:
                        self.view.mybuttons[c].config(relief=tk.RAISED, text=c)

                    # Remove labels and mark as error
                    self.newlabels[self.wid]['label'] = ['error']
                    self.view.mybuttons[class_name].config(
                        relief=tk.SUNKEN, text='['+class_name+']')

                elif relief == 'sunken':
                    # Unpress button and remove category from the list
                    self.newlabels[self.wid]['label'].remove(class_name)
                    self.view.mybuttons[class_name].config(
                        relief=tk.RAISED, text=class_name)
                else:
                    exit('---- OOPS: Unexpected relief value in button')

            elif class_name == 'end':

                for c in self.categories:
                    self.view.mybuttons[c].config(relief=tk.RAISED, text=c)
                # Annotate the date.
                self.newlabels[self.wid]['date'] = datetime.now()
                # Go to the next sample to label.
                self.takeandshow_sample()

                # # Change GUI label
                if self.url:
                    self.view.update_guilabel(self.url)

            else:
                exit('---- Unknown button name')

        else:

            exit("---- ERROR: Unknown category model")
