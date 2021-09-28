#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Python libraries
import copy
import sys
import pandas as pd
import ipdb

# Local imports


class LabelProcessor(object):

    """
    LabelProcessor is the class providing processing facilities to manage the
    labeled dataset

    The class provides facilities to:

        - Check if the category structure is a tree.
        - Give the appropriate format to label variables
        - Transfer new labels and preds to the whole data structure
        - Infer labels based on the tree category structure.
    """

    def __init__(self, categories, parentcat, log,
                 alphabet={'yes': 1, 'no': -1, 'unknown': 0, 'error': -99},
                 cat_model='single'):

        """
        States the alphabet of the Labels

        Args:
            :categories: Set of categories
            :parentcat:  Dictionary defining the tree category structure
            :log:        Logger
            :alphabet:   Possible labels for each category
            :cat_model:  Category model (monolabel or multilabel)
        """

        self.categories = categories
        self.parentcat = parentcat
        self.log = log
        self._yes = alphabet['yes']
        self._no = alphabet['no']
        self._unknown = alphabet['unknown']
        self._error = alphabet['error']
        self.cat_model = cat_model
        self.alphabet = alphabet

        # Check consistency of the label processing
        if not self.isTree():
            sys.exit("The structure of parent categories is not a tree")

    def isTree(self):

        """ Verifies it the graph structure defined by the parentcat dictionary
            is a tree and has no loops

            :Args:
                :-: The dictionary of categories is an object attribute
                            parentcat[c] is the parent of category c.
            :Returns:
                :isTree:    True is the graph is a Tree.
                            False if the graph has any loop.

            .. note:: For large trees, this code can be made much more
                      efficient by removing checked paths from the graph before
                      entering the next outer-loop step.
        """

        cats = self.parentcat.keys()
        isTree = True

        # Start an ascending path for every category in cats
        for c in cats:

            # The last visied node
            cold = c

            # A record of visited nodes
            FlagCats = dict.fromkeys(cats, 0)

            while isTree:

                # Register the last visited node
                FlagCats[cold] = 1

                # Move one step upwards
                cnew = self.parentcat[cold]

                if cnew not in cats:
                    # cnew is a root node. No loops here.
                    break
                else:
                    if FlagCats[cnew] == 1:
                        isTree = False
                        break
                cold = cnew

            if not isTree:
                break

        return isTree

    def formatData(self, df_labels):

        """ This method replaces any None or any unrecognized values
            in the labels from df_labels by an 'unknown' label.
            Also, it creates metadata columns if they do not exist.
        """

        if len(df_labels) > 0:
            df_labels.loc[df_labels['label'] not in self.alphabet,
                          self.categories] = self._unknown

            if ('info', 'marker') not in df_labels.columns:
                df_labels['info', 'marker'] = None
            if ('info', 'relabel') not in df_labels.columns:
                df_labels['info', 'relabel'] = None
            if ('info', 'weight') not in df_labels.columns:
                df_labels['info', 'weight'] = None
            if ('info', 'userId') not in df_labels.columns:
                df_labels['info', 'userId'] = None
            if ('info', 'date') not in df_labels.columns:
                df_labels['info', 'date'] = None

        return df_labels

    def cleanLabels(self, df_labels):

        """ Removes non recognized label values in the labels dictionary
            by the "unknown_label" codeword.

            WARNING: This method is likely obsolete, because the label cleaning
            is already being done by formatData.
        """

        if len(df_labels) > 0:
            df_labels.loc[df_labels['label'] not in self.alphabet,
                          self.categories] = self._unknown

        return df_labels

    def transferData(self, data2, data, mode='project'):

        """ Transfer predictions and labels to the pool of data managed by the
            DataMgr object.

            :Inputs:
                :data2: data to transfer.
                :data: destination dataset
                :mode: Type of transfer:
                    - 'project' (default): Only wids already existing in data
                                are taken from data2.
                    - 'expand': data is expanded with the new wids
                                in data2.

            :Returns:
                :-: No variables are returned. The methods modifies input
                    mutable object 'data'.

            WARNING: THIS METHOD IS DEPRECATED. IT HAS BEEN REPLACES BY
                     transferLabels() and transferPreds()
        """

        # Check consistency and infer labels from the tree structure

        # Data fusion
        for wid in data2:

            if wid in data:

                # Make sure that the label information is complete
                data2[wid]['label'] = self.inferLabels(data2[wid]['label'])

                # Only non "None" predictions are transferred
                for cat in self.categories:
                    if data2[wid]['pred'][cat] is not None:
                        data[wid]['pred'][cat] = data2[wid]['pred'][cat]

                # Transfer labels.
                label = data[wid]['label']
                label2 = data2[wid]['label']
                data[wid]['label'] = self.updateLabels(label2, label)

                # Transfer marker and relabel. Note that if there are no
                # marker or relabel variables in the new data, the original
                # value is preserved.
                if 'marker' in data2[wid]:
                    data[wid]['marker'] = data2[wid]['marker']

                if 'relabel' in data2[wid]:
                    data[wid]['relabel'] = data2[wid]['relabel']

                # Transfer weight. It is not clear if weights should be
                # transferred. Probably not, but probably the original value is
                # no longer valid ...
                if 'weight' in data2[wid]:
                    data[wid]['weight'] = data2[wid]['weight']

                # Transfer the identity of the human labelers
                if 'userId' in data2[wid]:
                    data[wid]['userId'] = data2[wid]['userId']

            elif mode == 'expand':

                data[wid] = {}

                # Transfer url
                data[wid]['url'] = data2[wid]['url']

                # Make sure that the label information is complete
                data2[wid]['label'] = self.inferLabels(data2[wid]['label'])

                # Transfer labels.
                data[wid]['label'] = data2[wid]['label']

                # Transfer predictions
                data[wid]['pred'] = data2[wid]['pred']

                # Transfer the rest of dictionary entries, if they exist
                if 'marker' in data2[wid]:
                    data[wid]['marker'] = data2[wid]['marker']
                else:
                    data[wid]['marker'] = None

                if 'relabel' in data2[wid]:
                    data[wid]['relabel'] = data2[wid]['relabel']
                else:
                    data[wid]['relabel'] = None

                if 'weight' in data2[wid]:
                    data[wid]['weight'] = data2[wid]['weight']
                else:
                    data[wid]['weight'] = None

                if 'userId' in data2[wid]:
                    data[wid]['userId'] = data2[wid]['userId']
                else:
                    data[wid]['userId'] = None

    def transferPreds(self, df2_preds, df_preds, mode='project'):

        """ Transfer predictions to the prediction dataframe managed by the
            DataMgr object.

            :Inputs:
                :df2_preds: data to transfer.
                :df_preds: destination dataset
                :mode: Type of transfer:
                    - 'project' (default): Only wids already existing in
                                df_preds are taken from df_preds2.
                    - 'expand': df_preds is expanded with the new wids
                                in df_preds2.
                    - 'contract': Wids that do not exist in df_preds2 are
                                removed

            :Returns:
                :df_preds:
        """

        # Remove useless rows.
        # All urls in df_preds of wids in df2_preds are copied into df2_preds
        # Wids in df2_preds that remain without url a removed

        # Original wid sets and their intersection
        widset = set(df_preds.index.tolist())
        widset2 = set(df2_preds.index.tolist())
        w_common = list(widset & widset2)

        # Copy urls from df_preds into df2_preds
        df2_preds.loc[w_common, 'url'] = df_preds.loc[w_common, 'url']

        # Remove predictions about wids whose url is unknown
        df2_preds = df2_preds[~df2_preds['url'].isnull()]

        # Transfer predicts
        if mode == 'contract':

            if len(df2_preds) == 0:
                exit("Transfer mode = contract is not permitted if " +
                     "df2_preds does not contain urls")

            # Remove the instances from df_preds that are not in df2_preds
            df_preds = df_preds[df_preds.index.isin(df2_preds.index)]

        if mode == 'contract' or mode == 'expand':

            # Add to df_preds the rows in df2_preds that are not in df_preds.
            df_preds = pd.concat(
                [df_preds, df2_preds[~df2_preds.index.isin(df_preds.index)]])

        # Update the rows in df_preds with the new data in df2_preds
        df_preds.update(df2_preds)

        return df_preds

    def transferLabels(self, df2_labels, df_labels):

        """ Transfer labels to the label dataframe managed by the DataMgr
            object.

            :Inputs:
                :df2_labels: data to transfer.
                :df_labels: destination dataset

            :Returns:
                :df_labels:
        """

        # Check consistency and infer labels from the tree structure

        # Data fusion
        for w in df2_labels.index:

            if w in df_labels.index:

                # Make sure that the label information is complete
                newlabel = df2_labels.loc[w].to_dict()

                df2_labels.loc[w]['label'] = self.inferLabels(newlabel)

                # Transfer labels.
                label = df_labels.loc[w]['label'].to_dict()
                label2 = df2_labels.loc[w]['label'].to_dict()
                df_labels.loc[w]['label'] = self.updateLabels(label2, label)

                # Transfer marker and relabel. Note that if there are no
                # marker or relabel variables in the new data, the original
                # value is preserved.
                if 'marker' in df2_labels.loc[w]['info']:
                    df_labels.loc[w, ('info', 'marker')] = df2_labels.loc[w, (
                        'info', 'marker')]

                if 'relabel' in df2_labels.loc[w]['info']:
                    df_labels.loc[w, ('info', 'relabel')] = df2_labels.loc[w, (
                        'info', 'relabel')]

                # Transfer weight. It is not clear if weights should be
                # transferred. Probably not, but probably the original value is
                # no longer valid ...
                if 'weight' in df2_labels.loc[w]['info']:
                    df_labels.loc[w, ('info', 'weight')] = df2_labels.loc[w, (
                        'info', 'weight')]

                # Transfer the identity of the human labelers
                if 'userId' in df2_labels.loc[w]['info']:
                    df_labels.loc[w, ('info', 'userId')] = df2_labels.loc[w, (
                        'info', 'userId')]

                # Transfer the identity of the human labelers
                if 'date' in df2_labels.loc[w]['info']:
                    df_labels.loc[w, ('info', 'date')] = df2_labels.loc[w, (
                        'info', 'date')]

            else:

                # All wids in df2_labels that are not in df_labels are added to
                # df_labels

                # Infer labels for all categories in the new label set
                newlabel = df2_labels.loc[w].to_dict()
                df2_labels.loc[w]['label'] = self.inferLabels(newlabel)

                # Transfer labels and info to df_labels
                df_labels.loc[w] = df2_labels.loc[w]

        return df_labels

    def transferNewLabels2(self, newlabels, df_labels, new_userId=None):

        """ Transfer a new set of labels contained in the labeling record.

            It assumes precise labeling: the category assigned to a wid is the
            finest subcategory containing the wid. Thus, the new category and
            all its ancestors are marked with the positive label, and any other
            category is marked with the negative label

            :Args:
                :newlabels: Dictionary of pairs wid: cat, where cat is the
                            new category assigned to the wid.
                :df_labels: The current label dataframe
                :new_userId: Name of the current human labeler

            :Returns:
                :df_labels:
        """

        for wid in newlabels:

            # New category
            cat = newlabels[wid]['label']

            # Only labels in the category set are transferred to the data
            # structure. This means that label "error" is ignored.
            # In case of "error", the pre-existing label information (i.e.
            # label value, marker and relabel) about the corresponding wid is
            # not modified.
            # [Despite of this, error events will be recorded in history files
            # (see transferLabelRecords())]
            if cat in self.categories:

                if wid not in df_labels:
                    if ('info', 'url') in df_labels.columns:
                        df_labels.loc[wid, ('info', 'url')] = (
                            newlabels[wid]['url'])

                # Insert the category in a label vector
                label_dict = dict.fromkeys(self.categories, self._unknown)
                label_dict[cat] = self._yes

                # Propagate the positive label to all ancestors. Also, a
                # negative label to the other non-descendant categories
                label_dict = self.inferLabels(label_dict)

                # Up to this point, the descendants of the given category have
                # not been modified. Since labeling is precise, they should
                # receive a negative label.
                # The following code assigns the negative class to all non
                # positive classes. This may be inefficient for large trees,
                # because some categories have been already labeled with the
                # negative class in the above lines.
                for c in self.categories:
                    if label_dict[c] != self._yes:
                        label_dict[c] = self._no

                for c in self.categories:
                    df_labels.loc[wid, ('label', c)] = label_dict[c]

                # Old "soft labeling". Insert new label in the data structure.
                # data[wid]['label'] = self.updateLabels(label_dict,
                #                                        label_orig)

                # Store the marker and relabel
                df_labels.loc[wid, ('info', 'marker')] = newlabels[wid][
                    'marker']
                df_labels.loc[wid, ('info', 'relabel')] = newlabels[wid][
                    'relabel']
                df_labels.loc[wid, ('info', 'date')] = newlabels[wid]['date']
                df_labels.loc[wid, ('info', 'userId')] = new_userId

        return df_labels

    def transferNewLabels(self, newlabels, data, new_userId=None):

        """ Transfer a new set of labels contained in the labeling record.

            It assumes precise labeling: the category assigned to a wid is the
            finest subcategory containing the wid. Thus, the new category and
            all its ancestors are marked with the positive label, and any other
            category is marked with the negative label

            :Args:
                :newlabels: Dictionary of pairs wid: cat, where cat is the
                            new category assigned to the wid.
                :data: The current data dictionary

            :Returns:
                :-: No variables are returned. The method modifies input
                    mutable object 'data'.
        """

        for wid in newlabels:

            # New category
            cat = newlabels[wid]['label']

            # Only labels in the category set are transferred to the data
            # structure. This means that label "error" is ignored.
            # In case of "error", the pre-existing label information (i.e.
            # label value, marker and relabel) about the corresponding wid is
            # not modified.
            # [Despite of this, error events will be recorded in history files
            # (see transferLabelRecords())]
            if cat in self.categories:

                if wid not in data:
                    data[wid] = {}
                    data[wid]['url'] = newlabels[wid]['url']

                # Insert the category in a label vector
                label_dict = dict.fromkeys(self.categories, self._unknown)
                label_dict[cat] = self._yes

                # Propagate the positive label to all ancestors. Also, a
                # negative label to the other non-descendant categories
                label_dict = self.inferLabels(label_dict)

                # Up to this point, the descendants of the given category have
                # not been modified. Since labeling is precise, they should
                # receive a negative label.
                # The following code assigns the negative class to all non
                # positive classes. This may be inefficient for large trees,
                # because some categories have been already labeled with the
                # negative class in the above lines.
                for c in self.categories:
                    if label_dict[c] != self._yes:
                        label_dict[c] = self._no
                data[wid]['label'] = label_dict

                # Old "soft labeling". Insert new label in the data structure.
                # data[wid]['label'] = self.updateLabels(label_dict,
                #                                        label_orig)

                # Store the marker and relabel
                data[wid]['marker'] = newlabels[wid]['marker']
                data[wid]['relabel'] = newlabels[wid]['relabel']
                data[wid]['date'] = newlabels[wid]['date']
                data[wid]['userId'] = new_userId

    def transferNewWeights2(self, newweights, df_labels):

        """ Transfer a weight dictionary to the dataset.

            :Args:
                :newweights: Dictionary of pairs wid: weight.
                :df_labels: The current label dataframe
                            Only the wids that are both in df_labels and
                            newweights are processed

            :Returns:
                :df_labels:
        """

        for wid in newweights:
            if wid in df_labels.index:
                df_labels.loc[wid, ('info', 'weight')] = newweights[wid]

        return df_labels

    def transferNewWeights(self, newweights, data):

        """ Transfer a weight dictionary to the dataset.

            :Args:
                :newweights: Dictionary of pairs wid: weight.
                :data: The current data dictionary.
                       Only the wids that are both in data and newweights are
                       processed

            :Returns:
                :-: No variables are returned. The method modifies input
                    mutable object 'data'.
        """

        for wid in newweights:
            if wid in data:
                data[wid]['weight'] = newweights[wid]

    def transferLabelRecords(self, newlabels, labelhistory, new_userId=None):

        """ Transfer a new set of labeling records contained in newlabels
            to the current data set.
        """

        for wid in newlabels:

            t = newlabels[wid]['date']
            tid = t.strftime("%Y%m%d%H%M%S%f")

            if wid not in labelhistory:
                labelhistory[wid] = {tid: newlabels[wid]}
            else:
                labelhistory[wid][tid] = newlabels[wid]

            # Add user Id.
            labelhistory[wid][tid]['userId'] = new_userId

    def updateLabels(self, label2, label):

        """ Updates label dictionary with the information provided by label2

            - If at least one category in label2 is ERROR, all categories are
              set to UNKNOWN
            - Any YES label in label2 is propagated through parent categories.
              Siblins of a node from category YES are set to zero.
              (i.e., sibling categories are assumed to be mutually exclusive)
            - Any NO label in label2 is propagated through child categories

            If the resulting label dictionary is not consistent (e.g. if all
            label components are NO), all categories are marked as UNKNOWN
        """

        # If thereis and "ERROR" label mark, all labels are removed.
        if any(label2[c] == self._error for c in self.categories):
            new_label = dict.fromkeys(label2, self._unknown)
        else:

            new_label = copy.copy(label2)

            # Preserve the old labels only if the new label is unknown
            for cat in self.categories:
                if new_label[cat] == self._unknown:
                    new_label[cat] = label[cat]

        label_out = self.inferLabels(new_label)

        if all(label_out[c] == self._unknown for c in self.categories):
            # self.inferLabels may have returned an 'all unknown' label
            # dictionary because all the input labels where 'no_label' (which
            # is inconsistent with the fact that all categories in the first
            # level of the tree are exhaustive).
            # For this reason, new_label is replaced by a copy of label2.
            label_out = copy.copy(label2)

        return label_out

    def inferLabels(self, label):

        """ Infer values of some label components based on the information
            of the observed labels and the category tree structure.
        """

        if any(label[c] == self._error for c in self.categories):
            # Any error in a label component invalidate the label
            # information
            new_label = dict.fromkeys(label, self._unknown)

        else:

            new_label = copy.copy(label)

            # Propagate labels
            for cat in self.categories:

                if new_label[cat] == self._yes:
                    new_label = self.upwards(new_label, cat)[0]
                elif new_label[cat] == self._no:
                    new_label = self.downwards(new_label, cat)[0]

        if all(label[c] == self._no for c in self.categories):
            # The category set is assumed to be exhaustive.
            # Therefore, if all values are NO, the labeling is not
            # consistent, and the label information must be removed
            new_label = dict.fromkeys(label, self._unknown)

        return new_label

    def upwards(self, label, cat):

        """ Propagates a label value "yes" up through the tree structure
        """

        isOK = True
        pcat = self.parentcat[cat]
        new_label = copy.copy(label)

        # Go up through the tree
        if pcat in self.categories:

            if new_label[pcat] == self._no:
                # Inconsistency. Return unknown to relabel
                isOK = False
                new_label = dict.fromkeys(new_label, self.unknown)
            elif new_label[pcat] == self._unknown:
                # Recursive call
                new_label[pcat] = self._yes
                new_label, isOK = self.upwards(new_label, pcat)

        if isOK:
            # Propagate a "NO" through the sibling categories
            for c in self.findSiblings(new_label, cat):
                if new_label[c] == self._yes:
                    # Labeling is inconsistent, because we are assuming
                    # that sibling categories are mutually exclusive
                    isOK = False
                    new_label = dict.fromkeys(new_label, self._unknown)
                    break
                else:
                    new_label[c] = self._no
                    new_label, isOK = self.downwards(new_label, c)
                    if not isOK:
                        break

        return new_label, isOK

    def downwards(self, label, cat):

        """ Propagates a label value "no" down through the tree structure
        """

        isOK = True
        new_label = copy.copy(label)

        for c in self.categories:

            pc = self.parentcat[c]

            # If c is a child of cat...
            if pc == cat:
                if new_label[c] == self._yes:
                    # Inconsistency. Return unknown to relabel
                    isOK = False
                    new_label = dict.fromkeys(new_label, self._unknown)
                    break
                elif new_label[c] == self._unknown:
                    new_label[c] = self._no
                    new_label, isOK = self.downwards(new_label, c)
                    if not isOK:
                        break

        return new_label, isOK

    def findSiblings(self, label, cat):

        """ Finds the siblins of node 'cat' in the label tree structure
            (two nodes are siblings if they share the same parent)
        """

        pc = self.parentcat[cat]
        sib = [c for c in self.categories
               if self.parentcat[c] == pc and c != cat]

        return sib
